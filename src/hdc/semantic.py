from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from ollama import Client, ResponseError
import torch
import torch.nn.functional as F
import torchhd

from hdc.core import DIMENSIONS, SEED


DOCUMENT_TEMPLATE_VERSION = 1
PROJECTION_TYPE = "rademacher-sign-v1"


class OllamaEmbeddingError(RuntimeError):
    """Raised when the local Ollama embedding service cannot satisfy a request."""


@dataclass(frozen=True)
class OllamaEmbedder:
    """Small batched client for Ollama's local embedding API."""

    model: str
    host: str
    timeout_seconds: float = 60.0
    _client: Client = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            Client(host=self.host, timeout=self.timeout_seconds),
        )

    @property
    def canonical_model(self) -> str:
        return self.model if ":" in self.model else f"{self.model}:latest"

    def model_digest(self) -> str:
        """Return the installed model digest so encoded data can reject model drift."""
        try:
            models = self._client.list().models
        except ResponseError as error:
            raise OllamaEmbeddingError(
                f"Ollama could not list local models: {error.error}"
            ) from error
        except ConnectionError as error:
            raise self._connection_error() from error

        for model in models:
            if model.model == self.canonical_model:
                digest = model.digest
                if not digest:
                    break
                return str(digest)
        raise OllamaEmbeddingError(
            f"Ollama model {self.canonical_model!r} is not installed at {self.host!r}"
        )

    def _embed(self, texts: Sequence[str], task: str) -> torch.Tensor:
        if not texts:
            raise ValueError("At least one text is required for embedding")
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding texts must be non-empty")

        try:
            result = self._client.embed(
                model=self.canonical_model,
                input=[f"{task}: {text}" for text in texts],
                truncate=False,
            )
        except ResponseError as error:
            raise OllamaEmbeddingError(
                f"Ollama embedding request failed: {error.error}"
            ) from error
        except ConnectionError as error:
            raise self._connection_error() from error

        values = result.embeddings
        if len(values) != len(texts):
            raise OllamaEmbeddingError(
                "Ollama returned an unexpected number of embedding vectors"
            )

        embeddings = torch.tensor(values, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise OllamaEmbeddingError("Ollama returned malformed embedding vectors")
        if not torch.isfinite(embeddings).all():
            raise OllamaEmbeddingError("Ollama returned non-finite embedding values")
        return F.normalize(embeddings, dim=1)

    def embed_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> torch.Tensor:
        return self._embed([text], "search_query")[0]

    def _connection_error(self) -> OllamaEmbeddingError:
        return OllamaEmbeddingError(
            f"Cannot reach Ollama at {self.host!r}; start Ollama and ensure "
            f"{self.canonical_model!r} is installed"
        )


@dataclass(frozen=True)
class RandomHyperplaneProjector:
    """Project normalized embeddings into deterministic bipolar MAP space."""

    matrix: torch.Tensor
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("Projection matrix must be two-dimensional")
        if self.matrix.dtype != torch.int8:
            raise TypeError("Projection matrix must use int8 Rademacher values")
        if set(torch.unique(self.matrix).tolist()) != {-1, 1}:
            raise ValueError("Projection matrix values must be exactly {-1, +1}")

    @property
    def input_dimensions(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def output_dimensions(self) -> int:
        return int(self.matrix.shape[1])

    @classmethod
    def create(
        cls,
        input_dimensions: int,
        output_dimensions: int = DIMENSIONS,
        seed: int = SEED,
    ) -> RandomHyperplaneProjector:
        if input_dimensions <= 0 or output_dimensions <= 0:
            raise ValueError("Projection dimensions must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        matrix = torch.randint(
            0,
            2,
            (input_dimensions, output_dimensions),
            generator=generator,
            dtype=torch.int8,
        )
        matrix.mul_(2).sub_(1)
        return cls(matrix=matrix, seed=seed)

    def project(self, embeddings: torch.Tensor) -> torchhd.MAPTensor:
        single = embeddings.ndim == 1
        batch = embeddings.unsqueeze(0) if single else embeddings
        if batch.ndim != 2 or batch.shape[1] != self.input_dimensions:
            raise ValueError(
                "Embedding dimension does not match projection matrix: "
                f"{tuple(batch.shape)} vs. {self.input_dimensions}"
            )
        batch = batch.to(dtype=torch.float32)
        projected = batch @ self.matrix.to(
            dtype=torch.float32,
            device=batch.device,
        )
        bipolar = torch.where(
            projected >= 0,
            torch.tensor(1.0, dtype=torch.float32, device=batch.device),
            torch.tensor(-1.0, dtype=torch.float32, device=batch.device),
        )
        result = torchhd.ensure_vsa_tensor(bipolar, vsa="MAP", dtype=torch.float32)
        return result[0] if single else result

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.matrix.cpu().numpy(), allow_pickle=False)
        return file_sha256(path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        seed: int = SEED,
        expected_sha256: str | None = None,
    ) -> RandomHyperplaneProjector:
        if expected_sha256 is not None:
            actual_sha256 = file_sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Projection artifact checksum mismatch: {actual_sha256} "
                    f"!= {expected_sha256}"
                )
        values = np.load(path, allow_pickle=False)
        return cls(torch.from_numpy(values), seed=seed)


@dataclass(frozen=True)
class SemanticHDCMetadata:
    model: str
    model_digest: str
    embedding_dimensions: int
    hdc_dimensions: int
    projection_seed: int
    projection_type: str
    projection_file: str
    projection_sha256: str
    semantic_weight: int
    document_template_version: int = DOCUMENT_TEMPLATE_VERSION

    def __post_init__(self) -> None:
        if self.embedding_dimensions <= 0 or self.hdc_dimensions <= 0:
            raise ValueError("Semantic HDC dimensions must be positive")
        if self.semantic_weight <= 0:
            raise ValueError("Semantic weight must be positive")

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> SemanticHDCMetadata:
        return cls(**json.loads(path.read_text()))

    def validate_runtime(
        self,
        *,
        model: str,
        model_digest: str,
        embedding_dimensions: int | None = None,
    ) -> None:
        if model != self.model:
            raise ValueError(f"Embedding model mismatch: {model!r} != {self.model!r}")
        if model_digest != self.model_digest:
            raise ValueError(
                "Embedding model digest changed; rebuild the HDC vectors before querying"
            )
        if (
            embedding_dimensions is not None
            and embedding_dimensions != self.embedding_dimensions
        ):
            raise ValueError(
                "Embedding dimension changed: "
                f"{embedding_dimensions} != {self.embedding_dimensions}"
            )
        if self.projection_type != PROJECTION_TYPE:
            raise ValueError(
                f"Unsupported projection type: {self.projection_type!r}"
            )
        if self.document_template_version != DOCUMENT_TEMPLATE_VERSION:
            raise ValueError(
                "Semantic document template changed; rebuild the HDC vectors"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
