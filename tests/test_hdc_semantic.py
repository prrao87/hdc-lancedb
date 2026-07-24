from __future__ import annotations

from pathlib import Path
from typing import Any

from ollama import EmbedResponse
import pytest
import torch
import torch.nn.functional as F

from hdc.semantic import (
    OllamaEmbedder,
    RandomHyperplaneProjector,
    SemanticHDCMetadata,
)


def test_embeddings_are_task_prefixed_and_normalized_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        host="http://localhost:11434",
    )
    request: dict[str, Any] = {}

    def embed(**kwargs: Any) -> EmbedResponse:
        request.update(kwargs)
        return EmbedResponse(embeddings=[[3.0, 4.0]])

    monkeypatch.setattr(embedder._client, "embed", embed)
    result = embedder.embed_query("mountainous Pacific cities")

    assert result.dtype == torch.float32
    assert float(torch.linalg.vector_norm(result)) == pytest.approx(1.0)
    assert request == {
        "model": "nomic-embed-text:latest",
        "input": ["search_query: mountainous Pacific cities"],
        "truncate": False,
    }


def test_projection_is_deterministic_bipolar_and_preserves_order() -> None:
    projector = RandomHyperplaneProjector.create(
        input_dimensions=3,
        output_dimensions=4096,
        seed=13,
    )
    embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    projected = projector.project(embeddings)

    assert projected.dtype == torch.float32
    assert set(torch.unique(projected).tolist()) == {-1.0, 1.0}
    close = F.cosine_similarity(projected[0], projected[1], dim=0)
    far = F.cosine_similarity(projected[0], projected[2], dim=0)
    assert float(close) > float(far)
    assert torch.equal(projected, projector.project(embeddings))


def test_projection_artifact_round_trips_with_checksum(
    tmp_path: Path,
) -> None:
    projector = RandomHyperplaneProjector.create(
        input_dimensions=4,
        output_dimensions=32,
        seed=7,
    )
    path = tmp_path / "projection.npy"
    checksum = projector.save(path)
    loaded = RandomHyperplaneProjector.load(
        path,
        seed=7,
        expected_sha256=checksum,
    )

    assert torch.equal(projector.matrix, loaded.matrix)


def test_model_digest_drift_is_rejected() -> None:
    metadata = SemanticHDCMetadata(
        model="nomic-embed-text:latest",
        model_digest="original",
        embedding_dimensions=768,
        hdc_dimensions=10_000,
        projection_seed=13,
        projection_type="rademacher-sign-v1",
        projection_file="projection.npy",
        projection_sha256="checksum",
        semantic_weight=8,
    )

    with pytest.raises(ValueError, match="digest changed"):
        metadata.validate_runtime(
            model="nomic-embed-text:latest",
            model_digest="changed",
        )


def test_semantic_weight_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Semantic weight"):
        SemanticHDCMetadata(
            model="nomic-embed-text:latest",
            model_digest="digest",
            embedding_dimensions=768,
            hdc_dimensions=10_000,
            projection_seed=13,
            projection_type="rademacher-sign-v1",
            projection_file="projection.npy",
            projection_sha256="checksum",
            semantic_weight=0,
        )
