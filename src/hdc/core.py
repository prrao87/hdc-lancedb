from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torchhd


DIMENSIONS = 10_000
SEED = 13
# Stored vectors and lazy image bytes are not structural graph properties.
NON_SYMBOLIC_COLUMNS = {"hv", "image"}


@dataclass(frozen=True)
class TorchHDEncoder:
    """Vocabulary-free wrapper around TorchHD primitives for graph records.

    Atomic structural symbols are generated independently from stable hashes,
    so adding a previously unseen token never requires rebuilding a vocabulary
    or changes any existing token's hypervector.

    Atomic MAP hypervectors and bound role/value associations are bipolar.
    Bundling retains the full-precision additive sum so associations can be
    added or removed exactly. Composite sums are projected back into bipolar
    MAP space only when they become factors in a larger binding operation.
    """

    dimensions: int = DIMENSIONS
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("TorchHD dimensions must be positive")
        object.__setattr__(self, "_token_cache", {})

    def token_hv(self, token: str) -> torchhd.MAPTensor:
        """Return a hash-derived bipolar hypervector for any symbolic token."""
        if not token:
            raise ValueError("Symbolic tokens must be non-empty")
        cached = self._token_cache.get(token)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._context_seed(f"token:{token}"))
        hypervector = torchhd.random(
            1,
            self.dimensions,
            "MAP",
            generator=generator,
            dtype=torch.float32,
        )[0]
        self._token_cache[token] = hypervector
        return hypervector

    def association_hv(self, role: str, value: str) -> torchhd.MAPTensor:
        """Bind a reusable role to one value, e.g. feature -> mountains."""
        return torchhd.bind(
            self.token_hv(f"key:{role}"),
            self.token_hv(f"value:{value}"),
        )

    def encode_properties(self, properties: Mapping[str, str]) -> torchhd.MAPTensor:
        """Encode a record as a raw sum of bound key/value associations."""
        sorted_items = sorted(
            (key, value)
            for key, value in properties.items()
            if key not in NON_SYMBOLIC_COLUMNS
        )
        return self.bundle(
            [self.association_hv(key, value) for key, value in sorted_items]
        )

    def bundle(self, hypervectors: Sequence[torchhd.MAPTensor]) -> torchhd.MAPTensor:
        """Bundle hypervectors as a full-precision, exactly updateable sum."""
        if not hypervectors:
            raise ValueError("Cannot bundle an empty hypervector sequence")

        # Keep this as the raw MAP superposition. Normalizing here would throw
        # away counts and make additive removal only approximate. Node-table
        # rows are accumulators: callers may add A with bundle(sum, A), or
        # remove it exactly with bundle(sum, negative(A)). Projection back to
        # bipolar {-1, +1} belongs at the later binding boundary below.
        return torchhd.multiset(torch.stack(list(hypervectors)))

    def tie_breaker_hv(self, context: str) -> torchhd.MAPTensor:
        """Return a deterministic random bipolar vector for zero-coordinate ties."""
        if not context:
            raise ValueError("Tie-breaking context must be non-empty")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._context_seed(f"tie:{context}"))
        return torchhd.random(
            1,
            self.dimensions,
            "MAP",
            generator=generator,
            dtype=torch.float32,
        )[0]

    def _context_seed(self, context: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{context}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    def normalize_for_binding(
        self,
        hypervector: torchhd.MAPTensor,
        context: str,
    ) -> torchhd.MAPTensor:
        """Project a sum into bipolar MAP space before using it as a factor.

        MAP's multiplication is self-inverse only for bipolar factors. Zero
        coordinates are resolved with a context-specific deterministic random
        vector instead of TorchHD's systematic zero -> -1 normalization.
        """
        tie_breaker = self.tie_breaker_hv(context).to(
            dtype=hypervector.dtype,
            device=hypervector.device,
        )
        positive = torch.tensor(1.0, dtype=hypervector.dtype, device=hypervector.device)
        negative = torch.tensor(-1.0, dtype=hypervector.dtype, device=hypervector.device)
        return torch.where(
            hypervector > 0,
            positive,
            torch.where(hypervector < 0, negative, tie_breaker),
        )

    def predicate_hv(self, predicate: str) -> torchhd.MAPTensor:
        """Encode a relationship type as its own symbolic hypervector."""
        return self.token_hv(f"predicate:{predicate}")

    def encode_triple(
        self,
        subject_hv: torchhd.MAPTensor,
        predicate_hv: torchhd.MAPTensor,
        object_hv: torchhd.MAPTensor,
        *,
        subject_context: str,
        object_context: str,
    ) -> torchhd.MAPTensor:
        """Normalize composite nodes on demand, then bind one S-P-O path."""
        subject_factor = self.normalize_for_binding(subject_hv, subject_context)
        object_factor = self.normalize_for_binding(object_hv, object_context)

        # A relationship-table vector is a different lifecycle stage from a
        # node-table sum. At this boundary the composite subject/object must be
        # bipolar so multiplying by the known factors recovers the remaining
        # factor exactly. Stable row identities supply the tie-break contexts.
        return torchhd.multibind(
            torch.stack([subject_factor, predicate_hv, object_factor])
        )
