from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torchhd


DIMENSIONS = 10_000
SEED = 13
VECTOR_COLUMNS = {"hv", "vibe_hv"}
# Raw multimodal assets (image bytes, …) live in the table but are not symbolic
# tokens, so they are skipped when building property bags and the vocabulary.
BLOB_COLUMNS = {"image"}
NON_SYMBOLIC_COLUMNS = VECTOR_COLUMNS | BLOB_COLUMNS


@dataclass(frozen=True)
class TorchHDEncoder:
    """Thin wrapper around TorchHD primitives for symbolic graph records.

    Atomic MAP hypervectors and bound role/value associations are bipolar.
    Bundling retains the full-precision additive sum so associations can be
    added or removed exactly. Composite sums are projected back into bipolar
    MAP space only when they become factors in a larger binding operation.
    """

    tokens: Sequence[str]
    dimensions: int = DIMENSIONS
    seed: int = SEED

    def __post_init__(self) -> None:
        token_to_index = {token: index for index, token in enumerate(self.tokens)}
        if len(token_to_index) != len(self.tokens):
            raise ValueError("TorchHDEncoder tokens must be unique")

        torch.manual_seed(self.seed)
        embedding = torchhd.embeddings.Random(
            len(self.tokens),
            self.dimensions,
            vsa="MAP",
        )

        object.__setattr__(self, "token_to_index", token_to_index)
        object.__setattr__(self, "embedding", embedding)

    def token_hv(self, token: str) -> torchhd.MAPTensor:
        """Return the deterministic random hypervector for one symbolic token."""
        return self.embedding.weight[self.token_to_index[token]]

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

    def encode_vibe_terms(self, terms: Mapping[str, str]) -> torchhd.MAPTensor:
        """Encode one evidence row as bound role/value associations.

        The feature association is repeated to preserve the demo's deliberate
        weighting of semantic content over provenance and strength metadata.
        """
        associations = []
        for role, value in sorted(terms.items()):
            association = self.association_hv(role, value)
            repetitions = 4 if role == "feature" else 1
            associations.extend([association] * repetitions)
        return self.bundle(associations)

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
        digest = hashlib.sha256(f"{self.seed}:{context}".encode()).digest()
        context_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device=self.embedding.weight.device)
        generator.manual_seed(context_seed)
        return torchhd.random(
            1,
            self.dimensions,
            "MAP",
            generator=generator,
            dtype=self.embedding.weight.dtype,
            device=self.embedding.weight.device,
        )[0]

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


def build_vocabulary(
    persons: Iterable[Mapping[str, str]],
    locations: Iterable[Mapping[str, str]],
    relationships: Iterable[Mapping[str, str]],
    predicate: str,
    location_vibes: Iterable[Mapping[str, str]] = (),
) -> list[str]:
    """Create the full symbolic vocabulary before random HVs are initialized."""
    tokens = {f"predicate:{predicate}"}
    for row in [*persons, *locations, *relationships]:
        for key, value in row.items():
            if key in NON_SYMBOLIC_COLUMNS:
                continue
            tokens.add(f"key:{key}")
            tokens.add(f"value:{value}")
    for vibe in location_vibes:
        for role in ("source", "feature", "strength"):
            tokens.add(f"key:{role}")
            tokens.add(f"value:{vibe[role]}")
    return sorted(tokens)


def hv_to_list(hv: torchhd.MAPTensor) -> list[float]:
    """Convert a TorchHD hypervector to a LanceDB-friendly float32 list."""
    return hv.detach().to(dtype=torch.float32).cpu().tolist()


def list_to_hv(values: list[float]) -> torchhd.MAPTensor:
    """Convert a stored LanceDB vector back into a TorchHD MAP tensor."""
    return torchhd.ensure_vsa_tensor(values, vsa="MAP", dtype=torch.float32)
