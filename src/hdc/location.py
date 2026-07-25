"""Combine structural properties and semantic evidence into one Location vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torchhd

from hdc.core import TorchHDEncoder


DEFAULT_SEMANTIC_WEIGHT = 8


@dataclass(frozen=True)
class LocationNodeEncoder:
    """Encode a complete Location row as one full node hypervector.

    Structural properties and semantic evidence are transient components. Their
    raw superposition is the only vector returned and stored. Repeating the
    bipolar semantic vector gives natural-language similarity enough influence
    to survive alongside the structural property associations.
    """

    encoder: TorchHDEncoder
    semantic_weight: int = DEFAULT_SEMANTIC_WEIGHT

    def __post_init__(self) -> None:
        if self.semantic_weight <= 0:
            raise ValueError("Semantic weight must be positive")

    def encode_node(
        self,
        properties: Mapping[str, str],
        semantic_hv: torchhd.MAPTensor,
    ) -> torchhd.MAPTensor:
        location_id = properties.get("id")
        if not location_id:
            raise ValueError("Location properties must include a non-empty id")
        if semantic_hv.ndim != 1 or semantic_hv.numel() != self.encoder.dimensions:
            raise ValueError(
                "Location semantic hypervector must be one-dimensional with "
                f"{self.encoder.dimensions} values"
            )
        if semantic_hv.dtype != torch.float32:
            raise TypeError("Location semantic hypervector must use float32")
        if not torch.all((semantic_hv == -1.0) | (semantic_hv == 1.0)):
            raise ValueError(
                "Location semantic hypervector must be bipolar {-1, +1}"
            )

        structural_hv = self.encoder.encode_properties(properties)
        return structural_hv + self.semantic_weight * semantic_hv
