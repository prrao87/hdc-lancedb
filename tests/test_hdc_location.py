from __future__ import annotations

import pytest
import torch
import torchhd

from hdc.core import TorchHDEncoder
from hdc.location import LocationNodeEncoder


@pytest.fixture
def base_encoder() -> TorchHDEncoder:
    return TorchHDEncoder(dimensions=1024)


@pytest.fixture
def properties() -> dict[str, str]:
    return {
        "id": "location-seattle",
        "name": "Seattle",
        "kind": "Location",
        "region": "pacific_northwest",
        "timezone": "pacific",
    }


def test_full_vector_is_structural_plus_weighted_semantic_superposition(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    semantic_hv = base_encoder.token_hv("test-semantic:seattle")
    result = LocationNodeEncoder(base_encoder).encode_node(
        properties,
        semantic_hv,
    )
    expected = base_encoder.encode_properties(properties) + 8 * semantic_hv

    assert torch.equal(result, expected)


def test_structural_and_semantic_changes_change_the_full_vector(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    location_encoder = LocationNodeEncoder(base_encoder)
    semantic_hv = base_encoder.token_hv("test-semantic:seattle")
    original = location_encoder.encode_node(properties, semantic_hv)
    changed_properties = location_encoder.encode_node(
        {**properties, "timezone": "mountain"},
        semantic_hv,
    )
    changed_semantics = location_encoder.encode_node(
        properties,
        base_encoder.token_hv("test-semantic:salt-lake-city"),
    )

    assert not torch.equal(original, changed_properties)
    assert not torch.equal(original, changed_semantics)


def test_encoding_is_deterministic_across_fresh_encoders(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    first = LocationNodeEncoder(base_encoder).encode_node(
        properties,
        base_encoder.token_hv("test-semantic:seattle"),
    )
    fresh_base = TorchHDEncoder(dimensions=1024)
    second = LocationNodeEncoder(fresh_base).encode_node(
        properties,
        fresh_base.token_hv("test-semantic:seattle"),
    )

    assert torch.equal(first, second)


def test_semantic_vector_must_be_dimensionally_compatible(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="1024 values"):
        LocationNodeEncoder(base_encoder).encode_node(
            properties,
            torch.ones(32, dtype=torch.float32).as_subclass(torchhd.MAPTensor),
        )


def test_semantic_vector_must_use_float32(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    semantic_hv = base_encoder.token_hv("test-semantic:seattle")
    with pytest.raises(TypeError, match="float32"):
        LocationNodeEncoder(base_encoder).encode_node(
            properties,
            semantic_hv.to(dtype=torch.float64),
        )


def test_semantic_vector_must_be_bipolar(
    base_encoder: TorchHDEncoder,
    properties: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="bipolar"):
        LocationNodeEncoder(base_encoder).encode_node(
            properties,
            torch.zeros(1024, dtype=torch.float32).as_subclass(
                torchhd.MAPTensor
            ),
        )


def test_semantic_weight_must_be_positive(
    base_encoder: TorchHDEncoder,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        LocationNodeEncoder(base_encoder, semantic_weight=0)
