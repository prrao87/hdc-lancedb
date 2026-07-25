from __future__ import annotations

import pytest
import torch
import torchhd

from hdc.core import TorchHDEncoder


@pytest.fixture
def encoder() -> TorchHDEncoder:
    return TorchHDEncoder()


def test_role_value_association_is_bipolar(encoder: TorchHDEncoder) -> None:
    association = encoder.association_hv("feature", "mountains")
    assert set(torch.unique(association).tolist()) == {-1.0, 1.0}


def test_raw_bundle_supports_exact_additive_removal(
    encoder: TorchHDEncoder,
) -> None:
    mountains = encoder.association_hv("feature", "mountains")
    coast = encoder.association_hv("feature", "pacific_coast")
    bundle = encoder.bundle([mountains, coast])
    recovered = torchhd.bundle(bundle, torchhd.negative(mountains))
    assert torch.equal(recovered, coast)


def test_symbols_are_stable_without_a_vocabulary(
    encoder: TorchHDEncoder,
) -> None:
    first = encoder.token_hv("value:Seattle").clone()
    unseen = encoder.token_hv("value:A previously unseen symbol")
    second = TorchHDEncoder().token_hv("value:Seattle")

    assert torch.equal(first, second)
    assert not torch.equal(first, unseen)


def test_tie_breaking_is_bipolar_and_context_deterministic(
    encoder: TorchHDEncoder,
) -> None:
    zero_bundle = torchhd.MAPTensor.empty(1, encoder.dimensions)[0]
    first = encoder.normalize_for_binding(zero_bundle, "Location:Seattle")
    again = encoder.normalize_for_binding(zero_bundle, "Location:Seattle")
    other = encoder.normalize_for_binding(zero_bundle, "Location:Portland")

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert set(torch.unique(first).tolist()) == {-1.0, 1.0}


def test_composite_nodes_unbind_exactly_from_triple(
    encoder: TorchHDEncoder,
) -> None:
    subject = encoder.encode_properties(
        {"name": "Seattle", "role": "designer"}
    )
    object_hv = encoder.bundle(
        [
            encoder.association_hv("feature", "mountains"),
            encoder.association_hv("feature", "pacific_coast"),
        ]
    )
    predicate = encoder.predicate_hv("VISITED")
    triple = encoder.encode_triple(
        subject,
        predicate,
        object_hv,
        subject_context="Person:test",
        object_context="Location:test",
    )
    subject_factor = encoder.normalize_for_binding(subject, "Person:test")
    recovered_object = torchhd.multibind(
        torch.stack([triple, subject_factor, predicate])
    )
    expected_object = encoder.normalize_for_binding(
        object_hv,
        "Location:test",
    )

    assert torch.equal(recovered_object, expected_object)
