from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torchhd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torchhd_encoder import TorchHDEncoder


class TorchHDEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = TorchHDEncoder(
            sorted(
                [
                    "key:feature",
                    "key:name",
                    "key:role",
                    "predicate:VISITED",
                    "value:Seattle",
                    "value:designer",
                    "value:mountains",
                    "value:pacific_coast",
                ]
            )
        )

    def test_role_value_association_is_bipolar(self) -> None:
        association = self.encoder.association_hv("feature", "mountains")
        self.assertEqual(set(torch.unique(association).tolist()), {-1.0, 1.0})

    def test_raw_bundle_supports_exact_additive_removal(self) -> None:
        mountains = self.encoder.association_hv("feature", "mountains")
        coast = self.encoder.association_hv("feature", "pacific_coast")
        bundle = self.encoder.bundle([mountains, coast])
        recovered = torchhd.bundle(bundle, torchhd.negative(mountains))
        self.assertTrue(torch.equal(recovered, coast))

    def test_vibe_feature_weight_is_retained_in_raw_sum(self) -> None:
        mountains = self.encoder.association_hv("feature", "mountains")
        encoded = self.encoder.encode_vibe_terms({"feature": "mountains"})
        self.assertTrue(torch.equal(encoded, mountains * 4))

    def test_tie_breaking_is_bipolar_and_context_deterministic(self) -> None:
        zero_bundle = torchhd.MAPTensor.empty(1, self.encoder.dimensions)[0]
        first = self.encoder.normalize_for_binding(zero_bundle, "Location:Seattle")
        again = self.encoder.normalize_for_binding(zero_bundle, "Location:Seattle")
        other = self.encoder.normalize_for_binding(zero_bundle, "Location:Portland")
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(set(torch.unique(first).tolist()), {-1.0, 1.0})

    def test_composite_nodes_unbind_exactly_from_triple(self) -> None:
        subject = self.encoder.encode_properties(
            {"name": "Seattle", "role": "designer"}
        )
        object_hv = self.encoder.bundle(
            [
                self.encoder.association_hv("feature", "mountains"),
                self.encoder.association_hv("feature", "pacific_coast"),
            ]
        )
        predicate = self.encoder.predicate_hv("VISITED")
        triple = self.encoder.encode_triple(
            subject,
            predicate,
            object_hv,
            subject_context="Person:test",
            object_context="Location:test",
        )
        subject_factor = self.encoder.normalize_for_binding(subject, "Person:test")
        recovered_object = torchhd.multibind(
            torch.stack([triple, subject_factor, predicate])
        )
        expected_object = self.encoder.normalize_for_binding(
            object_hv,
            "Location:test",
        )
        self.assertTrue(torch.equal(recovered_object, expected_object))


if __name__ == "__main__":
    unittest.main()
