from __future__ import annotations

from pathlib import Path

import lance
import polars as pl
import pyarrow as pa
import torch

from hdc.core import DIMENSIONS, TorchHDEncoder
from hdc.encode import (
    EncodedRows,
    EncodedTable,
    build_location_documents,
    encode_rows,
    hv_merge_table,
    merge_hv_column,
    read_table,
)
from hdc.location import LocationNodeEncoder


def test_read_table_projects_out_blobs_and_existing_vectors(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "id": ["row-1"],
            "name": ["Seattle"],
            "image": [b"image bytes"],
            "hv": pa.array(
                [[1.0] * DIMENSIONS],
                type=pa.list_(pa.float16(), DIMENSIONS),
            ),
        }
    )
    lance.write_dataset(table, tmp_path / "Location.lance")

    result = read_table(tmp_path, "Location")

    assert result.columns == ["id", "name"]


def test_hypervectors_are_stored_as_fixed_size_float16() -> None:
    table = hv_merge_table(
        ["row-1"],
        torch.ones((1, DIMENSIONS), dtype=torch.float32),
    )

    assert table.schema.field("hv").type == pa.list_(
        pa.float16(),
        DIMENSIONS,
    )


def test_merge_replaces_existing_hv_idempotently(tmp_path: Path) -> None:
    existing_vector = pa.array(
        [[1.0] * DIMENSIONS],
        type=pa.list_(pa.float16(), DIMENSIONS),
    )
    for table_name in ("Person", "Location", "VISITED"):
        lance.write_dataset(
            pa.table(
                {
                    "id": [f"{table_name.lower()}-1"],
                    "name": [table_name],
                    "hv": existing_vector,
                }
            ),
            tmp_path / f"{table_name}.lance",
        )

    replacement = torch.full(
        (1, DIMENSIONS),
        2.0,
        dtype=torch.float32,
    )
    encoded = EncodedRows(
        persons=EncodedTable(["person-1"], replacement),
        locations=EncodedTable(["location-1"], replacement),
        relationships=EncodedTable(["visited-1"], replacement),
    )
    merge_hv_column(tmp_path, encoded)

    for table_name in ("Person", "Location", "VISITED"):
        dataset = lance.dataset(tmp_path / f"{table_name}.lance")
        assert dataset.schema.names.count("hv") == 1
        stored = dataset.to_table(columns=["hv"]).to_pylist()[0]["hv"]
        assert stored[0] == 2.0


def test_location_documents_use_human_readable_features_and_evidence() -> None:
    documents = build_location_documents(
        [
            {
                "location_id": "location-seattle",
                "feature": "pacific_coast",
                "evidence": "city is on Puget Sound",
                "weight": 0.92,
            },
            {
                "location_id": "location-seattle",
                "feature": "mountains",
                "evidence": "mountain range is visible",
                "weight": 0.95,
            },
        ]
    )

    assert documents["location-seattle"] == (
        "mountains: mountain range is visible; "
        "pacific coast: city is on Puget Sound"
    )


def test_full_location_vector_is_persisted_and_used_by_relationship() -> None:
    persons = pl.DataFrame(
        [
            {
                "id": "person-maya",
                "name": "Maya",
                "kind": "Person",
                "role": "designer",
            }
        ]
    )
    locations = pl.DataFrame(
        [
            {
                "id": "location-seattle",
                "name": "Seattle",
                "kind": "Location",
                "region": "pacific_northwest",
                "timezone": "pacific",
            }
        ]
    )
    relationships = pl.DataFrame(
        [
            {
                "id": "visit-maya-seattle",
                "person_id": "person-maya",
                "location_id": "location-seattle",
                "predicate": "VISITED",
            }
        ]
    )
    encoder = TorchHDEncoder()
    semantic_hv = encoder.token_hv("test-semantic:seattle")

    encoded = encode_rows(
        persons,
        locations,
        relationships,
        {"location-seattle": semantic_hv},
        encoder,
        LocationNodeEncoder(encoder),
    )

    location_hv = encoded.locations.hvs[0]
    structural_hv = encoder.encode_properties(locations.row(0, named=True))
    assert torch.equal(location_hv, structural_hv + 8 * semantic_hv)

    person_hv = encoded.persons.hvs[0]
    predicate_hv = encoder.predicate_hv("VISITED")
    expected_relationship = encoder.encode_triple(
        person_hv,
        predicate_hv,
        location_hv,
        subject_context="Person:person-maya:properties",
        object_context="Location:location-seattle:full-node",
    )
    structural_only_relationship = encoder.encode_triple(
        person_hv,
        predicate_hv,
        structural_hv,
        subject_context="Person:person-maya:properties",
        object_context="Location:location-seattle:properties",
    )

    assert torch.equal(encoded.relationships.hvs[0], expected_relationship)
    assert not torch.equal(
        encoded.relationships.hvs[0],
        structural_only_relationship,
    )
