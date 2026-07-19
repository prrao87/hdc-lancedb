from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lance
import polars as pl
import pyarrow as pa

from person_location_data import (
    PREDICATE,
    location_vibe_records,
    validate_location_vibes,
)
from storage_paths import DEFAULT_DB_URI, DEFAULT_VOCAB_PATH, RAW_DATA_DIR
from torchhd_encoder import (
    DIMENSIONS,
    NON_SYMBOLIC_COLUMNS,
    TorchHDEncoder,
    build_vocabulary,
    hv_to_list,
)


@dataclass(frozen=True)
class EncodedRows:
    """Rows ready to merge into existing LanceDB tables with HDC vectors."""

    persons: pl.DataFrame
    locations: pl.DataFrame
    relationships: pl.DataFrame


def strength_bucket(weight: str | float) -> str | None:
    """Convert numeric vibe evidence weight into a symbolic strength bucket."""
    value = float(weight)
    if value >= 0.80:
        return "strong"
    if value >= 0.50:
        return "medium"
    if value >= 0.20:
        return "weak"
    return None


def prepare_vibes(vibes: pl.DataFrame) -> list[dict[str, str]]:
    """Normalize vibe evidence rows before vocabulary building and encoding."""
    prepared = []
    for row in vibes.iter_rows(named=True):
        strength = strength_bucket(row["weight"])
        if strength is None:
            continue
        prepared.append(
            {
                "location_id": row["location_id"],
                "source": row["source"],
                "feature": row["feature"],
                "strength": strength,
                "weight": row["weight"],
                "evidence": row["evidence"],
                "image_path": row["image_path"],
            }
        )
    return prepared


def dataset_path(db_uri: Path, table_name: str) -> str:
    """Filesystem path of one Lance table inside the shared dataset."""
    return str(db_uri / f"{table_name}.lance")


def read_table(db_uri: Path, table_name: str) -> pl.DataFrame:
    """Read symbolic graph columns, skipping blobs and prior HDC vectors.

    Neither image bytes nor already-encoded vectors are inputs to a fresh HDC
    encoding pass. Projecting both out avoids materializing large columns and
    keeps repeat encoding proportional to the source graph properties.
    """
    ds = lance.dataset(dataset_path(db_uri, table_name))
    columns = [
        name for name in ds.schema.names if name not in NON_SYMBOLIC_COLUMNS
    ]
    return pl.from_arrow(ds.to_table(columns=columns))


def encode_rows(
    persons: pl.DataFrame,
    locations: pl.DataFrame,
    relationships: pl.DataFrame,
    location_vibes: list[dict[str, str]],
    encoder: TorchHDEncoder,
) -> EncodedRows:
    """Attach entity and relationship hypervectors to existing graph rows."""
    person_hvs = {}
    person_rows = []
    for person in persons.iter_rows(named=True):
        person_hv = encoder.encode_properties(person)
        person_hvs[person["id"]] = person_hv
        person_rows.append({**person, "hv": hv_to_list(person_hv)})

    vibe_rows_by_location: dict[str, list[dict[str, str]]] = {}
    for vibe in location_vibes:
        vibe_rows_by_location.setdefault(vibe["location_id"], []).append(vibe)

    location_hvs = {}
    location_vibe_hvs = {}
    location_rows = []
    for location in locations.iter_rows(named=True):
        location_hv = encoder.encode_properties(location)
        location_hvs[location["id"]] = location_hv
        vibe_hvs = [
            encoder.encode_vibe_terms(
                {
                    "source": vibe["source"],
                    "feature": vibe["feature"],
                    "strength": vibe["strength"],
                }
            )
            for vibe in vibe_rows_by_location.get(location["id"], [])
        ]
        if not vibe_hvs:
            raise ValueError(f"Location has no vibe evidence: {location['id']}")
        location_vibe_hv = encoder.bundle(vibe_hvs)
        location_vibe_hvs[location["id"]] = location_vibe_hv
        location_rows.append(
            {
                **location,
                "hv": hv_to_list(location_hv),
                "vibe_hv": hv_to_list(location_vibe_hv),
            }
        )

    predicate_hv = encoder.predicate_hv(PREDICATE)
    relationship_rows = []
    for relationship in relationships.iter_rows(named=True):
        triple_hv = encoder.encode_triple(
            person_hvs[relationship["person_id"]],
            predicate_hv,
            location_hvs[relationship["location_id"]],
            subject_context=f"Person:{relationship['person_id']}:properties",
            object_context=f"Location:{relationship['location_id']}:properties",
        )
        vibe_triple_hv = encoder.encode_triple(
            person_hvs[relationship["person_id"]],
            predicate_hv,
            location_vibe_hvs[relationship["location_id"]],
            subject_context=f"Person:{relationship['person_id']}:properties",
            object_context=f"Location:{relationship['location_id']}:vibes",
        )
        relationship_rows.append(
            {
                **relationship,
                "hv": hv_to_list(triple_hv),
                "vibe_hv": hv_to_list(vibe_triple_hv),
            }
        )

    return EncodedRows(
        pl.DataFrame(person_rows),
        pl.DataFrame(location_rows),
        pl.DataFrame(relationship_rows),
    )


def merge_hv_columns(db_uri: Path, encoded: EncodedRows) -> None:
    """Join the new HDC vector columns onto existing graph rows by `id`.

    Uses Lance's column merge rather than a row upsert: `merge_insert` cannot
    round-trip a blob column (it reads existing values back as descriptors), so
    we only ever bring in the `id` key plus the freshly computed vectors and let
    Lance graft them on. Existing columns, including the lazy image blob, are
    untouched. Any prior HDC columns are dropped first so re-encoding is
    idempotent.
    """
    for table_name, rows in [
        ("Person", encoded.persons),
        ("Location", encoded.locations),
        (PREDICATE, encoded.relationships),
    ]:
        hv_columns = [column for column in ("hv", "vibe_hv") if column in rows.columns]
        path = dataset_path(db_uri, table_name)
        ds = lance.dataset(path)
        stale = [column for column in hv_columns if column in ds.schema.names]
        if stale:
            ds.drop_columns(stale)
            ds = lance.dataset(path)
        ds.merge(hv_merge_table(rows, hv_columns), left_on="id", right_on="id")


def hv_merge_table(rows: pl.DataFrame, hv_columns: list[str]) -> pa.Table:
    """Build the `id` + hypervector Arrow table Lance grafts onto a graph table.

    Polars would hand Lance `large_list<double>` columns, but LanceDB vector
    search needs fixed-size floating-point lists. TorchHD still performs its
    encoding algebra in float32, while Lance stores the finished vectors as
    float16 to halve their raw storage cost. That is a deliberate storage vs.
    precision tradeoff: it is lossless for this demo's small integer-valued MAP
    coordinates, but production workloads should validate search recall before
    adopting it for vectors with a wider numeric range.
    """
    merge_columns = rows.select(["id", *hv_columns]).to_arrow()
    fields = [
        pa.field(field.name, pa.list_(pa.float16(), DIMENSIONS))
        if field.name in hv_columns
        else field
        for field in merge_columns.schema
    ]
    return merge_columns.cast(pa.schema(fields))


def encode_hdc(
    db_uri: Path = DEFAULT_DB_URI,
    vocab_path: Path = DEFAULT_VOCAB_PATH,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> None:
    """Add HDC vector columns to graph tables in the shared LanceDB dataset.

    Run graph_ingest.py first. This step intentionally augments those same
    tables, demonstrating LanceDB's ability to evolve the schema with new
    feature columns.
    """
    persons = read_table(db_uri, "Person")
    locations = read_table(db_uri, "Location")
    relationships = read_table(db_uri, PREDICATE)
    vibes = location_vibe_records(raw_data_dir)
    validate_location_vibes(locations, vibes)
    prepared_vibes = prepare_vibes(vibes)

    tokens = build_vocabulary(
        persons.iter_rows(named=True),
        locations.iter_rows(named=True),
        relationships.iter_rows(named=True),
        PREDICATE,
        prepared_vibes,
    )
    encoder = TorchHDEncoder(tokens)
    encoded = encode_rows(persons, locations, relationships, prepared_vibes, encoder)
    merge_hv_columns(db_uri, encoded)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(tokens, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add TorchHD vectors to LanceDB tables.")
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_hdc(args.db_uri, args.vocab_path, args.raw_data_dir)
    print(f"Added HDC columns to shared LanceDB dataset: {args.db_uri}")
    print(f"Wrote HDC vocabulary: {args.vocab_path}")
