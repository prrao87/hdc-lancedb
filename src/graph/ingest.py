from __future__ import annotations

import shutil
from pathlib import Path

import lancedb
import polars as pl
import pyarrow as pa

from graph.data import (
    BLOB_COLUMNS,
    PREDICATE,
    location_records,
    person_records,
    relationship_records,
    validate_relationships,
)
from graph.paths import DEFAULT_DB_URI, RAW_DATA_DIR


def blob_encoded(data: pl.DataFrame) -> pa.Table:
    """Convert to Arrow, tagging asset columns for lazy Lance blob storage.

    Marking a binary column with `lance-encoding:blob` keeps its bytes out of
    normal scans: the value materializes only via `take_blobs`, so a query that
    never asks for the image never pays to read it. See the on-demand image
    endpoint in graph_api.py.
    """
    table = data.to_arrow()
    fields = [
        field.with_metadata({b"lance-encoding:blob": b"true"})
        if field.name in BLOB_COLUMNS
        else field
        for field in table.schema
    ]
    return pa.Table.from_arrays(table.columns, schema=pa.schema(fields))


def ingest_graph(db_uri: Path = DEFAULT_DB_URI, raw_data_dir: Path = RAW_DATA_DIR) -> None:
    """Load raw CSV node/relationship files into the shared LanceDB dataset."""
    persons = person_records(raw_data_dir)
    locations = location_records(raw_data_dir)
    relationships = relationship_records(raw_data_dir)
    validate_relationships(persons, locations, relationships)

    predicates = set(relationships.get_column("predicate").unique().to_list())
    if predicates != {PREDICATE}:
        raise ValueError(f"Expected only {PREDICATE} relationships, found {predicates}")

    if db_uri.exists():
        shutil.rmtree(db_uri)
    db_uri.parent.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(db_uri)
    db.create_table("Person", data=blob_encoded(persons))
    db.create_table("Location", data=blob_encoded(locations))
    db.create_table(PREDICATE, data=blob_encoded(relationships))
