from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import lancedb

from person_location_data import PREDICATE, location_records, person_records
from person_location_data import relationship_records, validate_relationships
from storage_paths import DEFAULT_DB_URI, RAW_DATA_DIR


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
    db.create_table("Person", data=persons)
    db.create_table("Location", data=locations)
    db.create_table(PREDICATE, data=relationships)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw CSV graph data into LanceDB.")
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_graph(args.db_uri, args.raw_data_dir)
    print(f"Wrote shared LanceDB graph tables: {args.db_uri}")
