from __future__ import annotations

import argparse
from pathlib import Path

from graph.ingest import ingest_graph
from graph.paths import DEFAULT_DB_URI, RAW_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw CSV graph data into LanceDB."
    )
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_graph(args.db_uri, args.raw_data_dir)
    print(f"Wrote shared LanceDB graph tables: {args.db_uri}")
