from __future__ import annotations

import argparse
from pathlib import Path

from graph.paths import (
    DEFAULT_DB_URI,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    RAW_DATA_DIR,
    SEMANTIC_HDC_METADATA_FILENAME,
)
from hdc.core import SEED
from hdc.encode import encode_hdc
from hdc.location import DEFAULT_SEMANTIC_WEIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add HDC vectors to LanceDB tables.")
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--projection-seed", type=int, default=SEED)
    parser.add_argument(
        "--semantic-weight",
        type=int,
        default=DEFAULT_SEMANTIC_WEIGHT,
        help="Weight of the semantic term in each full Location.hv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_hdc(
        db_uri=args.db_uri,
        raw_data_dir=args.raw_data_dir,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        projection_seed=args.projection_seed,
        semantic_weight=args.semantic_weight,
    )
    print(f"Added HDC columns to shared LanceDB dataset: {args.db_uri}")
    print(
        "Wrote semantic HDC metadata and projection: "
        f"{args.db_uri / SEMANTIC_HDC_METADATA_FILENAME}"
    )
