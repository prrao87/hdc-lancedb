from __future__ import annotations

import argparse
from pathlib import Path

from graph.paths import DEFAULT_DB_URI, DEFAULT_OLLAMA_HOST
from hdc.retrieve import hdc_fuzzy_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search full-node HDC vectors, then traverse graph paths."
    )
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "Arbitrary fuzzy location description, e.g. "
            "'mountainous cities near the pacific ocean'."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Optional calibrated minimum cosine score. By default retrieval is "
            "top-k only because semantic scores are not probabilities."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of locations returned by LanceDB vector search.",
    )
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for row in hdc_fuzzy_paths(
        args.query,
        args.db_uri,
        min_score=args.min_score,
        limit=args.limit,
        ollama_host=args.ollama_host,
    ):
        print({**row, "score": round(row["score"], 3)})
