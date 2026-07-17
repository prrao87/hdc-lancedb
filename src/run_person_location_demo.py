from __future__ import annotations

import argparse
from pathlib import Path

from graph_ingest import ingest_graph
from graph_retrieve import execute_query, visited_query
from hdc_encode import encode_hdc
from hdc_retrieve import hdc_fuzzy_paths
from storage_paths import DEFAULT_DB_URI, DEFAULT_VOCAB_PATH


def build_demo(db_uri: Path, vocab_path: Path) -> None:
    """Build graph tables first, then add HDC columns to those same tables."""
    ingest_graph(db_uri)
    encode_hdc(db_uri=db_uri, vocab_path=vocab_path)


def run_fuzzy_demo(db_uri: Path, vocab_path: Path, query: str) -> None:
    """Run fuzzy HDC retrieval and graph validation from one LanceDB dataset."""
    build_demo(db_uri, vocab_path)

    hdc_paths = hdc_fuzzy_paths(query, db_uri, vocab_path)
    graph_rows = execute_query(visited_query(), db_uri).to_pylist()

    print(f"Shared LanceDB dataset: {db_uri}")
    print(f"Question: {query}")
    print()
    print("LanceDB HDC matches, expanded through lance-graph:")
    for row in hdc_paths:
        feature_list = ", ".join(row["features"])
        print(
            f"  - {row['person']} -> {row['city']} "
            f"({row['timezone']}, score={row['score']:.3f}, features={feature_list})"
        )
    print()
    print("All lance-graph validation paths:")
    for row in graph_rows:
        print(f"  - {row['person']} -> {row['city']} ({row['timezone']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a Person-VISITED-Location LanceDB demo."
    )
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument(
        "--query",
        default="persons who visited cities on the pacific coast with mountains nearby",
        help=(
            "Fuzzy query, e.g. 'persons who visited cities on the pacific coast "
            "with mountains nearby'."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fuzzy_demo(args.db_uri, args.vocab_path, args.query)
