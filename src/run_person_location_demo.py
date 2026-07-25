from __future__ import annotations

import argparse
from pathlib import Path

from graph.ingest import ingest_graph
from graph.paths import (
    DEFAULT_DB_URI,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
)
from graph.retrieve import execute_query, visited_query
from hdc.encode import encode_hdc
from hdc.retrieve import hdc_fuzzy_paths


def build_demo(
    db_uri: Path,
    embedding_model: str,
    ollama_host: str,
) -> None:
    """Build graph tables first, then add HDC columns to those same tables."""
    ingest_graph(db_uri)
    encode_hdc(
        db_uri=db_uri,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
    )


def run_fuzzy_demo(
    db_uri: Path,
    query: str,
    *,
    embedding_model: str,
    ollama_host: str,
    min_score: float | None,
    limit: int,
) -> None:
    """Run fuzzy HDC retrieval and graph validation from one LanceDB dataset."""
    build_demo(db_uri, embedding_model, ollama_host)

    hdc_paths = hdc_fuzzy_paths(
        query,
        db_uri,
        min_score=min_score,
        limit=limit,
        ollama_host=ollama_host,
    )
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
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument(
        "--query",
        default="mountainous cities near the pacific ocean",
        help=(
            "Fuzzy location description, e.g. "
            "'mountainous cities near the pacific ocean'."
        ),
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fuzzy_demo(
        args.db_uri,
        args.query,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        min_score=args.min_score,
        limit=args.limit,
    )
