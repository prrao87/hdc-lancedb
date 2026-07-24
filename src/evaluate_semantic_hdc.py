from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.semantic import (
    DEFAULT_PATH_QUERY_SET,
    DEFAULT_QUERY_SET,
    EvaluationResult,
    evaluate,
)
from graph.paths import DEFAULT_DB_URI, DEFAULT_OLLAMA_HOST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic HDC ranking and graph-path retrieval."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_SET)
    parser.add_argument(
        "--path-queries",
        type=Path,
        default=DEFAULT_PATH_QUERY_SET,
        help="Semantic queries that also assert exact VISITED traversal results.",
    )
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def print_results(title: str, results: list[EvaluationResult]) -> None:
    print(title)
    for result in results:
        status = "PASS" if result.passes else "FAIL"
        ranking = ", ".join(
            f"{city}={score:.3f}" for city, score in result.ranked_cities
        )
        print(
            f"{status} expected={result.expected_city!r} "
            f"rank={result.expected_rank} "
            f"query={result.query!r}"
        )
        print(f"  {ranking}")
        if result.expected_people:
            visitors = " | ".join(
                result.people_by_city.get(result.expected_city, ())
            )
            print(f"  visitors={visitors}")


def main() -> int:
    args = parse_args()
    semantic_results = evaluate(
        args.queries,
        args.db_uri,
        args.ollama_host,
        args.limit,
    )
    path_results = evaluate(
        args.path_queries,
        args.db_uri,
        args.ollama_host,
        args.limit,
        require_expected_people=True,
    )
    print_results("Semantic city ranking", semantic_results)
    print()
    print_results("Semantic ranking + VISITED traversal", path_results)

    top_one = sum(result.expected_rank == 1 for result in semantic_results)
    path_passes = sum(result.passes for result in path_results)
    mrr = sum(
        result.reciprocal_rank for result in semantic_results
    ) / len(semantic_results)
    print()
    print(
        f"Semantic top-1 accuracy: {top_one}/{len(semantic_results)} "
        f"({top_one / len(semantic_results):.1%})"
    )
    print(f"Semantic mean reciprocal rank: {mrr:.3f}")
    print(
        f"VISITED path checks: {path_passes}/{len(path_results)} "
        f"({path_passes / len(path_results):.1%})"
    )
    return 0 if all(
        result.passes for result in [*semantic_results, *path_results]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
