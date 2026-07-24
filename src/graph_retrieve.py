from __future__ import annotations

import argparse
from pathlib import Path

from graph.paths import DEFAULT_DB_URI
from graph.retrieve import (
    city_query,
    execute_query,
    timezone_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cypher queries over the graph.")
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--city", default="Seattle")
    parser.add_argument("--timezone", help="Validate people by Location.timezone.")
    parser.add_argument("--query", help="Custom Cypher query. Overrides --city.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    query = (
        args.query
        or (timezone_query(args.timezone) if args.timezone else city_query(args.city))
    )
    rows = execute_query(query, args.db_uri).to_pylist()
    if not rows:
        print("(no rows)")
    for row in rows:
        print(row)
