from __future__ import annotations

from pathlib import Path

import lance
from lance_graph import CypherEngine, GraphConfig

from graph.paths import DEFAULT_DB_URI

# Single source of truth for the graph's shape. The CypherEngine's GraphConfig is
# built from this, and the visualizer API (graph_api.py) reads it directly — so the
# schema view, query builder, and engine never drift. Swap this spec to point the
# whole stack at a different lance-graph dataset.
GRAPH_SCHEMA = {
    "nodes": [
        {"label": "Person", "id": "id"},
        {"label": "Location", "id": "id"},
    ],
    "relationships": [
        {
            "type": "VISITED",
            "from_label": "Person",
            "from_key": "person_id",
            "to_label": "Location",
            "to_key": "location_id",
        },
    ],
}


def table_names() -> list[str]:
    """Lance table names backing the graph (one per node label and relationship)."""
    return [n["label"] for n in GRAPH_SCHEMA["nodes"]] + [
        r["type"] for r in GRAPH_SCHEMA["relationships"]
    ]


def build_engine(db_uri: Path = DEFAULT_DB_URI) -> CypherEngine:
    """Create a reusable CypherEngine with schema/catalog metadata cached."""
    builder = GraphConfig.builder()
    for node in GRAPH_SCHEMA["nodes"]:
        builder = builder.with_node_label(node["label"], node["id"])
    for rel in GRAPH_SCHEMA["relationships"]:
        builder = builder.with_relationship(rel["type"], rel["from_key"], rel["to_key"])
    config = builder.build()
    datasets = {
        name: lance.dataset(str(db_uri / f"{name}.lance")) for name in table_names()
    }
    return CypherEngine(config, datasets)


def city_query(city: str) -> str:
    """Build the default exact graph query for a city."""
    escaped_city = city.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "MATCH (p:Person)-[:VISITED]->(l:Location) "
        f"WHERE l.name = '{escaped_city}' "
        "RETURN p.name AS person, l.name AS city "
        "ORDER BY p.name"
    )


def timezone_query(timezone: str) -> str:
    """Build the graph validation query for a location timezone property."""
    escaped_timezone = timezone.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "MATCH (p:Person)-[:VISITED]->(l:Location) "
        f"WHERE l.timezone = '{escaped_timezone}' "
        "RETURN p.name AS person, l.name AS city, l.timezone AS timezone "
        "ORDER BY p.name"
    )


def visited_query() -> str:
    """Build the exact graph query for all person-location paths."""
    return (
        "MATCH (p:Person)-[:VISITED]->(l:Location) "
        "RETURN p.name AS person, l.name AS city, l.timezone AS timezone "
        "ORDER BY p.name"
    )


def execute_query(query: str, db_uri: Path = DEFAULT_DB_URI):
    """Execute an arbitrary Cypher query against the cached engine."""
    return build_engine(db_uri).execute(query)
