from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph_retrieve import GRAPH_SCHEMA, build_engine
from storage_paths import DEFAULT_DB_URI

# Cypher comparison operators the builder is allowed to emit. The query is constructed
# entirely server-side from structured selections, so this allow-list (plus validating
# every property name against the schema) is the whole injection surface.
ALLOWED_OPS = {"=", "<>", "<", "<=", ">", ">=", "CONTAINS", "STARTS WITH", "ENDS WITH"}

app = FastAPI(title="lance-graph visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- schema introspection -------------------------------------------------------------


def _is_vector(field_type: pa.DataType) -> bool:
    """True for the high-dimensional embedding columns (hv, vibe_hv)."""
    return (
        pa.types.is_fixed_size_list(field_type)
        or pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
    )


def _node_label(label: str) -> dict | None:
    return next((n for n in GRAPH_SCHEMA["nodes"] if n["label"] == label), None)


def _dataset(label_or_type: str, db_uri: Path = DEFAULT_DB_URI) -> lance.LanceDataset:
    return lance.dataset(str(db_uri / f"{label_or_type}.lance"))


def schema_payload(db_uri: Path = DEFAULT_DB_URI) -> dict:
    """Describe the graph from GRAPH_SCHEMA + each table's Arrow schema.

    Scalar columns are offered as projectable/filterable properties; vector columns
    are reported separately (badged "embedding") and never become projectable fields.
    No Cypher runs here.
    """
    nodes = []
    for node in GRAPH_SCHEMA["nodes"]:
        dataset = _dataset(node["label"], db_uri)
        properties, embeddings = [], []
        for field in dataset.schema:
            (embeddings if _is_vector(field.type) else properties).append(field.name)
        nodes.append(
            {
                "label": node["label"],
                "id_field": node["id"],
                "properties": properties,
                "embeddings": embeddings,
                "count": dataset.count_rows(),
            }
        )
    edges = [
        {"type": r["type"], "from": r["from_label"], "to": r["to_label"]}
        for r in GRAPH_SCHEMA["relationships"]
    ]
    return {"nodes": nodes, "edges": edges}


# --- query assembly -------------------------------------------------------------------


class RenderSpec(BaseModel):
    label: str  # display-label column
    tooltip: list[str] = Field(default_factory=list)


class FilterSpec(BaseModel):
    label: str
    prop: str
    op: str
    value: str


class QuerySpec(BaseModel):
    edge: str | None = None  # relationship type for a path query
    node: str | None = None  # single node label, when not querying an edge
    render: dict[str, RenderSpec] = Field(default_factory=dict)
    filters: list[FilterSpec] = Field(default_factory=list)
    limit: int = Field(default=25, ge=1, le=500)


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _plan(spec: QuerySpec) -> tuple[str, list[tuple[str, str]]]:
    """Resolve the spec into (MATCH pattern, [(var, label), ...] for each node)."""
    if spec.edge:
        rel = next(
            (r for r in GRAPH_SCHEMA["relationships"] if r["type"] == spec.edge), None
        )
        if rel is None:
            raise HTTPException(400, f"Unknown relationship type: {spec.edge}")
        vars_labels = [("a", rel["from_label"]), ("b", rel["to_label"])]
        pattern = (
            f"MATCH (a:{rel['from_label']})-[:{spec.edge}]->(b:{rel['to_label']})"
        )
        return pattern, vars_labels
    if spec.node:
        if _node_label(spec.node) is None:
            raise HTTPException(400, f"Unknown node label: {spec.node}")
        return f"MATCH (a:{spec.node})", [("a", spec.node)]
    raise HTTPException(400, "Query spec must set either 'edge' or 'node'.")


def assemble(spec: QuerySpec, db_uri: Path = DEFAULT_DB_URI) -> dict:
    """Build Cypher from the structured spec, run it, and return a node-link graph."""
    pattern, vars_labels = _plan(spec)
    label_to_var = {label: var for var, label in vars_labels}

    # Validate render/filter columns against the real (scalar) schema before emitting.
    scalar_props: dict[str, set[str]] = {}
    id_field: dict[str, str] = {}
    for _var, label in vars_labels:
        node = _node_label(label)
        id_field[label] = node["id"]
        dataset = _dataset(label, db_uri)
        scalar_props[label] = {
            f.name for f in dataset.schema if not _is_vector(f.type)
        }

    # Projection: only id + chosen label + tooltip columns. Embeddings can never enter.
    columns: list[str] = []
    for var, label in vars_labels:
        render = spec.render.get(label)
        wanted = {id_field[label]}
        if render:
            wanted.add(render.label)
            wanted.update(render.tooltip)
        for col in wanted:
            if col not in scalar_props[label]:
                raise HTTPException(400, f"{label} has no scalar property '{col}'")
            columns.append(f"{var}.{col}")

    where_clauses: list[str] = []
    for f in spec.filters:
        if f.label not in label_to_var:
            raise HTTPException(400, f"Filter label '{f.label}' is not in this pattern")
        if f.prop not in scalar_props[f.label]:
            raise HTTPException(400, f"{f.label} has no scalar property '{f.prop}'")
        if f.op not in ALLOWED_OPS:
            raise HTTPException(400, f"Operator not allowed: {f.op}")
        where_clauses.append(f"{label_to_var[f.label]}.{f.prop} {f.op} {_quote(f.value)}")

    where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    cypher = f"{pattern}{where} RETURN {', '.join(columns)} LIMIT {spec.limit}"

    rows = build_engine(db_uri).execute(cypher).to_pylist()

    nodes: list[dict] = []
    links: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        row_ids: list[str] = []
        for var, label in vars_labels:
            raw_id = row[f"{var}.{id_field[label]}"]
            node_id = f"{label}:{raw_id}"
            row_ids.append(node_id)
            if node_id in seen:
                continue
            seen.add(node_id)
            render = spec.render.get(label)
            display = row.get(f"{var}.{render.label}") if render else raw_id
            tooltip = (
                {p: row.get(f"{var}.{p}") for p in render.tooltip} if render else {}
            )
            nodes.append(
                {
                    "id": node_id,
                    "group": label,
                    "label": display if display is not None else raw_id,
                    "props": tooltip,
                }
            )
        # Pattern adjacency → edges (single hop today; generalizes to chains).
        for source, target in zip(row_ids, row_ids[1:]):
            links.append({"source": source, "target": target, "type": spec.edge})

    return {"nodes": nodes, "links": links, "cypher": cypher}


# --- endpoints ------------------------------------------------------------------------


@app.get("/schema")
def get_schema() -> dict:
    return schema_payload()


@app.post("/query")
def post_query(spec: QuerySpec) -> dict:
    try:
        return assemble(spec)
    except HTTPException:
        raise
    except ValueError as exc:  # Cypher parse/plan errors from the engine
        raise HTTPException(400, str(exc)) from exc
