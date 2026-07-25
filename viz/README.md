# lance-graph visualizer

A two-view graph visualizer over the lance-graph dataset:

- **Schema view** — the meta-graph (labels, properties, relationships), derived from
  `GraphConfig` + each table's Arrow schema. No query runs. Hypervector columns (`hv`)
  are badged, never drawn.
- **Instance view** — a query *builder* (pick relationship, per-label display/tooltip
  columns, structured filters, limit). The backend constructs Cypher from your
  selections, dedups nodes, and returns a node-link graph. The generated Cypher is shown
  read-only.

The Cypher engine (`lance_graph.CypherEngine`) is Python-only, so the browser never
touches it: a FastAPI backend serves JSON and the React frontend only draws it.

## Run

Two processes. From the repo root:

```bash
# 1. backend (port 8000) — uses build_engine() from src/graph/retrieve.py
uv run uvicorn graph_api:app --app-dir src --reload

# 2. frontend (port 5173) — must be 5173; the backend CORS allowlist pins it
cd viz && npm install && npm run dev
```

Open http://localhost:5173.

## Pointing at a different dataset

The whole stack reads one declarative spec — `GRAPH_SCHEMA` in
[src/graph/retrieve.py](../src/graph/retrieve.py). Edit the node labels / id fields /
relationships there and both the engine and the visualizer follow.

## API

- `GET /schema` → `{nodes:[{label,id_field,properties,embeddings,count}], edges:[{type,from,to}]}`
- `POST /query` with `{edge|node, render:{Label:{label,tooltip[]}}, filters:[{label,prop,op,value}], limit}`
  → `{nodes, links, cypher}`
