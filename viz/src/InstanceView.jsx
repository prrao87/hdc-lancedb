import { useCallback, useEffect, useMemo, useState } from "react";
import GraphCanvas from "./GraphCanvas.jsx";
import { imageUrl, runQuery } from "./api.js";

const OPS = ["=", "<>", "<", "<=", ">", ">=", "CONTAINS", "STARTS WITH", "ENDS WITH"];

const defaultLabelCol = (node) =>
  node.properties.includes("name") ? "name" : node.id_field;

export default function InstanceView({ schema, colorMap }) {
  const nodeByLabel = useMemo(
    () => Object.fromEntries(schema.nodes.map((n) => [n.label, n])),
    [schema]
  );

  const [edge, setEdge] = useState(schema.edges[0]?.type ?? "");
  const edgeDef = schema.edges.find((e) => e.type === edge);
  const involved = edgeDef ? [edgeDef.from, edgeDef.to] : [];

  const [render, setRender] = useState({});
  const [filters, setFilters] = useState([]);
  const [limit, setLimit] = useState(25);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);

  // A node's raw entity id, minus the "Label:" prefix used for graph node ids.
  const rawId = (node) => node.id.slice(node.group.length + 1);
  const hasImage = (label) => nodeByLabel[label]?.assets?.includes("image");

  // (Re)initialize render config whenever the chosen edge changes.
  useEffect(() => {
    const next = {};
    for (const label of involved) {
      next[label] = { label: defaultLabelCol(nodeByLabel[label]), tooltip: [] };
    }
    setRender(next);
    setFilters([]);
  }, [edge]); // eslint-disable-line react-hooks/exhaustive-deps

  const buildSpec = useCallback(
    () => ({
      edge,
      render: Object.fromEntries(
        involved.map((label) => [label, render[label] ?? { label: nodeByLabel[label].id_field, tooltip: [] }])
      ),
      filters: filters.filter((f) => f.value !== ""),
      limit: Number(limit) || 25,
    }),
    [edge, involved, render, filters, limit, nodeByLabel]
  );

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSelected(null);
      setResult(await runQuery(buildSpec()));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [buildSpec]);

  // Run the default query once render config is ready.
  useEffect(() => {
    if (Object.keys(render).length > 0 && !result) run();
  }, [render]); // eslint-disable-line react-hooks/exhaustive-deps

  const setLabelCol = (label, col) =>
    setRender((r) => ({ ...r, [label]: { ...r[label], label: col } }));

  const toggleTooltip = (label, col) =>
    setRender((r) => {
      const tip = r[label].tooltip;
      const next = tip.includes(col) ? tip.filter((c) => c !== col) : [...tip, col];
      return { ...r, [label]: { ...r[label], tooltip: next } };
    });

  const addFilter = () =>
    setFilters((f) => [
      ...f,
      { label: involved[0], prop: nodeByLabel[involved[0]].properties[0], op: "=", value: "" },
    ]);
  const updateFilter = (i, patch) =>
    setFilters((f) => f.map((row, j) => (j === i ? { ...row, ...patch } : row)));
  const removeFilter = (i) => setFilters((f) => f.filter((_, j) => j !== i));

  return (
    <div className="view-split">
      <GraphCanvas
        data={result ?? { nodes: [], links: [] }}
        colorMap={colorMap}
        onNodeClick={setSelected}
      />
      <aside className="side-panel builder">
        <h3>Query builder</h3>

        {selected && (
          <div className="builder-card node-preview">
            <div className="card-head">
              <span className="swatch" style={{ background: colorMap[selected.group] }} />
              {selected.label ?? rawId(selected)}
              <button type="button" className="mini" onClick={() => setSelected(null)}>
                ×
              </button>
            </div>
            {hasImage(selected.group) ? (
              <figure className="asset">
                <img
                  src={imageUrl(selected.group, rawId(selected))}
                  alt={`${selected.group} ${rawId(selected)}`}
                />
                <figcaption className="muted">
                  Image bytes fetched on demand from the lazy blob column.
                </figcaption>
              </figure>
            ) : (
              <p className="muted">No image asset for {selected.group} nodes.</p>
            )}
          </div>
        )}

        <label className="ctl">
          <span>Relationship</span>
          <select value={edge} onChange={(e) => setEdge(e.target.value)}>
            {schema.edges.map((e) => (
              <option key={e.type} value={e.type}>
                ({e.from})-[:{e.type}]→({e.to})
              </option>
            ))}
          </select>
        </label>

        {involved.map((label) => (
          <div className="builder-card" key={label}>
            <div className="card-head">
              <span className="swatch" style={{ background: colorMap[label] }} />
              {label}
            </div>
            <label className="ctl">
              <span>Display</span>
              <select
                value={render[label]?.label ?? ""}
                onChange={(e) => setLabelCol(label, e.target.value)}
              >
                {nodeByLabel[label].properties.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </label>
            <div className="ctl">
              <span>Tooltip</span>
              <div className="chips">
                {nodeByLabel[label].properties.map((p) => (
                  <button
                    key={p}
                    type="button"
                    className={`chip ${render[label]?.tooltip.includes(p) ? "on" : ""}`}
                    onClick={() => toggleTooltip(label, p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}

        <div className="builder-card">
          <div className="card-head">
            Filters
            <button type="button" className="mini" onClick={addFilter}>+ add</button>
          </div>
          {filters.length === 0 && <p className="muted">No filters — all rows.</p>}
          {filters.map((f, i) => (
            <div className="filter-row" key={i}>
              <select
                value={f.label}
                onChange={(e) =>
                  updateFilter(i, {
                    label: e.target.value,
                    prop: nodeByLabel[e.target.value].properties[0],
                  })
                }
              >
                {involved.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
              <select value={f.prop} onChange={(e) => updateFilter(i, { prop: e.target.value })}>
                {nodeByLabel[f.label].properties.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <select value={f.op} onChange={(e) => updateFilter(i, { op: e.target.value })}>
                {OPS.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
              <input
                value={f.value}
                placeholder="value"
                onChange={(e) => updateFilter(i, { value: e.target.value })}
              />
              <button type="button" className="mini danger" onClick={() => removeFilter(i)}>×</button>
            </div>
          ))}
        </div>

        <label className="ctl inline">
          <span>Limit</span>
          <input
            type="number"
            min="1"
            max="500"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
          />
        </label>

        <button className="run" onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run query"}
        </button>

        {error && <div className="error">{error}</div>}
        {result && (
          <>
            <div className="result-meta">
              {result.nodes.length} nodes · {result.links.length} edges
            </div>
            <div className="cypher-pane">
              <span className="cypher-head">Generated Cypher (read-only)</span>
              <pre>{result.cypher}</pre>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
