import { useMemo, useState } from "react";
import GraphCanvas from "./GraphCanvas.jsx";

// The meta-graph: one node per label, one link per relationship. Seeded into fixed
// positions so the small graph doesn't drift, then frozen on engine stop.
export default function SchemaView({ schema, colorMap }) {
  const [selected, setSelected] = useState(null);

  const data = useMemo(() => {
    const n = schema.nodes.length;
    const nodes = schema.nodes.map((node, i) => ({
      id: node.label,
      group: node.label,
      label: `${node.label} (${node.count})`,
      count: node.count,
      // Spread horizontally as a starting seed.
      x: (i - (n - 1) / 2) * 160,
      y: 0,
      meta: node,
    }));
    const links = schema.edges.map((e) => ({
      source: e.from,
      target: e.to,
      type: e.type,
    }));
    return { nodes, links };
  }, [schema]);

  const detail = selected
    ? schema.nodes.find((node) => node.label === selected)
    : null;

  return (
    <div className="view-split">
      <GraphCanvas
        data={data}
        colorMap={colorMap}
        nodeRadius={11}
        cooldownTicks={60}
        onNodeClick={(node) => setSelected(node.id)}
      />
      <aside className="side-panel">
        <h3>Schema</h3>
        <p className="hint">
          Derived from <code>GraphConfig</code> + Arrow schemas — no query runs. Click a
          node to inspect its columns.
        </p>
        {!detail && <p className="muted">No label selected.</p>}
        {detail && (
          <div className="label-detail">
            <h4>
              <span
                className="swatch"
                style={{ background: colorMap[detail.label] }}
              />
              {detail.label}
              <span className="count">{detail.count} rows</span>
            </h4>
            <div className="field-group">
              <span className="field-head">
                Properties · id = <code>{detail.id_field}</code>
              </span>
              <ul className="fields">
                {detail.properties.map((p) => (
                  <li key={p}>
                    <code>{p}</code>
                    {p === detail.id_field && <span className="tag id">id</span>}
                  </li>
                ))}
              </ul>
            </div>
            {detail.embeddings.length > 0 && (
              <div className="field-group">
                <span className="field-head">Embeddings</span>
                <ul className="fields">
                  {detail.embeddings.map((e) => (
                    <li key={e}>
                      <code>{e}</code>
                      <span className="tag emb">embedding · 10k-dim</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {detail.assets?.length > 0 && (
              <div className="field-group">
                <span className="field-head">Assets</span>
                <ul className="fields">
                  {detail.assets.map((a) => (
                    <li key={a}>
                      <code>{a}</code>
                      <span className="tag asset">blob · lazy</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
