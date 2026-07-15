import { useMemo, useState } from "react";
import { buildColorMap, useSchema } from "./api.js";
import SchemaView from "./SchemaView.jsx";
import InstanceView from "./InstanceView.jsx";

export default function App() {
  const { schema, error } = useSchema();
  const [tab, setTab] = useState("instance");

  const colorMap = useMemo(
    () => (schema ? buildColorMap(schema.nodes.map((n) => n.label)) : {}),
    [schema]
  );

  return (
    <div className="app">
      <header className="topbar">
        <h1>lance-graph visualizer</h1>
        <nav className="tabs">
          <button className={tab === "schema" ? "active" : ""} onClick={() => setTab("schema")}>
            Schema
          </button>
          <button className={tab === "instance" ? "active" : ""} onClick={() => setTab("instance")}>
            Instance
          </button>
        </nav>
      </header>

      {error && (
        <div className="boot-error">
          Couldn’t reach the API at <code>localhost:8000</code> — start it with
          <code> uv run uvicorn graph_api:app --app-dir src</code>.<br />
          <small>{error}</small>
        </div>
      )}

      {!schema && !error && <div className="boot">Loading schema…</div>}

      {schema && (
        <main className="content">
          {tab === "schema" ? (
            <SchemaView schema={schema} colorMap={colorMap} />
          ) : (
            <InstanceView schema={schema} colorMap={colorMap} />
          )}
        </main>
      )}
    </div>
  );
}
