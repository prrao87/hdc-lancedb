import { useEffect, useLayoutEffect, useRef, useState } from "react";

const BASE = "http://localhost:8000";

export async function getSchema() {
  const res = await fetch(`${BASE}/schema`);
  if (!res.ok) throw new Error(`/schema failed: ${res.status}`);
  return res.json();
}

// URL for one node's image asset. The bytes live in a lazy blob column and are
// fetched only when this URL is actually requested (e.g. an <img> is rendered).
export function imageUrl(label, rawId) {
  return `${BASE}/image/${encodeURIComponent(label)}/${encodeURIComponent(rawId)}`;
}

export async function runQuery(spec) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(spec),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `/query failed: ${res.status}`);
  return data;
}

// Color palette keyed by label order, shared across both views.
const PALETTE = [
  "#6ea8fe", "#f0a35e", "#5fd38a", "#c08cf0",
  "#e8688a", "#5ec8d8", "#d8c45e", "#a0aab8",
];

export function buildColorMap(labels) {
  return Object.fromEntries(labels.map((l, i) => [l, PALETTE[i % PALETTE.length]]));
}

// Measure a container so ForceGraph2D can be sized to fill it. Reacts to element
// resizes (panel/tab changes) and window resizes, and only commits real changes.
export function useSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () =>
      setSize((prev) =>
        prev.width === el.clientWidth && prev.height === el.clientHeight
          ? prev
          : { width: el.clientWidth, height: el.clientHeight }
      );
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    measure();
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);
  return [ref, size];
}

export function useSchema() {
  const [schema, setSchema] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    getSchema().then(setSchema).catch((e) => setError(e.message));
  }, []);
  return { schema, error };
}
