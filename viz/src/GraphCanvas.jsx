import { useCallback, useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useSize } from "./api.js";

// Shared force-graph renderer for both views: colored labeled nodes + edge labels.
export default function GraphCanvas({
  data,
  colorMap,
  onNodeClick,
  nodeRadius = 7,
  linkLabelKey = "type",
  cooldownTicks,
  onEngineStop,
}) {
  const [ref, size] = useSize();
  const fgRef = useRef(null);

  const handleEngineStop = useCallback(() => {
    fgRef.current?.zoomToFit(400, 48);
    onEngineStop?.();
  }, [onEngineStop]);

  // Spread nodes apart so labels don't overlap on these small graphs.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-260);
    fg.d3Force("link")?.distance(70);
  }, [data]);

  const paintNode = useCallback(
    (node, ctx, scale) => {
      const r = nodeRadius;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = colorMap[node.group] || "#9aa3b2";
      ctx.fill();
      ctx.lineWidth = 1.2 / scale;
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.stroke();

      const label = node.label ?? node.id;
      const fontSize = Math.max(11 / scale, 3);
      ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#e6e9ef";
      ctx.fillText(label, node.x, node.y + r + 2 / scale);
    },
    [colorMap, nodeRadius]
  );

  const nodeTooltip = useCallback((node) => {
    const lines = [`<b>${node.group}</b>: ${node.label ?? node.id}`];
    if (node.props) {
      for (const [k, v] of Object.entries(node.props)) {
        lines.push(`${k}: ${v ?? "∅"}`);
      }
    }
    if (node.count != null) lines.push(`${node.count} rows`);
    return `<div class="gtt">${lines.join("<br>")}</div>`;
  }, []);

  return (
    <div className="graph-canvas" ref={ref}>
      {size.width > 0 && (
        <ForceGraph2D
          ref={fgRef}
          width={size.width}
          height={size.height}
          graphData={data}
          backgroundColor="#0f1115"
          nodeCanvasObject={paintNode}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, nodeRadius + 3, 0, 2 * Math.PI);
            ctx.fill();
          }}
          nodeLabel={nodeTooltip}
          onNodeClick={onNodeClick}
          linkColor={() => "rgba(125,134,150,0.6)"}
          linkWidth={1.5}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkLabel={(l) => l[linkLabelKey] || ""}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={(link, ctx, scale) => {
            const label = link[linkLabelKey];
            if (!label || scale < 1.2) return;
            const { source: s, target: t } = link;
            if (!s.x || !t.x) return;
            const x = (s.x + t.x) / 2;
            const y = (s.y + t.y) / 2;
            const fontSize = 9 / scale;
            ctx.font = `${fontSize}px -apple-system, sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "rgba(230,233,239,0.7)";
            ctx.fillText(label, x, y);
          }}
          cooldownTicks={cooldownTicks}
          onEngineStop={handleEngineStop}
        />
      )}
    </div>
  );
}
