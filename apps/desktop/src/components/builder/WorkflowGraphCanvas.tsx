import { useEffect, useMemo, useState } from "react";
import type { EdgeSpec, NodeSpec } from "@saturday/shared/workflow";

const NODE_WIDTH = 188;
const NODE_HEIGHT = 104;

type DragState = {
  nodeId: string;
  offsetX: number;
  offsetY: number;
} | null;

type WorkflowGraphCanvasProps = {
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onNodePositionChange: (nodeId: string, x: number, y: number) => void;
};

function nodeColor(node: NodeSpec): string {
  if (node.type === "llm") {
    return "border-sky-500/50 bg-sky-500/10";
  }
  if (node.type === "tool") {
    return "border-emerald-500/50 bg-emerald-500/10";
  }
  if (node.type === "conditional") {
    return "border-amber-500/50 bg-amber-500/10";
  }
  if (node.type === "verify") {
    return "border-cyan-500/50 bg-cyan-500/10";
  }
  return "border-gold/60 bg-gold/10";
}

export default function WorkflowGraphCanvas({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  onNodePositionChange,
}: WorkflowGraphCanvasProps) {
  const [dragState, setDragState] = useState<DragState>(null);

  const nodeMap = useMemo(() => {
    const map = new Map<string, NodeSpec>();
    for (const node of nodes) {
      map.set(node.id, node);
    }
    return map;
  }, [nodes]);

  useEffect(() => {
    if (!dragState) {
      return;
    }
    const onMouseMove = (event: MouseEvent) => {
      const x = event.clientX - dragState.offsetX;
      const y = event.clientY - dragState.offsetY;
      onNodePositionChange(dragState.nodeId, Math.max(0, x), Math.max(0, y));
    };
    const onMouseUp = () => {
      setDragState(null);
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [dragState, onNodePositionChange]);

  const maxX = Math.max(1200, ...nodes.map((node) => (node.position?.x || 0) + NODE_WIDTH + 120));
  const maxY = Math.max(700, ...nodes.map((node) => (node.position?.y || 0) + NODE_HEIGHT + 120));

  return (
    <div
      className="relative min-h-[38rem] overflow-auto rounded-2xl border border-subtle bg-[#08080c]"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onSelectNode(null);
        }
      }}
    >
      <div className="relative" style={{ width: maxX, height: maxY }}>
        <svg className="absolute inset-0 h-full w-full">
          {edges.map((edge) => {
            const fromNode = nodeMap.get(edge.from);
            const toNode = nodeMap.get(edge.to);
            if (!fromNode || !toNode) {
              return null;
            }

            const fromX = (fromNode.position?.x || 0) + NODE_WIDTH;
            const fromY = (fromNode.position?.y || 0) + NODE_HEIGHT / 2;
            const toX = toNode.position?.x || 0;
            const toY = (toNode.position?.y || 0) + NODE_HEIGHT / 2;
            const bend = Math.max(30, Math.abs(toX - fromX) * 0.45);

            const path = `M ${fromX} ${fromY} C ${fromX + bend} ${fromY}, ${toX - bend} ${toY}, ${toX} ${toY}`;
            return (
              <g key={edge.id}>
                <path d={path} stroke="rgba(148,163,184,0.55)" fill="none" strokeWidth={2} />
                <text
                  x={(fromX + toX) / 2}
                  y={(fromY + toY) / 2 - 8}
                  fill="rgba(148,163,184,0.9)"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {edge.label || "always"}
                </text>
              </g>
            );
          })}
        </svg>

        {nodes.map((node) => {
          const selected = node.id === selectedNodeId;
          const x = node.position?.x || 0;
          const y = node.position?.y || 0;
          return (
            <button
              key={node.id}
              type="button"
              className={
                "absolute rounded-xl border p-3 text-left shadow-sm transition " +
                nodeColor(node) +
                (selected ? " ring-2 ring-gold/60" : "")
              }
              style={{ left: x, top: y, width: NODE_WIDTH, height: NODE_HEIGHT }}
              onClick={() => onSelectNode(node.id)}
              onMouseDown={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                setDragState({
                  nodeId: node.id,
                  offsetX: event.clientX - rect.left,
                  offsetY: event.clientY - rect.top,
                });
              }}
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-[10px] uppercase tracking-wide text-secondary">
                  {node.type}
                </span>
                <span className="text-[10px] text-secondary">{node.id}</span>
              </div>
              <p className="line-clamp-2 text-sm font-medium text-primary">
                {node.label || node.id}
              </p>
              <div className="mt-2 text-[11px] text-secondary">
                {node.reads.length} reads · {node.writes.length} writes
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
