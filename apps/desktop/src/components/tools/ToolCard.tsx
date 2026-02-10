import { Badge } from "../ui/badge";
import type { Tool } from "../../lib/api";

export type ToolCategory =
  | "local"
  | "external"
  | "vision"
  | "workflow"
  | "other";

export type CatalogTool = Tool & {
  category: ToolCategory;
};

type ToolCardProps = {
  tool: CatalogTool;
};

export default function ToolCard({ tool }: ToolCardProps) {
  const statusClass = tool.enabled
    ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-100"
    : "border-white/20 bg-white/5 text-secondary";
  const showId = tool.id.trim() && tool.id.trim() !== tool.name.trim();
  const kindLabel = formatKindLabel(tool.kind);

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.25)]">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold leading-snug text-primary">
            {tool.name}
          </h3>
          {showId ? <p className="truncate text-xs text-secondary">{tool.id}</p> : null}
        </div>
        <Badge
          className={`border text-[11px] ${CATEGORY_BADGE_CLASS[tool.category]}`}
        >
          {CATEGORY_LABEL[tool.category]}
        </Badge>
      </div>

      <p className="mt-3 min-h-10 text-sm leading-relaxed text-secondary line-clamp-2">
        {tool.description || "No description yet."}
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Kind</p>
          <p className="mt-1 truncate text-sm text-primary">{kindLabel}</p>
        </div>
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Status</p>
          <div className="mt-1">
            <Badge className={`border text-[11px] ${statusClass}`}>
              {tool.enabled ? "Enabled" : "Disabled"}
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}

const CATEGORY_BADGE_CLASS: Record<ToolCategory, string> = {
  local: "border-emerald-400/40 bg-emerald-500/10 text-emerald-100",
  external: "border-sky-400/40 bg-sky-500/10 text-sky-100",
  vision: "border-amber-400/40 bg-amber-500/10 text-amber-100",
  workflow: "border-violet-400/40 bg-violet-500/10 text-violet-100",
  other: "border-white/10 bg-white/5 text-secondary",
};

const CATEGORY_LABEL: Record<ToolCategory, string> = {
  local: "Local",
  external: "External",
  vision: "Vision",
  workflow: "Workflow",
  other: "Other",
};

function formatKindLabel(kind?: string): string {
  const raw = (kind ?? "").trim();
  if (!raw) {
    return "Local";
  }
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
