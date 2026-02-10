import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import type { Workflow } from "../../lib/api";

export type WorkflowCategory = "simple" | "moderate" | "complex" | "other";

export type CatalogWorkflow = Workflow & {
  category: WorkflowCategory;
};

type WorkflowCardProps = {
  workflow: CatalogWorkflow;
  selected?: boolean;
  onSelect?: (workflow: CatalogWorkflow) => void;
};

const CATEGORY_BADGE_CLASS: Record<WorkflowCategory, string> = {
  simple: "border-emerald-400/30 bg-emerald-500/10 text-emerald-100",
  moderate: "border-amber-400/30 bg-amber-500/10 text-amber-100",
  complex: "border-sky-400/30 bg-sky-500/10 text-sky-100",
  other: "border-white/20 bg-white/5 text-secondary",
};

export default function WorkflowCard({
  workflow,
  selected = false,
  onSelect,
}: WorkflowCardProps) {
  const statusLabel = formatLabel(workflow.status, "Available");
  const typeLabel = formatLabel(workflow.type, "Other");
  const showId = workflow.id.trim() && workflow.id.trim() !== workflow.title.trim();
  const versionLabel = formatLabel(workflow.version, "V1");

  return (
    <div
      className={
        "rounded-2xl border bg-[#0b0b10] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition " +
        (selected
          ? "border-gold shadow-[0_0_0_1px_rgba(212,175,55,0.25)]"
          : "border-subtle")
      }
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold leading-snug text-primary">
            {workflow.title}
          </h3>
          {showId ? (
            <p className="truncate text-xs text-secondary">{workflow.id}</p>
          ) : null}
        </div>
        <Badge
          className={`border text-[11px] ${CATEGORY_BADGE_CLASS[workflow.category]}`}
        >
          {typeLabel}
        </Badge>
      </div>

      <p className="mt-3 min-h-10 text-sm leading-relaxed text-secondary line-clamp-2">
        {workflow.description || "No description yet."}
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Version</p>
          <p className="mt-1 truncate text-sm text-primary">{versionLabel}</p>
        </div>
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Status</p>
          <p className="mt-1 truncate text-sm text-primary">{statusLabel}</p>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-end">
        <Button
          type="button"
          className={
            "h-8 rounded-full px-4 text-xs " +
            (selected
              ? "border border-gold/40 bg-gold/10 text-gold hover:bg-gold/20"
              : "border border-subtle bg-transparent text-secondary hover:text-primary")
          }
          onClick={() => onSelect?.(workflow)}
        >
          {selected ? "Selected" : "Select"}
        </Button>
      </div>
    </div>
  );
}

function formatLabel(value: string | undefined, fallback: string): string {
  const raw = (value ?? "").trim();
  if (!raw) {
    return fallback;
  }
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
