import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import type { Workflow } from "../../lib/api";

type WorkflowCardProps = {
  workflow: Workflow;
  selected?: boolean;
  onSelect?: (workflow: Workflow) => void;
};

const typeClasses: Record<string, string> = {
  simple: "border-emerald-400/30 bg-emerald-500/10 text-emerald-100",
  moderate: "border-amber-400/30 bg-amber-500/10 text-amber-100",
  complex: "border-sky-400/30 bg-sky-500/10 text-sky-100",
};

export default function WorkflowCard({
  workflow,
  selected = false,
  onSelect,
}: WorkflowCardProps) {
  const statusLabel = workflow.status ?? "available";
  const workflowType = workflow.type.toLowerCase();
  const badgeClass =
    typeClasses[workflowType] ?? "border-white/20 bg-white/5 text-secondary";

  return (
    <div
      className={
        "rounded-2xl border bg-[#0b0b10] p-5 shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition " +
        (selected ? "border-gold" : "border-subtle")
      }
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold leading-snug text-primary">
          {workflow.title}
        </h3>
        <Badge className={`border text-[11px] capitalize ${badgeClass}`}>
          {workflowType}
        </Badge>
      </div>

      <p className="mt-3 min-h-10 text-sm leading-relaxed text-secondary line-clamp-2">
        {workflow.description || "No description yet."}
      </p>

      <div className="mt-6 flex items-center justify-between border-t border-subtle pt-4">
        <span className="text-xs text-secondary">
          Status: <span className="capitalize text-primary">{statusLabel}</span>
        </span>
        <Button
          type="button"
          className="h-8 rounded-full bg-gold px-4 text-xs text-black hover:bg-[#e1c161]"
          onClick={() => onSelect?.(workflow)}
        >
          Select
        </Button>
      </div>
    </div>
  );
}
