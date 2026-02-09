import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import type { Tool } from "../../lib/api";

type ToolCardProps = {
  tool: Tool;
};

export default function ToolCard({ tool }: ToolCardProps) {
  const kindLabel = tool.kind ? `${tool.kind} tool` : "Local Tool";
  const statusClass = tool.enabled
    ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-100"
    : "border-white/20 bg-white/5 text-secondary";

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-5 shadow-[0_10px_30px_rgba(0,0,0,0.25)]">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold leading-snug text-primary">
          {tool.name}
        </h3>
        <Badge className="border border-white/15 bg-white/5 text-[11px] capitalize text-secondary">
          {kindLabel}
        </Badge>
      </div>

      <p className="mt-3 min-h-10 text-sm leading-relaxed text-secondary line-clamp-2">
        {tool.description || "No description yet."}
      </p>

      <div className="mt-6 flex items-center justify-between border-t border-subtle pt-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-secondary">Status</span>
          <Badge className={`border text-[11px] ${statusClass}`}>
            {tool.enabled ? "Enabled" : "Disabled"}
          </Badge>
        </div>
        <Button
          type="button"
          className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
        >
          Configure
        </Button>
      </div>
    </div>
  );
}
