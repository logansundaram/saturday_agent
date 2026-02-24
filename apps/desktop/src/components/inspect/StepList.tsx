import type { Step } from "../../lib/api";

type StepListProps = {
  steps: Step[];
  selectedIndex: number;
  onSelect: (index: number) => void;
};

function durationLabel(step: Step): string {
  const started = step.started_at ? Date.parse(step.started_at) : NaN;
  const ended = step.ended_at ? Date.parse(step.ended_at) : NaN;
  if (!Number.isFinite(started) || !Number.isFinite(ended)) {
    return "--";
  }
  const delta = Math.max(0, ended - started);
  if (delta < 1000) {
    return `${delta}ms`;
  }
  return `${(delta / 1000).toFixed(2)}s`;
}

export default function StepList({ steps, selectedIndex, onSelect }: StepListProps) {
  if (steps.length === 0) {
    return (
      <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-4 py-5 text-sm text-secondary">
        No steps recorded for this run.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-2">
      <div className="max-h-[32rem] space-y-1 overflow-auto pr-1">
        {steps.map((step, index) => {
          const selected = index === selectedIndex;
          const normalizedStatus = String(step.status || "").toLowerCase();
          const ok = normalizedStatus === "ok" || normalizedStatus === "success" || normalizedStatus === "completed";
          const running = normalizedStatus === "running" || normalizedStatus === "pending";
          const skipped = normalizedStatus === "skipped";
          return (
            <button
              key={`${step.step_index}-${step.name}-${index}`}
              type="button"
              onClick={() => onSelect(index)}
              className={
                "w-full rounded-xl border px-3 py-2 text-left transition " +
                (selected
                  ? "border-gold/40 bg-gold/10"
                  : "border-subtle bg-transparent hover:bg-white/5")
              }
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-secondary">
                  #{step.step_index}
                </span>
                <span
                  className={
                    "h-2 w-2 rounded-full " +
                    (ok ? "bg-emerald-400" : running ? "bg-sky-400" : skipped ? "bg-zinc-400" : "bg-rose-400")
                  }
                  aria-hidden="true"
                />
              </div>
              <div className="mt-1 truncate text-sm text-primary">{step.name}</div>
              <div className="mt-1 text-[11px] text-secondary">
                {normalizedStatus || "unknown"} · {durationLabel(step)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
