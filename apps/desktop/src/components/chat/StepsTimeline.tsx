import type { ChatRunTimelineStep } from "../../lib/api";

type StepsTimelineProps = {
  steps: ChatRunTimelineStep[];
};

function statusDotClass(status: ChatRunTimelineStep["status"]): string {
  if (status === "ok") {
    return "bg-emerald-400";
  }
  if (status === "error") {
    return "bg-rose-400";
  }
  return "bg-amber-300";
}

function stepDurationMs(step: ChatRunTimelineStep): number | null {
  if (typeof step.duration_ms === "number" && Number.isFinite(step.duration_ms)) {
    return Math.max(0, step.duration_ms);
  }
  if (!step.started_at || !step.ended_at) {
    return null;
  }
  const startedAt = Date.parse(step.started_at);
  const endedAt = Date.parse(step.ended_at);
  if (!Number.isFinite(startedAt) || !Number.isFinite(endedAt)) {
    return null;
  }
  return Math.max(0, endedAt - startedAt);
}

function formatDuration(step: ChatRunTimelineStep): string {
  const durationMs = stepDurationMs(step);
  if (durationMs === null) {
    return "";
  }
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  return `${(durationMs / 1000).toFixed(1)}s`;
}

export default function StepsTimeline({ steps }: StepsTimelineProps) {
  if (!Array.isArray(steps) || steps.length === 0) {
    return null;
  }

  const ordered = [...steps].sort((left, right) => left.step_index - right.step_index);

  return (
    <div className="mb-2 rounded-xl border border-subtle bg-black/20 px-3 py-2">
      <p className="mb-2 text-[11px] uppercase tracking-wide text-secondary">Steps</p>
      <div className="space-y-1.5">
        {ordered.map((step) => {
          const duration = formatDuration(step);
          const title = step.label || step.name;
          return (
            <div key={`${step.step_index}:${step.name}`} className="rounded-md bg-black/10 px-2 py-1.5">
              <div className="flex items-center gap-2 text-xs">
                <span className={`inline-block h-2 w-2 rounded-full ${statusDotClass(step.status)}`} />
                <span className="min-w-0 flex-1 truncate text-primary">{title}</span>
                {duration ? <span className="text-secondary">{duration}</span> : null}
              </div>
              {step.summary ? (
                <p className="mt-1 text-[11px] leading-4 text-secondary">{step.summary}</p>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
