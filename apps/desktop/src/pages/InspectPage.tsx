import { useCallback, useEffect, useMemo, useState } from "react";
import StepList from "../components/inspect/StepList";
import JsonView from "../components/inspect/JsonView";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { getRun, getRunLogs, getRunState } from "../lib/api";
import type { Run, RunSnapshot, Step } from "../lib/api";

type InspectPageProps = {
  runId?: string | null;
  onBack: () => void;
};

type InspectTab = "step" | "tools" | "state";

type ToolEntry = {
  key: string;
  tool_id: string;
  status: string;
  duration: string;
  step_name: string;
  step_index: number;
  input: unknown;
  output: unknown;
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

function collectToolEntries(step: Step): ToolEntry[] {
  const entries: ToolEntry[] = [];
  const stepDuration = durationLabel(step);
  const baseKey = `${step.step_index}:${step.name}`;
  const stepOutput = step.output;

  if (step.name.startsWith("tool.")) {
    entries.push({
      key: `${baseKey}:direct`,
      tool_id: step.name.replace("tool.", "") || step.name,
      status: step.status,
      duration: stepDuration,
      step_name: step.name,
      step_index: step.step_index,
      input: step.input,
      output: stepOutput,
    });
  }

  if (stepOutput && typeof stepOutput === "object") {
    const outputRecord = stepOutput as Record<string, unknown>;
    const toolCalls = Array.isArray(outputRecord.tool_calls)
      ? outputRecord.tool_calls
      : [];
    const toolResults = Array.isArray(outputRecord.tool_results)
      ? outputRecord.tool_results
      : [];

    toolCalls.forEach((item, index) => {
      if (!item || typeof item !== "object") {
        return;
      }
      const callRecord = item as Record<string, unknown>;
      const toolId = String(callRecord.tool_id ?? callRecord.id ?? "tool");
      entries.push({
        key: `${baseKey}:call:${index}:${toolId}`,
        tool_id: toolId,
        status: String(callRecord.status ?? step.status ?? "ok"),
        duration: stepDuration,
        step_name: step.name,
        step_index: step.step_index,
        input: callRecord.input,
        output: callRecord.output,
      });
    });

    toolResults.forEach((item, index) => {
      if (!item || typeof item !== "object") {
        return;
      }
      const resultRecord = item as Record<string, unknown>;
      const toolId = String(resultRecord.tool_id ?? resultRecord.id ?? "tool");
      entries.push({
        key: `${baseKey}:result:${index}:${toolId}`,
        tool_id: toolId,
        status: String(resultRecord.status ?? step.status ?? "ok"),
        duration: stepDuration,
        step_name: step.name,
        step_index: step.step_index,
        input: null,
        output: resultRecord.output ?? resultRecord,
      });
    });
  }

  return entries;
}

function summarizeRunId(runId: string): string {
  if (runId.length <= 18) {
    return runId;
  }
  return `${runId.slice(0, 10)}...${runId.slice(-6)}`;
}

export default function InspectPage({ runId, onBack }: InspectPageProps) {
  const [run, setRun] = useState<Run | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [snapshots, setSnapshots] = useState<RunSnapshot[]>([]);
  const [stateDerived, setStateDerived] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<InspectTab>("step");
  const [selectedStepIndex, setSelectedStepIndex] = useState<number>(0);
  const [selectedSnapshotIndex, setSelectedSnapshotIndex] = useState<number>(0);

  const loadData = useCallback(async () => {
    if (!runId) {
      setRun(null);
      setSteps([]);
      setSnapshots([]);
      setStateDerived(false);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [runData, logsData, stateData] = await Promise.all([
        getRun(runId),
        getRunLogs(runId),
        getRunState(runId),
      ]);
      setRun(runData);
      setSteps(logsData.steps);

      const nextSnapshots = stateData?.snapshots ?? [];
      setSnapshots(nextSnapshots);
      setStateDerived(Boolean(stateData?.derived));

      setSelectedStepIndex(0);
      setSelectedSnapshotIndex(nextSnapshots.length > 0 ? nextSnapshots.length - 1 : 0);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load run.";
      setError(message);
      setRun(null);
      setSteps([]);
      setSnapshots([]);
      setStateDerived(false);
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const selectedStep = steps[selectedStepIndex];
  const selectedSnapshot = snapshots[selectedSnapshotIndex];

  const toolEntries = useMemo(() => {
    if (selectedStep) {
      const selected = collectToolEntries(selectedStep);
      if (selected.length > 0) {
        return selected;
      }
    }
    return steps.flatMap((step) => collectToolEntries(step));
  }, [selectedStep, steps]);

  const runStatus = String(run?.status ?? "").toLowerCase() === "ok" ? "ok" : "error";
  const toolIds = Array.isArray(run?.tool_ids) ? run.tool_ids : [];

  const copyRunId = async () => {
    try {
      await navigator.clipboard.writeText(runId || "");
    } catch {
      // Non-blocking if clipboard access is unavailable.
    }
  };

  if (!runId) {
    return (
      <div className="bg-panel section-base pr-64 relative min-h-screen">
        <div className="section-hero pb-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="section-header text-5xl">Inspect</h1>
              <p className="section-framer text-secondary">
                Open a chat run to inspect step-by-step execution.
              </p>
            </div>
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={onBack}
            >
              Go to Chat
            </Button>
          </div>
        </div>
        <div className="px-4 pb-8">
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <p>No active run selected. Send a prompt in Chat, then click Inspect.</p>
              <Button
                type="button"
                className="h-8 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
                onClick={onBack}
              >
                Open Chat
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-panel section-base pr-64 relative min-h-screen">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Inspect Run</h1>
            <p className="section-framer text-secondary">
              Detailed execution trace for workflow runs.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={onBack}
            >
              Back to Chat
            </Button>
            <Badge
              className={
                "border " +
                (runStatus === "ok"
                  ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
                  : "border-rose-400/40 bg-rose-500/10 text-rose-100")
              }
            >
              {runStatus.toUpperCase()}
            </Badge>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-2xl border border-subtle bg-[#0b0b10] px-4 py-3">
          <span className="text-xs uppercase tracking-wide text-secondary">Run ID</span>
          <code className="rounded-md border border-subtle bg-[#08080c] px-2 py-1 text-xs text-primary">
            {summarizeRunId(runId)}
          </code>
          <Button
            type="button"
            className="h-7 rounded-full border border-subtle bg-transparent px-2.5 text-xs text-secondary hover:text-primary"
            onClick={() => void copyRunId()}
          >
            Copy
          </Button>
        </div>
      </div>

      <div className="px-4 pb-8">
        {error ? (
          <div className="mb-4 rounded-2xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span>{error}</span>
              <Button
                type="button"
                className="h-8 rounded-full border border-rose-400/40 bg-transparent px-3 text-xs text-rose-100 hover:bg-rose-500/10"
                onClick={() => void loadData()}
              >
                Retry
              </Button>
            </div>
          </div>
        ) : null}

        {loading ? (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
            <div className="space-y-3">
              <div className="h-24 animate-pulse rounded-2xl border border-subtle bg-[#0b0b10]" />
              <div className="h-24 animate-pulse rounded-2xl border border-subtle bg-[#0b0b10]" />
              <div className="h-80 animate-pulse rounded-2xl border border-subtle bg-[#0b0b10]" />
            </div>
            <div className="h-[36rem] animate-pulse rounded-2xl border border-subtle bg-[#0b0b10]" />
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
            <div className="space-y-3">
              <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
                <p className="text-xs uppercase tracking-wide text-secondary">Workflow</p>
                <p className="mt-2 text-sm text-primary">{run?.workflow_id || "N/A"}</p>
                <p className="mt-1 text-xs text-secondary">{run?.workflow_type || "N/A"}</p>
              </div>

              <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
                <p className="text-xs uppercase tracking-wide text-secondary">Model</p>
                <p className="mt-2 text-sm text-primary">{run?.model_id || "N/A"}</p>
              </div>

              <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
                <p className="text-xs uppercase tracking-wide text-secondary">Tools</p>
                {toolIds.length === 0 ? (
                  <p className="mt-2 text-sm text-secondary">No tools selected.</p>
                ) : (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {toolIds.map((toolId) => (
                      <span
                        key={toolId}
                        className="rounded-full border border-subtle bg-transparent px-2 py-1 text-[11px] text-secondary"
                      >
                        {toolId}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <StepList
                steps={steps}
                selectedIndex={selectedStepIndex}
                onSelect={setSelectedStepIndex}
              />
            </div>

            <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  className={
                    "h-8 rounded-full border px-3 text-xs " +
                    (activeTab === "step"
                      ? "border-gold/40 bg-gold/10 text-gold"
                      : "border-subtle bg-transparent text-secondary hover:text-primary")
                  }
                  onClick={() => setActiveTab("step")}
                >
                  Step
                </Button>
                <Button
                  type="button"
                  className={
                    "h-8 rounded-full border px-3 text-xs " +
                    (activeTab === "tools"
                      ? "border-gold/40 bg-gold/10 text-gold"
                      : "border-subtle bg-transparent text-secondary hover:text-primary")
                  }
                  onClick={() => setActiveTab("tools")}
                >
                  Tool Calls
                </Button>
                <Button
                  type="button"
                  className={
                    "h-8 rounded-full border px-3 text-xs " +
                    (activeTab === "state"
                      ? "border-gold/40 bg-gold/10 text-gold"
                      : "border-subtle bg-transparent text-secondary hover:text-primary")
                  }
                  onClick={() => setActiveTab("state")}
                >
                  State
                </Button>
              </div>

              {activeTab === "step" ? (
                selectedStep ? (
                  <div className="space-y-4">
                    <div className="rounded-2xl border border-subtle bg-[#08080c] px-3 py-2 text-xs text-secondary">
                      <span className="text-primary">{selectedStep.name}</span>
                      {" · "}
                      step #{selectedStep.step_index}
                      {" · "}
                      {selectedStep.status}
                      {" · "}
                      {durationLabel(selectedStep)}
                    </div>
                    <JsonView title="Input" value={selectedStep.input} />
                    <JsonView title="Output" value={selectedStep.output} />
                    {selectedStep.error ? (
                      <JsonView title="Error" value={selectedStep.error} />
                    ) : null}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-subtle bg-[#08080c] px-4 py-5 text-sm text-secondary">
                    No step selected.
                  </div>
                )
              ) : null}

              {activeTab === "tools" ? (
                toolEntries.length === 0 ? (
                  <div className="rounded-2xl border border-subtle bg-[#08080c] px-4 py-5 text-sm text-secondary">
                    No tool activity found in this run.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {toolEntries.map((entry) => (
                      <div
                        key={entry.key}
                        className="rounded-2xl border border-subtle bg-[#08080c] p-3"
                      >
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs text-secondary">
                          <span className="text-primary">{entry.tool_id}</span>
                          <span>
                            {entry.status} · {entry.duration}
                          </span>
                        </div>
                        <div className="mb-3 text-[11px] text-secondary">
                          Step #{entry.step_index}: {entry.step_name}
                        </div>
                        <div className="space-y-3">
                          <JsonView title="Tool Input" value={entry.input} initiallyCollapsed />
                          <JsonView title="Tool Output" value={entry.output} initiallyCollapsed />
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : null}

              {activeTab === "state" ? (
                snapshots.length === 0 ? (
                  <div className="rounded-2xl border border-subtle bg-[#08080c] px-4 py-5 text-sm text-secondary">
                    No state snapshots available for this run.
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-2xl border border-subtle bg-[#08080c] p-3">
                      <div className="mb-2 flex items-center justify-between text-xs text-secondary">
                        <span>Snapshot {selectedSnapshotIndex + 1} / {snapshots.length}</span>
                        <span>
                          Step #{selectedSnapshot?.step_index ?? "-"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, snapshots.length - 1)}
                        value={selectedSnapshotIndex}
                        onChange={(event) =>
                          setSelectedSnapshotIndex(Number(event.target.value))
                        }
                        className="w-full accent-[#d4af37]"
                      />
                      <div className="mt-2 text-[11px] text-secondary">
                        {selectedSnapshot?.timestamp || ""}
                      </div>
                    </div>
                    {stateDerived ? (
                      <div className="rounded-xl border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
                        Best-effort state snapshots were derived from step outputs.
                      </div>
                    ) : null}
                    <JsonView title="State Snapshot" value={selectedSnapshot?.state ?? {}} />
                  </div>
                )
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
