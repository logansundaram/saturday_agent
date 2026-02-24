import { useEffect, useMemo, useState } from "react";
import type {
  ReplayDiagnostic,
  ReplayPatchMode,
  RunStepDetail,
  RunStepSummary,
} from "@saturday/shared/run";
import { replayRun, replayRunDryRun } from "../../lib/api";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import {
  buildReplayRequestFromEditor,
  hasReplayDiagnosticErrors,
  normalizeReplayDiagnostics,
  stringifyJsonForEditor,
} from "./stateDiff";

type ReplayModalProps = {
  open: boolean;
  runId: string;
  step: RunStepSummary | null;
  stepDetail: RunStepDetail | null;
  defaultSandbox: boolean;
  onClose: () => void;
  onReplayCreated: (newRunId: string) => void;
};

function severityTone(severity: string): string {
  if (severity === "info") {
    return "border-sky-400/30 bg-sky-500/10 text-sky-100";
  }
  if (severity === "warning") {
    return "border-amber-400/30 bg-amber-500/10 text-amber-100";
  }
  return "border-rose-400/30 bg-rose-500/10 text-rose-100";
}

export default function ReplayModal({
  open,
  runId,
  step,
  stepDetail,
  defaultSandbox,
  onClose,
  onReplayCreated,
}: ReplayModalProps) {
  const [baseState, setBaseState] = useState<"pre" | "post">("post");
  const [patchMode, setPatchMode] = useState<ReplayPatchMode>("overlay");
  const [replayThisStep, setReplayThisStep] = useState<boolean>(false);
  const [sandbox, setSandbox] = useState<boolean>(defaultSandbox);
  const [patchText, setPatchText] = useState<string>("{}");
  const [parseError, setParseError] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<ReplayDiagnostic[]>([]);
  const [validating, setValidating] = useState<boolean>(false);
  const [running, setRunning] = useState<boolean>(false);

  useEffect(() => {
    if (!open) {
      return;
    }
    setBaseState("post");
    setPatchMode("overlay");
    setReplayThisStep(false);
    setSandbox(defaultSandbox);
    setPatchText("{}");
    setParseError(null);
    setDiagnostics([]);
    setValidating(false);
    setRunning(false);
  }, [defaultSandbox, open, step?.step_id]);

  const hasErrors = useMemo(
    () => hasReplayDiagnosticErrors(diagnostics),
    [diagnostics]
  );

  const canRun = Boolean(step && !parseError && !hasErrors && !running);

  const loadStateAsPatch = (snapshot: unknown, nextBaseState: "pre" | "post") => {
    setPatchMode("replace");
    setBaseState(nextBaseState);
    setPatchText(stringifyJsonForEditor(snapshot, "{}"));
    setParseError(null);
    setDiagnostics([]);
  };

  const buildRequest = () => {
    if (!step) {
      setParseError("No step selected.");
      return null;
    }
    const built = buildReplayRequestFromEditor({
      fromStepId: step.step_id,
      patchText,
      patchMode,
      sandbox,
      baseState,
      replayThisStep,
    });
    setParseError(built.parseError);
    return built.request;
  };

  const onValidate = async () => {
    const request = buildRequest();
    if (!request) {
      return;
    }
    setValidating(true);
    try {
      const response = await replayRunDryRun(runId, request);
      setDiagnostics(normalizeReplayDiagnostics(response.diagnostics));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to validate replay request.";
      setDiagnostics(
        normalizeReplayDiagnostics([
          {
            code: "REQUEST_FAILED",
            severity: "error",
            message,
          },
        ])
      );
    } finally {
      setValidating(false);
    }
  };

  const onRun = async () => {
    const request = buildRequest();
    if (!request) {
      return;
    }
    setRunning(true);
    try {
      const response = await replayRun(runId, request);
      const normalizedDiagnostics = normalizeReplayDiagnostics(response.diagnostics);
      setDiagnostics(normalizedDiagnostics);
      if (hasReplayDiagnosticErrors(normalizedDiagnostics) || !response.new_run_id) {
        return;
      }
      onReplayCreated(response.new_run_id);
      onClose();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to start replay run.";
      setDiagnostics(
        normalizeReplayDiagnostics([
          {
            code: "REQUEST_FAILED",
            severity: "error",
            message,
          },
        ])
      );
    } finally {
      setRunning(false);
    }
  };

  if (!open) {
    return null;
  }

  const preState = stepDetail?.pre_state;
  const postState = stepDetail?.post_state;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6">
      <div className="w-full max-w-4xl rounded-2xl border border-subtle bg-[#0b0b10] p-5 shadow-2xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base text-primary">Replay From Step</h2>
            <p className="mt-1 text-xs text-secondary">
              {step ? `#${step.step_index} ${step.name}` : "No step selected"}
            </p>
          </div>
          <Button
            type="button"
            className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
            onClick={onClose}
          >
            Close
          </Button>
        </div>

        <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
            <p className="text-[11px] uppercase tracking-wide text-secondary">Replay Base</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button
                type="button"
                className={
                  "h-7 rounded-full border px-3 text-xs " +
                  (baseState === "pre"
                    ? "border-gold/40 bg-gold/10 text-gold"
                    : "border-subtle bg-transparent text-secondary hover:text-primary")
                }
                onClick={() => setBaseState("pre")}
              >
                Pre State
              </Button>
              <Button
                type="button"
                className={
                  "h-7 rounded-full border px-3 text-xs " +
                  (baseState === "post"
                    ? "border-gold/40 bg-gold/10 text-gold"
                    : "border-subtle bg-transparent text-secondary hover:text-primary")
                }
                onClick={() => setBaseState("post")}
              >
                Post State
              </Button>
            </div>
          </div>

          <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
            <p className="text-[11px] uppercase tracking-wide text-secondary">Replay Mode</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button
                type="button"
                className={
                  "h-7 rounded-full border px-3 text-xs " +
                  (!replayThisStep
                    ? "border-gold/40 bg-gold/10 text-gold"
                    : "border-subtle bg-transparent text-secondary hover:text-primary")
                }
                onClick={() => setReplayThisStep(false)}
              >
                Replay Next Step
              </Button>
              <Button
                type="button"
                className={
                  "h-7 rounded-full border px-3 text-xs " +
                  (replayThisStep
                    ? "border-gold/40 bg-gold/10 text-gold"
                    : "border-subtle bg-transparent text-secondary hover:text-primary")
                }
                onClick={() => setReplayThisStep(true)}
              >
                Replay This Step
              </Button>
            </div>
          </div>
        </div>

        <div className="mb-4 rounded-xl border border-subtle bg-[#08080c] p-3">
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <label className="text-[11px] uppercase tracking-wide text-secondary" htmlFor="patchMode">
              Patch Mode
            </label>
            <select
              id="patchMode"
              value={patchMode}
              onChange={(event) => {
                const nextValue = event.target.value as ReplayPatchMode;
                setPatchMode(nextValue);
                if (nextValue === "jsonpatch" && patchText.trim() === "{}") {
                  setPatchText("[]");
                }
              }}
              className="h-8 rounded-md border border-subtle bg-[#0b0b10] px-2 text-xs text-primary"
            >
              <option value="overlay">overlay</option>
              <option value="replace">replace</option>
              <option value="jsonpatch">jsonpatch</option>
            </select>

            <label className="ml-auto flex items-center gap-2 text-xs text-secondary">
              <input
                type="checkbox"
                checked={sandbox}
                onChange={(event) => setSandbox(Boolean(event.target.checked))}
                className="accent-[#d4af37]"
              />
              Sandbox mode
            </label>
          </div>

          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Button
              type="button"
              className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => loadStateAsPatch(preState, "pre")}
              disabled={stepDetail == null}
            >
              Load Pre State
            </Button>
            <Button
              type="button"
              className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => loadStateAsPatch(postState, "post")}
              disabled={stepDetail == null}
            >
              Load Post State
            </Button>
          </div>

          <Textarea
            value={patchText}
            onChange={(event) => {
              setPatchText(event.target.value);
              setParseError(null);
            }}
            className="min-h-[16rem] font-mono text-xs leading-relaxed"
            spellCheck={false}
          />
          {parseError ? (
            <div className="mt-2 rounded-lg border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
              Invalid JSON: {parseError}
            </div>
          ) : null}
        </div>

        <div className="mb-4">
          {diagnostics.length > 0 ? (
            <div className="space-y-2">
              {diagnostics.map((item, index) => (
                <div
                  key={`${item.code}:${item.path ?? ""}:${index}`}
                  className={`rounded-lg border px-3 py-2 text-xs ${severityTone(item.severity)}`}
                >
                  <div className="font-medium">{item.message}</div>
                  <div className="mt-1 text-[11px] opacity-90">
                    {item.code}
                    {item.path ? ` · ${item.path}` : ""}
                    {item.expected ? ` · expected ${item.expected}` : ""}
                    {item.actual ? ` · actual ${item.actual}` : ""}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-subtle bg-[#08080c] px-3 py-2 text-xs text-secondary">
              Validate before running to preview schema issues.
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <Button
            type="button"
            className="h-8 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
            onClick={() => void onValidate()}
            disabled={validating || running || !step}
          >
            {validating ? "Validating..." : "Validate"}
          </Button>
          <Button
            type="button"
            className="h-8 rounded-full border border-gold/40 bg-gold/10 px-4 text-xs text-gold hover:bg-gold/20 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={() => void onRun()}
            disabled={!canRun}
          >
            {running ? "Starting Replay..." : "Run Replay"}
          </Button>
        </div>
      </div>
    </div>
  );
}
