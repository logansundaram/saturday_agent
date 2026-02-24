import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  NodeType,
  ValidationDiagnostic,
  WorkflowSpec,
} from "@saturday/shared/workflow";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import type { Tool } from "../../lib/api";
import {
  approveToolCall,
  compileWorkflowSpec,
  createWorkflowVersion,
  getPendingToolCalls,
  getRun,
  getRunLogs,
  getWorkflowDetail,
  getWorkflowVersions,
  runWorkflow,
  type PendingToolCall,
  type Step,
  type WorkflowVersionRecord,
} from "../../lib/api";
import WorkflowBuilderErrorBoundary from "./WorkflowBuilderErrorBoundary";
import WorkflowGraphCanvas from "./WorkflowGraphCanvas";
import {
  createInitialWorkflowSpec,
  defaultNodeForType,
  getNextEdgeId,
  getNextNodeId,
  normalizeDraftSpec,
  normalizeKeyList,
  slugify,
  validateDraftLocally,
} from "./workflowDraft";

const BUILDER_INTENT_STORAGE_KEY = "saturday.builder.intent";
const POLL_INTERVAL_MS = 1500;

type BuilderIntent = {
  tab?: "tools" | "workflows";
  workflowId?: string;
};

type WorkflowBuilderPanelProps = {
  tools: Tool[];
};

type ValidationMode = "button" | "save" | "run";

const NODE_TYPES: NodeType[] = ["llm", "tool", "conditional", "verify", "finalize"];

function parseBuilderIntent(raw: string | null): BuilderIntent | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as BuilderIntent;
    return {
      tab: parsed.tab === "workflows" ? "workflows" : "tools",
      workflowId:
        typeof parsed.workflowId === "string" && parsed.workflowId.trim()
          ? parsed.workflowId.trim()
          : undefined,
    };
  } catch {
    return null;
  }
}

function dedupeDiagnostics(
  diagnostics: ValidationDiagnostic[]
): ValidationDiagnostic[] {
  const seen = new Set<string>();
  const output: ValidationDiagnostic[] = [];
  for (const item of diagnostics) {
    const key = [
      item.code,
      item.severity,
      item.message,
      item.node_id || "",
      item.edge_id || "",
      item.path || "",
    ].join("|");
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    output.push(item);
  }
  return output;
}

function computePlannedTools(spec: WorkflowSpec): Array<{
  node_id: string;
  tool_name: string;
  args_map: Record<string, unknown>;
}> {
  const planned: Array<{
    node_id: string;
    tool_name: string;
    args_map: Record<string, unknown>;
  }> = [];
  for (const node of spec.nodes) {
    if (node.type !== "tool") {
      continue;
    }
    const toolName = String((node.config as Record<string, unknown>)?.tool_name || "").trim();
    planned.push({
      node_id: node.id,
      tool_name: toolName || "(unset)",
      args_map:
        ((node.config as Record<string, unknown>)?.args_map as Record<string, unknown>) || {},
    });
  }
  return planned;
}

function defaultRunInput(): string {
  return JSON.stringify(
    {
      task: "Test the workflow with this sample task.",
      context: {},
    },
    null,
    2
  );
}

function statusTone(status: string): string {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "ok" || normalized === "completed" || normalized === "success") {
    return "border-emerald-400/40 bg-emerald-500/10 text-emerald-100";
  }
  if (normalized === "running" || normalized === "pending") {
    return "border-sky-400/40 bg-sky-500/10 text-sky-100";
  }
  if (normalized === "skipped") {
    return "border-zinc-400/40 bg-zinc-500/10 text-zinc-200";
  }
  if (normalized === "rejected") {
    return "border-amber-400/40 bg-amber-500/10 text-amber-100";
  }
  return "border-rose-400/40 bg-rose-500/10 text-rose-100";
}

function patchNodeConfig(node: WorkflowSpec["nodes"][number], patch: Record<string, unknown>) {
  return {
    ...node,
    config: {
      ...(node.config as Record<string, unknown>),
      ...patch,
    },
  } as WorkflowSpec["nodes"][number];
}

export default function WorkflowBuilderPanel({ tools }: WorkflowBuilderPanelProps) {
  const [draft, setDraft] = useState<WorkflowSpec>(createInitialWorkflowSpec());
  const [workflowIdTouched, setWorkflowIdTouched] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>("llm_1");
  const [versions, setVersions] = useState<WorkflowVersionRecord[]>([]);
  const [latestSavedSpec, setLatestSavedSpec] = useState<WorkflowSpec | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string>("");
  const [diagnostics, setDiagnostics] = useState<ValidationDiagnostic[]>([]);
  const [isLoadingWorkflow, setIsLoadingWorkflow] = useState<boolean>(false);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [messageError, setMessageError] = useState<string>("");
  const [messageSuccess, setMessageSuccess] = useState<string>("");
  const [toolArgsMapText, setToolArgsMapText] = useState<string>("{}");
  const [toolArgsMapError, setToolArgsMapError] = useState<string>("");
  const [runInputText, setRunInputText] = useState<string>(defaultRunInput());
  const [runInputError, setRunInputError] = useState<string>("");
  const [runId, setRunId] = useState<string>("");
  const [runStatus, setRunStatus] = useState<string>("");
  const [runSteps, setRunSteps] = useState<Step[]>([]);
  const [runPendingToolCalls, setRunPendingToolCalls] = useState<PendingToolCall[]>([]);
  const [runOutput, setRunOutput] = useState<Record<string, unknown>>({});
  const [plannedToolCalls, setPlannedToolCalls] = useState<
    Array<{ node_id: string; tool_name: string; args_map: Record<string, unknown> }>
  >([]);
  const [newEdgeFrom, setNewEdgeFrom] = useState<string>("");
  const [newEdgeTo, setNewEdgeTo] = useState<string>("");
  const [newEdgeLabel, setNewEdgeLabel] = useState<string>("always");

  const pollTimerRef = useRef<number | null>(null);
  const pollAbortRef = useRef<AbortController | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const compileAbortRef = useRef<AbortController | null>(null);

  const availableToolIds = useMemo(
    () => tools.filter((tool) => tool.enabled).map((tool) => tool.id),
    [tools]
  );

  const selectedNode = useMemo(
    () => draft.nodes.find((node) => node.id === selectedNodeId) || null,
    [draft.nodes, selectedNodeId]
  );

  const hasBlockingDiagnostics = useMemo(
    () => diagnostics.some((item) => item.severity === "error"),
    [diagnostics]
  );

  const workflowVersionLabel = useMemo(() => {
    if (!selectedVersionId) {
      return "Draft";
    }
    const version = versions.find((item) => item.version_id === selectedVersionId);
    if (!version) {
      return "Draft";
    }
    return `v${version.version_num}`;
  }, [selectedVersionId, versions]);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (pollAbortRef.current) {
      pollAbortRef.current.abort();
      pollAbortRef.current = null;
    }
  }, []);

  const applyLoadedSpec = useCallback((spec: WorkflowSpec) => {
    const normalized = normalizeDraftSpec(spec);
    setDraft(normalized);
    setLatestSavedSpec(normalized);
    setSelectedNodeId(normalized.nodes[0]?.id || null);
    setNewEdgeFrom(normalized.nodes[0]?.id || "");
    setNewEdgeTo(normalized.nodes[1]?.id || normalized.nodes[0]?.id || "");
  }, []);

  const loadWorkflow = useCallback(
    async (workflowId: string) => {
      const trimmed = workflowId.trim();
      if (!trimmed) {
        return;
      }
      if (loadAbortRef.current) {
        loadAbortRef.current.abort();
      }
      const controller = new AbortController();
      loadAbortRef.current = controller;
      setIsLoadingWorkflow(true);
      setMessageError("");
      setMessageSuccess("");

      try {
        const [detailPayload, versionsPayload] = await Promise.all([
          getWorkflowDetail(trimmed, controller.signal),
          getWorkflowVersions(trimmed, controller.signal),
        ]);
        if (controller.signal.aborted) {
          return;
        }

        const versionItems =
          versionsPayload.length > 0 ? versionsPayload : detailPayload.versions || [];
        setVersions(versionItems);
        const latest = versionItems[0] || detailPayload.latest_version || null;
        if (latest && latest.spec) {
          applyLoadedSpec(latest.spec);
          setSelectedVersionId(latest.version_id || "");
          setWorkflowIdTouched(true);
          setMessageSuccess(
            `Loaded ${trimmed} ${latest.version_num ? `(v${latest.version_num})` : ""}.`
          );
        } else {
          setMessageError(`Workflow '${trimmed}' has no saved versions.`);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setMessageError(
          error instanceof Error ? error.message : "Failed to load workflow."
        );
      } finally {
        if (loadAbortRef.current === controller) {
          loadAbortRef.current = null;
        }
        setIsLoadingWorkflow(false);
      }
    },
    [applyLoadedSpec]
  );

  useEffect(() => {
    const intent = parseBuilderIntent(
      window.sessionStorage.getItem(BUILDER_INTENT_STORAGE_KEY)
    );
    if (!intent || intent.tab !== "workflows" || !intent.workflowId) {
      return;
    }
    window.sessionStorage.removeItem(BUILDER_INTENT_STORAGE_KEY);
    void loadWorkflow(intent.workflowId);
  }, [loadWorkflow]);

  useEffect(() => {
    return () => {
      stopPolling();
      if (loadAbortRef.current) {
        loadAbortRef.current.abort();
      }
      if (compileAbortRef.current) {
        compileAbortRef.current.abort();
      }
    };
  }, [stopPolling]);

  useEffect(() => {
    if (workflowIdTouched) {
      return;
    }
    const nextWorkflowId = `workflow.custom.${slugify(draft.name, "untitled_workflow")}`;
    setDraft((current) => ({
      ...current,
      workflow_id: nextWorkflowId,
    }));
  }, [draft.name, workflowIdTouched]);

  useEffect(() => {
    if (selectedNode?.type !== "tool") {
      setToolArgsMapText("{}");
      setToolArgsMapError("");
      return;
    }
    const toolConfig = selectedNode.config as Record<string, unknown>;
    setToolArgsMapText(JSON.stringify(toolConfig.args_map || {}, null, 2));
    setToolArgsMapError("");
  }, [selectedNode]);

  const updateDraft = useCallback((updater: (current: WorkflowSpec) => WorkflowSpec) => {
    setDraft((current) => normalizeDraftSpec(updater(current)));
  }, []);

  const runValidation = useCallback(
    async (mode: ValidationMode): Promise<{
      valid: boolean;
      normalized: WorkflowSpec;
      diagnostics: ValidationDiagnostic[];
    }> => {
      if (compileAbortRef.current) {
        compileAbortRef.current.abort();
      }
      const controller = new AbortController();
      compileAbortRef.current = controller;
      setIsValidating(true);
      setMessageError("");

      const normalizedDraft = normalizeDraftSpec(draft);
      const localDiagnostics = validateDraftLocally(normalizedDraft, availableToolIds);

      try {
        const compileResult = await compileWorkflowSpec(normalizedDraft, controller.signal);
        if (controller.signal.aborted) {
          return {
            valid: false,
            normalized: normalizedDraft,
            diagnostics: localDiagnostics,
          };
        }
        const mergedDiagnostics = dedupeDiagnostics([
          ...localDiagnostics,
          ...compileResult.diagnostics,
        ]);
        const hasErrors = mergedDiagnostics.some((item) => item.severity === "error");

        setDiagnostics(mergedDiagnostics);
        setDraft(normalizeDraftSpec(compileResult.workflow_spec));
        setPlannedToolCalls(computePlannedTools(compileResult.workflow_spec));
        if (!hasErrors && mode === "button") {
          setMessageSuccess("Validation passed. No blocking errors.");
        }
        if (hasErrors) {
          setMessageError("Validation failed. Resolve errors before save or run.");
        }
        return {
          valid: !hasErrors && compileResult.valid,
          normalized: normalizeDraftSpec(compileResult.workflow_spec),
          diagnostics: mergedDiagnostics,
        };
      } catch (error) {
        if (controller.signal.aborted) {
          return {
            valid: false,
            normalized: normalizedDraft,
            diagnostics: localDiagnostics,
          };
        }
        const message =
          error instanceof Error ? error.message : "Workflow validation failed.";
        setDiagnostics(localDiagnostics);
        setMessageError(message);
        return {
          valid: false,
          normalized: normalizedDraft,
          diagnostics: localDiagnostics,
        };
      } finally {
        if (compileAbortRef.current === controller) {
          compileAbortRef.current = null;
        }
        setIsValidating(false);
      }
    },
    [availableToolIds, draft]
  );

  const pollRunState = useCallback(
    async (currentRunId: string) => {
      if (!currentRunId) {
        return;
      }
      if (pollAbortRef.current) {
        pollAbortRef.current.abort();
      }
      const controller = new AbortController();
      pollAbortRef.current = controller;

      try {
        const [runPayload, logsPayload, pendingPayload] = await Promise.all([
          getRun(currentRunId, controller.signal),
          getRunLogs(currentRunId, controller.signal),
          getPendingToolCalls(currentRunId, controller.signal),
        ]);
        if (controller.signal.aborted) {
          return;
        }
        setRunStatus(String(runPayload.status || ""));
        setRunSteps(logsPayload.steps || []);
        setRunPendingToolCalls(pendingPayload || []);

        const resultPayload =
          runPayload.result && typeof runPayload.result === "object"
            ? (runPayload.result as Record<string, unknown>)
            : {};
        const nextOutput =
          resultPayload.output && typeof resultPayload.output === "object"
            ? (resultPayload.output as Record<string, unknown>)
            : {};
        setRunOutput(nextOutput);

        const terminalStatuses = new Set(["ok", "error", "rejected", "completed", "failed"]);
        if (terminalStatuses.has(String(runPayload.status || "").toLowerCase())) {
          stopPolling();
          setIsRunning(false);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setMessageError(error instanceof Error ? error.message : "Run polling failed.");
      } finally {
        if (pollAbortRef.current === controller) {
          pollAbortRef.current = null;
        }
      }
    },
    [stopPolling]
  );

  const startPolling = useCallback(
    (currentRunId: string) => {
      stopPolling();
      void pollRunState(currentRunId);
      pollTimerRef.current = window.setInterval(() => {
        void pollRunState(currentRunId);
      }, POLL_INTERVAL_MS);
    },
    [pollRunState, stopPolling]
  );

  const handleAddNode = useCallback(
    (type: NodeType) => {
      setMessageError("");
      setMessageSuccess("");
      updateDraft((current) => {
        const nodeId = getNextNodeId(current.nodes, type);
        const node = defaultNodeForType(type, nodeId, current.nodes.length);
        return {
          ...current,
          nodes: [...current.nodes, node],
        };
      });
      const previewId = getNextNodeId(draft.nodes, type);
      setSelectedNodeId(previewId);
    },
    [draft.nodes, updateDraft]
  );

  const handleRemoveNode = useCallback(
    (nodeId: string) => {
      const target = draft.nodes.find((node) => node.id === nodeId);
      if (!target) {
        return;
      }
      if (target.type === "finalize") {
        const finalizeCount = draft.nodes.filter((node) => node.type === "finalize").length;
        if (finalizeCount <= 1) {
          setMessageError("Workflow must keep at least one finalize node.");
          return;
        }
      }
      updateDraft((current) => ({
        ...current,
        nodes: current.nodes.filter((node) => node.id !== nodeId),
        edges: current.edges.filter((edge) => edge.from !== nodeId && edge.to !== nodeId),
      }));
      setSelectedNodeId((current) => (current === nodeId ? null : current));
    },
    [draft.nodes, updateDraft]
  );

  const handleValidate = useCallback(async () => {
    setMessageSuccess("");
    await runValidation("button");
  }, [runValidation]);

  const handleLoadVersion = useCallback(
    (version: WorkflowVersionRecord) => {
      if (!version.spec) {
        setMessageError("Selected version does not include workflow spec.");
        return;
      }
      applyLoadedSpec(version.spec);
      setSelectedVersionId(version.version_id);
      setWorkflowIdTouched(true);
      setMessageSuccess(`Loaded version v${version.version_num}.`);
      setDiagnostics([]);
    },
    [applyLoadedSpec]
  );

  const handleSave = useCallback(async () => {
    if (isSaving) {
      return;
    }
    const workflowId = String(draft.workflow_id || "").trim();
    if (!workflowId) {
      setMessageError("Workflow id is required.");
      return;
    }
    if (!draft.name.trim()) {
      setMessageError("Workflow name is required.");
      return;
    }

    setIsSaving(true);
    setMessageError("");
    setMessageSuccess("");
    try {
      const validation = await runValidation("save");
      if (!validation.valid) {
        return;
      }
      const created = await createWorkflowVersion(workflowId, validation.normalized, "builder");
      const refreshed = await getWorkflowVersions(workflowId);
      setVersions(refreshed);
      setLatestSavedSpec(validation.normalized);
      setSelectedVersionId(created.version_id);
      setWorkflowIdTouched(true);
      setMessageSuccess(`Saved ${workflowId} as version v${created.version_num}.`);
    } catch (error) {
      setMessageError(error instanceof Error ? error.message : "Failed to save workflow.");
    } finally {
      setIsSaving(false);
    }
  }, [draft, isSaving, runValidation]);

  const handleResetToLastSaved = useCallback(() => {
    if (!latestSavedSpec) {
      setMessageError("No saved snapshot available yet.");
      return;
    }
    applyLoadedSpec(latestSavedSpec);
    setDiagnostics([]);
    setMessageSuccess("Draft reset to last saved version.");
  }, [applyLoadedSpec, latestSavedSpec]);

  const handleExport = useCallback(() => {
    const blob = new Blob([JSON.stringify(draft, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const fileSlug = slugify(draft.name || "workflow", "workflow");
    anchor.href = url;
    anchor.download = `${fileSlug}.workflow.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, [draft]);

  const handleImport = useCallback(
    async (file: File) => {
      setMessageError("");
      setMessageSuccess("");
      try {
        const text = await file.text();
        const parsed = JSON.parse(text) as WorkflowSpec;
        const normalized = normalizeDraftSpec(parsed);
        const imported = await compileWorkflowSpec(normalized);
        const hasErrors = imported.diagnostics.some((item) => item.severity === "error");
        setDiagnostics(imported.diagnostics);
        if (!imported.valid || hasErrors) {
          setMessageError("Imported JSON has validation errors.");
          return;
        }
        const nextSpec = normalizeDraftSpec(imported.workflow_spec);
        setDraft(nextSpec);
        setSelectedNodeId(nextSpec.nodes[0]?.id || null);
        setSelectedVersionId("");
        setWorkflowIdTouched(Boolean(nextSpec.workflow_id));
        setMessageSuccess("Imported workflow JSON into draft.");
      } catch (error) {
        setMessageError(error instanceof Error ? error.message : "Invalid JSON import.");
      }
    },
    []
  );

  const handleRunSandbox = useCallback(async () => {
    if (isRunning) {
      return;
    }
    setMessageError("");
    setMessageSuccess("");
    setRunInputError("");
    setRunOutput({});
    setRunPendingToolCalls([]);
    setRunSteps([]);

    let parsedInput: Record<string, unknown> = {};
    try {
      const raw = JSON.parse(runInputText || "{}");
      parsedInput = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
    } catch {
      setRunInputError("Run input must be valid JSON.");
      return;
    }

    setIsRunning(true);
    try {
      const validation = await runValidation("run");
      if (!validation.valid) {
        setIsRunning(false);
        return;
      }
      const response = await runWorkflow({
        draft_spec: validation.normalized,
        input: parsedInput as Record<string, any>,
        sandbox_mode: true,
      });
      const nextRunId = String(response.run_id || "");
      setRunId(nextRunId);
      setRunStatus(String(response.status || "running"));
      setPlannedToolCalls(computePlannedTools(validation.normalized));
      setMessageSuccess(`Sandbox run started (${nextRunId}).`);
      startPolling(nextRunId);
    } catch (error) {
      setMessageError(error instanceof Error ? error.message : "Sandbox run failed.");
      setIsRunning(false);
    }
  }, [isRunning, runInputText, runValidation, startPolling]);

  const handleApproveToolCall = useCallback(
    async (toolCallId: string, approved: boolean) => {
      if (!runId) {
        return;
      }
      try {
        await approveToolCall(runId, toolCallId, approved);
        await pollRunState(runId);
      } catch (error) {
        setMessageError(
          error instanceof Error ? error.message : "Failed to submit tool approval."
        );
      }
    },
    [pollRunState, runId]
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.55fr)_minmax(0,0.95fr)]">
        <div className="space-y-4">
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm uppercase tracking-wide text-secondary">Graph</h2>
              <div className="flex flex-wrap items-center gap-2">
                {NODE_TYPES.map((type) => (
                  <Button
                    key={type}
                    type="button"
                    className="h-7 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                    onClick={() => handleAddNode(type)}
                  >
                    + {type}
                  </Button>
                ))}
              </div>
            </div>

            <WorkflowBuilderErrorBoundary>
              <WorkflowGraphCanvas
                nodes={draft.nodes}
                edges={draft.edges}
                selectedNodeId={selectedNodeId}
                onSelectNode={setSelectedNodeId}
                onNodePositionChange={(nodeId, x, y) => {
                  updateDraft((current) => ({
                    ...current,
                    nodes: current.nodes.map((node) =>
                      node.id === nodeId
                        ? {
                            ...node,
                            position: { x, y },
                          }
                        : node
                    ),
                  }));
                }}
              />
            </WorkflowBuilderErrorBoundary>

            <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-[1fr_1fr_1fr_auto]">
              <select
                value={newEdgeFrom}
                onChange={(event) => setNewEdgeFrom(event.target.value)}
                className="h-9 rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
              >
                <option value="">From node</option>
                {draft.nodes.map((node) => (
                  <option key={`from-${node.id}`} value={node.id}>
                    {node.id}
                  </option>
                ))}
              </select>
              <select
                value={newEdgeTo}
                onChange={(event) => setNewEdgeTo(event.target.value)}
                className="h-9 rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
              >
                <option value="">To node</option>
                {draft.nodes.map((node) => (
                  <option key={`to-${node.id}`} value={node.id}>
                    {node.id}
                  </option>
                ))}
              </select>
              <select
                value={newEdgeLabel}
                onChange={(event) => setNewEdgeLabel(event.target.value)}
                className="h-9 rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
              >
                <option value="always">always</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
              <Button
                type="button"
                className="h-9 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
                onClick={() => {
                  if (!newEdgeFrom || !newEdgeTo) {
                    setMessageError("Select both source and target nodes.");
                    return;
                  }
                  const duplicate = draft.edges.some(
                    (edge) =>
                      edge.from === newEdgeFrom &&
                      edge.to === newEdgeTo &&
                      String(edge.label || "always") === newEdgeLabel
                  );
                  if (duplicate) {
                    setMessageError("This edge already exists.");
                    return;
                  }
                  updateDraft((current) => ({
                    ...current,
                    edges: [
                      ...current.edges,
                      {
                        id: getNextEdgeId(current.edges),
                        from: newEdgeFrom,
                        to: newEdgeTo,
                        label: newEdgeLabel,
                      },
                    ],
                  }));
                  setMessageError("");
                }}
              >
                Add edge
              </Button>
            </div>

            <div className="mt-3 max-h-44 space-y-1 overflow-auto rounded-xl border border-subtle bg-[#08080c] p-2">
              {draft.edges.length === 0 ? (
                <p className="px-2 py-2 text-xs text-secondary">No edges yet.</p>
              ) : (
                draft.edges.map((edge) => (
                  <div
                    key={edge.id}
                    className="flex items-center justify-between gap-2 rounded-lg border border-subtle px-2 py-1.5 text-xs text-secondary"
                  >
                    <span className="truncate">
                      {edge.id}: {edge.from} -[{edge.label || "always"}]-&gt; {edge.to}
                    </span>
                    <Button
                      type="button"
                      className="h-6 rounded-full border border-subtle bg-transparent px-2 text-[10px] text-secondary hover:text-primary"
                      onClick={() => {
                        updateDraft((current) => ({
                          ...current,
                          edges: current.edges.filter((item) => item.id !== edge.id),
                        }));
                      }}
                    >
                      Remove
                    </Button>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm uppercase tracking-wide text-secondary">Actions</h2>
              <Badge className="border border-white/10 bg-white/5 text-xs text-secondary">
                {workflowVersionLabel}
              </Badge>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                className="h-9 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
                onClick={() => void handleValidate()}
                disabled={isValidating || isSaving || isRunning}
              >
                {isValidating ? "Validating..." : "Validate"}
              </Button>
              <Button
                type="button"
                className="h-9 rounded-full bg-gold px-4 text-xs text-black hover:bg-[#e0be55] disabled:opacity-60"
                onClick={() => void handleSave()}
                disabled={isSaving || isValidating || isRunning}
              >
                {isSaving ? "Saving..." : "Save New Version"}
              </Button>
              <Button
                type="button"
                className="h-9 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
                onClick={handleResetToLastSaved}
                disabled={!latestSavedSpec || isSaving || isRunning}
              >
                Reset To Last Saved
              </Button>
              <Button
                type="button"
                className="h-9 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
                onClick={handleExport}
              >
                Export JSON
              </Button>
              <label className="inline-flex h-9 cursor-pointer items-center rounded-full border border-subtle px-4 text-xs text-secondary hover:text-primary">
                Import JSON
                <input
                  type="file"
                  accept="application/json,.json"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (!file) {
                      return;
                    }
                    void handleImport(file);
                    event.target.value = "";
                  }}
                />
              </label>
            </div>

            <div className="mt-4">
              <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                Sandbox Run Input (JSON)
              </label>
              <Textarea
                value={runInputText}
                rows={5}
                onChange={(event) => setRunInputText(event.target.value)}
                className="border-subtle bg-[#08080c] font-mono text-xs"
              />
              {runInputError ? (
                <p className="mt-1 text-xs text-rose-200">{runInputError}</p>
              ) : null}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  className="h-9 rounded-full border border-sky-400/50 bg-sky-500/10 px-4 text-xs text-sky-100"
                  onClick={() => void handleRunSandbox()}
                  disabled={isRunning || isSaving || isValidating || hasBlockingDiagnostics}
                >
                  {isRunning ? "Running..." : "Sandbox Run"}
                </Button>
                {runId ? (
                  <code className="rounded-md border border-subtle bg-[#08080c] px-2 py-1 text-[11px] text-secondary">
                    {runId}
                  </code>
                ) : null}
                {runStatus ? (
                  <Badge className={statusTone(runStatus)}>{runStatus}</Badge>
                ) : null}
                {runId ? (
                  <Button
                    type="button"
                    className="h-8 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                    onClick={() => {
                      window.dispatchEvent(
                        new CustomEvent("dashboard:navigate", {
                          detail: {
                            page: "inspect",
                            runId,
                          },
                        })
                      );
                    }}
                  >
                    Open In Inspect
                  </Button>
                ) : null}
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <h2 className="mb-3 text-sm uppercase tracking-wide text-secondary">
              Diagnostics
            </h2>
            {diagnostics.length === 0 ? (
              <p className="text-sm text-secondary">No diagnostics yet.</p>
            ) : (
              <div className="max-h-56 space-y-2 overflow-auto">
                {diagnostics.map((item, index) => (
                  <div
                    key={`${item.code}-${index}`}
                    className={
                      "rounded-xl border px-3 py-2 text-xs " +
                      (item.severity === "error"
                        ? "border-rose-400/40 bg-rose-500/10 text-rose-100"
                        : item.severity === "warning"
                        ? "border-amber-400/40 bg-amber-500/10 text-amber-100"
                        : "border-sky-400/40 bg-sky-500/10 text-sky-100")
                    }
                  >
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <span className="font-medium">{item.code}</span>
                      <span>{item.severity}</span>
                    </div>
                    <p>{item.message}</p>
                    {item.node_id ? (
                      <p className="mt-1 text-[11px] opacity-90">node: {item.node_id}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </div>

          {runId ? (
            <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
              <h2 className="mb-3 text-sm uppercase tracking-wide text-secondary">
                Sandbox Activity
              </h2>

              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                  <h3 className="mb-2 text-xs uppercase tracking-wide text-secondary">
                    Planned Tool Calls
                  </h3>
                  {plannedToolCalls.length === 0 ? (
                    <p className="text-xs text-secondary">No tool nodes in this workflow.</p>
                  ) : (
                    <div className="space-y-2">
                      {plannedToolCalls.map((item) => (
                        <div key={`${item.node_id}:${item.tool_name}`} className="rounded-lg border border-subtle p-2">
                          <div className="text-xs text-primary">
                            {item.node_id} -&gt; {item.tool_name}
                          </div>
                          <pre className="mt-1 overflow-auto text-[10px] text-secondary">
                            {JSON.stringify(item.args_map, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                  <h3 className="mb-2 text-xs uppercase tracking-wide text-secondary">
                    Pending Approvals
                  </h3>
                  {runPendingToolCalls.length === 0 ? (
                    <p className="text-xs text-secondary">No pending tool approvals.</p>
                  ) : (
                    <div className="space-y-2">
                      {runPendingToolCalls.map((call) => (
                        <div key={call.tool_call_id} className="rounded-lg border border-subtle p-2">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="text-xs text-primary">{call.tool_name}</span>
                            <Badge className={statusTone(call.status)}>{call.status}</Badge>
                          </div>
                          <pre className="overflow-auto text-[10px] text-secondary">
                            {JSON.stringify(call.args, null, 2)}
                          </pre>
                          <div className="mt-2 flex items-center gap-2">
                            <Button
                              type="button"
                              className="h-7 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 text-[11px] text-emerald-100"
                              onClick={() => void handleApproveToolCall(call.tool_call_id, true)}
                            >
                              Approve
                            </Button>
                            <Button
                              type="button"
                              className="h-7 rounded-full border border-rose-400/40 bg-rose-500/10 px-3 text-[11px] text-rose-100"
                              onClick={() => void handleApproveToolCall(call.tool_call_id, false)}
                            >
                              Reject
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-3 rounded-xl border border-subtle bg-[#08080c] p-3">
                <h3 className="mb-2 text-xs uppercase tracking-wide text-secondary">Run Output</h3>
                <pre className="max-h-56 overflow-auto text-[11px] text-secondary">
                  {JSON.stringify(runOutput, null, 2)}
                </pre>
              </div>

              <div className="mt-3 rounded-xl border border-subtle bg-[#08080c] p-3">
                <h3 className="mb-2 text-xs uppercase tracking-wide text-secondary">Step Logs</h3>
                {runSteps.length === 0 ? (
                  <p className="text-xs text-secondary">No steps yet.</p>
                ) : (
                  <div className="max-h-72 space-y-2 overflow-auto">
                    {runSteps.map((step) => (
                      <div key={`${step.step_index}:${step.name}`} className="rounded-lg border border-subtle p-2 text-xs">
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <span className="text-primary">
                            #{step.step_index} {step.name}
                          </span>
                          <Badge className={statusTone(step.status)}>{step.status}</Badge>
                        </div>
                        <p className="text-secondary">{step.summary || "No summary"}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <h2 className="mb-3 text-sm uppercase tracking-wide text-secondary">
              Workflow Metadata
            </h2>
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                  Name
                </label>
                <Input
                  value={draft.name}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      name: event.target.value,
                    }))
                  }
                  className="h-9 border-subtle bg-[#08080c]"
                  placeholder="Order Resolution Workflow"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                  Workflow ID
                </label>
                <Input
                  value={draft.workflow_id || ""}
                  onChange={(event) => {
                    setWorkflowIdTouched(true);
                    setDraft((current) => ({
                      ...current,
                      workflow_id: event.target.value,
                    }));
                  }}
                  className="h-9 border-subtle bg-[#08080c] font-mono text-xs"
                  placeholder="workflow.custom.order_resolution"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                  Description
                </label>
                <Textarea
                  value={draft.description || ""}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      description: event.target.value,
                    }))
                  }
                  rows={2}
                  className="border-subtle bg-[#08080c]"
                />
              </div>
              <label className="inline-flex items-center gap-2 text-xs text-secondary">
                <input
                  type="checkbox"
                  checked={Boolean(draft.allow_cycles)}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      allow_cycles: event.target.checked,
                    }))
                  }
                />
                Allow cycles
              </label>
            </div>
          </div>

          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-sm uppercase tracking-wide text-secondary">State Schema</h2>
              <Button
                type="button"
                className="h-7 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                onClick={() => {
                  setDraft((current) => ({
                    ...current,
                    state_schema: [
                      ...current.state_schema,
                      {
                        key: "",
                        type: "string",
                        description: "",
                        required: false,
                      },
                    ],
                  }));
                }}
              >
                Add key
              </Button>
            </div>
            <div className="max-h-64 space-y-2 overflow-auto">
              {draft.state_schema.map((item, index) => (
                <div
                  key={`state-key-${index}`}
                  className="rounded-xl border border-subtle bg-[#08080c] p-2"
                >
                  <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_120px_auto]">
                    <Input
                      value={item.key}
                      onChange={(event) => {
                        const nextValue = event.target.value;
                        setDraft((current) => ({
                          ...current,
                          state_schema: current.state_schema.map((entry, entryIndex) =>
                            entryIndex === index ? { ...entry, key: nextValue } : entry
                          ),
                        }));
                      }}
                      className="h-8 border-subtle bg-[#0b0b10] text-xs"
                      placeholder="state_key"
                    />
                    <select
                      value={item.type}
                      onChange={(event) => {
                        const typeValue = event.target.value as
                          | "string"
                          | "number"
                          | "bool"
                          | "json";
                        setDraft((current) => ({
                          ...current,
                          state_schema: current.state_schema.map((entry, entryIndex) =>
                            entryIndex === index ? { ...entry, type: typeValue } : entry
                          ),
                        }));
                      }}
                      className="h-8 rounded-md border border-subtle bg-[#0b0b10] px-2 text-xs text-primary outline-none"
                    >
                      <option value="string">string</option>
                      <option value="number">number</option>
                      <option value="bool">bool</option>
                      <option value="json">json</option>
                    </select>
                    <Button
                      type="button"
                      className="h-8 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                      onClick={() => {
                        setDraft((current) => ({
                          ...current,
                          state_schema: current.state_schema.filter((_, entryIndex) => entryIndex !== index),
                        }));
                      }}
                    >
                      Remove
                    </Button>
                  </div>
                  <Input
                    value={item.description || ""}
                    onChange={(event) => {
                      const description = event.target.value;
                      setDraft((current) => ({
                        ...current,
                        state_schema: current.state_schema.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, description } : entry
                        ),
                      }));
                    }}
                    className="mt-2 h-8 border-subtle bg-[#0b0b10] text-xs"
                    placeholder="Description"
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <h2 className="mb-3 text-sm uppercase tracking-wide text-secondary">Node Inspector</h2>
            {!selectedNode ? (
              <p className="text-sm text-secondary">Select a node on the graph canvas.</p>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <Badge className="border border-white/10 bg-white/5 text-xs text-secondary">
                    {selectedNode.type}
                  </Badge>
                  <Button
                    type="button"
                    className="h-7 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                    onClick={() => handleRemoveNode(selectedNode.id)}
                  >
                    Remove Node
                  </Button>
                </div>

                <div>
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Node Label
                  </label>
                  <Input
                    value={selectedNode.label || ""}
                    onChange={(event) =>
                      updateDraft((current) => ({
                        ...current,
                        nodes: current.nodes.map((node) =>
                          node.id === selectedNode.id
                            ? { ...node, label: event.target.value }
                            : node
                        ),
                      }))
                    }
                    className="h-8 border-subtle bg-[#08080c]"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Reads (comma-separated keys)
                  </label>
                  <Input
                    value={(selectedNode.reads || []).join(", ")}
                    onChange={(event) =>
                      updateDraft((current) => ({
                        ...current,
                        nodes: current.nodes.map((node) =>
                          node.id === selectedNode.id
                            ? { ...node, reads: normalizeKeyList(event.target.value) }
                            : node
                        ),
                      }))
                    }
                    className="h-8 border-subtle bg-[#08080c]"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Writes (comma-separated keys)
                  </label>
                  <Input
                    value={(selectedNode.writes || []).join(", ")}
                    onChange={(event) =>
                      updateDraft((current) => ({
                        ...current,
                        nodes: current.nodes.map((node) =>
                          node.id === selectedNode.id
                            ? { ...node, writes: normalizeKeyList(event.target.value) }
                            : node
                        ),
                      }))
                    }
                    className="h-8 border-subtle bg-[#08080c]"
                  />
                </div>

                {selectedNode.type === "llm" ? (
                  <>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Prompt Template
                      </label>
                      <Textarea
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.prompt_template || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      prompt_template: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        rows={4}
                        className="border-subtle bg-[#08080c]"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Output Key
                      </label>
                      <Input
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.output_key || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      output_key: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}

                {selectedNode.type === "tool" ? (
                  <>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Tool
                      </label>
                      <select
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.tool_name || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      tool_name: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 w-full rounded-md border border-subtle bg-[#08080c] px-2 text-sm text-primary outline-none"
                      >
                        <option value="">Select tool</option>
                        {tools.map((tool) => (
                          <option key={tool.id} value={tool.id}>
                            {tool.name} ({tool.id})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Args Map (JSON)
                      </label>
                      <Textarea
                        value={toolArgsMapText}
                        onChange={(event) => {
                          setToolArgsMapText(event.target.value);
                          setToolArgsMapError("");
                        }}
                        onBlur={() => {
                          try {
                            const parsed = JSON.parse(toolArgsMapText || "{}");
                            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
                              setToolArgsMapError("args_map must be a JSON object.");
                              return;
                            }
                            updateDraft((current) => ({
                              ...current,
                              nodes: current.nodes.map((node) =>
                                node.id === selectedNode.id
                                  ? patchNodeConfig(node, { args_map: parsed })
                                  : node
                              ),
                            }));
                            setToolArgsMapError("");
                          } catch {
                            setToolArgsMapError("Invalid JSON for args_map.");
                          }
                        }}
                        rows={4}
                        className="border-subtle bg-[#08080c] font-mono text-xs"
                      />
                      {toolArgsMapError ? (
                        <p className="mt-1 text-xs text-rose-200">{toolArgsMapError}</p>
                      ) : null}
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Output Key
                      </label>
                      <Input
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.output_key || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      output_key: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}

                {selectedNode.type === "conditional" ? (
                  <>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Expression
                      </label>
                      <Input
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.expression || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      expression: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                      />
                    </div>
                    <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
                      <Input
                        value={String((selectedNode.config as Record<string, unknown>)?.field || "")}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      field: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                        placeholder="legacy field"
                      />
                      <select
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.operator || "equals"
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? patchNodeConfig(node, {
                                    operator: event.target.value as
                                      | "equals"
                                      | "contains"
                                      | "gt"
                                      | "lt"
                                      | "exists"
                                      | "not_exists"
                                      | "in",
                                  })
                                : node
                            ),
                          }))
                        }
                        className="h-8 rounded-md border border-subtle bg-[#08080c] px-2 text-sm text-primary outline-none"
                      >
                        <option value="equals">equals</option>
                        <option value="contains">contains</option>
                        <option value="gt">gt</option>
                        <option value="lt">lt</option>
                        <option value="exists">exists</option>
                        <option value="not_exists">not_exists</option>
                        <option value="in">in</option>
                      </select>
                      <Input
                        value={String((selectedNode.config as Record<string, unknown>)?.value || "")}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      value: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                        placeholder="legacy value"
                      />
                    </div>
                  </>
                ) : null}

                {selectedNode.type === "verify" ? (
                  <>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Mode
                      </label>
                      <select
                        value={String((selectedNode.config as Record<string, unknown>)?.mode || "rule")}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? patchNodeConfig(node, {
                                    mode: event.target.value as "rule" | "llm",
                                  })
                                : node
                            ),
                          }))
                        }
                        className="h-8 w-full rounded-md border border-subtle bg-[#08080c] px-2 text-sm text-primary outline-none"
                      >
                        <option value="rule">rule</option>
                        <option value="llm">llm</option>
                      </select>
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Rule Expression
                      </label>
                      <Input
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.expression || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      expression: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Verifier Prompt
                      </label>
                      <Textarea
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.prompt_template || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      prompt_template: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        rows={3}
                        className="border-subtle bg-[#08080c]"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Output Key
                      </label>
                      <Input
                        value={String(
                          (selectedNode.config as Record<string, unknown>)?.output_key || ""
                        )}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            nodes: current.nodes.map((node) =>
                              node.id === selectedNode.id
                                ? {
                                    ...node,
                                    config: {
                                      ...(node.config as Record<string, unknown>),
                                      output_key: event.target.value,
                                    },
                                  }
                                : node
                            ),
                          }))
                        }
                        className="h-8 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}

                {selectedNode.type === "finalize" ? (
                  <div>
                    <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                      Response Template
                    </label>
                    <Textarea
                      value={String(
                        (selectedNode.config as Record<string, unknown>)?.response_template || ""
                      )}
                      onChange={(event) =>
                        updateDraft((current) => ({
                          ...current,
                          nodes: current.nodes.map((node) =>
                            node.id === selectedNode.id
                              ? {
                                  ...node,
                                  config: {
                                    ...(node.config as Record<string, unknown>),
                                    response_template: event.target.value,
                                  },
                                }
                              : node
                          ),
                        }))
                      }
                      rows={3}
                      className="border-subtle bg-[#08080c]"
                    />
                  </div>
                ) : null}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-sm uppercase tracking-wide text-secondary">Versions</h2>
              {isLoadingWorkflow ? (
                <span className="text-xs text-secondary">Loading...</span>
              ) : null}
            </div>
            <div className="mb-3 flex gap-2">
              <Input
                value={draft.workflow_id || ""}
                onChange={(event) => {
                  setWorkflowIdTouched(true);
                  setDraft((current) => ({
                    ...current,
                    workflow_id: event.target.value,
                  }));
                }}
                className="h-8 border-subtle bg-[#08080c] font-mono text-xs"
                placeholder="workflow.custom.id"
              />
              <Button
                type="button"
                className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                onClick={() => void loadWorkflow(String(draft.workflow_id || ""))}
              >
                Load
              </Button>
            </div>
            {versions.length === 0 ? (
              <p className="text-sm text-secondary">No versions loaded.</p>
            ) : (
              <div className="max-h-72 space-y-2 overflow-auto">
                {versions.map((version) => (
                  <div
                    key={version.version_id}
                    className={
                      "rounded-xl border p-2 text-xs " +
                      (version.version_id === selectedVersionId
                        ? "border-gold/40 bg-gold/10"
                        : "border-subtle bg-[#08080c]")
                    }
                  >
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <span className="text-primary">v{version.version_num}</span>
                      <span className="text-secondary">{version.created_by}</span>
                    </div>
                    <div className="text-secondary">{version.created_at}</div>
                    <Button
                      type="button"
                      className="mt-2 h-7 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                      onClick={() => handleLoadVersion(version)}
                    >
                      Load Version
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {messageError ? (
        <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
          {messageError}
        </div>
      ) : null}
      {messageSuccess ? (
        <div className="rounded-xl border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-100">
          {messageSuccess}
        </div>
      ) : null}
    </div>
  );
}
