import type {
  ValidationDiagnostic,
  WorkflowCompileResult,
  WorkflowSpec,
} from "@saturday/shared/workflow";
import type {
  ReplayDryRunResponse as SharedReplayDryRunResponse,
  ReplayRequest as SharedReplayRequest,
  ReplayResponse as SharedReplayResponse,
  Run as SharedRun,
  RunStepDetail as SharedRunStepDetail,
  RunStepSummary as SharedRunStepSummary,
  ToolCall as SharedToolCall,
} from "@saturday/shared/run";
import { streamSSE } from "./sse";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// 0 disables client-side timeout (temporary global default for local runs).
const REQUEST_TIMEOUT_MS = 0;
// 0 disables client-side timeout for long-running chat/workflow runs.
const chatTimeoutMsRaw = Number(import.meta.env.VITE_CHAT_TIMEOUT_MS ?? "0");
const CHAT_REQUEST_TIMEOUT_MS = Number.isFinite(chatTimeoutMsRaw)
  ? chatTimeoutMsRaw
  : 0;

function resolveTimeout(
  timeoutMs: number,
  fallbackMs: number = REQUEST_TIMEOUT_MS
): number | null {
  if (!Number.isFinite(timeoutMs)) {
    if (!Number.isFinite(fallbackMs) || fallbackMs <= 0) {
      return null;
    }
    return fallbackMs;
  }
  if (timeoutMs <= 0) {
    return null;
  }
  return timeoutMs;
}

export type Workflow = {
  id: string;
  type: string;
  title: string;
  name?: string;
  description?: string;
  version?: string;
  status?: string;
  source?: "builtin" | "custom" | string;
  enabled?: boolean;
};

export type WorkflowNode =
  | { id: string; type: "start"; config?: Record<string, never> }
  | {
      id: string;
      type: "llm";
      config: { prompt_template: string; output_key?: string };
    }
  | {
      id: string;
      type: "tool";
      config: {
        tool_id: string;
        input_map?: Record<string, string>;
        output_key?: string;
      };
    }
  | {
      id: string;
      type: "condition";
      config: {
        field: string;
        operator:
          | "equals"
          | "contains"
          | "gt"
          | "lt"
          | "exists"
          | "not_exists"
          | "in";
        value?: any;
      };
    }
  | { id: string; type: "end"; config: { response_template?: string } };

export type WorkflowEdge = {
  from: string;
  to: string;
  condition?: "always" | "true" | "false";
};

export type WorkflowDefinition = {
  id: string;
  name: string;
  title?: string;
  description: string;
  enabled: boolean;
  source: "builtin" | "custom" | string;
  type: string;
  graph: {
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  };
  created_at: string;
  updated_at: string;
};

export type CreateWorkflowPayload = {
  id?: string;
  name: string;
  description?: string;
  enabled?: boolean;
  graph: {
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  };
};

export type UpdateWorkflowPayload = {
  name?: string;
  description?: string;
  enabled?: boolean;
  graph?: {
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  };
};

export type Model = {
  id: string;
  name: string;
  source?: string;
  status?: string;
};

export type HttpToolConfig = {
  url: string;
  method?: "GET" | "POST" | string;
  headers?: Record<string, string>;
  timeout_ms?: number;
};

export type PythonToolConfig = {
  code: string;
  timeout_ms?: number;
  allowed_imports?: string[];
};

export type PromptToolConfig = {
  prompt_template: string;
  system_prompt?: string;
  temperature?: number;
  timeout_ms?: number;
};

export type ToolConfig = HttpToolConfig | PythonToolConfig | PromptToolConfig;

export type Tool = {
  id: string;
  name: string;
  kind: "local" | "external" | string;
  type: "http" | "python" | "prompt" | "builtin" | string;
  description: string;
  enabled: boolean;
  source?: "builtin" | "custom" | string;
  config: ToolConfig;
  input_schema?: any;
  output_schema?: any;
  created_at: string;
  updated_at: string;
};

export type CreateToolPayload = {
  name: string;
  id?: string;
  kind: "local" | "external";
  description: string;
  type: "http" | "python" | "prompt";
  enabled?: boolean;
  config: ToolConfig;
  input_schema?: any;
  output_schema?: any;
};

export type ChatRunRequest = {
  message: string;
  workflow_id: string;
  model_id: string;
  tool_ids: string[];
  vision_model_id?: string;
  artifact_ids?: string[];
  context?: any;
  thread_id?: string;
  stream?: boolean;
};

export type ChatRunStep = {
  name: string;
  status: "ok" | "error";
  started_at: string;
  ended_at: string;
  summary?: string;
};

export type ChatRunResponse = {
  run_id: string;
  output_text: string;
  steps?: ChatRunStep[];
  workflow_id: string;
  model_id: string;
  tool_ids: string[];
};

export type StreamStepStatus = "running" | "ok" | "error";

export type ChatRunTimelineStep = {
  step_index: number;
  name: string;
  label?: string;
  status: StreamStepStatus;
  started_at?: string;
  ended_at?: string;
  summary?: string;
  duration_ms?: number;
};

export type ChatRunStreamRunStartEvent = {
  type: "run_start";
  run_id: string;
  workflow_id: string;
  model_id: string;
  tool_ids: string[];
  started_at: string;
};

export type ChatRunStreamStepStartEvent = {
  type: "step_start";
  run_id: string;
  step_index: number;
  name: string;
  started_at: string;
  label?: string;
};

export type ChatRunStreamStepEndEvent = {
  type: "step_end";
  run_id: string;
  step_index: number;
  name: string;
  status: "ok" | "error";
  ended_at: string;
  summary: string;
  meta?: Record<string, any>;
};

export type ChatRunStreamToolCallEvent = {
  type: "tool_call";
  run_id: string;
  step_index: number;
  tool_id: string;
  input_summary?: string;
  input?: Record<string, any>;
};

export type ChatRunStreamToolResultEvent = {
  type: "tool_result";
  run_id: string;
  step_index: number;
  tool_id: string;
  status: "ok" | "error";
  output_summary?: string;
  output?: Record<string, any>;
};

export type ChatRunStreamTokenEvent = {
  type: "token";
  run_id: string;
  text: string;
};

export type ChatRunStreamFinalEvent = {
  type: "final";
  run_id: string;
  status: "ok" | "error";
  output_text: string;
  ended_at: string;
};

export type ChatRunStreamErrorEvent = {
  type: "error";
  run_id: string;
  message: string;
};

export type ChatRunStreamEvent =
  | ChatRunStreamRunStartEvent
  | ChatRunStreamStepStartEvent
  | ChatRunStreamStepEndEvent
  | ChatRunStreamToolCallEvent
  | ChatRunStreamToolResultEvent
  | ChatRunStreamTokenEvent
  | ChatRunStreamFinalEvent
  | ChatRunStreamErrorEvent;

export type WorkflowRunRequest = {
  workflow_version_id?: string;
  workflow_id?: string;
  draft_spec?: WorkflowSpec;
  input: Record<string, any>;
  sandbox_mode?: boolean;
  created_by?: string;
};

export type WorkflowRunResponse = {
  run_id: string;
  workflow_id: string;
  workflow_type?: string;
  workflow_version_id?: string;
  workflow_version_num?: number;
  status: string;
  sandbox_mode?: boolean;
  output: Record<string, any>;
  steps: Array<Record<string, any>>;
  diagnostics?: ValidationDiagnostic[];
};

export type Run = SharedRun;
export type StepToolCall = SharedToolCall;

export type Step = {
  step_id?: string;
  step_index: number;
  name: string;
  status: string;
  node_id?: string;
  node_type?: string;
  started_at?: string;
  ended_at?: string;
  summary?: string;
  input?: any;
  output?: any;
  error?: any;
  tool_calls?: StepToolCall[];
};

export type RunLogs = {
  run_id: string;
  steps: Step[];
};

export type RunSnapshot = {
  step_index: number;
  timestamp: string;
  state: any;
};

export type RunState = {
  run_id: string;
  snapshots: RunSnapshot[];
  derived?: boolean;
};

export type RunStepSummary = SharedRunStepSummary;
export type RunStepDetail = SharedRunStepDetail;
export type ReplayRequest = SharedReplayRequest;
export type ReplayResponse = SharedReplayResponse;
export type ReplayDryRunResponse = SharedReplayDryRunResponse;
export type RerunFromStateRequest = {
  step_index: number;
  state_json: Record<string, unknown>;
  resume?: "next" | "same";
  sandbox?: boolean | null;
};
export type RerunFromStateResponse = {
  new_run_id: string | null;
  diagnostics: ReplayResponse["diagnostics"];
};

type WorkflowsResponse = {
  workflows?: Workflow[];
};

type ModelsResponse = {
  models?: Model[];
  default_model?: string;
  ollama_status?: string;
};

export type ArtifactUploadResponse = {
  artifact_id: string;
  mime: string;
  size: number;
  sha256: string;
};

type ToolsResponse = {
  tools?: Tool[];
};

type RunLogsResponse = {
  run_id?: string;
  steps?: Step[];
};

type RunStateResponse = {
  run_id?: string;
  snapshots?: RunSnapshot[];
  derived?: boolean;
};

type RunStepsResponse = {
  run_id?: string;
  steps?: RunStepSummary[];
};

type RunStepDetailResponse = {
  run_id?: string;
  step?: RunStepDetail;
};

export type WorkflowVersionRecord = {
  version_id: string;
  workflow_id: string;
  version_num: number;
  spec?: WorkflowSpec;
  compiled?: Record<string, unknown>;
  created_at: string;
  created_by: string;
  workflow?: {
    workflow_id: string;
    name: string;
    description: string;
    enabled: boolean;
    created_at: string;
    updated_at: string;
  };
};

export type WorkflowDetail = {
  workflow_id: string;
  name: string;
  description: string;
  enabled: boolean;
  latest_version?: WorkflowVersionRecord | null;
  versions: WorkflowVersionRecord[];
};

export type PendingToolCall = {
  tool_call_id: string;
  run_id: string;
  step_id?: number | null;
  tool_name: string;
  args: Record<string, unknown>;
  approved_bool?: boolean | null;
  started_at?: string;
  status: string;
};

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<T> {
  const controller = new AbortController();
  const externalSignal = init?.signal;
  const abortFromExternal = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener("abort", abortFromExternal);
    }
  }
  const resolvedTimeoutMs = resolveTimeout(timeoutMs);
  const timeoutId =
    resolvedTimeoutMs === null
      ? null
      : window.setTimeout(() => controller.abort(), resolvedTimeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      method: init?.method ?? "GET",
      body: init?.body,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Request to ${path} failed (${response.status})`;
      try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string" && payload.detail.trim()) {
          detail = `${detail}: ${payload.detail}`;
        } else if (payload.detail !== undefined) {
          detail = `${detail}: ${JSON.stringify(payload.detail)}`;
        }
      } catch {
        // Keep default message if error payload is not JSON.
      }
      throw new Error(detail);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    if (error instanceof Error) {
      throw new Error(error.message || "Unable to reach backend API.");
    }
    throw new Error("Unable to reach backend API.");
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    if (externalSignal) {
      externalSignal.removeEventListener("abort", abortFromExternal);
    }
  }
}

export async function getWorkflows(): Promise<Workflow[]> {
  const payload = await fetchJson<WorkflowsResponse | Workflow[]>("/workflows");
  if (Array.isArray(payload)) {
    return payload;
  }
  if (!Array.isArray(payload.workflows)) {
    return [];
  }
  return payload.workflows;
}

export async function createWorkflow(
  payload: CreateWorkflowPayload
): Promise<WorkflowDefinition> {
  return fetchJson<WorkflowDefinition>("/builder/workflows", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getWorkflowDefinition(
  workflowId: string
): Promise<WorkflowDefinition> {
  return fetchJson<WorkflowDefinition>(
    `/builder/workflows/${encodeURIComponent(workflowId)}`
  );
}

export async function updateWorkflow(
  workflowId: string,
  payload: UpdateWorkflowPayload
): Promise<WorkflowDefinition> {
  return fetchJson<WorkflowDefinition>(
    `/builder/workflows/${encodeURIComponent(workflowId)}`,
    {
      method: "PATCH",
      body: JSON.stringify(payload),
    }
  );
}

export async function setWorkflowEnabled(
  workflowId: string,
  enabled: boolean
): Promise<WorkflowDefinition> {
  return updateWorkflow(workflowId, { enabled });
}

export async function compileWorkflowSpec(
  workflowSpec: WorkflowSpec,
  signal?: AbortSignal
): Promise<WorkflowCompileResult> {
  const payload = await fetchJson<WorkflowCompileResult>("/workflow/compile", {
    method: "POST",
    body: JSON.stringify({
      workflow_spec: workflowSpec,
    }),
    signal,
  });
  return {
    valid: Boolean(payload.valid),
    workflow_spec: payload.workflow_spec,
    compiled: payload.compiled || {},
    diagnostics: Array.isArray(payload.diagnostics) ? payload.diagnostics : [],
  };
}

export async function getWorkflowDetail(
  workflowId: string,
  signal?: AbortSignal
): Promise<WorkflowDetail> {
  const payload = await fetchJson<WorkflowDetail>(
    `/workflow/${encodeURIComponent(workflowId)}`,
    { signal }
  );
  return {
    workflow_id: payload.workflow_id,
    name: payload.name,
    description: payload.description || "",
    enabled: Boolean(payload.enabled),
    latest_version: payload.latest_version || null,
    versions: Array.isArray(payload.versions) ? payload.versions : [],
  };
}

export async function getWorkflowVersions(
  workflowId: string,
  signal?: AbortSignal
): Promise<WorkflowVersionRecord[]> {
  const payload = await fetchJson<{
    workflow_id: string;
    versions?: WorkflowVersionRecord[];
  }>(`/workflow/${encodeURIComponent(workflowId)}/versions`, { signal });
  return Array.isArray(payload.versions) ? payload.versions : [];
}

export async function createWorkflowVersion(
  workflowId: string,
  workflowSpec: WorkflowSpec,
  createdBy: string = "builder",
  signal?: AbortSignal
): Promise<WorkflowVersionRecord> {
  return fetchJson<WorkflowVersionRecord>(
    `/workflow/${encodeURIComponent(workflowId)}/versions`,
    {
      method: "POST",
      body: JSON.stringify({
        workflow_spec: workflowSpec,
        created_by: createdBy,
      }),
      signal,
    }
  );
}

export async function runWorkflow(
  payload: WorkflowRunRequest,
  signal?: AbortSignal
): Promise<WorkflowRunResponse> {
  return fetchJson<WorkflowRunResponse>("/workflow/run", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  }, CHAT_REQUEST_TIMEOUT_MS);
}

export async function getModels(): Promise<{
  models: Model[];
  default_model?: string;
  ollama_status?: string;
}> {
  const payload = await fetchJson<ModelsResponse | Model[]>("/models");
  if (Array.isArray(payload)) {
    return { models: payload };
  }

  const models = Array.isArray(payload.models) ? payload.models : [];
  return {
    models,
    default_model: payload.default_model,
    ollama_status: payload.ollama_status,
  };
}

export async function getVisionModels(): Promise<{
  models: Model[];
  default_model?: string;
  ollama_status?: string;
}> {
  const payload = await fetchJson<ModelsResponse | Model[]>("/models/vision");
  if (Array.isArray(payload)) {
    return { models: payload };
  }

  const models = Array.isArray(payload.models) ? payload.models : [];
  return {
    models,
    default_model: payload.default_model,
    ollama_status: payload.ollama_status,
  };
}

export async function getTools(): Promise<Tool[]> {
  const payload = await fetchJson<ToolsResponse | Tool[]>("/tools");
  if (Array.isArray(payload)) {
    return payload;
  }
  if (!Array.isArray(payload.tools)) {
    return [];
  }
  return payload.tools;
}

export async function createTool(payload: CreateToolPayload): Promise<Tool> {
  return fetchJson<Tool>("/builder/tools", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function setToolEnabled(id: string, enabled: boolean): Promise<Tool> {
  return fetchJson<Tool>(`/tools/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ enabled }),
  });
}

export async function chatRun(req: ChatRunRequest): Promise<ChatRunResponse> {
  return fetchJson<ChatRunResponse>(
    "/chat",
    {
      method: "POST",
      body: JSON.stringify(req),
    },
    CHAT_REQUEST_TIMEOUT_MS
  );
}

export async function chatRunStream(
  req: ChatRunRequest,
  onEvent: (event: ChatRunStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  await streamSSE<ChatRunStreamEvent>(
    `${API_BASE_URL}/chat/stream`,
    { ...req, stream: true },
    onEvent,
    {
      signal,
      timeoutMs: CHAT_REQUEST_TIMEOUT_MS,
      headers: {
        Accept: "text/event-stream",
      },
    }
  );
}

export async function getRun(runId: string, signal?: AbortSignal): Promise<Run> {
  return fetchJson<Run>(`/runs/${encodeURIComponent(runId)}`, { signal });
}

export async function getRunLogs(
  runId: string,
  signal?: AbortSignal
): Promise<RunLogs> {
  const payload = await fetchJson<RunLogsResponse>(
    `/runs/${encodeURIComponent(runId)}/logs`,
    { signal }
  );
  return {
    run_id: typeof payload.run_id === "string" ? payload.run_id : runId,
    steps: Array.isArray(payload.steps) ? payload.steps : [],
  };
}

export async function getRunSteps(
  runId: string,
  signal?: AbortSignal
): Promise<{ run_id: string; steps: RunStepSummary[] }> {
  const payload = await fetchJson<RunStepsResponse>(
    `/runs/${encodeURIComponent(runId)}/steps`,
    { signal }
  );
  return {
    run_id: typeof payload.run_id === "string" ? payload.run_id : runId,
    steps: Array.isArray(payload.steps) ? payload.steps : [],
  };
}

export async function getRunStepDetail(
  runId: string,
  stepId: string,
  signal?: AbortSignal
): Promise<{ run_id: string; step: RunStepDetail | null }> {
  const payload = await fetchJson<RunStepDetailResponse>(
    `/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepId)}`,
    { signal }
  );
  return {
    run_id: typeof payload.run_id === "string" ? payload.run_id : runId,
    step: payload.step ?? null,
  };
}

export async function replayRunDryRun(
  runId: string,
  request: ReplayRequest,
  signal?: AbortSignal
): Promise<ReplayDryRunResponse> {
  const payload = await fetchJson<ReplayDryRunResponse>(
    `/runs/${encodeURIComponent(runId)}/replay/dry_run`,
    {
      method: "POST",
      body: JSON.stringify(request),
      signal,
    }
  );
  return {
    new_run_id: payload.new_run_id ?? null,
    diagnostics: Array.isArray(payload.diagnostics) ? payload.diagnostics : [],
    fork_start_state:
      payload.fork_start_state && typeof payload.fork_start_state === "object"
        ? payload.fork_start_state
        : null,
    resume_node_id: payload.resume_node_id ?? null,
  };
}

export async function replayRun(
  runId: string,
  request: ReplayRequest,
  signal?: AbortSignal
): Promise<ReplayResponse> {
  const payload = await fetchJson<ReplayResponse>(
    `/runs/${encodeURIComponent(runId)}/replay`,
    {
      method: "POST",
      body: JSON.stringify(request),
      signal,
    }
  );
  return {
    new_run_id: payload.new_run_id ?? null,
    diagnostics: Array.isArray(payload.diagnostics) ? payload.diagnostics : [],
    fork_start_state:
      payload.fork_start_state && typeof payload.fork_start_state === "object"
        ? payload.fork_start_state
        : null,
    resume_node_id: payload.resume_node_id ?? null,
  };
}

export async function rerunFromState(
  runId: string,
  request: RerunFromStateRequest,
  signal?: AbortSignal
): Promise<RerunFromStateResponse> {
  const payload = await fetchJson<{
    new_run_id?: string | null;
    diagnostics?: ReplayResponse["diagnostics"];
  }>(`/runs/${encodeURIComponent(runId)}/rerun_from_state`, {
    method: "POST",
    body: JSON.stringify(request),
    signal,
  });
  return {
    new_run_id: payload.new_run_id ?? null,
    diagnostics: Array.isArray(payload.diagnostics) ? payload.diagnostics : [],
  };
}

export async function getPendingToolCalls(
  runId: string,
  signal?: AbortSignal
): Promise<PendingToolCall[]> {
  const payload = await fetchJson<{
    run_id?: string;
    pending?: PendingToolCall[];
  }>(`/runs/${encodeURIComponent(runId)}/pending_tool_calls`, { signal });
  return Array.isArray(payload.pending) ? payload.pending : [];
}

export async function approveToolCall(
  runId: string,
  toolCallId: string,
  approved: boolean
): Promise<{
  run_id: string;
  tool_call_id: string;
  approved: boolean;
  status: string;
}> {
  return fetchJson(
    `/runs/${encodeURIComponent(runId)}/tool_calls/${encodeURIComponent(
      toolCallId
    )}/approve`,
    {
      method: "POST",
      body: JSON.stringify({ approved }),
    }
  );
}

export async function getRunState(runId: string): Promise<RunState | null> {
  try {
    const payload = await fetchJson<RunStateResponse>(
      `/runs/${encodeURIComponent(runId)}/state`
    );
    return {
      run_id: typeof payload.run_id === "string" ? payload.run_id : runId,
      snapshots: Array.isArray(payload.snapshots) ? payload.snapshots : [],
      derived: Boolean(payload.derived),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    if (message.includes("(404)")) {
      return null;
    }
    throw error;
  }
}

export async function uploadArtifact(file: File): Promise<ArtifactUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const controller = new AbortController();
  const resolvedTimeoutMs = resolveTimeout(REQUEST_TIMEOUT_MS);
  const timeoutId =
    resolvedTimeoutMs === null
      ? null
      : window.setTimeout(() => controller.abort(), resolvedTimeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}/artifacts/upload`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Artifact upload failed (${response.status})`;
      try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string" && payload.detail.trim()) {
          detail = `${detail}: ${payload.detail}`;
        }
      } catch {
        // Keep default message.
      }
      throw new Error(detail);
    }

    return (await response.json()) as ArtifactUploadResponse;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    if (error instanceof Error) {
      throw new Error(error.message || "Unable to upload artifact.");
    }
    throw new Error("Unable to upload artifact.");
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
  }
}
