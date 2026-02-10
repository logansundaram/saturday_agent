export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const REQUEST_TIMEOUT_MS = 10000;
const CHAT_REQUEST_TIMEOUT_MS = Number(
  import.meta.env.VITE_CHAT_TIMEOUT_MS ?? "180000"
);

function resolveTimeout(timeoutMs: number): number {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return REQUEST_TIMEOUT_MS;
  }
  return timeoutMs;
}

export type Workflow = {
  id: string;
  type: string;
  title: string;
  description?: string;
  version?: string;
  status?: string;
};

export type Model = {
  id: string;
  name: string;
  source?: string;
  status?: string;
};

export type Tool = {
  id: string;
  name: string;
  kind?: string;
  description?: string;
  enabled: boolean;
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
};

export type ChatRunStep = {
  name: string;
  status: "ok" | "error";
  started_at: string;
  ended_at: string;
};

export type ChatRunResponse = {
  run_id: string;
  output_text: string;
  steps?: ChatRunStep[];
  workflow_id: string;
  model_id: string;
  tool_ids: string[];
};

export type Run = {
  run_id: string;
  kind?: string;
  status: string;
  workflow_id?: string;
  workflow_type?: string;
  model_id?: string;
  tool_ids?: string[];
  started_at?: string;
  ended_at?: string;
  payload?: any;
  result?: any;
};

export type Step = {
  step_index: number;
  name: string;
  status: "ok" | "error";
  started_at?: string;
  ended_at?: string;
  input?: any;
  output?: any;
  error?: any;
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

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    resolveTimeout(timeoutMs)
  );

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
    window.clearTimeout(timeoutId);
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

export async function chatRun(req: ChatRunRequest): Promise<ChatRunResponse> {
  return fetchJson<ChatRunResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(req),
  }, CHAT_REQUEST_TIMEOUT_MS);
}

export async function getRun(runId: string): Promise<Run> {
  return fetchJson<Run>(`/runs/${encodeURIComponent(runId)}`);
}

export async function getRunLogs(runId: string): Promise<RunLogs> {
  const payload = await fetchJson<RunLogsResponse>(`/runs/${encodeURIComponent(runId)}/logs`);
  return {
    run_id: typeof payload.run_id === "string" ? payload.run_id : runId,
    steps: Array.isArray(payload.steps) ? payload.steps : [],
  };
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
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    resolveTimeout(REQUEST_TIMEOUT_MS)
  );

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
    window.clearTimeout(timeoutId);
  }
}
