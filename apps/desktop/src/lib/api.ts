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

type WorkflowsResponse = {
  workflows?: Workflow[];
};

type ModelsResponse = {
  models?: Model[];
  default_model?: string;
  ollama_status?: string;
};

type ToolsResponse = {
  tools?: Tool[];
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
