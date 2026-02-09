import type { ChatOptions, ChatResult, ChatTransport, Message } from "../chatTransport";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 30000;

type ApiChatTiming = {
  started_at?: string;
  ended_at?: string;
  latency_ms?: number;
};

type ApiChatResponse = {
  run_id: string;
  model: string;
  output_text: string;
  raw?: unknown;
  timing?: ApiChatTiming;
};

type ApiHealthResponse = {
  api?: "ok" | "down";
  ollama?: "ok" | "down";
};

export class LangGraphApiTransport implements ChatTransport {
  name = "langgraph";
  private baseUrl: string;

  constructor(baseUrl: string = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async health(): Promise<{ ok: boolean; detail?: string }> {
    try {
      const response = await fetchWithTimeout(`${this.baseUrl}/health`, {
        method: "GET",
      });
      if (!response.ok) {
        return { ok: false, detail: `HTTP ${response.status}` };
      }

      const payload = (await response.json()) as ApiHealthResponse;
      if (payload.api !== "ok") {
        return { ok: false, detail: "API is down" };
      }
      if (payload.ollama !== "ok") {
        return { ok: false, detail: "Ollama is down" };
      }
      return { ok: true };
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return { ok: false, detail: "Timeout" };
      }
      const detail = error instanceof Error ? error.message : "Network error";
      return { ok: false, detail };
    }
  }

  async send(messages: Message[], opts?: ChatOptions): Promise<ChatResult> {
    const startedAt = new Date();

    const payload: Record<string, unknown> = {
      messages,
      stream: false,
    };
    if (opts?.model) {
      payload.model = opts.model;
    }
    if (opts?.temperature !== undefined) {
      payload.temperature = opts.temperature;
    }
    if (opts?.seed !== undefined) {
      payload.seed = opts.seed;
    }

    const response = await fetchWithTimeout(`${this.baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `Backend error (${response.status})`;
      try {
        const errorPayload = (await response.json()) as { detail?: unknown };
        if (typeof errorPayload?.detail === "string" && errorPayload.detail.trim()) {
          detail = errorPayload.detail;
        }
      } catch {
        // Response may not be JSON on failures.
      }
      throw new Error(detail);
    }

    const data = (await response.json()) as ApiChatResponse;
    const endedAt = new Date();

    return {
      runId: data.run_id,
      model: data.model,
      outputText: data.output_text ?? "",
      raw: data.raw,
      timing: {
        startedAt: data.timing?.started_at ?? startedAt.toISOString(),
        endedAt: data.timing?.ended_at ?? endedAt.toISOString(),
        latencyMs:
          data.timing?.latency_ms ??
          Math.max(0, endedAt.getTime() - startedAt.getTime()),
      },
    };
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
}
