// TODO: Swap implementation to FastAPI/LangChain later (single choke point).

const DEFAULT_BASE_URL = "http://localhost:11434";
const DEFAULT_TIMEOUT_MS = 3000;

export type OllamaModel = {
  name: string;
  model?: string;
  modified_at?: string;
  size?: number;
  digest?: string;
  details?: {
    family?: string;
    format?: string;
    parameter_size?: string;
    quantization_level?: string;
  };
};

export async function listModels(): Promise<OllamaModel[]> {
  const baseUrl = import.meta.env.VITE_OLLAMA_BASE_URL ?? DEFAULT_BASE_URL;

  try {
    const response = await fetchWithTimeout(`${baseUrl}/api/tags`);
    if (!response.ok) {
      throw new Error(`Ollama error (${response.status})`);
    }

    const data = (await response.json()) as { models?: OllamaModel[] };
    return Array.isArray(data?.models) ? data.models : [];
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Ollama request timed out");
    }

    if (error instanceof Error && error.message.startsWith("Ollama error")) {
      throw error;
    }

    throw new Error("Unable to reach Ollama");
  }
}

export async function ollamaHealth(): Promise<{ ok: boolean; detail?: string }> {
  const baseUrl = import.meta.env.VITE_OLLAMA_BASE_URL ?? DEFAULT_BASE_URL;

  try {
    const response = await fetchWithTimeout(`${baseUrl}/api/tags`);
    if (!response.ok) {
      return { ok: false, detail: `HTTP ${response.status}` };
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

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
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
