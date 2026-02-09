import type { ChatOptions, ChatResult, ChatTransport, Message, Role } from "../chatTransport";

const DEFAULT_BASE_URL = "http://localhost:11434";
const DEFAULT_MODEL = "llama3.1:8b";

const roleLabels: Record<Role, string> = {
  system: "System",
  user: "User",
  assistant: "Assistant",
};

export class OllamaTransport implements ChatTransport {
  name = "ollama";
  private baseUrl: string;
  private defaultModel: string;

  constructor(
    baseUrl: string = import.meta.env.VITE_OLLAMA_BASE_URL ?? DEFAULT_BASE_URL,
    defaultModel: string = import.meta.env.VITE_OLLAMA_MODEL ?? DEFAULT_MODEL
  ) {
    this.baseUrl = baseUrl;
    this.defaultModel = defaultModel;
  }

  async health(): Promise<{ ok: boolean; detail?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) {
        return { ok: false, detail: `HTTP ${response.status}` };
      }
      return { ok: true };
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Network error";
      return { ok: false, detail };
    }
  }

  async send(messages: Message[], opts?: ChatOptions): Promise<ChatResult> {
    const startedAt = new Date();
    const prompt = buildPrompt(messages);
    const model = opts?.model ?? this.defaultModel;

    const options: Record<string, number> = {};
    if (opts?.temperature !== undefined) {
      options.temperature = opts.temperature;
    }
    if (opts?.seed !== undefined) {
      options.seed = opts.seed;
    }

    const payload: Record<string, unknown> = {
      model,
      prompt,
      stream: false,
    };

    if (Object.keys(options).length > 0) {
      payload.options = options;
    }

    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Ollama error (${response.status})`);
    }

    const data = (await response.json()) as { response?: string };
    const endedAt = new Date();

    return {
      runId: crypto.randomUUID(),
      model,
      outputText: data.response ?? "",
      raw: data,
      timing: {
        startedAt: startedAt.toISOString(),
        endedAt: endedAt.toISOString(),
        latencyMs: endedAt.getTime() - startedAt.getTime(),
      },
    };
  }
}

function buildPrompt(messages: Message[]): string {
  const parts = messages.map((message) => {
    const label = roleLabels[message.role];
    return `${label}:\n${message.content}`;
  });

  parts.push("Assistant:");
  return parts.join("\n\n").trim();
}

// TODO: Add a FastAPI/LangChain transport and select via env flag.
