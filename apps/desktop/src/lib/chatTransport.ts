export type Role = "system" | "user" | "assistant";

export type Message = {
  role: Role;
  content: string;
};

export type ChatOptions = {
  model?: string;
  temperature?: number;
  seed?: number;
};

export type ChatResult = {
  runId: string;
  model: string;
  outputText: string;
  raw?: unknown;
  timing?: {
    startedAt: string;
    endedAt: string;
    latencyMs: number;
  };
};

export interface ChatTransport {
  name: string;
  health(): Promise<{ ok: boolean; detail?: string }>;
  send(messages: Message[], opts?: ChatOptions): Promise<ChatResult>;
}
