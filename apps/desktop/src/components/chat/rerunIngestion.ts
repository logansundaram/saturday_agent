import type { ChatRunTimelineStep, Run, Step } from "../../lib/api";
import type { ChatRunMetaMap } from "../../lib/chatRunMetaStore";
import type { ChatMessage, ChatStoreState } from "../../lib/chatStore";

const OK_STEP_STATUSES = new Set(["success", "ok", "completed", "skipped"]);
const ERROR_STEP_STATUSES = new Set(["error", "failed", "rejected"]);
const OK_RUN_STATUSES = new Set(["ok", "completed", "success"]);
const ERROR_RUN_STATUSES = new Set(["error", "failed", "rejected"]);

type ChatStateSlice = Pick<ChatStoreState, "messagesByThread" | "activeThreadId">;

type ResolveRerunTargetThreadInput = {
  chatState: ChatStateSlice;
  runMetaByMessageId: ChatRunMetaMap;
  sourceRunId?: string | null;
};

type ResolveSourcePromptMessageInput = {
  messages: ChatMessage[];
  runMetaByMessageId: ChatRunMetaMap;
  sourceRunId?: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalized(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim();
}

export function normalizePromptText(value: string): string {
  return normalized(value);
}

function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value.trim();
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => {
        if (typeof item === "string") {
          return item.trim();
        }
        const record = asRecord(item);
        if (!record) {
          return "";
        }
        return normalized(record.text || record.content || record.message);
      })
      .filter(Boolean);
    return parts.join("\n").trim();
  }
  const record = asRecord(value);
  if (!record) {
    return "";
  }
  return normalized(record.text || record.content || record.message);
}

function extractLastUserMessage(messages: unknown): string {
  if (!Array.isArray(messages)) {
    return "";
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const item = messages[index];
    if (typeof item === "string") {
      const text = item.trim();
      if (text) {
        return text;
      }
      continue;
    }
    const record = asRecord(item);
    if (!record) {
      continue;
    }
    const role = normalized(record.role).toLowerCase();
    const content = extractText(record.content ?? record.text ?? record.message);
    if (!content) {
      continue;
    }
    if (!role || role === "user" || role === "human") {
      return content;
    }
  }
  return "";
}

function mapStepStatus(status: unknown): ChatRunTimelineStep["status"] {
  const value = normalized(status).toLowerCase();
  if (OK_STEP_STATUSES.has(value)) {
    return "ok";
  }
  if (ERROR_STEP_STATUSES.has(value)) {
    return "error";
  }
  return "running";
}

function durationMs(startedAt?: string, endedAt?: string): number | undefined {
  if (!startedAt || !endedAt) {
    return undefined;
  }
  const started = Date.parse(startedAt);
  const ended = Date.parse(endedAt);
  if (!Number.isFinite(started) || !Number.isFinite(ended)) {
    return undefined;
  }
  return Math.max(0, ended - started);
}

export function extractRerunPromptCandidate(run: Run): string {
  const forkState = asRecord(run.forked_from_state_json);
  const payload = asRecord(run.payload);
  const payloadInput = asRecord(payload?.input);

  const candidates = [
    normalized(forkState?.task),
    extractLastUserMessage(forkState?.messages),
    normalized(payloadInput?.task),
    normalized(payload?.message),
    extractLastUserMessage(payloadInput?.messages),
  ];

  for (const candidate of candidates) {
    if (candidate) {
      return candidate;
    }
  }
  return "";
}

export function extractRerunOutputText(run: Run): string {
  const result = asRecord(run.result);
  const outputText = normalized(result?.output_text);
  if (outputText) {
    return outputText;
  }

  const output = asRecord(result?.output);
  const answer = normalized(output?.answer);
  if (answer) {
    return answer;
  }
  const error = normalized(output?.error);
  if (error) {
    return error;
  }

  const status = normalized(run.status).toLowerCase();
  if (ERROR_RUN_STATUSES.has(status)) {
    return `Error: Run ended with status ${status || "error"}.`;
  }
  if (OK_RUN_STATUSES.has(status)) {
    return "Run completed.";
  }
  return "";
}

export function latestUserMessageText(messages: ChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user") {
      continue;
    }
    const content = normalized(message.content);
    if (content) {
      return content;
    }
  }
  return "";
}

export function resolveSourcePromptUserMessageId(
  input: ResolveSourcePromptMessageInput
): string | null {
  const sourceRunId = normalized(input.sourceRunId);
  if (!sourceRunId) {
    return null;
  }

  let assistantIndex = -1;
  for (let index = input.messages.length - 1; index >= 0; index -= 1) {
    const message = input.messages[index];
    if (message.role !== "assistant") {
      continue;
    }
    const runIdFromMessage = normalized(message.runId);
    const runIdFromMeta = normalized(input.runMetaByMessageId[message.id]?.runId);
    if (runIdFromMessage === sourceRunId || runIdFromMeta === sourceRunId) {
      assistantIndex = index;
      break;
    }
  }

  if (assistantIndex < 0) {
    return null;
  }

  for (let pointer = assistantIndex - 1; pointer >= 0; pointer -= 1) {
    const candidate = input.messages[pointer];
    if (candidate.role !== "user") {
      continue;
    }
    if (normalized(candidate.id)) {
      return candidate.id;
    }
  }
  return null;
}

export function mapRunLogStepsToTimeline(steps: Step[]): ChatRunTimelineStep[] {
  return [...steps]
    .sort((left, right) => left.step_index - right.step_index)
    .map((step) => {
      const name = normalized(step.name) || `step_${step.step_index}`;
      const startedAt = normalized(step.started_at) || undefined;
      const endedAt = normalized(step.ended_at) || undefined;
      return {
        step_index: Number(step.step_index),
        name,
        label: name.replace(/_/g, " "),
        status: mapStepStatus(step.status),
        started_at: startedAt,
        ended_at: endedAt,
        summary: normalized(step.summary) || undefined,
        duration_ms: durationMs(startedAt, endedAt),
      };
    });
}

export function isTerminalRunStatus(status: string): boolean {
  const normalizedStatus = normalized(status).toLowerCase();
  return OK_RUN_STATUSES.has(normalizedStatus) || ERROR_RUN_STATUSES.has(normalizedStatus);
}

export function toRunMetaStatus(status: string): "running" | "ok" | "error" {
  const normalizedStatus = normalized(status).toLowerCase();
  if (OK_RUN_STATUSES.has(normalizedStatus)) {
    return "ok";
  }
  if (ERROR_RUN_STATUSES.has(normalizedStatus)) {
    return "error";
  }
  return "running";
}

export function resolveRerunTargetThreadId(
  input: ResolveRerunTargetThreadInput
): string | null {
  const sourceRunId = normalized(input.sourceRunId);
  if (sourceRunId) {
    for (const [threadId, messages] of Object.entries(input.chatState.messagesByThread)) {
      const hasSourceRunMessage = messages.some(
        (message) => normalized(message.runId) === sourceRunId
      );
      if (hasSourceRunMessage) {
        return threadId;
      }
    }

    const sourceMessageId = Object.entries(input.runMetaByMessageId).find(
      ([, meta]) => normalized(meta?.runId) === sourceRunId
    )?.[0];
    if (sourceMessageId) {
      for (const [threadId, messages] of Object.entries(input.chatState.messagesByThread)) {
        if (messages.some((message) => message.id === sourceMessageId)) {
          return threadId;
        }
      }
    }
  }

  const activeThreadId = normalized(input.chatState.activeThreadId);
  return activeThreadId || null;
}
