import type { ChatRunTimelineStep } from "./api";

export type ChatRunMetaStatus = "running" | "ok" | "error";

export type ChatRunMetaRecord = {
  runId?: string;
  status: ChatRunMetaStatus;
  endedAt?: string;
  steps: ChatRunTimelineStep[];
  workflowId?: string;
  modelId?: string;
  toolIds?: string[];
};

export type ChatRunMetaMap = Record<string, ChatRunMetaRecord>;

const STORAGE_KEY = "saturday.chat.runmeta.v1";

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function sanitizeStep(raw: unknown): ChatRunTimelineStep | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const item = raw as Partial<ChatRunTimelineStep>;
  const stepIndex = Number(item.step_index);
  if (!Number.isFinite(stepIndex)) {
    return null;
  }
  const status =
    item.status === "ok" || item.status === "error" || item.status === "running"
      ? item.status
      : "running";
  const duration = Number(item.duration_ms);

  return {
    step_index: stepIndex,
    name: typeof item.name === "string" ? item.name : "",
    label: typeof item.label === "string" ? item.label : undefined,
    status,
    started_at: typeof item.started_at === "string" ? item.started_at : undefined,
    ended_at: typeof item.ended_at === "string" ? item.ended_at : undefined,
    summary: typeof item.summary === "string" ? item.summary : undefined,
    duration_ms: Number.isFinite(duration) ? duration : undefined,
  };
}

function sanitizeRecord(raw: unknown): ChatRunMetaRecord | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const item = raw as Partial<ChatRunMetaRecord>;
  const status: ChatRunMetaStatus =
    item.status === "ok" || item.status === "error" ? item.status : "running";
  const steps = Array.isArray(item.steps)
    ? item.steps
        .map((step) => sanitizeStep(step))
        .filter((step): step is ChatRunTimelineStep => step !== null)
        .sort((left, right) => left.step_index - right.step_index)
    : [];

  return {
    runId: typeof item.runId === "string" ? item.runId : undefined,
    status,
    endedAt: typeof item.endedAt === "string" ? item.endedAt : undefined,
    steps,
    workflowId: typeof item.workflowId === "string" ? item.workflowId : undefined,
    modelId: typeof item.modelId === "string" ? item.modelId : undefined,
    toolIds: isStringArray(item.toolIds) ? item.toolIds : undefined,
  };
}

function persist(value: ChatRunMetaMap): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Ignore storage write errors.
  }
}

export function loadChatRunMetaMap(): ChatRunMetaMap {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const sanitized: ChatRunMetaMap = {};
    for (const [messageId, value] of Object.entries(parsed)) {
      if (!messageId.trim()) {
        continue;
      }
      const record = sanitizeRecord(value);
      if (record) {
        sanitized[messageId] = record;
      }
    }
    return sanitized;
  } catch {
    return {};
  }
}

export function setChatRunMeta(messageId: string, record: ChatRunMetaRecord): ChatRunMetaMap {
  if (!messageId.trim()) {
    return loadChatRunMetaMap();
  }
  const next = loadChatRunMetaMap();
  const sanitized = sanitizeRecord(record);
  if (sanitized) {
    next[messageId] = sanitized;
    persist(next);
  }
  return next;
}

export function getChatRunMeta(messageId: string): ChatRunMetaRecord | null {
  const map = loadChatRunMetaMap();
  return map[messageId] ?? null;
}
