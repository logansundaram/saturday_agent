export type Role = "system" | "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  runId?: string;
  workflowId?: string;
  modelId?: string;
  toolIds?: string[];
  artifactIds?: string[];
};

export type ChatThread = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  pinned?: boolean;
};

export type ChatStoreState = {
  threads: ChatThread[];
  messagesByThread: Record<string, ChatMessage[]>;
  activeThreadId: string;
};

const STORAGE_KEY = "saturday.chat.history.v1";
export const NEW_CHAT_TITLE = "New Chat";

// TODO: Swap to SQLite later via Electron main process IPC.
// TODO: Add full-text search later.

const createId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const sanitizeMessage = (value: unknown): ChatMessage | null => {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Partial<ChatMessage>;
  if (
    typeof record.id !== "string" ||
    (record.role !== "system" &&
      record.role !== "user" &&
      record.role !== "assistant") ||
    typeof record.content !== "string" ||
    !isFiniteNumber(record.createdAt)
  ) {
    return null;
  }

  return {
    id: record.id,
    role: record.role,
    content: record.content,
    createdAt: record.createdAt,
    runId: typeof record.runId === "string" ? record.runId : undefined,
    workflowId:
      typeof record.workflowId === "string" ? record.workflowId : undefined,
    modelId: typeof record.modelId === "string" ? record.modelId : undefined,
    toolIds: isStringArray(record.toolIds) ? record.toolIds : undefined,
    artifactIds: isStringArray(record.artifactIds) ? record.artifactIds : undefined,
  };
};

const sanitizeThread = (value: unknown): ChatThread | null => {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Partial<ChatThread>;
  if (
    typeof record.id !== "string" ||
    typeof record.title !== "string" ||
    !isFiniteNumber(record.createdAt) ||
    !isFiniteNumber(record.updatedAt)
  ) {
    return null;
  }

  return {
    id: record.id,
    title: record.title.trim() || NEW_CHAT_TITLE,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    pinned: typeof record.pinned === "boolean" ? record.pinned : undefined,
  };
};

const createThreadRecord = (
  id: string = createId(),
  title: string = NEW_CHAT_TITLE
): ChatThread => {
  const now = Date.now();
  return {
    id,
    title,
    createdAt: now,
    updatedAt: now,
  };
};

const sortThreads = (threads: ChatThread[]): ChatThread[] =>
  [...threads].sort(
    (left, right) =>
      Number(Boolean(right.pinned)) - Number(Boolean(left.pinned)) ||
      right.updatedAt - left.updatedAt ||
      right.createdAt - left.createdAt
  );

const createEmptyState = (): ChatStoreState => {
  return {
    threads: [],
    messagesByThread: {},
    activeThreadId: "",
  };
};

const normalizeState = (value: unknown): ChatStoreState => {
  if (!value || typeof value !== "object") {
    return createEmptyState();
  }

  const record = value as Partial<ChatStoreState>;
  const rawThreads = Array.isArray(record.threads) ? record.threads : [];
  const threads: ChatThread[] = rawThreads
    .map((thread) => sanitizeThread(thread))
    .filter((thread): thread is ChatThread => thread !== null);

  const messagesByThread: Record<string, ChatMessage[]> = {};
  const rawMessagesByThread =
    record.messagesByThread && typeof record.messagesByThread === "object"
      ? record.messagesByThread
      : {};

  for (const thread of threads) {
    const entry = (rawMessagesByThread as Record<string, unknown>)[thread.id];
    const messages = Array.isArray(entry)
      ? entry
          .map((message) => sanitizeMessage(message))
          .filter((message): message is ChatMessage => message !== null)
      : [];
    messagesByThread[thread.id] = messages;
  }

  if (threads.length === 0) {
    return createEmptyState();
  }

  const activeThreadId =
    typeof record.activeThreadId === "string" &&
    threads.some((thread) => thread.id === record.activeThreadId)
      ? record.activeThreadId
      : sortThreads(threads)[0]?.id ?? "";

  return {
    threads,
    messagesByThread,
    activeThreadId,
  };
};

const persistState = (state: ChatStoreState): void => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore write failures (e.g. quota exceeded).
  }
};

const mutateState = (mutator: (state: ChatStoreState) => void): ChatStoreState => {
  const next = loadState();
  mutator(next);
  const normalized = normalizeState(next);
  persistState(normalized);
  return normalized;
};

export function loadState(): ChatStoreState {
  if (typeof window === "undefined") {
    return createEmptyState();
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      const initial = createEmptyState();
      persistState(initial);
      return initial;
    }
    const parsed = JSON.parse(raw) as unknown;
    const normalized = normalizeState(parsed);
    persistState(normalized);
    return normalized;
  } catch {
    const initial = createEmptyState();
    persistState(initial);
    return initial;
  }
}

export function saveState(state: ChatStoreState): void {
  const normalized = normalizeState(state);
  persistState(normalized);
}

export function createThread(): string {
  const threadId = createId();
  mutateState((state) => {
    const thread = createThreadRecord(threadId);
    state.threads.push(thread);
    state.messagesByThread[threadId] = [];
    state.activeThreadId = threadId;
  });
  return threadId;
}

export function renameThread(threadId: string, title: string): void {
  const nextTitle = title.trim();
  if (!threadId || !nextTitle) {
    return;
  }

  mutateState((state) => {
    const thread = state.threads.find((item) => item.id === threadId);
    if (!thread) {
      return;
    }
    thread.title = nextTitle;
    thread.updatedAt = Date.now();
  });
}

export function deleteThread(threadId: string): void {
  if (!threadId) {
    return;
  }

  mutateState((state) => {
    state.threads = state.threads.filter((item) => item.id !== threadId);
    delete state.messagesByThread[threadId];

    if (state.threads.length === 0) {
      state.activeThreadId = "";
      return;
    }

    if (state.activeThreadId === threadId) {
      state.activeThreadId = sortThreads(state.threads)[0].id;
    }
  });
}

export function appendMessage(threadId: string, message: ChatMessage): void {
  if (!threadId) {
    return;
  }

  mutateState((state) => {
    const now = Date.now();
    if (!state.threads.some((item) => item.id === threadId)) {
      state.threads.push(createThreadRecord(threadId));
      state.messagesByThread[threadId] = [];
    }

    const nextMessage: ChatMessage = {
      ...message,
      id: message.id || createId(),
      createdAt: isFiniteNumber(message.createdAt) ? message.createdAt : now,
      content: message.content ?? "",
      toolIds: isStringArray(message.toolIds) ? message.toolIds : undefined,
      artifactIds: isStringArray(message.artifactIds)
        ? message.artifactIds
        : undefined,
    };

    const current = state.messagesByThread[threadId] ?? [];
    state.messagesByThread[threadId] = [...current, nextMessage];

    const thread = state.threads.find((item) => item.id === threadId);
    if (thread) {
      thread.updatedAt = now;
    }
    state.activeThreadId = threadId;
  });
}

export function setActiveThread(threadId: string): void {
  if (!threadId) {
    return;
  }

  mutateState((state) => {
    if (!state.threads.some((item) => item.id === threadId)) {
      state.threads.push(createThreadRecord(threadId));
      state.messagesByThread[threadId] = [];
    }
    state.activeThreadId = threadId;
  });
}
