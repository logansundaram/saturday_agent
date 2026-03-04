export type ProjectChatRole = "system" | "user" | "assistant";

export type ProjectChatMessage = {
  id: string;
  role: ProjectChatRole;
  content: string;
  createdAt: number;
  runId?: string;
  workflowId?: string;
  modelId?: string;
  toolIds?: string[];
  artifactIds?: string[];
};

export type ProjectChatMessagePatch = Partial<
  Pick<
    ProjectChatMessage,
    "content" | "runId" | "workflowId" | "modelId" | "toolIds" | "artifactIds"
  >
>;

type ProjectChatStoreState = {
  messagesByChatId: Record<string, ProjectChatMessage[]>;
};

const STORAGE_KEY = "saturday.project.chat.history.v1";

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const sanitizeMessage = (value: unknown): ProjectChatMessage | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Partial<ProjectChatMessage>;
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

const createEmptyState = (): ProjectChatStoreState => ({
  messagesByChatId: {},
});

const persist = (state: ProjectChatStoreState): void => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore quota errors.
  }
};

const loadState = (): ProjectChatStoreState => {
  if (typeof window === "undefined") {
    return createEmptyState();
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      const initial = createEmptyState();
      persist(initial);
      return initial;
    }
    const parsed = JSON.parse(raw) as ProjectChatStoreState;
    const messagesByChatId: Record<string, ProjectChatMessage[]> = {};
    const source =
      parsed && typeof parsed === "object" && parsed.messagesByChatId
        ? parsed.messagesByChatId
        : {};
    for (const [chatId, messages] of Object.entries(source)) {
      if (!chatId.trim() || !Array.isArray(messages)) {
        continue;
      }
      messagesByChatId[chatId] = messages
        .map((message) => sanitizeMessage(message))
        .filter((message): message is ProjectChatMessage => message !== null);
    }
    const normalized = { messagesByChatId };
    persist(normalized);
    return normalized;
  } catch {
    const initial = createEmptyState();
    persist(initial);
    return initial;
  }
};

const mutate = (
  mutator: (state: ProjectChatStoreState) => void
): ProjectChatStoreState => {
  const next = loadState();
  mutator(next);
  persist(next);
  return next;
};

export function getProjectChatMessages(chatId: string): ProjectChatMessage[] {
  return loadState().messagesByChatId[chatId] ?? [];
}

export function appendProjectChatMessage(
  chatId: string,
  message: ProjectChatMessage
): ProjectChatMessage[] {
  if (!chatId.trim()) {
    return [];
  }
  const next = mutate((state) => {
    const current = state.messagesByChatId[chatId] ?? [];
    state.messagesByChatId[chatId] = [...current, message];
  });
  return next.messagesByChatId[chatId] ?? [];
}

export function updateProjectChatMessage(
  chatId: string,
  messageId: string,
  patch: ProjectChatMessagePatch
): ProjectChatMessage[] {
  if (!chatId.trim() || !messageId.trim()) {
    return getProjectChatMessages(chatId);
  }
  const next = mutate((state) => {
    const current = state.messagesByChatId[chatId] ?? [];
    state.messagesByChatId[chatId] = current.map((message) =>
      message.id === messageId
        ? {
            ...message,
            ...patch,
            toolIds: isStringArray(patch.toolIds)
              ? patch.toolIds
              : message.toolIds,
            artifactIds: isStringArray(patch.artifactIds)
              ? patch.artifactIds
              : message.artifactIds,
          }
        : message
    );
  });
  return next.messagesByChatId[chatId] ?? [];
}

export function deleteProjectChatMessages(chatId: string): void {
  if (!chatId.trim()) {
    return;
  }
  mutate((state) => {
    delete state.messagesByChatId[chatId];
  });
}
