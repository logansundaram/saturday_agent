import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AttachmentsBar, {
  type ChatAttachment,
} from "../components/chat/AttachmentsBar";
import DropzoneOverlay from "../components/chat/DropzoneOverlay";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import {
  API_BASE_URL,
  chatRun,
  getModels,
  getTools,
  getVisionModels,
  getWorkflows,
  uploadArtifact,
} from "../lib/api";
import type { Model, Tool, Workflow } from "../lib/api";
import {
  NEW_CHAT_TITLE,
  appendMessage,
  createThread,
  deleteThread,
  loadState,
  renameThread,
  setActiveThread,
  type ChatMessage,
  type ChatStoreState,
  type ChatThread,
} from "../lib/chatStore";

const MAX_TEXTAREA_HEIGHT = 160;
const VISION_TOOL_ID = "vision.analyze";

const toImageFiles = (files: FileList | File[] | null): File[] => {
  if (!files) {
    return [];
  }
  return Array.from(files).filter((file) => file.type.startsWith("image/"));
};

const artifactPreviewUrl = (artifactId: string): string =>
  `${API_BASE_URL}/artifacts/${encodeURIComponent(artifactId)}`;

const hasFilePayload = (dataTransfer: DataTransfer | null): boolean => {
  if (!dataTransfer) {
    return false;
  }
  return Array.from(dataTransfer.types || []).includes("Files");
};

const createId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const sortThreads = (threads: ChatThread[]): ChatThread[] => {
  return [...threads].sort(
    (left, right) =>
      Number(Boolean(right.pinned)) - Number(Boolean(left.pinned)) ||
      right.updatedAt - left.updatedAt ||
      right.createdAt - left.createdAt
  );
};

const formatThreadTime = (updatedAt: number): string => {
  const diff = Date.now() - updatedAt;
  if (diff < 60_000) {
    return "now";
  }
  if (diff < 3_600_000) {
    return `${Math.floor(diff / 60_000)}m`;
  }
  if (diff < 86_400_000) {
    return `${Math.floor(diff / 3_600_000)}h`;
  }
  if (diff < 604_800_000) {
    return `${Math.floor(diff / 86_400_000)}d`;
  }
  return new Date(updatedAt).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
};

const deriveThreadTitle = (text: string): string => {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) {
    return NEW_CHAT_TITLE;
  }
  return clean.length > 48 ? `${clean.slice(0, 48)}...` : clean;
};

type ChatPageProps = {
  onInspectRun?: (runId: string) => void;
};

export default function ChatPage({ onInspectRun }: ChatPageProps) {
  const [chatState, setChatState] = useState<ChatStoreState>(() => loadState());
  const [input, setInput] = useState<string>("");
  const [isSending, setIsSending] = useState<boolean>(false);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [visionModels, setVisionModels] = useState<Model[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);

  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [selectedVisionModelId, setSelectedVisionModelId] = useState<string>("");
  const [selectedToolIds, setSelectedToolIds] = useState<string[]>([]);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [isUploadingArtifacts, setIsUploadingArtifacts] = useState<boolean>(false);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [visionModelsFallback, setVisionModelsFallback] = useState<boolean>(false);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editingThreadTitle, setEditingThreadTitle] = useState<string>("");
  const [threadPendingDelete, setThreadPendingDelete] = useState<ChatThread | null>(
    null
  );

  const [optionsLoading, setOptionsLoading] = useState<boolean>(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [toolsWarning, setToolsWarning] = useState<string | null>(null);
  const [toolsMenuOpen, setToolsMenuOpen] = useState<boolean>(false);

  const listRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const toolsMenuRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef<number>(0);

  const backendReady =
    !optionsLoading &&
    !optionsError &&
    workflows.length > 0 &&
    models.length > 0 &&
    selectedWorkflowId.length > 0 &&
    selectedModelId.length > 0;

  const refreshChatState = useCallback(() => {
    setChatState(loadState());
  }, []);

  const sortedThreads = useMemo(
    () => sortThreads(chatState.threads),
    [chatState.threads]
  );
  const activeThread = useMemo(
    () =>
      chatState.threads.find((thread) => thread.id === chatState.activeThreadId) ??
      null,
    [chatState.threads, chatState.activeThreadId]
  );
  const activeMessages = useMemo(() => {
    if (!chatState.activeThreadId) {
      return [];
    }
    return chatState.messagesByThread[chatState.activeThreadId] ?? [];
  }, [chatState.activeThreadId, chatState.messagesByThread]);

  useEffect(() => {
    if (chatState.threads.length > 0 && chatState.activeThreadId) {
      return;
    }
    createThread();
    refreshChatState();
  }, [chatState.threads.length, chatState.activeThreadId, refreshChatState]);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const container = listRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [activeMessages, autoScroll, isSending]);

  useEffect(() => {
    if (textareaRef.current) {
      resizeTextarea(textareaRef.current);
    }
  }, [input]);

  useEffect(() => {
    if (!toolsMenuOpen) {
      return;
    }

    const handleClickOutside = (event: MouseEvent) => {
      if (!toolsMenuRef.current) {
        return;
      }
      if (!toolsMenuRef.current.contains(event.target as Node)) {
        setToolsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [toolsMenuOpen]);

  const loadOptions = useCallback(async () => {
    setOptionsLoading(true);
    setOptionsError(null);

    const [workflowResult, modelResult, toolResult, visionResult] = await Promise.allSettled([
      getWorkflows(),
      getModels(),
      getTools(),
      getVisionModels(),
    ]);

    const criticalErrors: string[] = [];
    const nonCriticalErrors: string[] = [];

    const workflowItems =
      workflowResult.status === "fulfilled" ? workflowResult.value : [];
    if (workflowResult.status === "rejected") {
      criticalErrors.push(
        workflowResult.reason instanceof Error
          ? workflowResult.reason.message
          : "Unable to load workflows."
      );
    }

    const modelPayload =
      modelResult.status === "fulfilled"
        ? modelResult.value
        : { models: [] as Model[], default_model: undefined as string | undefined };
    if (modelResult.status === "rejected") {
      criticalErrors.push(
        modelResult.reason instanceof Error
          ? modelResult.reason.message
          : "Unable to load models."
      );
    }

    const toolItems = toolResult.status === "fulfilled" ? toolResult.value : [];
    if (toolResult.status === "rejected") {
      nonCriticalErrors.push(
        toolResult.reason instanceof Error
          ? toolResult.reason.message
          : "Unable to load tools."
      );
    }

    const visionPayload =
      visionResult.status === "fulfilled"
        ? visionResult.value
        : { models: [] as Model[], default_model: undefined as string | undefined };
    if (visionResult.status === "rejected") {
      nonCriticalErrors.push(
        visionResult.reason instanceof Error
          ? visionResult.reason.message
          : "Unable to load vision models."
      );
    }

    setWorkflows(workflowItems);
    setModels(modelPayload.models);
    setVisionModels(visionPayload.models);
    setTools(toolItems);
    setVisionModelsFallback(
      visionResult.status !== "fulfilled" || visionPayload.models.length === 0
    );

    const workflowIds = new Set(workflowItems.map((item) => item.id));
    const modelIds = new Set(modelPayload.models.map((item) => item.id));
    const visionModelIds = new Set(visionPayload.models.map((item) => item.id));
    const toolIds = new Set(toolItems.map((item) => item.id));

    const preferredWorkflowId = workflowItems.find((item) => item.id === "simple.v1")?.id;
    setSelectedWorkflowId((prev) => {
      if (workflowIds.size === 0) {
        return "";
      }
      if (workflowIds.has(prev)) {
        return prev;
      }
      return preferredWorkflowId ?? workflowItems[0]?.id ?? "";
    });

    setSelectedModelId((prev) => {
      if (modelIds.size === 0) {
        return "";
      }
      if (modelIds.has(prev)) {
        return prev;
      }
      if (modelPayload.default_model && modelIds.has(modelPayload.default_model)) {
        return modelPayload.default_model;
      }
      return modelPayload.models[0]?.id ?? "";
    });

    setSelectedVisionModelId((prev) => {
      if (visionModelIds.size === 0) {
        return prev || visionPayload.default_model || "";
      }
      if (visionModelIds.has(prev)) {
        return prev;
      }
      if (
        visionPayload.default_model &&
        visionModelIds.has(visionPayload.default_model)
      ) {
        return visionPayload.default_model;
      }
      return visionPayload.models[0]?.id ?? "";
    });

    setSelectedToolIds((prev) => {
      if (toolIds.size === 0) {
        return [];
      }
      const selectedFromStorage = prev.filter((id) => toolIds.has(id));
      if (selectedFromStorage.length > 0) {
        return selectedFromStorage;
      }
      return toolItems
        .filter((tool) => tool.enabled && tool.id !== VISION_TOOL_ID)
        .map((tool) => tool.id);
    });

    setOptionsError(criticalErrors.length > 0 ? criticalErrors.join(" ") : null);
    setToolsWarning(nonCriticalErrors.length > 0 ? nonCriticalErrors.join(" ") : null);
    setOptionsLoading(false);
  }, []);

  useEffect(() => {
    void loadOptions();
  }, [loadOptions]);

  const handleScroll = useCallback(() => {
    const container = listRef.current;
    if (!container) {
      return;
    }
    const threshold = 48;
    const atBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold;
    setAutoScroll(atBottom);
  }, []);

  const toggleTool = useCallback((toolId: string) => {
    setSelectedToolIds((prev) => {
      if (prev.includes(toolId)) {
        return prev.filter((id) => id !== toolId);
      }
      return [...prev, toolId];
    });
  }, []);

  const uploadFiles = useCallback(async (files: File[]) => {
    const imageFiles = toImageFiles(files);
    if (imageFiles.length === 0) {
      return;
    }

    setAttachmentError(null);
    setIsUploadingArtifacts(true);

    const uploaded: ChatAttachment[] = [];
    const failures: string[] = [];

    for (const file of imageFiles) {
      try {
        const payload = await uploadArtifact(file);
        uploaded.push({
          artifactId: payload.artifact_id,
          name: file.name || payload.artifact_id,
          mime: payload.mime,
          size: payload.size,
          sha256: payload.sha256,
          previewUrl: artifactPreviewUrl(payload.artifact_id),
        });
      } catch (error) {
        failures.push(
          error instanceof Error ? error.message : `Failed to upload ${file.name}`
        );
      }
    }

    if (uploaded.length > 0) {
      setAttachments((prev) => [...prev, ...uploaded]);
    }
    if (failures.length > 0) {
      setAttachmentError(failures.join(" "));
    }
    setIsUploadingArtifacts(false);
  }, []);

  const removeAttachment = useCallback((artifactId: string) => {
    setAttachments((prev) => prev.filter((item) => item.artifactId !== artifactId));
  }, []);

  const handlePickFiles = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const imageFiles = toImageFiles(event.target.files);
      if (imageFiles.length > 0) {
        void uploadFiles(imageFiles);
      }
      event.target.value = "";
    },
    [uploadFiles]
  );

  const handlePaste = useCallback(
    (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const files: File[] = [];
      for (const item of Array.from(event.clipboardData.items)) {
        if (item.kind !== "file") {
          continue;
        }
        const file = item.getAsFile();
        if (file && file.type.startsWith("image/")) {
          files.push(file);
        }
      }
      if (files.length > 0) {
        event.preventDefault();
        void uploadFiles(files);
      }
    },
    [uploadFiles]
  );

  const handleDragEnter = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasFilePayload(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current += 1;
    setDragActive(true);
  }, []);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasFilePayload(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasFilePayload(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLElement>) => {
      if (!hasFilePayload(event.dataTransfer)) {
        return;
      }
      event.preventDefault();
      dragDepthRef.current = 0;
      setDragActive(false);

      const files = toImageFiles(event.dataTransfer?.files ?? null);
      if (files.length === 0) {
        return;
      }
      void uploadFiles(files);
    },
    [uploadFiles]
  );

  const handleCreateThread = useCallback(() => {
    createThread();
    refreshChatState();
    setInput("");
    setAttachments([]);
    setAttachmentError(null);
    setEditingThreadId(null);
    setEditingThreadTitle("");
    setAutoScroll(true);
  }, [refreshChatState]);

  const handleSelectThread = useCallback(
    (threadId: string) => {
      if (!threadId || threadId === chatState.activeThreadId) {
        return;
      }
      setActiveThread(threadId);
      refreshChatState();
      setAutoScroll(true);
      setEditingThreadId(null);
      setEditingThreadTitle("");
    },
    [chatState.activeThreadId, refreshChatState]
  );

  const startRenameThread = useCallback((thread: ChatThread) => {
    setEditingThreadId(thread.id);
    setEditingThreadTitle(thread.title);
  }, []);

  const cancelRenameThread = useCallback(() => {
    setEditingThreadId(null);
    setEditingThreadTitle("");
  }, []);

  const commitRenameThread = useCallback(
    (threadId: string) => {
      const nextTitle = editingThreadTitle.trim() || NEW_CHAT_TITLE;
      renameThread(threadId, nextTitle);
      refreshChatState();
      setEditingThreadId(null);
      setEditingThreadTitle("");
    },
    [editingThreadTitle, refreshChatState]
  );

  const confirmDeleteThread = useCallback(() => {
    if (!threadPendingDelete) {
      return;
    }
    const deletingThreadId = threadPendingDelete.id;
    const deletingActive = deletingThreadId === chatState.activeThreadId;

    deleteThread(deletingThreadId);

    let nextState = loadState();
    if (nextState.threads.length === 0) {
      createThread();
      nextState = loadState();
    }

    setChatState(nextState);
    setThreadPendingDelete(null);
    setEditingThreadId(null);
    setEditingThreadTitle("");

    if (deletingActive) {
      setInput("");
      setAttachments([]);
      setAttachmentError(null);
      setAutoScroll(true);
    }
  }, [threadPendingDelete, chatState.activeThreadId]);

  const handleSend = useCallback(async () => {
    const trimmed = input.trim();
    const attachmentIds = attachments.map((item) => item.artifactId);
    const hasAttachments = attachmentIds.length > 0;
    const messageToSend =
      trimmed || (hasAttachments ? "Analyze the attached image(s)." : "");
    const activeThreadId = chatState.activeThreadId;

    if (
      !messageToSend ||
      !activeThreadId ||
      isSending ||
      isUploadingArtifacts ||
      !backendReady
    ) {
      return;
    }

    const threadMessages = chatState.messagesByThread[activeThreadId] ?? [];
    const shouldUpdateTitle =
      (activeThread?.title ?? NEW_CHAT_TITLE) === NEW_CHAT_TITLE &&
      threadMessages.every((message) => message.role !== "user");
    const selectedVision = selectedVisionModelId.trim();
    const userContent = hasAttachments
      ? `${messageToSend}\n\n[Attached ${attachmentIds.length} image(s)]`
      : messageToSend;
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: userContent,
      createdAt: Date.now(),
      workflowId: selectedWorkflowId || undefined,
      modelId: selectedModelId || undefined,
      toolIds: selectedToolIds,
      artifactIds: attachmentIds,
    };

    appendMessage(activeThreadId, userMessage);
    if (shouldUpdateTitle) {
      renameThread(activeThreadId, deriveThreadTitle(messageToSend));
    }
    refreshChatState();

    setAutoScroll(true);
    setInput("");
    setAttachments([]);
    setAttachmentError(null);
    setIsSending(true);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    try {
      const result = await chatRun({
        message: messageToSend,
        thread_id: activeThreadId,
        workflow_id: selectedWorkflowId,
        model_id: selectedModelId,
        tool_ids: selectedToolIds,
        vision_model_id: selectedVision || undefined,
        artifact_ids: attachmentIds,
      });
      appendMessage(activeThreadId, {
        id: createId(),
        role: "assistant",
        content: result.output_text || "",
        createdAt: Date.now(),
        runId: result.run_id,
        workflowId: result.workflow_id || selectedWorkflowId || undefined,
        modelId: result.model_id || selectedModelId || undefined,
        toolIds: Array.isArray(result.tool_ids) ? result.tool_ids : selectedToolIds,
        artifactIds: attachmentIds,
      });
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : "Failed to reach backend";
      appendMessage(activeThreadId, {
        id: createId(),
        role: "assistant",
        content: `Error: ${detail}`,
        createdAt: Date.now(),
        workflowId: selectedWorkflowId || undefined,
        modelId: selectedModelId || undefined,
        toolIds: selectedToolIds,
        artifactIds: attachmentIds,
      });
    } finally {
      refreshChatState();
      setIsSending(false);
    }
  }, [
    input,
    attachments,
    chatState.activeThreadId,
    chatState.messagesByThread,
    activeThread,
    isSending,
    isUploadingArtifacts,
    backendReady,
    selectedWorkflowId,
    selectedModelId,
    selectedVisionModelId,
    selectedToolIds,
    refreshChatState,
  ]);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void handleSend();
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const openInspect = useCallback((runId?: string) => {
    if (!runId || !runId.trim()) {
      return;
    }
    if (onInspectRun) {
      onInspectRun(runId);
      return;
    }
    if (typeof window !== "undefined") {
      window.location.hash = `#/inspect/${encodeURIComponent(runId)}`;
    }
  }, [onInspectRun]);

  const selectedToolCount = selectedToolIds.length;
  const hasAttachments = attachments.length > 0;
  const visionToolSelected = selectedToolIds.includes(VISION_TOOL_ID);
  const visionToolEnabled = tools.some(
    (tool) => tool.id === VISION_TOOL_ID && tool.enabled
  );
  const visionSelectorEnabled = hasAttachments || visionToolSelected || visionToolEnabled;
  const canSend =
    backendReady &&
    chatState.activeThreadId.length > 0 &&
    !isSending &&
    !isUploadingArtifacts &&
    (input.trim().length > 0 || hasAttachments) &&
    selectedModelId.length > 0 &&
    (!hasAttachments || selectedVisionModelId.trim().length > 0);
  const availabilityHint =
    optionsError ??
    (!optionsLoading && workflows.length === 0
      ? "No workflows are available from the backend."
      : !optionsLoading && models.length === 0
      ? "No models detected. Install or start an Ollama model, then refresh."
      : null);

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 flex-col">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Chat</h1>
            <p className="section-framer text-secondary">
              Select model, workflow, and tools for each run.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge
              className={
                "border " +
                (backendReady
                  ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
                  : "border-rose-400/40 bg-rose-500/10 text-rose-100")
              }
            >
              Backend {backendReady ? "Connected" : "Unavailable"}
            </Badge>
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={() => void loadOptions()}
            >
              Refresh
            </Button>
          </div>
        </div>

        <div className="mt-4 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          {availabilityHint ? (
            <div className="mb-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
              {availabilityHint}
            </div>
          ) : null}
          {toolsWarning ? (
            <div className="mb-4 rounded-xl border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
              {toolsWarning}
            </div>
          ) : null}

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)]">
            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                Workflow
              </label>
              <select
                value={selectedWorkflowId}
                onChange={(event) => setSelectedWorkflowId(event.target.value)}
                className="h-9 w-full rounded-md border border-subtle bg-[#0b0b10] px-3 text-sm text-primary outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40"
                disabled={optionsLoading || workflows.length === 0}
              >
                {workflows.length === 0 ? (
                  <option value="">No workflows</option>
                ) : (
                  workflows.map((workflow) => (
                    <option key={workflow.id} value={workflow.id}>
                      {workflow.title}
                    </option>
                  ))
                )}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                Model
              </label>
              <select
                value={selectedModelId}
                onChange={(event) => setSelectedModelId(event.target.value)}
                className="h-9 w-full rounded-md border border-subtle bg-[#0b0b10] px-3 text-sm text-primary outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40"
                disabled={optionsLoading || models.length === 0}
              >
                {models.length === 0 ? (
                  <option value="">No models</option>
                ) : (
                  models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))
                )}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                Vision Model
              </label>
              {visionModelsFallback ? (
                <input
                  value={selectedVisionModelId}
                  onChange={(event) => setSelectedVisionModelId(event.target.value)}
                  placeholder="e.g. llava:latest"
                  className="h-9 w-full rounded-md border border-subtle bg-[#0b0b10] px-3 text-sm text-primary outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40 disabled:opacity-60"
                  disabled={optionsLoading || !visionSelectorEnabled}
                />
              ) : (
                <select
                  value={selectedVisionModelId}
                  onChange={(event) => setSelectedVisionModelId(event.target.value)}
                  className="h-9 w-full rounded-md border border-subtle bg-[#0b0b10] px-3 text-sm text-primary outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40 disabled:opacity-60"
                  disabled={
                    optionsLoading || visionModels.length === 0 || !visionSelectorEnabled
                  }
                >
                  {visionModels.length === 0 ? (
                    <option value="">No vision models</option>
                  ) : (
                    visionModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name}
                      </option>
                    ))
                  )}
                </select>
              )}
            </div>

            <div ref={toolsMenuRef} className="relative">
              <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                Tools
              </label>
              <Button
                type="button"
                className="h-9 w-full justify-between rounded-md border border-subtle bg-[#0b0b10] px-3 text-sm text-primary hover:text-primary"
                onClick={() => setToolsMenuOpen((prev) => !prev)}
                disabled={optionsLoading}
              >
                <span>
                  {selectedToolCount > 0
                    ? `${selectedToolCount} selected`
                    : "No tools selected"}
                </span>
                <span className="text-secondary">▾</span>
              </Button>

              {toolsMenuOpen ? (
                <div className="absolute z-20 mt-2 w-full rounded-xl border border-subtle bg-[#0b0b10] p-3 shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <Button
                      type="button"
                      className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                      onClick={() =>
                        setSelectedToolIds(
                          tools
                            .filter((tool) => tool.enabled && tool.id !== VISION_TOOL_ID)
                            .map((tool) => tool.id)
                        )
                      }
                    >
                      Enabled
                    </Button>
                    <Button
                      type="button"
                      className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                      onClick={() => setSelectedToolIds([])}
                    >
                      None
                    </Button>
                  </div>

                  <div className="max-h-44 space-y-2 overflow-y-auto pr-1">
                    {tools.length === 0 ? (
                      <p className="text-xs text-secondary">No tools available.</p>
                    ) : (
                      tools.map((tool) => {
                        const checked = selectedToolIds.includes(tool.id);
                        return (
                          <label
                            key={tool.id}
                            className="flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-1.5 hover:bg-white/5"
                          >
                            <div className="min-w-0">
                              <p className="truncate text-sm text-primary">{tool.name}</p>
                              {tool.description ? (
                                <p className="truncate text-xs text-secondary">
                                  {tool.description}
                                </p>
                              ) : null}
                            </div>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleTool(tool.id)}
                              className="h-4 w-4 rounded border-subtle bg-transparent"
                            />
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 gap-4 px-4">
        <aside className="w-64 shrink-0 min-h-0 pb-6">
          <div className="flex h-full flex-col rounded-2xl border border-subtle bg-[#0b0b10] p-3 shadow-[0_10px_30px_rgba(0,0,0,0.3)]">
            <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-secondary">
              Conversations
            </p>
            <p className="text-[11px] text-secondary">
              {sortedThreads.length} total
            </p>
          </div>
          <Button
            type="button"
            className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
            onClick={handleCreateThread}
          >
            New Chat
          </Button>
        </div>

            <div className="flex-1 overflow-y-auto pr-1">
              <div className="space-y-2">
                {sortedThreads.map((thread) => {
              const isActive = thread.id === chatState.activeThreadId;
              const isEditing = editingThreadId === thread.id;

              return (
                <div
                  key={thread.id}
                  className={
                    "group flex items-center gap-2 rounded-full border px-2 py-1.5 transition " +
                    (isActive
                      ? "border-[#6d28d9]/50 bg-[#171721] shadow-[0_0_0_1px_rgba(109,40,217,0.25)]"
                      : "border-subtle bg-[#11111a] hover:border-[#2a2638]")
                  }
                >
                  {isEditing ? (
                    <Input
                      value={editingThreadTitle}
                      autoFocus
                      onChange={(event) => setEditingThreadTitle(event.target.value)}
                      onBlur={() => commitRenameThread(thread.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          commitRenameThread(thread.id);
                        } else if (event.key === "Escape") {
                          event.preventDefault();
                          cancelRenameThread();
                        }
                      }}
                      className="h-8 flex-1 rounded-full border-subtle bg-transparent text-sm"
                    />
                  ) : (
                    <button
                      type="button"
                      className="min-w-0 flex-1 rounded-full px-2 py-1 text-left"
                      onClick={() => handleSelectThread(thread.id)}
                    >
                      <p className="truncate text-sm text-primary">{thread.title}</p>
                      <p className="mt-0.5 text-[11px] text-secondary">
                        {formatThreadTime(thread.updatedAt)}
                      </p>
                    </button>
                  )}

                  {!isEditing ? (
                    <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                      <Button
                        type="button"
                        className="h-6 rounded-full border border-subtle bg-transparent px-2.5 text-[10px] text-secondary hover:text-primary"
                        onClick={() => startRenameThread(thread)}
                      >
                        Rename
                      </Button>
                      <Button
                        type="button"
                        className="h-6 rounded-full border border-rose-400/40 bg-transparent px-2.5 text-[10px] text-rose-200 hover:text-rose-100"
                        onClick={() => setThreadPendingDelete(thread)}
                      >
                        Delete
                      </Button>
                    </div>
                  ) : null}
                </div>
              );
                })}
              </div>
            </div>
          </div>
        </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-hidden px-4">
          <div
            ref={listRef}
            onScroll={handleScroll}
            className="h-full overflow-y-auto pr-2 [scrollbar-color:#242438_#0b0b10] [scrollbar-width:thin] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-[#0b0b10] [&::-webkit-scrollbar-thumb]:bg-[#242438] [&::-webkit-scrollbar-thumb:hover]:bg-[#242438]"
          >
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 pb-6">
              {activeMessages.length === 0 ? (
                <div className="rounded-3xl border border-subtle bg-[#0b0b10] px-6 py-5 text-sm text-secondary">
                  Start a new conversation. Runs are executed through the backend
                  with your selected model, workflow, and tools.
                </div>
              ) : (
                activeMessages.map((message) => {
                  const isUser = message.role === "user";
                  const isError =
                    message.role === "assistant" &&
                    message.content.trim().toLowerCase().startsWith("error:");

                  return (
                    <div
                      key={message.id}
                      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      <div className="max-w-[75%]">
                        <div
                          className={
                            `rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm ` +
                            (isUser
                              ? "border-[#2a2638] bg-[#171721]"
                              : isError
                              ? "border-rose-400/40 bg-rose-500/10 text-rose-100"
                              : "border-subtle bg-[#0b0b10]")
                          }
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>
                        </div>

                        {!isUser &&
                        (message.runId ||
                          message.workflowId ||
                          message.modelId ||
                          (message.toolIds?.length ?? 0) > 0) ? (
                          <div className="mt-1 flex items-center justify-between gap-3 text-[11px] text-secondary">
                            <span className="truncate">
                              {message.runId ? `Run ID: ${message.runId}` : "Assistant reply"}
                              {message.workflowId ? ` · ${message.workflowId}` : ""}
                              {message.modelId ? ` · ${message.modelId}` : ""}
                              {message.toolIds && message.toolIds.length > 0
                                ? ` · ${message.toolIds.length} tools`
                                : ""}
                            </span>
                            {message.runId ? (
                              <Button
                                type="button"
                                className="h-6 rounded-full border border-subtle bg-transparent px-2.5 text-[11px] text-secondary hover:text-primary"
                                onClick={() => openInspect(message.runId)}
                              >
                                Inspect
                              </Button>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  );
                })
              )}

              {isSending ? (
                <div className="flex justify-start">
                  <div className="max-w-[75%] rounded-2xl border border-subtle bg-[#0b0b10] px-4 py-3 text-sm text-secondary">
                    Thinking...
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <form
          onSubmit={handleSubmit}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className="px-4 pb-6 pt-3"
        >
          <div className="relative mx-auto max-w-3xl">
            <DropzoneOverlay visible={dragActive} />
            <div className="rounded-2xl border border-subtle bg-[#0b0b10]/90 px-3 py-2 shadow-[0_10px_30px_rgba(0,0,0,0.35)] ring-1 ring-white/10 transition focus-within:ring-2 focus-within:ring-[#6d28d9]/40">
              <AttachmentsBar attachments={attachments} onRemove={removeAttachment} />
              {attachmentError ? (
                <p className="mb-2 rounded-md border border-rose-400/40 bg-rose-500/10 px-2 py-1 text-xs text-rose-100">
                  {attachmentError}
                </p>
              ) : null}
              {isUploadingArtifacts ? (
                <p className="mb-2 text-xs text-secondary">Uploading image(s)...</p>
              ) : null}
              <div className="flex items-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={handlePickFiles}
                />
                <Button
                  type="button"
                  className="h-9 rounded-full border border-subtle bg-transparent px-3 text-sm text-secondary hover:text-primary"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!backendReady || isUploadingArtifacts || isSending}
                >
                  Attach
                </Button>
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  onInput={(event) => resizeTextarea(event.currentTarget)}
                  placeholder={backendReady ? "Ask anything" : "Connect backend to chat"}
                  rows={1}
                  className="min-h-9 max-h-40 flex-1 resize-none border-none bg-transparent px-0 py-2 text-sm leading-5 text-primary placeholder:text-secondary shadow-none outline-none focus-visible:ring-0"
                  disabled={!backendReady || !chatState.activeThreadId}
                />
                <Button
                  type="submit"
                  className="h-9 rounded-full bg-gold px-4 text-black hover:bg-[#e1c161] disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={!canSend}
                >
                  {backendReady
                    ? isSending
                      ? "Sending..."
                      : isUploadingArtifacts
                      ? "Uploading..."
                      : "Send"
                    : "Connect backend"}
                </Button>
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-secondary">
              <span>
                Enter to send · Shift+Enter for a new line · Paste or drop screenshots
              </span>
              {!backendReady && !optionsLoading ? (
                <span>{availabilityHint ?? "Chat is temporarily unavailable."}</span>
              ) : hasAttachments && !selectedVisionModelId.trim() ? (
                <span>Select a vision model to send attached images.</span>
              ) : null}
            </div>
          </div>
        </form>
      </div>
      </div>

      {threadPendingDelete ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
          <div className="w-full max-w-md rounded-2xl border border-subtle bg-[#0b0b10] p-5 shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
            <h2 className="text-base text-primary">Delete conversation?</h2>
            <p className="mt-2 text-sm text-secondary">
              This will permanently remove "{threadPendingDelete.title}" and all
              messages in it.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button
                type="button"
                className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
                onClick={() => setThreadPendingDelete(null)}
              >
                Cancel
              </Button>
              <Button
                type="button"
                className="h-9 rounded-full border border-rose-400/40 bg-rose-500/10 px-4 text-sm text-rose-100 hover:bg-rose-500/20"
                onClick={confirmDeleteThread}
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function resizeTextarea(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  const nextHeight = Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY =
    textarea.scrollHeight > MAX_TEXTAREA_HEIGHT ? "auto" : "hidden";
}
