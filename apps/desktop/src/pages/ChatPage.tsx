import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AttachmentsBar, {
  type ChatAttachment,
} from "../components/chat/AttachmentsBar";
import DropzoneOverlay from "../components/chat/DropzoneOverlay";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
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

const STORAGE_KEY = "saturday.chat.v2";
const MAX_TEXTAREA_HEIGHT = 160;
const VISION_TOOL_ID = "vision.analyze";

type ChatRole = "user" | "assistant";

type ChatBubble = {
  role: ChatRole;
  content: string;
  runId?: string;
  stepCount?: number;
  error?: boolean;
};

type StoredChatState = {
  messages: ChatBubble[];
  workflowId?: string;
  modelId?: string;
  visionModelId?: string;
  toolIds?: string[];
  threadId?: string;
};

const isChatBubble = (value: unknown): value is ChatBubble => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as {
    role?: unknown;
    content?: unknown;
    runId?: unknown;
    stepCount?: unknown;
    error?: unknown;
  };
  if (record.role !== "user" && record.role !== "assistant") {
    return false;
  }
  if (typeof record.content !== "string") {
    return false;
  }
  if (record.runId !== undefined && typeof record.runId !== "string") {
    return false;
  }
  if (record.stepCount !== undefined && typeof record.stepCount !== "number") {
    return false;
  }
  if (record.error !== undefined && typeof record.error !== "boolean") {
    return false;
  }
  return true;
};

const loadStoredState = (): StoredChatState => {
  if (typeof window === "undefined") {
    return { messages: [] };
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { messages: [] };
    }
    const parsed = JSON.parse(raw) as StoredChatState;
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.filter(isChatBubble)
      : [];
    const toolIds = Array.isArray(parsed.toolIds)
      ? parsed.toolIds.filter((item): item is string => typeof item === "string")
      : [];

    return {
      messages,
      workflowId:
        typeof parsed.workflowId === "string" ? parsed.workflowId : undefined,
      modelId: typeof parsed.modelId === "string" ? parsed.modelId : undefined,
      visionModelId:
        typeof parsed.visionModelId === "string"
          ? parsed.visionModelId
          : undefined,
      toolIds,
      threadId: typeof parsed.threadId === "string" ? parsed.threadId : undefined,
    };
  } catch {
    return { messages: [] };
  }
};

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

type ChatPageProps = {
  onInspectRun?: (runId: string) => void;
};

export default function ChatPage({ onInspectRun }: ChatPageProps) {
  const stored = useMemo(() => loadStoredState(), []);

  const [messages, setMessages] = useState<ChatBubble[]>(stored.messages);
  const [input, setInput] = useState<string>("");
  const [isSending, setIsSending] = useState<boolean>(false);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [visionModels, setVisionModels] = useState<Model[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);

  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>(
    stored.workflowId ?? ""
  );
  const [selectedModelId, setSelectedModelId] = useState<string>(
    stored.modelId ?? ""
  );
  const [selectedVisionModelId, setSelectedVisionModelId] = useState<string>(
    stored.visionModelId ?? ""
  );
  const [selectedToolIds, setSelectedToolIds] = useState<string[]>(
    stored.toolIds ?? []
  );
  const [threadId] = useState<string | undefined>(stored.threadId);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [isUploadingArtifacts, setIsUploadingArtifacts] = useState<boolean>(false);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [visionModelsFallback, setVisionModelsFallback] = useState<boolean>(false);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);

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

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const payload: StoredChatState = {
      messages,
      workflowId: selectedWorkflowId || undefined,
      modelId: selectedModelId || undefined,
      visionModelId: selectedVisionModelId || undefined,
      toolIds: selectedToolIds,
      threadId,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [
    messages,
    selectedWorkflowId,
    selectedModelId,
    selectedVisionModelId,
    selectedToolIds,
    threadId,
  ]);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const container = listRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages, autoScroll]);

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

  const handleSend = useCallback(async () => {
    const trimmed = input.trim();
    const attachmentIds = attachments.map((item) => item.artifactId);
    const hasAttachments = attachmentIds.length > 0;
    const messageToSend =
      trimmed || (hasAttachments ? "Analyze the attached image(s)." : "");

    if (!messageToSend || isSending || isUploadingArtifacts || !backendReady) {
      return;
    }

    const userMessage: ChatBubble = {
      role: "user",
      content: hasAttachments
        ? `${messageToSend}\n\n[Attached ${attachmentIds.length} image(s)]`
        : messageToSend,
    };
    const thinkingMessage: ChatBubble = {
      role: "assistant",
      content: "Thinking...",
    };

    const placeholderIndex = messages.length + 1;
    const selectedVision = selectedVisionModelId.trim();

    setAutoScroll(true);
    setInput("");
    setAttachments([]);
    setAttachmentError(null);
    setIsSending(true);
    setMessages((prev) => [...prev, userMessage, thinkingMessage]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    try {
      const result = await chatRun({
        message: messageToSend,
        thread_id: threadId,
        workflow_id: selectedWorkflowId,
        model_id: selectedModelId,
        tool_ids: selectedToolIds,
        vision_model_id: selectedVision || undefined,
        artifact_ids: attachmentIds,
      });

      setMessages((prev) => {
        const next = [...prev];
        next[placeholderIndex] = {
          role: "assistant",
          content: result.output_text || "",
          runId: result.run_id,
          stepCount: Array.isArray(result.steps) ? result.steps.length : 0,
        };
        return next;
      });
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : "Failed to reach backend";
      setMessages((prev) => {
        const next = [...prev];
        next[placeholderIndex] = {
          role: "assistant",
          content: `Error: ${detail}`,
          error: true,
        };
        return next;
      });
    } finally {
      setIsSending(false);
    }
  }, [
    input,
    attachments,
    isSending,
    isUploadingArtifacts,
    backendReady,
    messages.length,
    threadId,
    selectedWorkflowId,
    selectedModelId,
    selectedVisionModelId,
    selectedToolIds,
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
    <div className="bg-panel section-base pr-64 relative flex h-screen flex-col">
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

      <div className="flex-1 overflow-hidden px-4 pb-32">
        <div
          ref={listRef}
          onScroll={handleScroll}
          className="h-full overflow-y-auto pr-2"
        >
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 pb-6">
            {messages.length === 0 ? (
              <div className="rounded-3xl border border-subtle bg-[#0b0b10] px-6 py-5 text-sm text-secondary">
                Start a new conversation. Runs are executed through the backend
                with your selected model, workflow, and tools.
              </div>
            ) : (
              messages.map((message, index) => {
                const isUser = message.role === "user";

                return (
                  <div
                    key={`${message.role}-${index}`}
                    className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    <div className="max-w-[75%]">
                      <div
                        className={
                          `rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm ` +
                          (isUser
                            ? "border-[#2a2638] bg-[#171721]"
                            : message.error
                            ? "border-rose-400/40 bg-rose-500/10 text-rose-100"
                            : "border-subtle bg-[#0b0b10]")
                        }
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      </div>

                      {!isUser && message.runId ? (
                        <div className="mt-1 flex items-center justify-between gap-3 text-[11px] text-secondary">
                          <span className="truncate">
                            Run ID: {message.runId}
                            {typeof message.stepCount === "number"
                              ? ` · ${message.stepCount} steps`
                              : ""}
                          </span>
                          <Button
                            type="button"
                            className="h-6 rounded-full border border-subtle bg-transparent px-2.5 text-[11px] text-secondary hover:text-primary"
                            onClick={() => openInspect(message.runId)}
                          >
                            Inspect
                          </Button>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className="fixed bottom-0 left-[12.5rem] right-64 pb-6"
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
                disabled={!backendReady}
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
  );
}

function resizeTextarea(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  const nextHeight = Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY =
    textarea.scrollHeight > MAX_TEXTAREA_HEIGHT ? "auto" : "hidden";
}
