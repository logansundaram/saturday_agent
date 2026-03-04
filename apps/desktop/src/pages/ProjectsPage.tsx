import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { NodeType, ValidationDiagnostic, WorkflowSpec } from "@saturday/shared/workflow";
import WorkflowGraphCanvas from "../components/builder/WorkflowGraphCanvas";
import {
  createInitialWorkflowSpec,
  defaultNodeForType,
  getNextNodeId,
  normalizeDraftSpec,
  validateDraftLocally,
} from "../components/builder/workflowDraft";
import AttachmentsBar, { type ChatAttachment } from "../components/chat/AttachmentsBar";
import StepsTimeline from "../components/chat/StepsTimeline";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import {
  API_BASE_URL,
  compileWorkflowSpec,
  createProject,
  createProjectChat,
  deleteProject,
  deleteProjectDocument,
  getModels,
  getProjects,
  getProject,
  getVisionModels,
  replaceProjectTools,
  runProjectStream,
  saveProjectWorkflow,
  updateProjectGroundTruth,
  uploadArtifact,
  uploadProjectDocument,
  type ChatRunStreamEvent,
  type ChatRunTimelineStep,
  type Model,
  type ProjectDetail,
  type ProjectDocument,
  type ProjectToolBinding,
  type ProjectWorkflow,
  type ProjectSummary,
  type Workflow,
  getWorkflows,
} from "../lib/api";
import {
  appendProjectChatMessage,
  deleteProjectChatMessages,
  getProjectChatMessages,
  type ProjectChatMessage,
} from "../lib/projectChatStore";
import {
  loadChatRunMetaMap,
  setChatRunMeta,
  type ChatRunMetaMap,
} from "../lib/chatRunMetaStore";

type ProjectsPageProps = {
  onInspectRun?: (runId: string) => void;
};

type ProjectTab = "chats" | "docs" | "ground_truth" | "workflow" | "tools";

type PendingAssistantMessage = {
  id: string;
  content: string;
  runId?: string;
  steps: ChatRunTimelineStep[];
};

const PROJECT_TABS: ProjectTab[] = [
  "chats",
  "docs",
  "ground_truth",
  "workflow",
  "tools",
];
const WORKFLOW_NODE_TYPES: NodeType[] = [
  "llm",
  "tool",
  "conditional",
  "verify",
  "finalize",
];

function createId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function artifactPreviewUrl(artifactId: string): string {
  return `${API_BASE_URL}/artifacts/${encodeURIComponent(artifactId)}`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return "Never";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function statusBadgeClass(status: string): string {
  const normalized = status.trim().toLowerCase();
  if (normalized === "ok" || normalized === "success" || normalized === "indexed") {
    return "border-emerald-400/40 bg-emerald-500/10 text-emerald-100";
  }
  if (normalized === "running" || normalized === "pending" || normalized === "ingesting") {
    return "border-sky-400/40 bg-sky-500/10 text-sky-100";
  }
  return "border-rose-400/40 bg-rose-500/10 text-rose-100";
}

function normalizeDiagnostics(
  diagnostics: Array<ValidationDiagnostic | Record<string, unknown>>
): ValidationDiagnostic[] {
  return diagnostics
    .filter((item) => item && typeof item === "object")
    .map((item) => item as ValidationDiagnostic);
}

function upsertTimelineStep(
  steps: ChatRunTimelineStep[],
  next: ChatRunTimelineStep
): ChatRunTimelineStep[] {
  const index = steps.findIndex((step) => step.step_index === next.step_index);
  if (index < 0) {
    return [...steps, next].sort((left, right) => left.step_index - right.step_index);
  }
  const merged = [...steps];
  merged[index] = { ...merged[index], ...next };
  return merged.sort((left, right) => left.step_index - right.step_index);
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

function buildWorkflowDraft(workflow: ProjectWorkflow | null | undefined): WorkflowSpec {
  if (workflow && workflow.workflow_spec && typeof workflow.workflow_spec === "object") {
    return normalizeDraftSpec(workflow.workflow_spec as WorkflowSpec);
  }
  const initial = createInitialWorkflowSpec();
  initial.name = "Project Workflow";
  initial.description = "Project-scoped workflow.";
  return normalizeDraftSpec(initial);
}

function normalizeMessagesForRun(messages: ProjectChatMessage[]): Array<Record<string, any>> {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}

function workflowLabel(projectDetail: ProjectDetail | null, selectedWorkflowId: string): string {
  if (projectDetail?.workflow?.valid) {
    return "Project Workflow";
  }
  return selectedWorkflowId || "No workflow";
}

function sortedProjects(projects: ProjectSummary[]): ProjectSummary[] {
  return [...projects].sort((left, right) =>
    String(right.updated_at || "").localeCompare(String(left.updated_at || ""))
  );
}

export default function ProjectsPage({ onInspectRun }: ProjectsPageProps) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [projectDetail, setProjectDetail] = useState<ProjectDetail | null>(null);
  const [activeTab, setActiveTab] = useState<ProjectTab>("chats");
  const [activeChatId, setActiveChatId] = useState<string>("");
  const [messages, setMessages] = useState<ProjectChatMessage[]>([]);
  const [runMetaByMessageId, setRunMetaByMessageId] = useState<ChatRunMetaMap>(() =>
    loadChatRunMetaMap()
  );
  const [pendingAssistant, setPendingAssistant] = useState<PendingAssistantMessage | null>(
    null
  );
  const [input, setInput] = useState<string>("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [isUploadingArtifacts, setIsUploadingArtifacts] = useState<boolean>(false);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [pageError, setPageError] = useState<string>("");
  const [pageMessage, setPageMessage] = useState<string>("");
  const [createProjectName, setCreateProjectName] = useState<string>("");
  const [createProjectDescription, setCreateProjectDescription] = useState<string>("");
  const [isCreatingProject, setIsCreatingProject] = useState<boolean>(false);
  const [isDeletingProject, setIsDeletingProject] = useState<boolean>(false);
  const [createChatName, setCreateChatName] = useState<string>("");
  const [isCreatingChat, setIsCreatingChat] = useState<boolean>(false);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [visionModels, setVisionModels] = useState<Model[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("complex.v1");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [selectedVisionModelId, setSelectedVisionModelId] = useState<string>("");
  const [selectedToolIds, setSelectedToolIds] = useState<string[]>([]);
  const [groundTruthDraft, setGroundTruthDraft] = useState<string>("");
  const [groundTruthSavedValue, setGroundTruthSavedValue] = useState<string>("");
  const [groundTruthStatus, setGroundTruthStatus] = useState<string>("");
  const [groundTruthSaving, setGroundTruthSaving] = useState<boolean>(false);
  const [docUploadError, setDocUploadError] = useState<string>("");
  const [isUploadingDocs, setIsUploadingDocs] = useState<boolean>(false);
  const [deletingDocId, setDeletingDocId] = useState<string>("");
  const [toolBindingsDraft, setToolBindingsDraft] = useState<ProjectToolBinding[]>([]);
  const [isSavingTools, setIsSavingTools] = useState<boolean>(false);
  const [workflowDraft, setWorkflowDraft] = useState<WorkflowSpec>(() =>
    buildWorkflowDraft(null)
  );
  const [workflowJsonText, setWorkflowJsonText] = useState<string>(() =>
    JSON.stringify(buildWorkflowDraft(null), null, 2)
  );
  const [workflowJsonError, setWorkflowJsonError] = useState<string>("");
  const [workflowDiagnostics, setWorkflowDiagnostics] = useState<ValidationDiagnostic[]>([]);
  const [workflowSaveStatus, setWorkflowSaveStatus] = useState<string>("");
  const [workflowSaving, setWorkflowSaving] = useState<boolean>(false);
  const [selectedWorkflowNodeId, setSelectedWorkflowNodeId] = useState<string | null>(null);

  const groundTruthDebounceRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const docInputRef = useRef<HTMLInputElement | null>(null);

  const refreshProjects = useCallback(
    async (nextSelectedProjectId?: string) => {
      const items = await getProjects();
      const ordered = sortedProjects(items);
      setProjects(ordered);
      const targetId =
        nextSelectedProjectId && ordered.some((item) => item.id === nextSelectedProjectId)
          ? nextSelectedProjectId
          : selectedProjectId && ordered.some((item) => item.id === selectedProjectId)
          ? selectedProjectId
          : ordered[0]?.id || "";
      setSelectedProjectId(targetId);
      return targetId;
    },
    [selectedProjectId]
  );

  const loadProjectDetail = useCallback(async (projectId: string) => {
    if (!projectId.trim()) {
      setProjectDetail(null);
      setActiveChatId("");
      setMessages([]);
      return;
    }
    const detail = await getProject(projectId);
    setProjectDetail(detail);
    setToolBindingsDraft(detail.tools || []);
    setGroundTruthDraft(detail.ground_truth?.content || "");
    setGroundTruthSavedValue(detail.ground_truth?.content || "");
    setWorkflowDraft(buildWorkflowDraft(detail.workflow));
    setWorkflowJsonText(JSON.stringify(buildWorkflowDraft(detail.workflow), null, 2));
    setWorkflowDiagnostics(
      normalizeDiagnostics((detail.workflow?.diagnostics as ValidationDiagnostic[]) || [])
    );
    setWorkflowJsonError("");
    if (!activeChatId || !(detail.chats || []).some((chat) => chat.id === activeChatId)) {
      setActiveChatId(detail.chats?.[0]?.id || "");
    }
    return detail;
  }, [activeChatId]);

  useEffect(() => {
    const load = async () => {
      try {
        setPageError("");
        const [workflowPayload, modelPayload, visionPayload] = await Promise.all([
          getWorkflows(),
          getModels(),
          getVisionModels(),
        ]);
        setWorkflows(workflowPayload);
        setModels(modelPayload.models);
        setVisionModels(visionPayload.models);
        const defaultWorkflow =
          workflowPayload.find((item) => item.id === "complex.v1")?.id ||
          workflowPayload[0]?.id ||
          "";
        setSelectedWorkflowId(defaultWorkflow);
        setSelectedModelId(modelPayload.default_model || modelPayload.models[0]?.id || "");
        setSelectedVisionModelId(
          visionPayload.default_model || visionPayload.models[0]?.id || ""
        );
        const targetProjectId = await refreshProjects();
        if (targetProjectId) {
          await loadProjectDetail(targetProjectId);
        }
      } catch (error) {
        setPageError(error instanceof Error ? error.message : "Unable to load Projects.");
      }
    };
    void load();
  }, [loadProjectDetail, refreshProjects]);

  useEffect(() => {
    if (!selectedProjectId) {
      return;
    }
    void loadProjectDetail(selectedProjectId).catch((error) => {
      setPageError(error instanceof Error ? error.message : "Unable to load project.");
    });
  }, [selectedProjectId, loadProjectDetail]);

  useEffect(() => {
    if (!activeChatId) {
      setMessages([]);
      return;
    }
    setMessages(getProjectChatMessages(activeChatId));
  }, [activeChatId]);

  useEffect(() => {
    if (projectDetail?.workflow?.valid) {
      return;
    }
    if (!selectedWorkflowId && workflows.length > 0) {
      setSelectedWorkflowId(
        workflows.find((item) => item.id === "complex.v1")?.id || workflows[0].id
      );
    }
  }, [projectDetail?.workflow?.valid, selectedWorkflowId, workflows]);

  useEffect(() => {
    if (!projectDetail) {
      return;
    }
    const enabledTools = projectDetail.tools
      .filter((tool) => tool.enabled)
      .map((tool) => tool.tool_name);
    setSelectedToolIds(enabledTools);
  }, [projectDetail?.id]);

  useEffect(() => {
    if (!selectedProjectId || groundTruthDraft === groundTruthSavedValue) {
      return;
    }
    setGroundTruthStatus("Saving...");
    if (groundTruthDebounceRef.current !== null) {
      window.clearTimeout(groundTruthDebounceRef.current);
    }
    groundTruthDebounceRef.current = window.setTimeout(() => {
      setGroundTruthSaving(true);
      void updateProjectGroundTruth(selectedProjectId, groundTruthDraft)
        .then((payload) => {
          setGroundTruthSavedValue(payload.content);
          setGroundTruthStatus(`Saved ${formatTimestamp(payload.updated_at)}`);
          setProjectDetail((current) =>
            current
              ? {
                  ...current,
                  ground_truth: payload,
                }
              : current
          );
        })
        .catch((error) => {
          setGroundTruthStatus(error instanceof Error ? error.message : "Unable to save.");
        })
        .finally(() => {
          setGroundTruthSaving(false);
        });
    }, 700);

    return () => {
      if (groundTruthDebounceRef.current !== null) {
        window.clearTimeout(groundTruthDebounceRef.current);
      }
    };
  }, [groundTruthDraft, groundTruthSavedValue, selectedProjectId]);

  const activeChat = useMemo(
    () => projectDetail?.chats?.find((chat) => chat.id === activeChatId) || null,
    [activeChatId, projectDetail?.chats]
  );
  const activeMessages = useMemo(() => {
    if (!pendingAssistant) {
      return messages;
    }
    const pendingMessage: ProjectChatMessage = {
      id: pendingAssistant.id,
      role: "assistant",
      content: pendingAssistant.content,
      createdAt: Date.now(),
      runId: pendingAssistant.runId,
    };
    return [...messages, pendingMessage];
  }, [messages, pendingAssistant]);
  const availableToolIds = useMemo(
    () => (projectDetail?.tools || []).map((tool) => tool.tool_name),
    [projectDetail?.tools]
  );
  const workflowLocalDiagnostics = useMemo(
    () => validateDraftLocally(workflowDraft, availableToolIds),
    [availableToolIds, workflowDraft]
  );

  const handleCreateProject = useCallback(async () => {
    if (!createProjectName.trim() || isCreatingProject) {
      return;
    }
    try {
      setIsCreatingProject(true);
      setPageError("");
      const created = await createProject({
        name: createProjectName.trim(),
        description: createProjectDescription.trim(),
      });
      setCreateProjectName("");
      setCreateProjectDescription("");
      setPageMessage(`Created ${created.name}.`);
      const targetProjectId = await refreshProjects(created.id);
      if (targetProjectId) {
        await loadProjectDetail(targetProjectId);
      }
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Unable to create project.");
    } finally {
      setIsCreatingProject(false);
    }
  }, [
    createProjectDescription,
    createProjectName,
    isCreatingProject,
    loadProjectDetail,
    refreshProjects,
  ]);

  const handleDeleteProject = useCallback(async () => {
    if (!projectDetail || isDeletingProject) {
      return;
    }
    if (!window.confirm(`Delete project "${projectDetail.name}"?`)) {
      return;
    }
    try {
      setIsDeletingProject(true);
      await deleteProject(projectDetail.id);
      for (const chat of projectDetail.chats || []) {
        deleteProjectChatMessages(chat.id);
      }
      setPageMessage(`Deleted ${projectDetail.name}.`);
      setProjectDetail(null);
      setSelectedProjectId("");
      const nextProjectId = await refreshProjects();
      if (nextProjectId) {
        await loadProjectDetail(nextProjectId);
      }
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Unable to delete project.");
    } finally {
      setIsDeletingProject(false);
    }
  }, [isDeletingProject, loadProjectDetail, projectDetail, refreshProjects]);

  const handleCreateChat = useCallback(async () => {
    if (!selectedProjectId || isCreatingChat) {
      return;
    }
    try {
      setIsCreatingChat(true);
      const created = await createProjectChat(selectedProjectId, {
        name: createChatName.trim() || "New Chat",
      });
      setCreateChatName("");
      const detail = await loadProjectDetail(selectedProjectId);
      setActiveChatId(created.id);
      if (detail) {
        setProjectDetail(detail);
      }
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Unable to create chat.");
    } finally {
      setIsCreatingChat(false);
    }
  }, [createChatName, isCreatingChat, loadProjectDetail, selectedProjectId]);

  const handleAttachmentInput = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) {
        return;
      }
      setIsUploadingArtifacts(true);
      setPageError("");
      try {
        const nextAttachments: ChatAttachment[] = [];
        for (const file of Array.from(files).filter((item) =>
          item.type.startsWith("image/")
        )) {
          const uploaded = await uploadArtifact(file);
          nextAttachments.push({
            artifactId: uploaded.artifact_id,
            name: file.name,
            mime: uploaded.mime,
            size: uploaded.size,
            sha256: uploaded.sha256,
            previewUrl: artifactPreviewUrl(uploaded.artifact_id),
          });
        }
        setAttachments((current) => [...current, ...nextAttachments]);
      } catch (error) {
        setPageError(error instanceof Error ? error.message : "Unable to upload attachment.");
      } finally {
        setIsUploadingArtifacts(false);
      }
    },
    []
  );

  const handleSend = useCallback(async () => {
    if (!selectedProjectId || !activeChatId || !projectDetail) {
      return;
    }
    if (!input.trim() || isSending) {
      return;
    }
    const messageId = createId();
    const userMessage: ProjectChatMessage = {
      id: createId(),
      role: "user",
      content: input.trim(),
      createdAt: Date.now(),
      artifactIds: attachments.map((item) => item.artifactId),
    };
    const nextMessages = appendProjectChatMessage(activeChatId, userMessage);
    setMessages(nextMessages);
    setInput("");
    setIsSending(true);
    setPendingAssistant({
      id: messageId,
      content: "",
      steps: [],
    });

    try {
      await runProjectStream(
        selectedProjectId,
        {
          message: userMessage.content,
          chat_id: activeChatId,
          workflow_id: projectDetail.workflow?.valid ? "" : selectedWorkflowId,
          model_id: selectedModelId,
          tool_ids: selectedToolIds,
          messages: normalizeMessagesForRun(nextMessages),
          vision_model_id: selectedVisionModelId || undefined,
          artifact_ids: attachments.map((item) => item.artifactId),
          context: {},
        },
        (event: ChatRunStreamEvent) => {
          if (event.type === "run_start") {
            setPendingAssistant((current) =>
              current
                ? {
                    ...current,
                    runId: event.run_id,
                  }
                : current
            );
            return;
          }
          if (event.type === "token") {
            setPendingAssistant((current) =>
              current
                ? {
                    ...current,
                    content: `${current.content}${event.text}`,
                  }
                : current
            );
            return;
          }
          if (event.type === "step_start") {
            setPendingAssistant((current) =>
              current
                ? {
                    ...current,
                    steps: upsertTimelineStep(current.steps, {
                      step_index: event.step_index,
                      name: event.name,
                      label: event.label,
                      status: "running",
                      started_at: event.started_at,
                    }),
                  }
                : current
            );
            return;
          }
          if (event.type === "step_end") {
            setPendingAssistant((current) =>
              current
                ? {
                    ...current,
                    steps: upsertTimelineStep(current.steps, {
                      step_index: event.step_index,
                      name: event.name,
                      label: String(event.meta?.label || event.name),
                      status: event.status === "ok" ? "ok" : "error",
                      ended_at: event.ended_at,
                      summary: event.summary,
                    }),
                  }
                : current
            );
            return;
          }
          if (event.type === "final") {
            setPendingAssistant((current) => {
              if (!current) {
                return current;
              }
              const assistantMessage: ProjectChatMessage = {
                id: messageId,
                role: "assistant",
                content: event.output_text,
                createdAt: Date.now(),
                runId: current.runId,
                workflowId: projectDetail.workflow?.valid ? "project-workflow" : selectedWorkflowId,
                modelId: selectedModelId,
                toolIds: [...selectedToolIds],
                artifactIds: attachments.map((item) => item.artifactId),
              };
              const persisted = appendProjectChatMessage(activeChatId, assistantMessage);
              setMessages(persisted);
              const completedSteps = current.steps.map((step) => ({
                ...step,
                duration_ms: durationMs(step.started_at, step.ended_at),
              }));
              setRunMetaByMessageId(
                setChatRunMeta(messageId, {
                  runId: current.runId,
                  status: event.status === "ok" ? "ok" : "error",
                  endedAt: event.ended_at,
                  steps: completedSteps,
                  workflowId: assistantMessage.workflowId,
                  modelId: selectedModelId,
                  toolIds: [...selectedToolIds],
                })
              );
              return null;
            });
            setAttachments([]);
            void loadProjectDetail(selectedProjectId).catch(() => undefined);
          }
        }
      );
    } catch (error) {
      const content = error instanceof Error ? error.message : "Project run failed.";
      const assistantMessage: ProjectChatMessage = {
        id: messageId,
        role: "assistant",
        content,
        createdAt: Date.now(),
        workflowId: projectDetail.workflow?.valid ? "project-workflow" : selectedWorkflowId,
        modelId: selectedModelId,
        toolIds: [...selectedToolIds],
      };
      const persisted = appendProjectChatMessage(activeChatId, assistantMessage);
      setMessages(persisted);
      setPageError(content);
      setPendingAssistant(null);
    } finally {
      setIsSending(false);
    }
  }, [
    activeChatId,
    attachments,
    input,
    isSending,
    loadProjectDetail,
    projectDetail,
    selectedModelId,
    selectedProjectId,
    selectedToolIds,
    selectedVisionModelId,
    selectedWorkflowId,
  ]);

  const handleUploadDocuments = useCallback(
    async (files: FileList | null) => {
      if (!selectedProjectId || !files || files.length === 0) {
        return;
      }
      setIsUploadingDocs(true);
      setDocUploadError("");
      try {
        for (const file of Array.from(files)) {
          await uploadProjectDocument(selectedProjectId, file);
        }
        await loadProjectDetail(selectedProjectId);
      } catch (error) {
        setDocUploadError(
          error instanceof Error ? error.message : "Unable to upload project documents."
        );
      } finally {
        setIsUploadingDocs(false);
      }
    },
    [loadProjectDetail, selectedProjectId]
  );

  const handleDeleteDocument = useCallback(
    async (doc: ProjectDocument) => {
      if (!selectedProjectId || deletingDocId) {
        return;
      }
      try {
        setDeletingDocId(doc.id);
        await deleteProjectDocument(selectedProjectId, doc.id);
        await loadProjectDetail(selectedProjectId);
      } catch (error) {
        setDocUploadError(
          error instanceof Error ? error.message : "Unable to delete project document."
        );
      } finally {
        setDeletingDocId("");
      }
    },
    [deletingDocId, loadProjectDetail, selectedProjectId]
  );

  const handleSaveTools = useCallback(async () => {
    if (!selectedProjectId || isSavingTools) {
      return;
    }
    try {
      setIsSavingTools(true);
      const payload = await replaceProjectTools(
        selectedProjectId,
        toolBindingsDraft.map((tool) => ({
          id: tool.project_binding_id || undefined,
          tool_name: tool.tool_name,
          enabled: tool.enabled,
          version: tool.version,
        }))
      );
      setToolBindingsDraft(payload);
      await loadProjectDetail(selectedProjectId);
      setPageMessage("Saved project tools.");
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Unable to save tools.");
    } finally {
      setIsSavingTools(false);
    }
  }, [isSavingTools, loadProjectDetail, selectedProjectId, toolBindingsDraft]);

  const handleValidateWorkflow = useCallback(async () => {
    try {
      const server = await compileWorkflowSpec(workflowDraft);
      setWorkflowDiagnostics([
        ...workflowLocalDiagnostics,
        ...normalizeDiagnostics(server.diagnostics),
      ]);
      setWorkflowSaveStatus("Validated project workflow.");
    } catch (error) {
      setWorkflowSaveStatus(error instanceof Error ? error.message : "Validation failed.");
    }
  }, [workflowDraft, workflowLocalDiagnostics]);

  const handleSaveWorkflow = useCallback(async () => {
    if (!selectedProjectId || workflowSaving || workflowJsonError) {
      return;
    }
    try {
      setWorkflowSaving(true);
      const server = await saveProjectWorkflow(selectedProjectId, workflowDraft);
      setWorkflowDiagnostics([
        ...workflowLocalDiagnostics,
        ...normalizeDiagnostics((server.diagnostics as ValidationDiagnostic[]) || []),
      ]);
      setWorkflowSaveStatus("Saved project workflow.");
      await loadProjectDetail(selectedProjectId);
    } catch (error) {
      setWorkflowSaveStatus(error instanceof Error ? error.message : "Unable to save workflow.");
    } finally {
      setWorkflowSaving(false);
    }
  }, [
    loadProjectDetail,
    selectedProjectId,
    workflowDraft,
    workflowJsonError,
    workflowLocalDiagnostics,
    workflowSaving,
  ]);

  const updateWorkflowDraft = useCallback((nextDraft: WorkflowSpec) => {
    const normalized = normalizeDraftSpec(nextDraft);
    setWorkflowDraft(normalized);
    setWorkflowJsonText(JSON.stringify(normalized, null, 2));
  }, []);

  const addWorkflowNode = useCallback(
    (type: NodeType) => {
      const nextNodeId = getNextNodeId(workflowDraft.nodes, type);
      const nextNode = defaultNodeForType(type, nextNodeId, workflowDraft.nodes.length);
      updateWorkflowDraft({
        ...workflowDraft,
        nodes: [...workflowDraft.nodes, nextNode],
      });
      setSelectedWorkflowNodeId(nextNode.id);
    },
    [updateWorkflowDraft, workflowDraft]
  );

  const renderChatsTab = () => {
    if (!projectDetail) {
      return null;
    }

    return (
      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <p className="text-xs uppercase tracking-wide text-secondary">Chats</p>
            <Badge className="border border-subtle bg-black/20 text-secondary">
              {projectDetail.chats.length}
            </Badge>
          </div>
          <div className="space-y-2">
            {projectDetail.chats.map((chat) => (
              <button
                key={chat.id}
                type="button"
                className={
                  "w-full rounded-xl border px-3 py-2 text-left transition " +
                  (chat.id === activeChatId
                    ? "border-gold/50 bg-gold/10 text-primary"
                    : "border-subtle bg-black/20 text-secondary hover:text-primary")
                }
                onClick={() => setActiveChatId(chat.id)}
              >
                <div className="text-sm font-medium">{chat.name}</div>
                <div className="mt-1 text-[11px] text-secondary">
                  {chat.last_run_at ? formatTimestamp(chat.last_run_at) : "No runs yet"}
                </div>
              </button>
            ))}
          </div>
          <div className="mt-4 space-y-2">
            <Input
              value={createChatName}
              onChange={(event) => setCreateChatName(event.target.value)}
              placeholder="New chat name"
            />
            <Button
              type="button"
              className="w-full rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
              onClick={() => void handleCreateChat()}
              disabled={isCreatingChat}
            >
              {isCreatingChat ? "Creating..." : "Create Chat"}
            </Button>
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-2xl border border-subtle bg-[#0b0b10]">
          <div className="border-b border-subtle px-4 py-3">
            <div className="flex flex-wrap items-center gap-3">
              <Badge className="border border-subtle bg-black/20 text-secondary">
                {workflowLabel(projectDetail, selectedWorkflowId)}
              </Badge>
              <div className="flex items-center gap-2">
                <span className="text-xs text-secondary">Workflow</span>
                <select
                  value={selectedWorkflowId}
                  onChange={(event) => setSelectedWorkflowId(event.target.value)}
                  className="rounded-lg border border-subtle bg-black/20 px-2 py-1 text-sm text-primary"
                  disabled={Boolean(projectDetail.workflow?.valid)}
                >
                  {workflows.map((workflow) => (
                    <option key={workflow.id} value={workflow.id}>
                      {workflow.title}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-secondary">Model</span>
                <select
                  value={selectedModelId}
                  onChange={(event) => setSelectedModelId(event.target.value)}
                  className="rounded-lg border border-subtle bg-black/20 px-2 py-1 text-sm text-primary"
                >
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-secondary">Vision</span>
                <select
                  value={selectedVisionModelId}
                  onChange={(event) => setSelectedVisionModelId(event.target.value)}
                  className="rounded-lg border border-subtle bg-black/20 px-2 py-1 text-sm text-primary"
                >
                  <option value="">Off</option>
                  {visionModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              {(projectDetail.tools || []).map((tool) => (
                <label
                  key={tool.tool_name}
                  className="flex items-center gap-2 rounded-full border border-subtle bg-black/20 px-3 py-1.5 text-xs text-secondary"
                >
                  <input
                    type="checkbox"
                    checked={selectedToolIds.includes(tool.tool_name)}
                    onChange={(event) => {
                      setSelectedToolIds((current) =>
                        event.target.checked
                          ? [...new Set([...current, tool.tool_name])]
                          : current.filter((item) => item !== tool.tool_name)
                      );
                    }}
                    disabled={!tool.enabled}
                  />
                  <span>{tool.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            {!activeChat ? (
              <div className="rounded-xl border border-subtle bg-black/20 px-4 py-6 text-sm text-secondary">
                Create a project chat to start.
              </div>
            ) : activeMessages.length === 0 ? (
              <div className="rounded-xl border border-subtle bg-black/20 px-4 py-6 text-sm text-secondary">
                No messages yet. Send the first project-scoped prompt.
              </div>
            ) : (
              <div className="space-y-4">
                {activeMessages.map((message) => {
                  const runMeta = runMetaByMessageId[message.id];
                  return (
                    <div
                      key={message.id}
                      className={
                        "rounded-2xl border p-4 " +
                        (message.role === "user"
                          ? "border-sky-400/20 bg-sky-500/10"
                          : "border-subtle bg-black/20")
                      }
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <div className="text-xs uppercase tracking-wide text-secondary">
                          {message.role}
                        </div>
                        {message.runId ? (
                          <Button
                            type="button"
                            className="h-7 rounded-full border border-subtle bg-transparent px-3 text-[11px] text-secondary hover:text-primary"
                            onClick={() => onInspectRun?.(message.runId as string)}
                          >
                            Inspect
                          </Button>
                        ) : null}
                      </div>
                      {runMeta?.steps?.length ? <StepsTimeline steps={runMeta.steps} /> : null}
                      <div className="whitespace-pre-wrap text-sm leading-6 text-primary">
                        {message.content}
                      </div>
                    </div>
                  );
                })}
                {pendingAssistant?.steps?.length ? (
                  <StepsTimeline steps={pendingAssistant.steps} />
                ) : null}
              </div>
            )}
          </div>

          <div className="border-t border-subtle px-4 py-4">
            <AttachmentsBar
              attachments={attachments}
              onRemove={(artifactId) =>
                setAttachments((current) =>
                  current.filter((item) => item.artifactId !== artifactId)
                )
              }
            />
            <div className="mb-3 flex gap-2">
              <Button
                type="button"
                className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploadingArtifacts}
              >
                {isUploadingArtifacts ? "Uploading..." : "Attach Images"}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                multiple
                accept="image/*"
                onChange={(event) => void handleAttachmentInput(event.target.files)}
              />
            </div>
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask within this project..."
              className="min-h-[7rem]"
            />
            <div className="mt-3 flex items-center justify-between gap-3">
              <p className="text-xs text-secondary">
                Shared context: project docs, ground truth, and tool overrides.
              </p>
              <Button
                type="button"
                className="rounded-full border border-gold/40 bg-gold/10 text-gold hover:bg-gold/15"
                onClick={() => void handleSend()}
                disabled={isSending || !activeChat}
              >
                {isSending ? "Running..." : "Send"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderProjectContextBoxes = () => {
    if (!projectDetail) {
      return null;
    }
    const documentPreview = projectDetail.documents.slice(0, 3);
    return (
      <div className="mb-4 grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="rounded-2xl border border-subtle bg-black/20 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-primary">Local Docs</h3>
              <p className="text-sm text-secondary">
                Project-scoped files for retrieval inside this workspace.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge className="border border-subtle bg-black/20 text-secondary">
                {projectDetail.document_count} docs
              </Badge>
              <Button
                type="button"
                className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
                onClick={() => docInputRef.current?.click()}
                disabled={isUploadingDocs}
              >
                {isUploadingDocs ? "Uploading..." : "Upload Files"}
              </Button>
              <Button
                type="button"
                className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
                onClick={() => setActiveTab("docs")}
              >
                Open Docs
              </Button>
            </div>
          </div>
          <input
            ref={docInputRef}
            type="file"
            className="hidden"
            multiple
            accept=".pdf,.txt,.md,.json,.csv,.py,.ts,.tsx,.js"
            onChange={(event) => {
              const files = event.target.files;
              void handleUploadDocuments(files);
              event.currentTarget.value = "";
            }}
          />
          {docUploadError ? (
            <div className="mt-3 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
              {docUploadError}
            </div>
          ) : null}
          <div className="mt-3 space-y-2">
            {documentPreview.length === 0 ? (
              <div className="rounded-xl border border-dashed border-subtle px-4 py-5 text-sm text-secondary">
                No local docs yet. Upload files to build this project's private RAG index.
              </div>
            ) : (
              documentPreview.map((doc) => (
                <div
                  key={doc.id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-subtle bg-[#0b0b10] px-3 py-3"
                >
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-primary">{doc.filename}</div>
                    <div className="text-xs text-secondary">
                      {doc.embedding_model} · {doc.chunk_count || 0} chunks
                    </div>
                  </div>
                  <Badge className={`border ${statusBadgeClass(doc.status)}`}>{doc.status}</Badge>
                </div>
              ))
            )}
            {projectDetail.documents.length > documentPreview.length ? (
              <button
                type="button"
                className="text-xs text-secondary transition hover:text-primary"
                onClick={() => setActiveTab("docs")}
              >
                View all {projectDetail.documents.length} documents
              </button>
            ) : null}
          </div>
        </div>

        <div className="rounded-2xl border border-subtle bg-black/20 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-primary">Ground Truth</h3>
              <p className="text-sm text-secondary">
                Persistent project facts injected into every run.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge className="border border-subtle bg-black/20 text-secondary">
                Updated {formatTimestamp(projectDetail.ground_truth.updated_at)}
              </Badge>
              <Badge
                className={`border ${
                  projectDetail.ground_truth.used_in_last_run
                    ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
                    : "border-subtle bg-black/20 text-secondary"
                }`}
              >
                {projectDetail.ground_truth.used_in_last_run ? "Used in last run" : "Draft"}
              </Badge>
              <Button
                type="button"
                className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
                onClick={() => setActiveTab("ground_truth")}
              >
                Open Full Editor
              </Button>
            </div>
          </div>
          <Textarea
            value={groundTruthDraft}
            onChange={(event) => setGroundTruthDraft(event.target.value)}
            className="mt-3 min-h-[13rem]"
            placeholder="Write stable facts, constraints, definitions, and canonical language for this project."
          />
          <p className="mt-3 text-xs text-secondary">
            {groundTruthStatus || "Autosaves after edits and is injected at run start."}
          </p>
        </div>
      </div>
    );
  };

  const renderDocsTab = () => {
    if (!projectDetail) {
      return null;
    }
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          <div>
            <h2 className="text-lg font-semibold text-primary">Project RAG Docs</h2>
            <p className="text-sm text-secondary">
              Upload files into the project-scoped collection `{projectDetail.id}`.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
              onClick={() => docInputRef.current?.click()}
              disabled={isUploadingDocs}
            >
              {isUploadingDocs ? "Uploading..." : "Upload Files"}
            </Button>
          </div>
        </div>
        {docUploadError ? (
          <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {docUploadError}
          </div>
        ) : null}
        <div className="space-y-3">
          {projectDetail.documents.length === 0 ? (
            <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-4 py-6 text-sm text-secondary">
              No project documents yet.
            </div>
          ) : (
            projectDetail.documents.map((doc) => (
              <div
                key={doc.id}
                className="rounded-2xl border border-subtle bg-[#0b0b10] p-4"
              >
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-base font-semibold text-primary">{doc.filename}</div>
                    <div className="text-xs text-secondary">
                      {doc.embedding_model} · {doc.chunk_count || 0} chunks
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className={`border ${statusBadgeClass(doc.status)}`}>
                      {doc.status}
                    </Badge>
                    <Button
                      type="button"
                      className="rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-1 text-xs text-rose-100 hover:bg-rose-500/20"
                      onClick={() => void handleDeleteDocument(doc)}
                      disabled={deletingDocId === doc.id}
                    >
                      {deletingDocId === doc.id ? "Deleting..." : "Delete"}
                    </Button>
                  </div>
                </div>
                <div className="grid gap-2 text-xs text-secondary md:grid-cols-2">
                  <div>Created: {formatTimestamp(doc.created_at)}</div>
                  <div>Updated: {formatTimestamp(doc.updated_at)}</div>
                  <div className="md:col-span-2 break-all">{doc.filepath}</div>
                  {doc.error_message ? (
                    <div className="md:col-span-2 text-rose-200">{doc.error_message}</div>
                  ) : null}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  };

  const renderGroundTruthTab = () => {
    if (!projectDetail) {
      return null;
    }
    return (
      <div className="space-y-4 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-primary">Ground Truth</h2>
            <p className="text-sm text-secondary">
              Persistent project knowledge injected into every run.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge className="border border-subtle bg-black/20 text-secondary">
              Updated {formatTimestamp(projectDetail.ground_truth.updated_at)}
            </Badge>
            <Badge
              className={`border ${
                projectDetail.ground_truth.used_in_last_run
                  ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
                  : "border-subtle bg-black/20 text-secondary"
              }`}
            >
              {projectDetail.ground_truth.used_in_last_run
                ? "Used in last run"
                : "Not used yet"}
            </Badge>
            {groundTruthSaving ? (
              <Badge className="border border-sky-400/40 bg-sky-500/10 text-sky-100">
                Saving...
              </Badge>
            ) : null}
          </div>
        </div>
        <Textarea
          value={groundTruthDraft}
          onChange={(event) => setGroundTruthDraft(event.target.value)}
          className="min-h-[24rem]"
          placeholder="Write the project's stable facts, assumptions, constraints, and canonical language."
        />
        <p className="text-xs text-secondary">{groundTruthStatus || "Autosaves after edits."}</p>
      </div>
    );
  };

  const renderWorkflowTab = () => {
    const allDiagnostics = [...workflowLocalDiagnostics, ...workflowDiagnostics];
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          <div>
            <h2 className="text-lg font-semibold text-primary">
              {projectDetail?.workflow ? "Workflow Inspector" : "Create Workflow"}
            </h2>
            <p className="text-sm text-secondary">
              Uses the shared workflow spec, graph canvas, and validation pipeline.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
              onClick={() => void handleValidateWorkflow()}
            >
              Validate
            </Button>
            <Button
              type="button"
              className="rounded-full border border-gold/40 bg-gold/10 text-gold hover:bg-gold/15"
              onClick={() => void handleSaveWorkflow()}
              disabled={workflowSaving || Boolean(workflowJsonError) || !selectedProjectId}
            >
              {workflowSaving ? "Saving..." : "Save Workflow"}
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {WORKFLOW_NODE_TYPES.map((type) => (
            <Button
              key={type}
              type="button"
              className="rounded-full border border-subtle bg-transparent text-secondary hover:text-primary"
              onClick={() => addWorkflowNode(type)}
            >
              Add {type}
            </Button>
          ))}
        </div>

        <WorkflowGraphCanvas
          nodes={workflowDraft.nodes}
          edges={workflowDraft.edges}
          selectedNodeId={selectedWorkflowNodeId}
          onSelectNode={setSelectedWorkflowNodeId}
          onNodePositionChange={(nodeId, x, y) => {
            updateWorkflowDraft({
              ...workflowDraft,
              nodes: workflowDraft.nodes.map((node) =>
                node.id === nodeId
                  ? {
                      ...node,
                      position: { x, y },
                    }
                  : node
              ),
            });
          }}
        />

        <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Workflow JSON</p>
          <Textarea
            value={workflowJsonText}
            onChange={(event) => {
              const nextValue = event.target.value;
              setWorkflowJsonText(nextValue);
              try {
                const parsed = JSON.parse(nextValue) as WorkflowSpec;
                setWorkflowJsonError("");
                setWorkflowDraft(normalizeDraftSpec(parsed));
              } catch (error) {
                setWorkflowJsonError(
                  error instanceof Error ? error.message : "Invalid workflow JSON."
                );
              }
            }}
            className="min-h-[22rem] font-mono text-xs"
          />
          {workflowJsonError ? (
            <p className="mt-2 text-xs text-rose-200">{workflowJsonError}</p>
          ) : null}
        </div>

        {allDiagnostics.length > 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
            <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Diagnostics</p>
            <div className="space-y-2">
              {allDiagnostics.map((item, index) => (
                <div
                  key={`${item.code}:${item.message}:${index}`}
                  className="rounded-lg border border-subtle bg-black/20 px-3 py-2"
                >
                  <div className="text-sm text-primary">
                    [{item.severity}] {item.code}
                  </div>
                  <div className="text-xs text-secondary">{item.message}</div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
        {workflowSaveStatus ? (
          <div className="rounded-xl border border-subtle bg-[#0b0b10] px-4 py-3 text-sm text-secondary">
            {workflowSaveStatus}
          </div>
        ) : null}
      </div>
    );
  };

  const renderToolsTab = () => {
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          <div>
            <h2 className="text-lg font-semibold text-primary">Project Tools</h2>
            <p className="text-sm text-secondary">
              Project bindings override global tool enabled state without duplicating implementations.
            </p>
          </div>
          <Button
            type="button"
            className="rounded-full border border-gold/40 bg-gold/10 text-gold hover:bg-gold/15"
            onClick={() => void handleSaveTools()}
            disabled={isSavingTools || !selectedProjectId}
          >
            {isSavingTools ? "Saving..." : "Save Tools"}
          </Button>
        </div>
        <div className="space-y-3">
          {toolBindingsDraft.map((tool) => (
            <div
              key={tool.tool_name}
              className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-subtle bg-[#0b0b10] p-4"
            >
              <div>
                <div className="text-base font-semibold text-primary">{tool.name}</div>
                <div className="text-xs text-secondary">
                  {tool.tool_name} · v{tool.version} · {tool.source}
                </div>
                <div className="mt-1 text-sm text-secondary">{tool.description}</div>
              </div>
              <label className="flex items-center gap-2 text-sm text-primary">
                <input
                  type="checkbox"
                  checked={tool.enabled}
                  onChange={(event) =>
                    setToolBindingsDraft((current) =>
                      current.map((item) =>
                        item.tool_name === tool.tool_name
                          ? { ...item, enabled: event.target.checked }
                          : item
                      )
                    )
                  }
                />
                Enabled
              </label>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 gap-4">
      <aside className="w-80 shrink-0 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
        <div className="mb-4">
          <h1 className="section-header text-4xl">Projects</h1>
          <p className="section-framer text-secondary">
            Long-lived workspaces with project-scoped RAG, workflows, tools, and chats.
          </p>
        </div>
        <div className="space-y-2 rounded-2xl border border-subtle bg-black/20 p-3">
          <Input
            value={createProjectName}
            onChange={(event) => setCreateProjectName(event.target.value)}
            placeholder="Project name"
          />
          <Textarea
            value={createProjectDescription}
            onChange={(event) => setCreateProjectDescription(event.target.value)}
            placeholder="Description"
            className="min-h-[7rem]"
          />
          <Button
            type="button"
            className="w-full rounded-full border border-gold/40 bg-gold/10 text-gold hover:bg-gold/15"
            onClick={() => void handleCreateProject()}
            disabled={isCreatingProject}
          >
            {isCreatingProject ? "Creating..." : "Create Project"}
          </Button>
        </div>

        <div className="mt-4 space-y-2">
          {projects.map((project) => (
            <button
              key={project.id}
              type="button"
              className={
                "w-full rounded-2xl border px-4 py-3 text-left transition " +
                (project.id === selectedProjectId
                  ? "border-gold/50 bg-gold/10"
                  : "border-subtle bg-black/20 hover:border-subtle hover:bg-black/30")
              }
              onClick={() => {
                setSelectedProjectId(project.id);
                setPageMessage("");
                setPageError("");
              }}
            >
              <div className="text-base font-semibold text-primary">{project.name}</div>
              <div className="mt-1 text-sm text-secondary">{project.description}</div>
              <div className="mt-2 text-[11px] text-secondary">
                {project.chat_count || 0} chats · {project.document_count || 0} docs
              </div>
            </button>
          ))}
        </div>
      </aside>

      <main className="min-w-0 flex-1 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
        {pageMessage ? (
          <div className="mb-3 rounded-xl border border-emerald-400/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
            {pageMessage}
          </div>
        ) : null}
        {pageError ? (
          <div className="mb-3 rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {pageError}
          </div>
        ) : null}

        {!projectDetail ? (
          <div className="rounded-2xl border border-subtle bg-black/20 px-6 py-10 text-center text-sm text-secondary">
            Select or create a project to begin.
          </div>
        ) : (
          <div className="flex h-full min-h-0 flex-col">
            <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-3xl font-semibold text-primary">{projectDetail.name}</h2>
                <p className="mt-2 max-w-3xl text-sm text-secondary">
                  {projectDetail.description}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge className="border border-subtle bg-black/20 text-secondary">
                  {projectDetail.chat_count} chats
                </Badge>
                <Badge className="border border-subtle bg-black/20 text-secondary">
                  {projectDetail.document_count} docs
                </Badge>
                <Button
                  type="button"
                  className="rounded-full border border-rose-400/30 bg-rose-500/10 text-rose-100 hover:bg-rose-500/20"
                  onClick={() => void handleDeleteProject()}
                  disabled={isDeletingProject}
                >
                  {isDeletingProject ? "Deleting..." : "Delete Project"}
                </Button>
              </div>
            </div>

            {renderProjectContextBoxes()}

            <div className="mb-4 flex flex-wrap gap-2">
              {PROJECT_TABS.map((tab) => (
                <Button
                  key={tab}
                  type="button"
                  className={
                    "rounded-full border px-4 py-2 text-sm " +
                    (tab === activeTab
                      ? "border-gold/50 bg-gold/10 text-gold"
                      : "border-subtle bg-black/20 text-secondary hover:text-primary")
                  }
                  onClick={() => setActiveTab(tab)}
                >
                  {tab === "docs"
                    ? "RAG Docs"
                    : tab === "ground_truth"
                    ? "Ground Truth"
                    : tab.charAt(0).toUpperCase() + tab.slice(1)}
                </Button>
              ))}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto">
              {activeTab === "chats"
                ? renderChatsTab()
                : activeTab === "docs"
                ? renderDocsTab()
                : activeTab === "ground_truth"
                ? renderGroundTruthTab()
                : activeTab === "workflow"
                ? renderWorkflowTab()
                : renderToolsTab()}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
