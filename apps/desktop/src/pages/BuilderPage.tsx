import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import {
  createTool,
  createWorkflow,
  getTools,
  getWorkflowDefinition,
  getWorkflows,
  runWorkflow,
  type CreateToolPayload,
  type Tool,
  type Workflow,
  type WorkflowEdge,
  type WorkflowNode,
} from "../lib/api";

type BuilderTab = "tools" | "workflows";

type HeaderRow = {
  id: string;
  key: string;
  value: string;
};

type ToolType = "http" | "python" | "prompt";

type EditableNodeType = "start" | "llm" | "tool" | "condition" | "end";

type EditableNode = {
  id: string;
  type: EditableNodeType;
  config: Record<string, any>;
};

type BuilderIntent = {
  tab?: BuilderTab;
  workflowId?: string;
};

const BUILDER_INTENT_STORAGE_KEY = "saturday.builder.intent";

const createId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const defaultHeaderRow = (): HeaderRow => ({
  id: createId(),
  key: "",
  value: "",
});

const slugify = (value: string): string => {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || "custom_resource";
};

const buildToolId = (name: string): string => `tool.custom.${slugify(name)}`;
const buildWorkflowId = (name: string): string => `workflow.custom.${slugify(name)}`;

const defaultWorkflowNodes = (): EditableNode[] => [
  { id: "start", type: "start", config: {} },
  { id: "end", type: "end", config: {} },
];

const defaultWorkflowEdges = (): WorkflowEdge[] => [
  { from: "start", to: "end", condition: "always" },
];

const isCustomTool = (tool: Tool): boolean =>
  String(tool.source || "").toLowerCase() === "custom" ||
  ["http", "python", "prompt"].includes(String(tool.type || "").toLowerCase());

const isCustomWorkflow = (workflow: Workflow): boolean =>
  String(workflow.source || "").toLowerCase() === "custom" ||
  String(workflow.type || "").toLowerCase() === "custom";

function parseBuilderIntent(raw: string | null): BuilderIntent | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as BuilderIntent;
    return {
      tab: parsed.tab === "workflows" ? "workflows" : "tools",
      workflowId:
        typeof parsed.workflowId === "string" && parsed.workflowId.trim()
          ? parsed.workflowId
          : undefined,
    };
  } catch {
    return null;
  }
}

function validateWorkflowGraph(nodes: EditableNode[], edges: WorkflowEdge[]): string[] {
  const errors: string[] = [];
  const nodeIdSet = new Set<string>();
  let startCount = 0;
  let endCount = 0;

  for (const node of nodes) {
    if (!node.id.trim()) {
      errors.push("Each node requires an id.");
      continue;
    }
    if (nodeIdSet.has(node.id)) {
      errors.push(`Duplicate node id '${node.id}'.`);
    }
    nodeIdSet.add(node.id);

    if (node.type === "start") {
      startCount += 1;
    }
    if (node.type === "end") {
      endCount += 1;
    }

    if (node.type === "llm") {
      if (!String(node.config.prompt_template || "").trim()) {
        errors.push(`Node '${node.id}' requires a prompt_template.`);
      }
    }
    if (node.type === "tool") {
      if (!String(node.config.tool_id || "").trim()) {
        errors.push(`Node '${node.id}' requires a tool_id.`);
      }
    }
    if (node.type === "condition") {
      const field = String(node.config.field || "").trim();
      const operator = String(node.config.operator || "").trim().toLowerCase();
      if (!field) {
        errors.push(`Condition node '${node.id}' requires a field.`);
      }
      if (
        ![
          "equals",
          "contains",
          "gt",
          "lt",
          "exists",
          "not_exists",
          "in",
        ].includes(operator)
      ) {
        errors.push(`Condition node '${node.id}' has an invalid operator.`);
      }
    }
  }

  if (startCount !== 1) {
    errors.push("Workflow must have exactly one Start node.");
  }
  if (endCount !== 1) {
    errors.push("Workflow must have exactly one End node.");
  }

  const outgoing = new Map<string, WorkflowEdge[]>();
  for (const edge of edges) {
    if (!nodeIdSet.has(edge.from)) {
      errors.push(`Edge references unknown from node '${edge.from}'.`);
      continue;
    }
    if (!nodeIdSet.has(edge.to)) {
      errors.push(`Edge references unknown to node '${edge.to}'.`);
      continue;
    }
    const condition = edge.condition || "always";
    if (!["always", "true", "false"].includes(condition)) {
      errors.push(`Edge '${edge.from}' -> '${edge.to}' has invalid condition.`);
      continue;
    }
    const list = outgoing.get(edge.from) || [];
    list.push(edge);
    outgoing.set(edge.from, list);
  }

  for (const node of nodes) {
    const nextEdges = outgoing.get(node.id) || [];
    if (node.type === "end") {
      if (nextEdges.length > 0) {
        errors.push("End node cannot have outgoing edges.");
      }
      continue;
    }

    if (node.type === "condition") {
      const hasTrue = nextEdges.some((edge) => (edge.condition || "always") === "true");
      const hasFalse = nextEdges.some(
        (edge) => (edge.condition || "always") === "false"
      );
      if (!hasTrue || !hasFalse) {
        errors.push(`Condition node '${node.id}' needs both true and false edges.`);
      }
      continue;
    }

    if (nextEdges.length === 0) {
      errors.push(`Node '${node.id}' must have at least one outgoing edge.`);
    }
    for (const edge of nextEdges) {
      if ((edge.condition || "always") !== "always") {
        errors.push(
          `Only Condition nodes can use true/false edges (node '${node.id}').`
        );
        break;
      }
    }
  }

  return errors;
}

function nodeTitle(node: EditableNode): string {
  return `${node.type.toUpperCase()} · ${node.id}`;
}

export default function BuilderPage() {
  const [activeTab, setActiveTab] = useState<BuilderTab>("tools");

  const [tools, setTools] = useState<Tool[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loadingIndex, setLoadingIndex] = useState<boolean>(true);

  const [toolName, setToolName] = useState<string>("");
  const [toolId, setToolId] = useState<string>("tool.custom.custom_resource");
  const [toolIdTouched, setToolIdTouched] = useState<boolean>(false);
  const [toolDescription, setToolDescription] = useState<string>("");
  const [toolKind, setToolKind] = useState<"local" | "external">("external");
  const [toolType, setToolType] = useState<ToolType>("http");
  const [toolHeaders, setToolHeaders] = useState<HeaderRow[]>([defaultHeaderRow()]);
  const [httpUrl, setHttpUrl] = useState<string>("");
  const [httpMethod, setHttpMethod] = useState<"GET" | "POST">("POST");
  const [httpTimeoutMs, setHttpTimeoutMs] = useState<string>("8000");
  const [pythonCode, setPythonCode] = useState<string>(
    "def run(input, context):\n    query = str(input.get('query', ''))\n    return {'echo': query, 'source': 'python_tool'}\n"
  );
  const [pythonTimeoutMs, setPythonTimeoutMs] = useState<string>("5000");
  const [pythonAllowedImports, setPythonAllowedImports] = useState<string>(
    "json,math,datetime,re"
  );
  const [promptTemplate, setPromptTemplate] = useState<string>(
    "Summarize this query clearly: {{query}}"
  );
  const [promptSystemPrompt, setPromptSystemPrompt] = useState<string>("");
  const [promptTemperature, setPromptTemperature] = useState<string>("0.2");
  const [promptTimeoutMs, setPromptTimeoutMs] = useState<string>("30000");

  const [toolError, setToolError] = useState<string | null>(null);
  const [toolSuccess, setToolSuccess] = useState<string | null>(null);
  const [toolSubmitting, setToolSubmitting] = useState<boolean>(false);

  const [workflowName, setWorkflowName] = useState<string>("");
  const [workflowId, setWorkflowId] = useState<string>("workflow.custom.custom_resource");
  const [workflowIdTouched, setWorkflowIdTouched] = useState<boolean>(false);
  const [workflowDescription, setWorkflowDescription] = useState<string>("");
  const [workflowEnabled, setWorkflowEnabled] = useState<boolean>(true);
  const [workflowNodes, setWorkflowNodes] = useState<EditableNode[]>(
    defaultWorkflowNodes()
  );
  const [workflowEdges, setWorkflowEdges] = useState<WorkflowEdge[]>(
    defaultWorkflowEdges()
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string>("start");
  const [edgeFrom, setEdgeFrom] = useState<string>("start");
  const [edgeTo, setEdgeTo] = useState<string>("end");
  const [edgeCondition, setEdgeCondition] = useState<"always" | "true" | "false">(
    "always"
  );
  const [workflowError, setWorkflowError] = useState<string | null>(null);
  const [workflowSuccess, setWorkflowSuccess] = useState<string | null>(null);
  const [workflowSubmitting, setWorkflowSubmitting] = useState<boolean>(false);
  const [workflowTesting, setWorkflowTesting] = useState<boolean>(false);
  const [workflowTestInput, setWorkflowTestInput] = useState<string>(
    "Summarize the latest support ticket"
  );
  const [workflowTestResult, setWorkflowTestResult] = useState<string>("");
  const [workflowLoading, setWorkflowLoading] = useState<boolean>(false);
  const [toolInputMapText, setToolInputMapText] = useState<string>("{}");
  const [toolInputMapError, setToolInputMapError] = useState<string | null>(null);

  useEffect(() => {
    if (!toolIdTouched) {
      setToolId(buildToolId(toolName));
    }
  }, [toolName, toolIdTouched]);

  useEffect(() => {
    if (!workflowIdTouched) {
      setWorkflowId(buildWorkflowId(workflowName));
    }
  }, [workflowName, workflowIdTouched]);

  const loadIndex = useCallback(async () => {
    setLoadingIndex(true);
    try {
      const [toolItems, workflowItems] = await Promise.all([getTools(), getWorkflows()]);
      setTools(toolItems);
      setWorkflows(workflowItems);
    } finally {
      setLoadingIndex(false);
    }
  }, []);

  const loadWorkflowIntoEditor = useCallback(async (id: string) => {
    if (!id.trim()) {
      return;
    }

    setWorkflowLoading(true);
    setWorkflowError(null);
    setWorkflowSuccess(null);
    try {
      const payload = await getWorkflowDefinition(id);
      const nodes = Array.isArray(payload.graph?.nodes)
        ? (payload.graph.nodes as EditableNode[])
        : defaultWorkflowNodes();
      const edges = Array.isArray(payload.graph?.edges)
        ? (payload.graph.edges as WorkflowEdge[])
        : defaultWorkflowEdges();

      setActiveTab("workflows");
      setWorkflowName(payload.name || payload.title || payload.id);
      setWorkflowId(payload.id);
      setWorkflowIdTouched(true);
      setWorkflowDescription(payload.description || "");
      setWorkflowEnabled(Boolean(payload.enabled));
      setWorkflowNodes(nodes.length > 0 ? nodes : defaultWorkflowNodes());
      setWorkflowEdges(edges.length > 0 ? edges : defaultWorkflowEdges());
      setSelectedNodeId(nodes[0]?.id || "start");
      setEdgeFrom(nodes[0]?.id || "start");
      setEdgeTo(nodes[1]?.id || "end");
      setWorkflowSuccess(`Loaded workflow ${payload.id}.`);
    } catch (error) {
      setWorkflowError(
        error instanceof Error ? error.message : "Unable to load workflow."
      );
    } finally {
      setWorkflowLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadIndex();
  }, [loadIndex]);

  useEffect(() => {
    const refreshTools = () => {
      void loadIndex();
    };
    const refreshWorkflows = () => {
      void loadIndex();
    };
    const openWorkflow = (event: Event) => {
      const customEvent = event as CustomEvent<BuilderIntent>;
      const workflowIdFromEvent = String(customEvent.detail?.workflowId || "").trim();
      if (!workflowIdFromEvent) {
        return;
      }
      void loadWorkflowIntoEditor(workflowIdFromEvent);
    };

    window.addEventListener("tools:updated", refreshTools);
    window.addEventListener("workflows:updated", refreshWorkflows);
    window.addEventListener("builder:open-workflow", openWorkflow as EventListener);

    return () => {
      window.removeEventListener("tools:updated", refreshTools);
      window.removeEventListener("workflows:updated", refreshWorkflows);
      window.removeEventListener(
        "builder:open-workflow",
        openWorkflow as EventListener
      );
    };
  }, [loadIndex, loadWorkflowIntoEditor]);

  useEffect(() => {
    const intent = parseBuilderIntent(
      window.sessionStorage.getItem(BUILDER_INTENT_STORAGE_KEY)
    );
    if (!intent) {
      return;
    }

    window.sessionStorage.removeItem(BUILDER_INTENT_STORAGE_KEY);
    if (intent.tab) {
      setActiveTab(intent.tab);
    }
    if (intent.workflowId) {
      void loadWorkflowIntoEditor(intent.workflowId);
    }
  }, [loadWorkflowIntoEditor]);

  const customTools = useMemo(() => tools.filter((tool) => isCustomTool(tool)), [tools]);
  const customWorkflows = useMemo(
    () => workflows.filter((workflow) => isCustomWorkflow(workflow)),
    [workflows]
  );

  const selectedNode = useMemo(
    () => workflowNodes.find((node) => node.id === selectedNodeId) || null,
    [workflowNodes, selectedNodeId]
  );

  const workflowGraphErrors = useMemo(
    () => validateWorkflowGraph(workflowNodes, workflowEdges),
    [workflowNodes, workflowEdges]
  );

  useEffect(() => {
    if (selectedNode?.type === "tool") {
      const mapValue =
        selectedNode.config.input_map && typeof selectedNode.config.input_map === "object"
          ? selectedNode.config.input_map
          : {};
      setToolInputMapText(JSON.stringify(mapValue, null, 2));
      setToolInputMapError(null);
    }
  }, [selectedNode]);

  const updateHeader = useCallback(
    (rowId: string, field: "key" | "value", value: string) => {
      setToolHeaders((prev) =>
        prev.map((row) => (row.id === rowId ? { ...row, [field]: value } : row))
      );
    },
    []
  );

  const addHeaderRow = useCallback(() => {
    setToolHeaders((prev) => [...prev, defaultHeaderRow()]);
  }, []);

  const removeHeaderRow = useCallback((rowId: string) => {
    setToolHeaders((prev) => {
      const next = prev.filter((row) => row.id !== rowId);
      return next.length > 0 ? next : [defaultHeaderRow()];
    });
  }, []);

  const handleCreateTool = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (toolSubmitting) {
        return;
      }

      const name = toolName.trim();
      if (!name) {
        setToolError("Tool name is required.");
        setToolSuccess(null);
        return;
      }

      let config: CreateToolPayload["config"] | null = null;

      if (toolType === "http") {
        const url = httpUrl.trim();
        if (!url) {
          setToolError("HTTP tools require a URL.");
          setToolSuccess(null);
          return;
        }

        const headers = toolHeaders.reduce<Record<string, string>>((acc, row) => {
          const key = row.key.trim();
          if (!key) {
            return acc;
          }
          acc[key] = row.value.trim();
          return acc;
        }, {});

        config = {
          url,
          method: httpMethod,
          headers,
          timeout_ms: Math.max(1, Number.parseInt(httpTimeoutMs, 10) || 8000),
        };
      }

      if (toolType === "python") {
        const code = pythonCode;
        if (!code.trim()) {
          setToolError("Python tools require code.");
          setToolSuccess(null);
          return;
        }

        const allowedImports = pythonAllowedImports
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item.length > 0);

        config = {
          code,
          timeout_ms: Math.max(1, Number.parseInt(pythonTimeoutMs, 10) || 5000),
          allowed_imports: allowedImports,
        };
      }

      if (toolType === "prompt") {
        const template = promptTemplate;
        if (!template.trim()) {
          setToolError("Prompt tools require a prompt template.");
          setToolSuccess(null);
          return;
        }

        config = {
          prompt_template: template,
          system_prompt: promptSystemPrompt,
          temperature: Number.parseFloat(promptTemperature) || 0.2,
          timeout_ms: Math.max(1, Number.parseInt(promptTimeoutMs, 10) || 30000),
        };
      }

      if (!config) {
        setToolError("Unsupported tool configuration.");
        setToolSuccess(null);
        return;
      }

      const payload: CreateToolPayload = {
        name,
        id: toolId.trim() || undefined,
        kind: toolKind,
        description: toolDescription.trim(),
        type: toolType,
        enabled: true,
        config,
      };

      setToolSubmitting(true);
      setToolError(null);
      setToolSuccess(null);

      try {
        const created = await createTool(payload);
        setToolSuccess(`Created ${created.name} (${created.id}).`);
        setToolName("");
        setToolDescription("");
        setToolIdTouched(false);
        setHttpUrl("");
        setToolHeaders([defaultHeaderRow()]);
        setHttpTimeoutMs("8000");
        await loadIndex();
        window.dispatchEvent(new CustomEvent("tools:updated"));
      } catch (error) {
        setToolError(error instanceof Error ? error.message : "Failed to create tool.");
      } finally {
        setToolSubmitting(false);
      }
    },
    [
      httpMethod,
      httpTimeoutMs,
      httpUrl,
      loadIndex,
      promptSystemPrompt,
      promptTemplate,
      promptTemperature,
      promptTimeoutMs,
      pythonAllowedImports,
      pythonCode,
      pythonTimeoutMs,
      toolDescription,
      toolHeaders,
      toolId,
      toolKind,
      toolName,
      toolSubmitting,
      toolType,
    ]
  );

  const resetWorkflowEditor = useCallback(() => {
    setWorkflowName("");
    setWorkflowId("workflow.custom.custom_resource");
    setWorkflowIdTouched(false);
    setWorkflowDescription("");
    setWorkflowEnabled(true);
    setWorkflowNodes(defaultWorkflowNodes());
    setWorkflowEdges(defaultWorkflowEdges());
    setSelectedNodeId("start");
    setEdgeFrom("start");
    setEdgeTo("end");
    setWorkflowError(null);
    setWorkflowSuccess(null);
    setWorkflowTestResult("");
  }, []);

  const addNode = useCallback(
    (type: EditableNodeType) => {
      if (type === "start" && workflowNodes.some((node) => node.type === "start")) {
        setWorkflowError("Workflow can only have one Start node.");
        return;
      }
      if (type === "end" && workflowNodes.some((node) => node.type === "end")) {
        setWorkflowError("Workflow can only have one End node.");
        return;
      }

      const newNodeId = `${type}_${slugify(createId())}`;
      const defaults: Record<string, any> =
        type === "llm"
          ? { prompt_template: "", output_key: "" }
          : type === "tool"
          ? { tool_id: "", input_map: {}, output_key: "" }
          : type === "condition"
          ? { field: "", operator: "equals", value: "" }
          : type === "end"
          ? { response_template: "" }
          : {};

      setWorkflowNodes((prev) => [...prev, { id: newNodeId, type, config: defaults }]);
      setSelectedNodeId(newNodeId);
      setEdgeFrom(newNodeId);
      setWorkflowError(null);
    },
    [workflowNodes]
  );

  const removeNode = useCallback(
    (nodeId: string) => {
      const target = workflowNodes.find((node) => node.id === nodeId);
      if (!target) {
        return;
      }
      if (target.type === "start" || target.type === "end") {
        setWorkflowError("Start and End nodes cannot be removed.");
        return;
      }
      setWorkflowNodes((prev) => prev.filter((node) => node.id !== nodeId));
      setWorkflowEdges((prev) =>
        prev.filter((edge) => edge.from !== nodeId && edge.to !== nodeId)
      );
      setSelectedNodeId("start");
    },
    [workflowNodes]
  );

  const updateNodeConfig = useCallback(
    (nodeId: string, key: string, value: any) => {
      setWorkflowNodes((prev) =>
        prev.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                config: {
                  ...node.config,
                  [key]: value,
                },
              }
            : node
        )
      );
    },
    []
  );

  const addEdge = useCallback(() => {
    const from = edgeFrom.trim();
    const to = edgeTo.trim();
    if (!from || !to) {
      setWorkflowError("Edge requires from and to nodes.");
      return;
    }
    const alreadyExists = workflowEdges.some(
      (edge) => edge.from === from && edge.to === to && (edge.condition || "always") === edgeCondition
    );
    if (alreadyExists) {
      setWorkflowError("This edge already exists.");
      return;
    }
    setWorkflowEdges((prev) => [...prev, { from, to, condition: edgeCondition }]);
    setWorkflowError(null);
  }, [edgeCondition, edgeFrom, edgeTo, workflowEdges]);

  const removeEdge = useCallback((index: number) => {
    setWorkflowEdges((prev) => prev.filter((_, edgeIndex) => edgeIndex !== index));
  }, []);

  const handleToolInputMapCommit = useCallback(() => {
    if (!selectedNode || selectedNode.type !== "tool") {
      return;
    }

    try {
      const parsed = JSON.parse(toolInputMapText || "{}");
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setToolInputMapError("input_map must be a JSON object.");
        return;
      }
      updateNodeConfig(selectedNode.id, "input_map", parsed);
      setToolInputMapError(null);
    } catch {
      setToolInputMapError("Invalid JSON for input_map.");
    }
  }, [selectedNode, toolInputMapText, updateNodeConfig]);

  const handleSaveWorkflow = useCallback(async () => {
    if (workflowSubmitting) {
      return;
    }

    const name = workflowName.trim();
    if (!name) {
      setWorkflowError("Workflow name is required.");
      setWorkflowSuccess(null);
      return;
    }

    if (workflowGraphErrors.length > 0) {
      setWorkflowError("Fix graph validation errors before saving.");
      setWorkflowSuccess(null);
      return;
    }

    const payload = {
      id: workflowId.trim() || undefined,
      name,
      description: workflowDescription.trim(),
      enabled: workflowEnabled,
      graph: {
        nodes: workflowNodes as WorkflowNode[],
        edges: workflowEdges,
      },
    };

    setWorkflowSubmitting(true);
    setWorkflowError(null);
    setWorkflowSuccess(null);

    try {
      const saved = await createWorkflow(payload);
      setWorkflowId(saved.id);
      setWorkflowIdTouched(true);
      setWorkflowSuccess(`Saved workflow ${saved.id}.`);
      await loadIndex();
      window.dispatchEvent(new CustomEvent("workflows:updated"));
    } catch (error) {
      setWorkflowError(
        error instanceof Error ? error.message : "Failed to save workflow."
      );
    } finally {
      setWorkflowSubmitting(false);
    }
  }, [
    loadIndex,
    workflowDescription,
    workflowEnabled,
    workflowEdges,
    workflowGraphErrors,
    workflowId,
    workflowName,
    workflowNodes,
    workflowSubmitting,
  ]);

  const handleTestWorkflow = useCallback(async () => {
    const id = workflowId.trim();
    if (!id) {
      setWorkflowError("Save the workflow first before test run.");
      return;
    }

    if (!workflowTestInput.trim()) {
      setWorkflowError("Provide sample input for test run.");
      return;
    }

    setWorkflowTesting(true);
    setWorkflowError(null);
    setWorkflowTestResult("");

    try {
      const result = await runWorkflow({
        workflow_id: id,
        input: {
          task: workflowTestInput,
          context: {},
        },
      });
      const answer = result.output?.answer;
      const normalized =
        typeof answer === "string" && answer.trim()
          ? answer
          : JSON.stringify(result.output || {}, null, 2);
      setWorkflowTestResult(normalized);
    } catch (error) {
      setWorkflowError(error instanceof Error ? error.message : "Workflow run failed.");
    } finally {
      setWorkflowTesting(false);
    }
  }, [workflowId, workflowTestInput]);

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 flex-col">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Builder</h1>
            <p className="section-framer text-secondary">
              Build custom tools and design workflow graphs in one place.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge className="border border-sky-400/40 bg-sky-500/10 text-sky-100">
              {customTools.length} custom tools
            </Badge>
            <Badge className="border border-emerald-400/40 bg-emerald-500/10 text-emerald-100">
              {customWorkflows.length} custom workflows
            </Badge>
          </div>
        </div>

        <div className="mt-4 inline-flex rounded-full border border-subtle bg-[#0b0b10] p-1">
          <button
            type="button"
            className={
              "rounded-full px-4 py-1.5 text-sm transition " +
              (activeTab === "tools"
                ? "bg-gold text-black"
                : "text-secondary hover:text-primary")
            }
            onClick={() => setActiveTab("tools")}
          >
            Tool Builder
          </button>
          <button
            type="button"
            className={
              "rounded-full px-4 py-1.5 text-sm transition " +
              (activeTab === "workflows"
                ? "bg-gold text-black"
                : "text-secondary hover:text-primary")
            }
            onClick={() => setActiveTab("workflows")}
          >
            Workflow Builder
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-8">
        {activeTab === "tools" ? (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <form
              onSubmit={(event) => {
                void handleCreateTool(event);
              }}
              className="rounded-2xl border border-subtle bg-[#0b0b10] p-5"
            >
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Name
                  </label>
                  <Input
                    value={toolName}
                    onChange={(event) => setToolName(event.target.value)}
                    placeholder="Customer Search"
                    className="h-10 border-subtle bg-[#08080c]"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Tool ID (Auto)
                  </label>
                  <Input
                    value={toolId}
                    onChange={(event) => {
                      setToolIdTouched(true);
                      setToolId(event.target.value);
                    }}
                    className="h-10 border-subtle bg-[#08080c] font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Kind
                  </label>
                  <select
                    value={toolKind}
                    onChange={(event) =>
                      setToolKind(event.target.value === "local" ? "local" : "external")
                    }
                    className="h-10 w-full rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
                  >
                    <option value="external">External</option>
                    <option value="local">Local</option>
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Type
                  </label>
                  <select
                    value={toolType}
                    onChange={(event) =>
                      setToolType((event.target.value as ToolType) || "http")
                    }
                    className="h-10 w-full rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
                  >
                    <option value="http">HTTP</option>
                    <option value="python">Python</option>
                    <option value="prompt">Prompt</option>
                  </select>
                </div>

                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Description
                  </label>
                  <Textarea
                    value={toolDescription}
                    onChange={(event) => setToolDescription(event.target.value)}
                    rows={2}
                    placeholder="Describe what this tool does."
                    className="border-subtle bg-[#08080c]"
                  />
                </div>

                {toolType === "http" ? (
                  <>
                    <div className="md:col-span-2">
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        URL
                      </label>
                      <Input
                        value={httpUrl}
                        onChange={(event) => setHttpUrl(event.target.value)}
                        placeholder="https://api.example.com/search"
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Method
                      </label>
                      <select
                        value={httpMethod}
                        onChange={(event) =>
                          setHttpMethod(event.target.value === "GET" ? "GET" : "POST")
                        }
                        className="h-10 w-full rounded-md border border-subtle bg-[#08080c] px-3 text-sm text-primary outline-none"
                      >
                        <option value="POST">POST</option>
                        <option value="GET">GET</option>
                      </select>
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Timeout (ms)
                      </label>
                      <Input
                        type="number"
                        min={1}
                        value={httpTimeoutMs}
                        onChange={(event) => setHttpTimeoutMs(event.target.value)}
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}

                {toolType === "python" ? (
                  <>
                    <div className="md:col-span-2">
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Python Code (`run(input, context)`)
                      </label>
                      <Textarea
                        value={pythonCode}
                        onChange={(event) => setPythonCode(event.target.value)}
                        rows={8}
                        className="border-subtle bg-[#08080c] font-mono text-xs"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Timeout (ms)
                      </label>
                      <Input
                        type="number"
                        min={1}
                        value={pythonTimeoutMs}
                        onChange={(event) => setPythonTimeoutMs(event.target.value)}
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Allowed Imports (comma-separated)
                      </label>
                      <Input
                        value={pythonAllowedImports}
                        onChange={(event) => setPythonAllowedImports(event.target.value)}
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}

                {toolType === "prompt" ? (
                  <>
                    <div className="md:col-span-2">
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Prompt Template
                      </label>
                      <Textarea
                        value={promptTemplate}
                        onChange={(event) => setPromptTemplate(event.target.value)}
                        rows={4}
                        className="border-subtle bg-[#08080c]"
                      />
                    </div>

                    <div className="md:col-span-2">
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        System Prompt (optional)
                      </label>
                      <Textarea
                        value={promptSystemPrompt}
                        onChange={(event) => setPromptSystemPrompt(event.target.value)}
                        rows={2}
                        className="border-subtle bg-[#08080c]"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Temperature
                      </label>
                      <Input
                        type="number"
                        step={0.1}
                        value={promptTemperature}
                        onChange={(event) => setPromptTemperature(event.target.value)}
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                        Timeout (ms)
                      </label>
                      <Input
                        type="number"
                        min={1}
                        value={promptTimeoutMs}
                        onChange={(event) => setPromptTimeoutMs(event.target.value)}
                        className="h-10 border-subtle bg-[#08080c]"
                      />
                    </div>
                  </>
                ) : null}
              </div>

              {toolType === "http" ? (
                <div className="mt-4 rounded-xl border border-subtle bg-[#08080c] p-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="text-xs uppercase tracking-wide text-secondary">Headers</p>
                    <Button
                      type="button"
                      className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                      onClick={addHeaderRow}
                    >
                      Add Header
                    </Button>
                  </div>

                  <div className="space-y-2">
                    {toolHeaders.map((row) => (
                      <div key={row.id} className="grid grid-cols-[1fr_1fr_auto] gap-2">
                        <Input
                          value={row.key}
                          onChange={(event) => updateHeader(row.id, "key", event.target.value)}
                          placeholder="Header"
                          className="h-9 border-subtle bg-[#0b0b10]"
                        />
                        <Input
                          value={row.value}
                          onChange={(event) =>
                            updateHeader(row.id, "value", event.target.value)
                          }
                          placeholder="Value"
                          className="h-9 border-subtle bg-[#0b0b10]"
                        />
                        <Button
                          type="button"
                          className="h-9 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                          onClick={() => removeHeaderRow(row.id)}
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {toolError ? (
                <div className="mt-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                  {toolError}
                </div>
              ) : null}
              {toolSuccess ? (
                <div className="mt-4 rounded-xl border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-100">
                  {toolSuccess}
                </div>
              ) : null}

              <div className="mt-5 flex justify-end">
                <Button
                  type="submit"
                  className="h-10 rounded-full bg-gold px-5 text-sm text-black hover:bg-[#e1c161] disabled:opacity-60"
                  disabled={toolSubmitting}
                >
                  {toolSubmitting ? "Creating..." : "Create Tool"}
                </Button>
              </div>
            </form>

            <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-5">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 className="text-sm uppercase tracking-wide text-secondary">Custom Tools</h2>
                <Button
                  type="button"
                  className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                  onClick={() => void loadIndex()}
                >
                  Refresh
                </Button>
              </div>

              {loadingIndex ? (
                <p className="text-sm text-secondary">Loading tools...</p>
              ) : customTools.length === 0 ? (
                <p className="text-sm text-secondary">No custom tools yet.</p>
              ) : (
                <div className="space-y-3">
                  {customTools.map((tool) => (
                    <div
                      key={tool.id}
                      className="rounded-xl border border-subtle bg-[#08080c] px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <p className="truncate text-sm text-primary">{tool.name}</p>
                        <Badge
                          className={
                            tool.enabled
                              ? "border border-emerald-400/30 bg-emerald-500/10 text-[11px] text-emerald-100"
                              : "border border-white/10 bg-white/5 text-[11px] text-secondary"
                          }
                        >
                          {tool.enabled ? "Enabled" : "Disabled"}
                        </Badge>
                      </div>
                      <p className="mt-1 truncate font-mono text-[11px] text-secondary">
                        {tool.id}
                      </p>
                      <p className="mt-1 text-[11px] uppercase tracking-wide text-secondary">
                        {tool.type}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,0.9fr)]">
            <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-5">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Workflow Name
                  </label>
                  <Input
                    value={workflowName}
                    onChange={(event) => setWorkflowName(event.target.value)}
                    placeholder="Customer Triage"
                    className="h-10 border-subtle bg-[#08080c]"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Workflow ID (Auto)
                  </label>
                  <Input
                    value={workflowId}
                    onChange={(event) => {
                      setWorkflowIdTouched(true);
                      setWorkflowId(event.target.value);
                    }}
                    className="h-10 border-subtle bg-[#08080c] font-mono text-xs"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="mb-1 block text-xs uppercase tracking-wide text-secondary">
                    Description
                  </label>
                  <Textarea
                    value={workflowDescription}
                    onChange={(event) => setWorkflowDescription(event.target.value)}
                    rows={2}
                    className="border-subtle bg-[#08080c]"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <input
                    id="workflow-enabled"
                    type="checkbox"
                    checked={workflowEnabled}
                    onChange={(event) => setWorkflowEnabled(event.target.checked)}
                    className="h-4 w-4"
                  />
                  <label htmlFor="workflow-enabled" className="text-sm text-secondary">
                    Enabled
                  </label>
                </div>

                <div className="flex justify-end gap-2 md:justify-start">
                  <Button
                    type="button"
                    className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                    onClick={resetWorkflowEditor}
                  >
                    New
                  </Button>
                  <Button
                    type="button"
                    className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                    onClick={() => void loadIndex()}
                  >
                    Refresh
                  </Button>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[220px_minmax(0,1fr)]">
                <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                  <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Node Palette</p>
                  <div className="space-y-2">
                    {(["start", "llm", "tool", "condition", "end"] as EditableNodeType[]).map(
                      (type) => (
                        <Button
                          key={type}
                          type="button"
                          className="h-8 w-full justify-start rounded-lg border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                          onClick={() => addNode(type)}
                        >
                          + {type.toUpperCase()}
                        </Button>
                      )
                    )}
                  </div>

                  <div className="mt-4 border-t border-subtle pt-3">
                    <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Nodes</p>
                    <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
                      {workflowNodes.map((node) => (
                        <div
                          key={node.id}
                          className={
                            "rounded-lg border px-2 py-2 " +
                            (selectedNodeId === node.id
                              ? "border-gold/40 bg-gold/10"
                              : "border-subtle bg-[#0b0b10]")
                          }
                        >
                          <button
                            type="button"
                            className="w-full text-left"
                            onClick={() => setSelectedNodeId(node.id)}
                          >
                            <p className="truncate text-xs text-primary">{nodeTitle(node)}</p>
                          </button>
                          {node.type !== "start" && node.type !== "end" ? (
                            <div className="mt-1 flex justify-end">
                              <Button
                                type="button"
                                className="h-6 rounded-full border border-rose-400/30 bg-transparent px-2 text-[10px] text-rose-200 hover:text-rose-100"
                                onClick={() => removeNode(node.id)}
                              >
                                Remove
                              </Button>
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                    <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Canvas</p>
                    <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
                      <div className="rounded-lg border border-subtle bg-[#0b0b10] p-3">
                        <p className="mb-2 text-xs text-secondary">Node Connections</p>
                        <div className="space-y-2">
                          {workflowEdges.length === 0 ? (
                            <p className="text-xs text-secondary">No edges yet.</p>
                          ) : (
                            workflowEdges.map((edge, index) => (
                              <div
                                key={`${edge.from}-${edge.to}-${edge.condition}-${index}`}
                                className="flex items-center justify-between gap-2 rounded-md border border-subtle bg-[#08080c] px-2 py-1.5"
                              >
                                <p className="truncate text-xs text-primary">
                                  {edge.from}
                                  {" -> "}
                                  {edge.to} ({edge.condition || "always"})
                                </p>
                                <Button
                                  type="button"
                                  className="h-6 rounded-full border border-subtle bg-transparent px-2 text-[10px] text-secondary hover:text-primary"
                                  onClick={() => removeEdge(index)}
                                >
                                  Remove
                                </Button>
                              </div>
                            ))
                          )}
                        </div>

                        <div className="mt-3 rounded-md border border-subtle bg-[#08080c] p-2">
                          <p className="mb-2 text-[11px] uppercase tracking-wide text-secondary">
                            Add Edge
                          </p>
                          <div className="grid grid-cols-1 gap-2 md:grid-cols-4">
                            <select
                              value={edgeFrom}
                              onChange={(event) => setEdgeFrom(event.target.value)}
                              className="h-8 rounded-md border border-subtle bg-[#0b0b10] px-2 text-xs text-primary"
                            >
                              {workflowNodes.map((node) => (
                                <option key={`from-${node.id}`} value={node.id}>
                                  {node.id}
                                </option>
                              ))}
                            </select>
                            <select
                              value={edgeTo}
                              onChange={(event) => setEdgeTo(event.target.value)}
                              className="h-8 rounded-md border border-subtle bg-[#0b0b10] px-2 text-xs text-primary"
                            >
                              {workflowNodes.map((node) => (
                                <option key={`to-${node.id}`} value={node.id}>
                                  {node.id}
                                </option>
                              ))}
                            </select>
                            <select
                              value={edgeCondition}
                              onChange={(event) =>
                                setEdgeCondition(
                                  (event.target.value as "always" | "true" | "false") ||
                                    "always"
                                )
                              }
                              className="h-8 rounded-md border border-subtle bg-[#0b0b10] px-2 text-xs text-primary"
                            >
                              <option value="always">always</option>
                              <option value="true">true</option>
                              <option value="false">false</option>
                            </select>
                            <Button
                              type="button"
                              className="h-8 rounded-md border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                              onClick={addEdge}
                            >
                              Add
                            </Button>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-lg border border-subtle bg-[#0b0b10] p-3">
                        <p className="mb-2 text-xs uppercase tracking-wide text-secondary">
                          Inspector
                        </p>
                        {selectedNode ? (
                          <div className="space-y-3">
                            <div>
                              <p className="text-xs text-secondary">Node</p>
                              <p className="text-sm text-primary">{nodeTitle(selectedNode)}</p>
                            </div>

                            {selectedNode.type === "llm" ? (
                              <>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    prompt_template
                                  </label>
                                  <Textarea
                                    value={String(selectedNode.config.prompt_template || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(
                                        selectedNode.id,
                                        "prompt_template",
                                        event.target.value
                                      )
                                    }
                                    rows={4}
                                    className="border-subtle bg-[#08080c]"
                                  />
                                </div>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    output_key
                                  </label>
                                  <Input
                                    value={String(selectedNode.config.output_key || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(
                                        selectedNode.id,
                                        "output_key",
                                        event.target.value
                                      )
                                    }
                                    className="h-9 border-subtle bg-[#08080c]"
                                  />
                                </div>
                              </>
                            ) : null}

                            {selectedNode.type === "tool" ? (
                              <>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    tool_id
                                  </label>
                                  <select
                                    value={String(selectedNode.config.tool_id || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(
                                        selectedNode.id,
                                        "tool_id",
                                        event.target.value
                                      )
                                    }
                                    className="h-9 w-full rounded-md border border-subtle bg-[#08080c] px-2 text-xs text-primary"
                                  >
                                    <option value="">Select tool</option>
                                    {tools.map((tool) => (
                                      <option key={`tool-node-${tool.id}`} value={tool.id}>
                                        {tool.name} ({tool.id})
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    input_map (JSON)
                                  </label>
                                  <Textarea
                                    value={toolInputMapText}
                                    onChange={(event) => setToolInputMapText(event.target.value)}
                                    onBlur={handleToolInputMapCommit}
                                    rows={4}
                                    className="border-subtle bg-[#08080c] font-mono text-xs"
                                  />
                                  {toolInputMapError ? (
                                    <p className="mt-1 text-[11px] text-rose-200">{toolInputMapError}</p>
                                  ) : null}
                                </div>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    output_key
                                  </label>
                                  <Input
                                    value={String(selectedNode.config.output_key || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(
                                        selectedNode.id,
                                        "output_key",
                                        event.target.value
                                      )
                                    }
                                    className="h-9 border-subtle bg-[#08080c]"
                                  />
                                </div>
                              </>
                            ) : null}

                            {selectedNode.type === "condition" ? (
                              <>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">field</label>
                                  <Input
                                    value={String(selectedNode.config.field || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(selectedNode.id, "field", event.target.value)
                                    }
                                    className="h-9 border-subtle bg-[#08080c]"
                                  />
                                </div>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">
                                    operator
                                  </label>
                                  <select
                                    value={String(selectedNode.config.operator || "equals")}
                                    onChange={(event) =>
                                      updateNodeConfig(
                                        selectedNode.id,
                                        "operator",
                                        event.target.value
                                      )
                                    }
                                    className="h-9 w-full rounded-md border border-subtle bg-[#08080c] px-2 text-xs text-primary"
                                  >
                                    {[
                                      "equals",
                                      "contains",
                                      "gt",
                                      "lt",
                                      "exists",
                                      "not_exists",
                                      "in",
                                    ].map((operator) => (
                                      <option key={operator} value={operator}>
                                        {operator}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                <div>
                                  <label className="mb-1 block text-xs text-secondary">value</label>
                                  <Input
                                    value={String(selectedNode.config.value || "")}
                                    onChange={(event) =>
                                      updateNodeConfig(selectedNode.id, "value", event.target.value)
                                    }
                                    className="h-9 border-subtle bg-[#08080c]"
                                  />
                                </div>
                              </>
                            ) : null}

                            {selectedNode.type === "end" ? (
                              <div>
                                <label className="mb-1 block text-xs text-secondary">
                                  response_template
                                </label>
                                <Textarea
                                  value={String(selectedNode.config.response_template || "")}
                                  onChange={(event) =>
                                    updateNodeConfig(
                                      selectedNode.id,
                                      "response_template",
                                      event.target.value
                                    )
                                  }
                                  rows={3}
                                  className="border-subtle bg-[#08080c]"
                                />
                              </div>
                            ) : null}
                          </div>
                        ) : (
                          <p className="text-xs text-secondary">Select a node to edit config.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                    <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Validation</p>
                    {workflowGraphErrors.length === 0 ? (
                      <p className="text-sm text-emerald-100">Graph looks valid.</p>
                    ) : (
                      <div className="space-y-1">
                        {workflowGraphErrors.map((error) => (
                          <p key={error} className="text-xs text-rose-200">
                            - {error}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="rounded-xl border border-subtle bg-[#08080c] p-3">
                    <p className="mb-2 text-xs uppercase tracking-wide text-secondary">Test Run</p>
                    <Textarea
                      value={workflowTestInput}
                      onChange={(event) => setWorkflowTestInput(event.target.value)}
                      rows={2}
                      className="border-subtle bg-[#0b0b10]"
                    />
                    <div className="mt-2 flex items-center gap-2">
                      <Button
                        type="button"
                        className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                        onClick={() => void handleTestWorkflow()}
                        disabled={workflowTesting || workflowLoading}
                      >
                        {workflowTesting ? "Running..." : "Test Run"}
                      </Button>
                    </div>
                    {workflowTestResult ? (
                      <pre className="mt-2 max-h-36 overflow-auto rounded-md border border-subtle bg-[#0b0b10] p-2 text-[11px] text-secondary">
                        {workflowTestResult}
                      </pre>
                    ) : null}
                  </div>
                </div>
              </div>

              {workflowError ? (
                <div className="mt-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                  {workflowError}
                </div>
              ) : null}
              {workflowSuccess ? (
                <div className="mt-4 rounded-xl border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-100">
                  {workflowSuccess}
                </div>
              ) : null}

              <div className="mt-5 flex justify-end">
                <Button
                  type="button"
                  className="h-10 rounded-full bg-gold px-5 text-sm text-black hover:bg-[#e1c161] disabled:opacity-60"
                  onClick={() => void handleSaveWorkflow()}
                  disabled={workflowSubmitting || workflowLoading}
                >
                  {workflowSubmitting ? "Saving..." : "Save Workflow"}
                </Button>
              </div>
            </div>

            <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-5">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 className="text-sm uppercase tracking-wide text-secondary">Custom Workflows</h2>
                <Button
                  type="button"
                  className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                  onClick={() => void loadIndex()}
                >
                  Refresh
                </Button>
              </div>

              {loadingIndex ? (
                <p className="text-sm text-secondary">Loading workflows...</p>
              ) : customWorkflows.length === 0 ? (
                <p className="text-sm text-secondary">No custom workflows yet.</p>
              ) : (
                <div className="space-y-3">
                  {customWorkflows.map((workflow) => (
                    <div
                      key={workflow.id}
                      className="rounded-xl border border-subtle bg-[#08080c] px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <p className="truncate text-sm text-primary">
                          {workflow.title || workflow.name || workflow.id}
                        </p>
                        <Badge
                          className={
                            workflow.status === "disabled"
                              ? "border border-white/10 bg-white/5 text-[11px] text-secondary"
                              : "border border-emerald-400/30 bg-emerald-500/10 text-[11px] text-emerald-100"
                          }
                        >
                          {workflow.status === "disabled" ? "Disabled" : "Enabled"}
                        </Badge>
                      </div>
                      <p className="mt-1 truncate font-mono text-[11px] text-secondary">
                        {workflow.id}
                      </p>
                      <div className="mt-2 flex justify-end">
                        <Button
                          type="button"
                          className="h-7 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
                          onClick={() => void loadWorkflowIntoEditor(workflow.id)}
                        >
                          Edit
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
