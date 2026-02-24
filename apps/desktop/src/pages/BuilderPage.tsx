import { type FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import WorkflowBuilderPanel from "../components/builder/WorkflowBuilderPanel";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import {
  createTool,
  getTools,
  getWorkflows,
  type CreateToolPayload,
  type Tool,
  type Workflow,
} from "../lib/api";

type BuilderTab = "tools" | "workflows";
type ToolType = "http" | "python" | "prompt";

type HeaderRow = {
  id: string;
  key: string;
  value: string;
};

type BuilderIntent = {
  tab?: BuilderTab;
  workflowId?: string;
};

const BUILDER_INTENT_STORAGE_KEY = "saturday.builder.intent";

function createId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function defaultHeaderRow(): HeaderRow {
  return { id: createId(), key: "", value: "" };
}

function slugify(value: string): string {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || "custom_resource";
}

function buildToolId(name: string): string {
  return `tool.custom.${slugify(name)}`;
}

function isCustomTool(tool: Tool): boolean {
  return (
    String(tool.source || "").toLowerCase() === "custom" ||
    ["http", "python", "prompt"].includes(String(tool.type || "").toLowerCase())
  );
}

function isCustomWorkflow(workflow: Workflow): boolean {
  return (
    String(workflow.source || "").toLowerCase() === "custom" ||
    String(workflow.type || "").toLowerCase() === "custom"
  );
}

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
          ? parsed.workflowId.trim()
          : undefined,
    };
  } catch {
    return null;
  }
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

  const [toolSubmitting, setToolSubmitting] = useState<boolean>(false);
  const [toolError, setToolError] = useState<string>("");
  const [toolSuccess, setToolSuccess] = useState<string>("");

  useEffect(() => {
    if (!toolIdTouched) {
      setToolId(buildToolId(toolName));
    }
  }, [toolName, toolIdTouched]);

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
    window.addEventListener("tools:updated", refreshTools);
    window.addEventListener("workflows:updated", refreshWorkflows);
    return () => {
      window.removeEventListener("tools:updated", refreshTools);
      window.removeEventListener("workflows:updated", refreshWorkflows);
    };
  }, [loadIndex]);

  useEffect(() => {
    const intent = parseBuilderIntent(
      window.sessionStorage.getItem(BUILDER_INTENT_STORAGE_KEY)
    );
    if (!intent) {
      return;
    }
    if (intent.tab) {
      setActiveTab(intent.tab);
    }
  }, []);

  const customTools = useMemo(() => tools.filter((tool) => isCustomTool(tool)), [tools]);
  const customWorkflows = useMemo(
    () => workflows.filter((workflow) => isCustomWorkflow(workflow)),
    [workflows]
  );

  const addHeaderRow = useCallback(() => {
    setToolHeaders((prev) => [...prev, defaultHeaderRow()]);
  }, []);

  const updateHeader = useCallback(
    (rowId: string, field: "key" | "value", value: string) => {
      setToolHeaders((prev) =>
        prev.map((row) => (row.id === rowId ? { ...row, [field]: value } : row))
      );
    },
    []
  );

  const removeHeaderRow = useCallback((rowId: string) => {
    setToolHeaders((prev) => {
      const next = prev.filter((row) => row.id !== rowId);
      return next.length > 0 ? next : [defaultHeaderRow()];
    });
  }, []);

  const handleCreateTool = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (toolSubmitting) {
        return;
      }

      const name = toolName.trim();
      if (!name) {
        setToolError("Tool name is required.");
        setToolSuccess("");
        return;
      }

      let config: CreateToolPayload["config"] | null = null;
      if (toolType === "http") {
        const url = httpUrl.trim();
        if (!url) {
          setToolError("HTTP tools require a URL.");
          setToolSuccess("");
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
      } else if (toolType === "python") {
        if (!pythonCode.trim()) {
          setToolError("Python tools require code.");
          setToolSuccess("");
          return;
        }
        config = {
          code: pythonCode,
          timeout_ms: Math.max(1, Number.parseInt(pythonTimeoutMs, 10) || 5000),
          allowed_imports: pythonAllowedImports
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
        };
      } else {
        if (!promptTemplate.trim()) {
          setToolError("Prompt tools require a prompt template.");
          setToolSuccess("");
          return;
        }
        config = {
          prompt_template: promptTemplate,
          system_prompt: promptSystemPrompt,
          temperature: Number.parseFloat(promptTemperature) || 0.2,
          timeout_ms: Math.max(1, Number.parseInt(promptTimeoutMs, 10) || 30000),
        };
      }

      const payload: CreateToolPayload = {
        id: toolId.trim() || undefined,
        name,
        description: toolDescription.trim(),
        kind: toolKind,
        type: toolType,
        enabled: true,
        config,
      };

      setToolSubmitting(true);
      setToolError("");
      setToolSuccess("");
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

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 flex-col">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Builder</h1>
            <p className="section-framer text-secondary">
              Create tools and build versioned DAG workflows with sandbox execution.
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
                    Tool ID
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
                        Python Code
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
                        Allowed Imports
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
                        System Prompt
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
                <p className="text-sm text-secondary">Loading...</p>
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
                      <p className="mt-1 text-xs text-secondary">{tool.description || "No description"}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <WorkflowBuilderPanel tools={tools} />
        )}
      </div>
    </div>
  );
}
