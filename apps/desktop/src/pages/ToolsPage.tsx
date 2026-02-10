import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ToolCard, {
  type CatalogTool,
  type ToolCategory,
} from "../components/tools/ToolCard";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { getTools, type Tool } from "../lib/api";

const TOOL_CATEGORY_ORDER: ToolCategory[] = [
  "local",
  "external",
  "vision",
  "workflow",
  "other",
];

const TOOL_CATEGORY_META: Record<
  ToolCategory,
  { title: string; summary: string; description: string }
> = {
  local: {
    title: "Local Tools",
    summary: "Local",
    description: "Tools that run directly on local runtime resources.",
  },
  external: {
    title: "External/API Tools",
    summary: "External",
    description: "Tools that call web services, APIs, or other remote systems.",
  },
  vision: {
    title: "Vision Tools",
    summary: "Vision",
    description: "Image and multimodal analysis tools.",
  },
  workflow: {
    title: "Workflow Tools",
    summary: "Workflow",
    description: "Tools focused on workflow planning, tracing, and inspection.",
  },
  other: {
    title: "Other Tools",
    summary: "Other",
    description: "Tools that do not match current classification heuristics.",
  },
};

const sortCatalogTools = (tools: CatalogTool[]): CatalogTool[] => {
  return [...tools].sort(
    (left, right) =>
      Number(right.enabled) - Number(left.enabled) ||
      left.name.localeCompare(right.name, undefined, { sensitivity: "base" }) ||
      left.id.localeCompare(right.id, undefined, { sensitivity: "base" })
  );
};

const normalize = (value?: string): string => (value ?? "").trim().toLowerCase();

const classifyTool = (tool: Tool): ToolCategory => {
  const id = normalize(tool.id);
  const name = normalize(tool.name);
  const kind = normalize(tool.kind);

  if (id.includes("vision") || name.includes("vision") || kind.includes("vision")) {
    return "vision";
  }

  if (
    id.startsWith("workflow.") ||
    name.includes("workflow") ||
    kind.includes("workflow")
  ) {
    return "workflow";
  }

  if (
    ["external", "api", "remote", "web", "saas", "cloud"].some((token) =>
      kind.includes(token)
    )
  ) {
    return "external";
  }

  if (!kind || kind.includes("local")) {
    return "local";
  }

  return "other";
};

const buildCatalogTools = (tools: Tool[]): CatalogTool[] => {
  return sortCatalogTools(
    tools.map((tool) => ({
      ...tool,
      category: classifyTool(tool),
    }))
  );
};

export default function ToolsPage() {
  const [tools, setTools] = useState<CatalogTool[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; detail?: string }>({
    ok: false,
  });

  const isFetchingRef = useRef(false);

  const fetchTools = useCallback(async (showLoading = true) => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;

    if (showLoading) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const data = await getTools();
      setTools(buildCatalogTools(data));
      setHealth({ ok: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load tools.";
      setError(message);
      setTools([]);
      setHealth({ ok: false, detail: message });
    } finally {
      setIsLoading(false);
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void fetchTools(true);
  }, [fetchTools]);

  const groupedTools = useMemo(() => {
    const groups: Record<ToolCategory, CatalogTool[]> = {
      local: [],
      external: [],
      vision: [],
      workflow: [],
      other: [],
    };

    for (const tool of tools) {
      groups[tool.category].push(tool);
    }

    for (const category of TOOL_CATEGORY_ORDER) {
      groups[category] = sortCatalogTools(groups[category]);
    }

    return groups;
  }, [tools]);

  const visibleCategories = useMemo(
    () => TOOL_CATEGORY_ORDER.filter((category) => groupedTools[category].length > 0),
    [groupedTools]
  );

  const totalTools = tools.length;
  const enabledTools = tools.filter((tool) => tool.enabled).length;
  const statusLabel = health.ok ? "Backend Connected" : "Backend Unavailable";
  const statusClass = health.ok
    ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
    : "border-rose-400/40 bg-rose-500/10 text-rose-100";

  const skeletons = useMemo(() => Array.from({ length: 6 }), []);

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 flex-col">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Tools</h1>
            <p className="section-framer text-secondary">
              Local tools by default, with clear visibility into what each tool can do.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className={`border ${statusClass}`}>{statusLabel}</Badge>
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={() => void fetchTools(true)}
            >
              Refresh
            </Button>
          </div>
        </div>

        <div className="mt-4 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          {error ? (
            <div className="mb-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
              {error}
            </div>
          ) : null}

          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs uppercase tracking-wide text-secondary">Tool Inventory</p>
            <p className="text-xs text-secondary">
              {enabledTools}/{totalTools} enabled
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {TOOL_CATEGORY_ORDER.map((category) => (
              <div
                key={category}
                className="rounded-full border border-subtle bg-[#11111a] px-3 py-1 text-xs"
              >
                <span className="text-primary">{TOOL_CATEGORY_META[category].summary}</span>
                <span className="ml-2 text-secondary">{groupedTools[category].length}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-8">
        {isLoading ? (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {skeletons.map((_, index) => (
              <div
                key={`tool-skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-subtle bg-[#0b0b10] p-4"
              >
                <div className="mb-4 h-4 w-2/3 rounded bg-white/10" />
                <div className="mb-3 h-3 w-1/3 rounded bg-white/5" />
                <div className="mb-4 h-12 rounded-lg bg-white/5" />
                <div className="space-y-2">
                  <div className="h-3 w-full rounded bg-white/5" />
                  <div className="h-3 w-5/6 rounded bg-white/5" />
                </div>
              </div>
            ))}
          </div>
        ) : tools.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            No tools are available. Register tools in the backend and refresh.
          </div>
        ) : (
          <div className="space-y-6">
            {visibleCategories.map((category) => (
              <section
                key={category}
                className="rounded-2xl border border-subtle bg-[#0b0b10] p-4"
              >
                <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-primary">
                      {TOOL_CATEGORY_META[category].title}
                    </h2>
                    <p className="text-xs text-secondary">
                      {TOOL_CATEGORY_META[category].description}
                    </p>
                  </div>
                  <Badge className="border border-white/10 bg-white/5 text-[11px] text-secondary">
                    {groupedTools[category].length} tool
                    {groupedTools[category].length === 1 ? "" : "s"}
                  </Badge>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                  {groupedTools[category].map((tool) => (
                    <ToolCard key={`${category}-${tool.id}`} tool={tool} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
