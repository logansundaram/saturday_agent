import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WorkflowCard, {
  type CatalogWorkflow,
  type WorkflowCategory,
} from "../components/workflows/WorkflowCard";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { getWorkflows, type Workflow } from "../lib/api";

const WORKFLOW_CATEGORY_ORDER: WorkflowCategory[] = [
  "simple",
  "moderate",
  "complex",
  "other",
];

const WORKFLOW_CATEGORY_META: Record<
  WorkflowCategory,
  { title: string; summary: string; description: string }
> = {
  simple: {
    title: "Simple Workflows",
    summary: "Simple",
    description: "Fast, direct flows for straightforward tasks.",
  },
  moderate: {
    title: "Moderate Workflows",
    summary: "Moderate",
    description: "Balanced workflows with planning and selective tooling.",
  },
  complex: {
    title: "Complex Workflows",
    summary: "Complex",
    description: "Multi-step orchestration for advanced or high-context tasks.",
  },
  other: {
    title: "Other Workflows",
    summary: "Other",
    description: "Workflows with custom or unknown type definitions.",
  },
};

const sortCatalogWorkflows = (workflows: CatalogWorkflow[]): CatalogWorkflow[] => {
  return [...workflows].sort(
    (left, right) =>
      left.title.localeCompare(right.title, undefined, { sensitivity: "base" }) ||
      left.id.localeCompare(right.id, undefined, { sensitivity: "base" })
  );
};

const normalize = (value?: string): string => (value ?? "").trim().toLowerCase();

const classifyWorkflow = (workflow: Workflow): WorkflowCategory => {
  const type = normalize(workflow.type);
  if (type === "simple") {
    return "simple";
  }
  if (type === "moderate") {
    return "moderate";
  }
  if (type === "complex") {
    return "complex";
  }
  return "other";
};

const buildCatalogWorkflows = (workflows: Workflow[]): CatalogWorkflow[] => {
  return sortCatalogWorkflows(
    workflows.map((workflow) => ({
      ...workflow,
      category: classifyWorkflow(workflow),
    }))
  );
};

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<CatalogWorkflow[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; detail?: string }>({
    ok: false,
  });
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(
    null
  );

  const isFetchingRef = useRef(false);

  const fetchWorkflows = useCallback(async (showLoading = true) => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;

    if (showLoading) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const data = await getWorkflows();
      const catalog = buildCatalogWorkflows(data);
      setWorkflows(catalog);
      setHealth({ ok: true });
      setSelectedWorkflowId((previous) => {
        if (!previous) {
          return previous;
        }
        return catalog.some((item) => item.id === previous) ? previous : null;
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to load workflows.";
      setError(message);
      setWorkflows([]);
      setHealth({ ok: false, detail: message });
      setSelectedWorkflowId(null);
    } finally {
      setIsLoading(false);
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void fetchWorkflows(true);
  }, [fetchWorkflows]);

  const groupedWorkflows = useMemo(() => {
    const groups: Record<WorkflowCategory, CatalogWorkflow[]> = {
      simple: [],
      moderate: [],
      complex: [],
      other: [],
    };

    for (const workflow of workflows) {
      groups[workflow.category].push(workflow);
    }

    for (const category of WORKFLOW_CATEGORY_ORDER) {
      groups[category] = sortCatalogWorkflows(groups[category]);
    }

    return groups;
  }, [workflows]);

  const visibleCategories = useMemo(
    () =>
      WORKFLOW_CATEGORY_ORDER.filter(
        (category) => groupedWorkflows[category].length > 0
      ),
    [groupedWorkflows]
  );

  const totalWorkflows = workflows.length;
  const selectedWorkflow = workflows.find(
    (workflow) => workflow.id === selectedWorkflowId
  );
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
            <h1 className="section-header text-5xl">Workflows</h1>
            <p className="section-framer text-secondary">
              Local, transparent workflows by default. Inspect and select a flow
              shape for each task.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className={`border ${statusClass}`}>{statusLabel}</Badge>
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={() => void fetchWorkflows(true)}
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
            <p className="text-xs uppercase tracking-wide text-secondary">
              Workflow Inventory
            </p>
            <p className="text-xs text-secondary">{totalWorkflows} total</p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {WORKFLOW_CATEGORY_ORDER.map((category) => (
              <div
                key={category}
                className="rounded-full border border-subtle bg-[#11111a] px-3 py-1 text-xs"
              >
                <span className="text-primary">
                  {WORKFLOW_CATEGORY_META[category].summary}
                </span>
                <span className="ml-2 text-secondary">
                  {groupedWorkflows[category].length}
                </span>
              </div>
            ))}
            {selectedWorkflow ? (
              <Badge className="border border-gold/40 bg-gold/10 text-[11px] text-gold">
                Selected: {selectedWorkflow.title}
              </Badge>
            ) : null}
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-8">
        {isLoading ? (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {skeletons.map((_, index) => (
              <div
                key={`workflow-skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-subtle bg-[#0b0b10] p-4"
              >
                <div className="mb-4 h-4 w-2/3 rounded bg-white/10" />
                <div className="mb-3 h-3 w-1/3 rounded bg-white/5" />
                <div className="mb-4 grid grid-cols-2 gap-2">
                  <div className="h-12 rounded-lg bg-white/5" />
                  <div className="h-12 rounded-lg bg-white/5" />
                </div>
                <div className="space-y-2">
                  <div className="h-3 w-full rounded bg-white/5" />
                  <div className="h-3 w-5/6 rounded bg-white/5" />
                </div>
              </div>
            ))}
          </div>
        ) : workflows.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            No workflows are available. Add workflow definitions in the backend
            and refresh.
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
                      {WORKFLOW_CATEGORY_META[category].title}
                    </h2>
                    <p className="text-xs text-secondary">
                      {WORKFLOW_CATEGORY_META[category].description}
                    </p>
                  </div>
                  <Badge className="border border-white/10 bg-white/5 text-[11px] text-secondary">
                    {groupedWorkflows[category].length} workflow
                    {groupedWorkflows[category].length === 1 ? "" : "s"}
                  </Badge>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                  {groupedWorkflows[category].map((workflow) => (
                    <WorkflowCard
                      key={`${category}-${workflow.id}`}
                      workflow={workflow}
                      selected={selectedWorkflowId === workflow.id}
                      onSelect={(item) => {
                        setSelectedWorkflowId(item.id);
                      }}
                    />
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
