import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import WorkflowCard from "../components/workflows/WorkflowCard";
import { getWorkflows } from "../lib/api";
import type { Workflow } from "../lib/api";

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(
    null
  );

  const skeletons = useMemo(() => Array.from({ length: 4 }), []);

  const fetchWorkflows = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getWorkflows();
      setWorkflows(data);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to load workflows.";
      setError(message);
      setWorkflows([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchWorkflows();
  }, [fetchWorkflows]);

  return (
    <div className="bg-panel section-base pr-64">
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
            {selectedWorkflowId ? (
              <Badge className="border border-gold/40 bg-gold/10 text-[11px] text-gold">
                Selected: {selectedWorkflowId}
              </Badge>
            ) : null}
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={() => void fetchWorkflows()}
            >
              Refresh
            </Button>
          </div>
        </div>
      </div>

      <div className="px-4 pb-10">
        {error ? (
          <div className="mb-6 rounded-2xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span>{error}</span>
              <Button
                type="button"
                className="h-8 rounded-full border border-rose-400/40 bg-transparent px-3 text-xs text-rose-100 hover:bg-rose-500/10"
                onClick={() => void fetchWorkflows()}
              >
                Retry
              </Button>
            </div>
          </div>
        ) : null}

        {isLoading ? (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {skeletons.map((_, index) => (
              <div
                key={`workflow-skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-subtle bg-[#0b0b10] p-5"
              >
                <div className="mb-4 h-4 w-2/3 rounded bg-white/10" />
                <div className="mb-5 h-3 w-1/3 rounded bg-white/5" />
                <div className="space-y-2">
                  <div className="h-3 w-full rounded bg-white/5" />
                  <div className="h-3 w-5/6 rounded bg-white/5" />
                </div>
                <div className="mt-6 h-8 w-24 rounded-full bg-white/10" />
              </div>
            ))}
          </div>
        ) : workflows.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <p>No workflows are available yet.</p>
              <Button
                type="button"
                className="h-8 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
              >
                Create Workflow
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {workflows.map((workflow) => (
              <WorkflowCard
                key={workflow.id}
                workflow={workflow}
                selected={selectedWorkflowId === workflow.id}
                onSelect={(item) => {
                  setSelectedWorkflowId(item.id);
                  console.log("Selected workflow", item.id);
                }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
