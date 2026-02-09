import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "../components/ui/button";
import ToolCard from "../components/tools/ToolCard";
import { getTools } from "../lib/api";
import type { Tool } from "../lib/api";

export default function ToolsPage() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const skeletons = useMemo(() => Array.from({ length: 4 }), []);

  const fetchTools = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getTools();
      setTools(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load tools.";
      setError(message);
      setTools([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchTools();
  }, [fetchTools]);

  return (
    <div className="bg-panel section-base pr-64">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Tools</h1>
            <p className="section-framer text-secondary">
              Local tools by default, with clear visibility into what each tool
              can do.
            </p>
          </div>
          <Button
            type="button"
            className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
            onClick={() => void fetchTools()}
          >
            Refresh
          </Button>
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
                onClick={() => void fetchTools()}
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
                key={`tool-skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-subtle bg-[#0b0b10] p-5"
              >
                <div className="mb-4 h-4 w-2/3 rounded bg-white/10" />
                <div className="mb-5 h-3 w-1/3 rounded bg-white/5" />
                <div className="space-y-2">
                  <div className="h-3 w-full rounded bg-white/5" />
                  <div className="h-3 w-5/6 rounded bg-white/5" />
                </div>
                <div className="mt-6 h-8 w-28 rounded-full bg-white/10" />
              </div>
            ))}
          </div>
        ) : tools.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <p>No tools are available yet.</p>
              <Button
                type="button"
                className="h-8 rounded-full border border-subtle bg-transparent px-4 text-xs text-secondary hover:text-primary"
              >
                Create Tool
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {tools.map((tool) => (
              <ToolCard key={tool.id} tool={tool} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
