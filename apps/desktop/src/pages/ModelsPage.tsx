import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import ModelCard from "../components/models/ModelCard";
import { listModels } from "../lib/ollama";
import type { OllamaModel } from "../lib/ollama";

const POLL_INTERVAL_MS = 2000;

const sortModels = (models: OllamaModel[]): OllamaModel[] => {
  return [...models].sort((a, b) => {
    const aTime = a.modified_at ? Date.parse(a.modified_at) : NaN;
    const bTime = b.modified_at ? Date.parse(b.modified_at) : NaN;

    if (!Number.isNaN(aTime) && !Number.isNaN(bTime)) {
      return bTime - aTime;
    }
    if (!Number.isNaN(aTime)) {
      return -1;
    }
    if (!Number.isNaN(bTime)) {
      return 1;
    }

    return a.name.localeCompare(b.name);
  });
};

export default function ModelsPage() {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; detail?: string }>({
    ok: false,
  });

  const initialLoadRef = useRef(true);
  const isFetchingRef = useRef(false);

  const fetchModels = useCallback(async (showLoading?: boolean) => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;

    const shouldShowLoading = showLoading ?? initialLoadRef.current;
    if (shouldShowLoading) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const data = await listModels();
      setModels(sortModels(data));
      setHealth({ ok: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to reach Ollama";
      setError(message);
      setModels([]);
      setHealth({ ok: false, detail: message });
    } finally {
      setIsLoading(false);
      initialLoadRef.current = false;
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void fetchModels(true);
    const intervalId = window.setInterval(() => {
      void fetchModels(false);
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [fetchModels]);

  const statusLabel = health.ok ? "Ollama Connected" : "Ollama Down";
  const statusClass = health.ok
    ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
    : "border-rose-400/40 bg-rose-500/10 text-rose-200";

  const skeletons = useMemo(() => Array.from({ length: 4 }), []);

  return (
    <div className="bg-panel section-base pr-64">
      <div className="section-hero">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Models</h1>
            <p className="section-framer text-secondary">
              Local-first models, tuned for fast iteration and private work.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className={`border ${statusClass}`}>{statusLabel}</Badge>
            <Button
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              type="button"
              onClick={() => void fetchModels(true)}
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
                className="h-8 rounded-full border border-rose-400/40 bg-transparent px-3 text-xs text-rose-100 hover:bg-rose-500/10"
                type="button"
                onClick={() => void fetchModels(true)}
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
                key={`skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-subtle bg-[#0b0b10] p-5"
              >
                <div className="mb-4 h-4 w-2/3 rounded bg-white/10" />
                <div className="mb-6 h-3 w-1/3 rounded bg-white/5" />
                <div className="space-y-2">
                  <div className="h-3 w-full rounded bg-white/5" />
                  <div className="h-3 w-5/6 rounded bg-white/5" />
                  <div className="h-3 w-2/3 rounded bg-white/5" />
                </div>
                <div className="mt-6 h-8 w-24 rounded-full bg-white/10" />
              </div>
            ))}
          </div>
        ) : models.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            {health.ok
              ? "No local models found yet. Pull a model in Ollama to get started."
              : "Ollama is not reachable. Start Ollama to load local models."}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {models.map((model) => (
              <ModelCard key={model.name} model={model} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
