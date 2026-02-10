import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ModelCard, {
  type CatalogModel,
  type ModelCategory,
} from "../components/models/ModelCard";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { getModels, getVisionModels, type Model } from "../lib/api";

const MODEL_CATEGORY_ORDER: ModelCategory[] = [
  "cloud",
  "embedding",
  "vision",
  "local",
  "other",
];

const MODEL_CATEGORY_META: Record<
  ModelCategory,
  { title: string; summary: string; description: string }
> = {
  cloud: {
    title: "Cloud Models (APIs)",
    summary: "Cloud",
    description: "Hosted providers and API-backed models available from the backend.",
  },
  embedding: {
    title: "Embedding Models",
    summary: "Embedding",
    description: "Vector/embedding-focused models used for retrieval and indexing.",
  },
  vision: {
    title: "Vision Models",
    summary: "Vision",
    description: "Models configured for multimodal or image understanding tasks.",
  },
  local: {
    title: "Local Models",
    summary: "Local",
    description: "Models served locally (for example through Ollama).",
  },
  other: {
    title: "Other Models",
    summary: "Other",
    description: "Models that do not match the current source/capability heuristics.",
  },
};

const EMBEDDING_HINTS = [
  "embed",
  "embedding",
  "bge",
  "e5",
  "gte",
  "nomic-embed",
  "mxbai",
  "minilm",
  "text-embedding",
];

const sortCatalogModels = (models: CatalogModel[]): CatalogModel[] => {
  return [...models].sort(
    (left, right) =>
      left.name.localeCompare(right.name, undefined, { sensitivity: "base" }) ||
      left.id.localeCompare(right.id, undefined, { sensitivity: "base" })
  );
};

const normalizeModelKey = (model: Model): string => {
  const id = typeof model.id === "string" ? model.id.trim() : "";
  const name = typeof model.name === "string" ? model.name.trim() : "";
  return (id || name).toLowerCase();
};

const isLocalSource = (source?: string): boolean => {
  const normalized = (source ?? "").trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return normalized === "local" || normalized.includes("ollama");
};

const isEmbeddingModel = (model: Model): boolean => {
  const value = `${model.id ?? ""} ${model.name ?? ""}`.toLowerCase();
  return EMBEDDING_HINTS.some((hint) => value.includes(hint));
};

const classifyModel = (model: Model, isVision: boolean): ModelCategory => {
  if (isVision) {
    return "vision";
  }
  if (isEmbeddingModel(model)) {
    return "embedding";
  }
  if (isLocalSource(model.source)) {
    return "local";
  }
  if ((model.source ?? "").trim()) {
    return "cloud";
  }
  return "other";
};

const buildCatalogModels = (
  textModels: Model[],
  visionModels: Model[]
): CatalogModel[] => {
  const merged = new Map<string, Model>();
  const visionKeys = new Set<string>();

  for (const model of visionModels) {
    const key = normalizeModelKey(model);
    if (!key) {
      continue;
    }
    visionKeys.add(key);
  }

  for (const model of textModels) {
    const key = normalizeModelKey(model);
    if (!key) {
      continue;
    }
    merged.set(key, model);
  }

  for (const model of visionModels) {
    const key = normalizeModelKey(model);
    if (!key || merged.has(key)) {
      continue;
    }
    merged.set(key, model);
  }

  const catalog: CatalogModel[] = [];
  for (const [key, model] of merged.entries()) {
    const id = (model.id ?? "").trim() || (model.name ?? "").trim();
    const name = (model.name ?? "").trim() || (model.id ?? "").trim();
    if (!id || !name) {
      continue;
    }
    catalog.push({
      id,
      name,
      source: model.source,
      status: model.status,
      category: classifyModel(model, visionKeys.has(key)),
    });
  }

  return sortCatalogModels(catalog);
};

export default function ModelsPage() {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; detail?: string }>({
    ok: false,
  });

  const isFetchingRef = useRef(false);

  const fetchModels = useCallback(async (showLoading = true) => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;

    if (showLoading) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const [textResult, visionResult] = await Promise.allSettled([
        getModels(),
        getVisionModels(),
      ]);

      let textModels: Model[] = [];
      let visionModels: Model[] = [];
      const messages: string[] = [];
      let backendConnected = false;

      if (textResult.status === "fulfilled") {
        textModels = textResult.value.models;
        backendConnected = true;
      } else {
        messages.push(
          textResult.reason instanceof Error
            ? textResult.reason.message
            : "Unable to load models."
        );
      }

      if (visionResult.status === "fulfilled") {
        visionModels = visionResult.value.models;
        backendConnected = true;
      } else {
        messages.push(
          visionResult.reason instanceof Error
            ? visionResult.reason.message
            : "Unable to load vision models."
        );
      }

      setModels(buildCatalogModels(textModels, visionModels));
      setHealth({
        ok: backendConnected,
        detail: backendConnected ? undefined : messages[0],
      });

      if (messages.length > 0) {
        setError(messages.join(" "));
      }
    } finally {
      setIsLoading(false);
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void fetchModels(true);
  }, [fetchModels]);

  const groupedModels = useMemo(() => {
    const groups: Record<ModelCategory, CatalogModel[]> = {
      cloud: [],
      embedding: [],
      vision: [],
      local: [],
      other: [],
    };

    for (const model of models) {
      groups[model.category].push(model);
    }

    for (const category of MODEL_CATEGORY_ORDER) {
      groups[category] = sortCatalogModels(groups[category]);
    }

    return groups;
  }, [models]);

  const visibleCategories = useMemo(
    () => MODEL_CATEGORY_ORDER.filter((category) => groupedModels[category].length > 0),
    [groupedModels]
  );

  const totalModels = models.length;
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

        <div className="mt-4 rounded-2xl border border-subtle bg-[#0b0b10] p-4">
          {error ? (
            <div className="mb-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
              {error}
            </div>
          ) : null}

          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs uppercase tracking-wide text-secondary">
              Model Inventory
            </p>
            <p className="text-xs text-secondary">{totalModels} total</p>
          </div>

          <div className="flex flex-wrap gap-2">
            {MODEL_CATEGORY_ORDER.map((category) => (
              <div
                key={category}
                className="rounded-full border border-subtle bg-[#11111a] px-3 py-1 text-xs"
              >
                <span className="text-primary">{MODEL_CATEGORY_META[category].summary}</span>
                <span className="ml-2 text-secondary">
                  {groupedModels[category].length}
                </span>
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
                key={`skeleton-${index}`}
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
        ) : models.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            No models detected. Connect a provider or start Ollama, then refresh.
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
                      {MODEL_CATEGORY_META[category].title}
                    </h2>
                    <p className="text-xs text-secondary">
                      {MODEL_CATEGORY_META[category].description}
                    </p>
                  </div>
                  <Badge className="border border-white/10 bg-white/5 text-[11px] text-secondary">
                    {groupedModels[category].length} model
                    {groupedModels[category].length === 1 ? "" : "s"}
                  </Badge>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                  {groupedModels[category].map((model) => (
                    <ModelCard key={`${category}-${model.id}`} model={model} />
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
