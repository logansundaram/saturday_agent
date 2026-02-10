import { Badge } from "../ui/badge";

export type ModelCategory = "cloud" | "embedding" | "vision" | "local" | "other";

export type CatalogModel = {
  id: string;
  name: string;
  source?: string;
  status?: string;
  category: ModelCategory;
};

interface ModelCardProps {
  model: CatalogModel;
}

const CATEGORY_BADGE_CLASS: Record<ModelCategory, string> = {
  cloud: "border-sky-400/40 bg-sky-500/10 text-sky-100",
  embedding: "border-cyan-400/40 bg-cyan-500/10 text-cyan-100",
  vision: "border-amber-400/40 bg-amber-500/10 text-amber-100",
  local: "border-emerald-400/40 bg-emerald-500/10 text-emerald-100",
  other: "border-white/10 bg-white/5 text-secondary",
};

const CATEGORY_LABEL: Record<ModelCategory, string> = {
  cloud: "Cloud",
  embedding: "Embedding",
  vision: "Vision",
  local: "Local",
  other: "Other",
};

export default function ModelCard({ model }: ModelCardProps) {
  const source = formatValue(model.source, "Unknown");
  const status = formatValue(model.status, "Available");
  const showId = model.id.trim() && model.id.trim() !== model.name.trim();

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.25)]">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-primary">{model.name}</h3>
          {showId ? (
            <p className="truncate text-xs text-secondary">{model.id}</p>
          ) : null}
        </div>
        <Badge
          className={`border text-[11px] ${CATEGORY_BADGE_CLASS[model.category]}`}
        >
          {CATEGORY_LABEL[model.category]}
        </Badge>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Source</p>
          <p className="mt-1 truncate text-sm text-primary">{source}</p>
        </div>
        <div className="rounded-xl border border-subtle bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-secondary">Status</p>
          <p className="mt-1 truncate text-sm text-primary">{status}</p>
        </div>
      </div>
    </div>
  );
}

function formatValue(value: string | undefined, fallback: string): string {
  const raw = (value ?? "").trim();
  if (!raw) {
    return fallback;
  }
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
