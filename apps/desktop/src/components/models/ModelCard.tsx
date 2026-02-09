import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import type { OllamaModel } from "../../lib/ollama";

interface ModelCardProps {
  model: OllamaModel;
  onSelect?: (modelName: string) => void;
}

export default function ModelCard({ model, onSelect }: ModelCardProps) {
  const details = model.details;

  const rows = [
    { label: "Family", value: details?.family },
    { label: "Params", value: details?.parameter_size },
    { label: "Quant", value: details?.quantization_level },
    { label: "Size", value: formatBytes(model.size) },
    { label: "Updated", value: formatUpdated(model.modified_at) },
  ].filter((row) => row.value);

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-5 shadow-[0_10px_30px_rgba(0,0,0,0.25)]">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-primary">{model.name}</h3>
          {model.model && model.model !== model.name ? (
            <p className="text-xs text-secondary">{model.model}</p>
          ) : null}
        </div>
        <Badge className="border border-white/10 bg-white/5 text-[11px] text-secondary">
          Local
        </Badge>
      </div>

      <div className="mt-4 grid gap-2 text-xs">
        {rows.length === 0 ? (
          <p className="text-secondary">No metadata available.</p>
        ) : (
          rows.map((row) => (
            <div key={row.label} className="flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-wide text-secondary">
                {row.label}
              </span>
              <span className="text-right text-sm text-primary">
                {row.value}
              </span>
            </div>
          ))
        )}
      </div>

      <div className="mt-5 flex items-center justify-end gap-2">
        <Button
          className="h-8 rounded-full border border-subtle bg-transparent px-3 text-xs text-secondary hover:text-primary"
          type="button"
        >
          Details
        </Button>
        <Button
          className="h-8 rounded-full bg-gold px-4 text-xs text-black hover:bg-[#e1c161]"
          type="button"
          onClick={() => onSelect?.(model.name)}
        >
          Select
        </Button>
      </div>
    </div>
  );
}

function formatBytes(value?: number): string | undefined {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }
  const gb = 1024 * 1024 * 1024;
  const mb = 1024 * 1024;
  if (value >= gb) {
    return `${(value / gb).toFixed(2)} GB`;
  }
  if (value >= mb) {
    return `${(value / mb).toFixed(0)} MB`;
  }
  return `${value} B`;
}

function formatUpdated(value?: string): string | undefined {
  if (!value) {
    return undefined;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}
