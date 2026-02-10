import { useMemo, useState } from "react";
import { Button } from "../ui/button";

type JsonViewProps = {
  title: string;
  value: unknown;
  initiallyCollapsed?: boolean;
};

const COLLAPSE_THRESHOLD = 1800;

function stringifyValue(value: unknown): string {
  if (value === undefined) {
    return "undefined";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export default function JsonView({
  title,
  value,
  initiallyCollapsed = false,
}: JsonViewProps) {
  const serialized = useMemo(() => stringifyValue(value), [value]);
  const isLarge = serialized.length > COLLAPSE_THRESHOLD;
  const [expanded, setExpanded] = useState<boolean>(() =>
    initiallyCollapsed ? false : !isLarge
  );

  const displayValue =
    expanded || !isLarge
      ? serialized
      : `${serialized.slice(0, COLLAPSE_THRESHOLD)}\n... (truncated)`;

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(serialized);
    } catch {
      // Non-blocking if clipboard is unavailable.
    }
  };

  return (
    <div className="rounded-2xl border border-subtle bg-[#0b0b10] p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-primary">{title}</h3>
        <div className="flex items-center gap-2">
          {isLarge ? (
            <Button
              type="button"
              className="h-7 rounded-full border border-subtle bg-transparent px-2.5 text-xs text-secondary hover:text-primary"
              onClick={() => setExpanded((prev) => !prev)}
            >
              {expanded ? "Collapse" : "Expand"}
            </Button>
          ) : null}
          <Button
            type="button"
            className="h-7 rounded-full border border-subtle bg-transparent px-2.5 text-xs text-secondary hover:text-primary"
            onClick={() => void onCopy()}
          >
            Copy
          </Button>
        </div>
      </div>
      <pre className="max-h-[28rem] overflow-auto rounded-xl border border-subtle bg-[#08080c] p-3 text-xs leading-relaxed text-secondary">
        {displayValue || "{}"}
      </pre>
    </div>
  );
}

