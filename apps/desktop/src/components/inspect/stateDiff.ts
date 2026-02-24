import type {
  ReplayDiagnostic,
  ReplayPatchMode,
  ReplayRequest,
} from "@saturday/shared/run";

export type StateDiffKind = "added" | "removed" | "changed";

export type StateDiffEntry = {
  path: string;
  kind: StateDiffKind;
  before?: unknown;
  after?: unknown;
};

export type ReplayRequestBuildInput = {
  fromStepId: string;
  patchText: string;
  patchMode: ReplayPatchMode;
  sandbox: boolean;
  baseState: "pre" | "post";
  replayThisStep: boolean;
};

export type ReplayRequestBuildResult = {
  request: ReplayRequest | null;
  parseError: string | null;
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function joinPath(parent: string, key: string): string {
  if (!parent) {
    return key;
  }
  return `${parent}.${key}`;
}

export function diffState(before: unknown, after: unknown): StateDiffEntry[] {
  const changes: StateDiffEntry[] = [];

  const walk = (left: unknown, right: unknown, path: string): void => {
    if (isPlainObject(left) && isPlainObject(right)) {
      const keys = new Set<string>([
        ...Object.keys(left).map((item) => String(item)),
        ...Object.keys(right).map((item) => String(item)),
      ]);
      Array.from(keys)
        .sort((a, b) => a.localeCompare(b))
        .forEach((key) => {
          const nextPath = joinPath(path, key);
          if (!(key in left)) {
            changes.push({
              path: nextPath,
              kind: "added",
              after: right[key],
            });
            return;
          }
          if (!(key in right)) {
            changes.push({
              path: nextPath,
              kind: "removed",
              before: left[key],
            });
            return;
          }
          walk(left[key], right[key], nextPath);
        });
      return;
    }

    if (Array.isArray(left) && Array.isArray(right)) {
      if (JSON.stringify(left) !== JSON.stringify(right)) {
        changes.push({
          path: path || "$",
          kind: "changed",
          before: left,
          after: right,
        });
      }
      return;
    }

    if (!Object.is(left, right)) {
      changes.push({
        path: path || "$",
        kind: "changed",
        before: left,
        after: right,
      });
    }
  };

  walk(before, after, "");
  return changes;
}

export function stringifyJsonForEditor(value: unknown, fallback: string = "{}"): string {
  if (value === undefined) {
    return fallback;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return fallback;
  }
}

export function buildReplayRequestFromEditor(
  input: ReplayRequestBuildInput
): ReplayRequestBuildResult {
  const trimmed = input.patchText.trim();
  const defaultPayload: unknown = input.patchMode === "jsonpatch" ? [] : {};
  const rawPayload = trimmed.length > 0 ? trimmed : JSON.stringify(defaultPayload);
  try {
    const parsedPatch = JSON.parse(rawPayload) as unknown;
    return {
      request: {
        from_step_id: input.fromStepId,
        state_patch: parsedPatch,
        patch_mode: input.patchMode,
        sandbox: input.sandbox,
        base_state: input.baseState,
        replay_this_step: input.replayThisStep,
      },
      parseError: null,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid JSON payload.";
    return {
      request: null,
      parseError: message,
    };
  }
}

export function normalizeReplayDiagnostics(value: unknown): ReplayDiagnostic[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: ReplayDiagnostic[] = [];
  value.forEach((item, index) => {
    if (!item || typeof item !== "object") {
      normalized.push({
        code: "UNKNOWN",
        severity: "error",
        message: `Invalid diagnostic payload at index ${index}.`,
      });
      return;
    }
    const source = item as Record<string, unknown>;
    const severity =
      source.severity === "warning" || source.severity === "info"
        ? source.severity
        : "error";
    normalized.push({
      code: String(source.code ?? "UNKNOWN"),
      severity,
      message: String(source.message ?? "Unknown replay validation issue."),
      path: source.path == null ? null : String(source.path),
      expected: source.expected == null ? null : String(source.expected),
      actual: source.actual == null ? null : String(source.actual),
    });
  });
  return normalized;
}

export function hasReplayDiagnosticErrors(diagnostics: ReplayDiagnostic[]): boolean {
  return diagnostics.some((item) => item.severity === "error");
}
