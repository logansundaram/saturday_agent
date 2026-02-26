import type { ReplayDiagnostic } from "@saturday/shared/run";
import type { RerunFromStateRequest } from "../../lib/api";

export type RerunFromStateBuildInput = {
  stepIndex: number;
  editorText: string;
  resume?: "next" | "same";
  sandbox?: boolean | null;
};

export type RerunFromStateBuildResult = {
  request: RerunFromStateRequest | null;
  parseError: string | null;
};

export type RerunDiagnostic = ReplayDiagnostic;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function stringifyRerunStateForEditor(
  value: unknown,
  fallback: string = "{}"
): string {
  if (value === undefined) {
    return fallback;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return fallback;
  }
}

export function buildRerunFromStateRequest(
  input: RerunFromStateBuildInput
): RerunFromStateBuildResult {
  const trimmed = input.editorText.trim();
  const raw = trimmed.length > 0 ? trimmed : "{}";
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!isPlainObject(parsed)) {
      return {
        request: null,
        parseError: "State JSON must be a JSON object.",
      };
    }

    const request: RerunFromStateRequest = {
      step_index: input.stepIndex,
      state_json: parsed,
      resume: input.resume ?? "next",
    };
    if (typeof input.sandbox === "boolean") {
      request.sandbox = input.sandbox;
    }
    return {
      request,
      parseError: null,
    };
  } catch (error) {
    return {
      request: null,
      parseError:
        error instanceof Error ? error.message : "Invalid JSON payload.",
    };
  }
}

export function normalizeRerunDiagnostics(value: unknown): RerunDiagnostic[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: RerunDiagnostic[] = [];
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
      message: String(source.message ?? "Unknown rerun validation issue."),
      path: source.path == null ? null : String(source.path),
      expected: source.expected == null ? null : String(source.expected),
      actual: source.actual == null ? null : String(source.actual),
    });
  });
  return normalized;
}

export function hasRerunDiagnosticErrors(
  diagnostics: RerunDiagnostic[]
): boolean {
  return diagnostics.some((item) => item.severity === "error");
}
