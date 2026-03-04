import { ipcMain } from "electron";
import { mkdir, writeFile } from "node:fs/promises";
import { appendFileSync } from "node:fs";
import path from "node:path";

const TRACE_PATH = (process.env.SATURDAY_USAGE_TRACE_PATH ?? "").trim();

type TracePayload = Record<string, unknown>;

function traceEnabled(): boolean {
  return TRACE_PATH.length > 0;
}

export function traceDesktopEvent(
  eventType: string,
  payload: TracePayload = {}
): void {
  if (!traceEnabled()) {
    return;
  }

  const entry = {
    ts: new Date().toISOString(),
    source: "desktop",
    event_type: eventType.trim() || "unknown",
    ...payload,
  };

  mkdir(path.dirname(TRACE_PATH), { recursive: true })
    .then(() => {
      appendFileSync(TRACE_PATH, `${JSON.stringify(entry)}\n`, "utf-8");
    })
    .catch(() => {
      // Ignore tracing failures in normal app flow.
    });
}

export function registerUsageTraceIpc(): void {
  ipcMain.removeAllListeners("usage-trace:event");
  ipcMain.on("usage-trace:event", (_event, payload: unknown) => {
    if (!payload || typeof payload !== "object") {
      return;
    }
    const record = payload as TracePayload;
    traceDesktopEvent(
      typeof record.event_type === "string" ? record.event_type : "renderer.event",
      record
    );
  });
}

export async function writeSmokeReport(
  payload: Record<string, unknown>
): Promise<void> {
  const reportPath = (process.env.SATURDAY_SMOKE_REPORT_PATH ?? "").trim();
  if (!reportPath) {
    return;
  }
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, JSON.stringify(payload, null, 2), "utf-8");
}
