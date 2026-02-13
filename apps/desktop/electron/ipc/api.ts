import { dialog, ipcMain } from "electron";
import { stat } from "node:fs/promises";
import path from "node:path";

const API_BASE_URL = process.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ToolInvokeResponse = {
  run_id: string;
  tool_id: string;
  status: string;
  output: unknown;
  steps: Array<Record<string, unknown>>;
};

async function invokeTool(
  toolId: string,
  input: Record<string, unknown>,
  context?: Record<string, unknown>
): Promise<ToolInvokeResponse> {
  const response = await fetch(`${API_BASE_URL}/tools/invoke`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      tool_id: toolId,
      input,
      context: context ?? {},
    }),
  });

  if (!response.ok) {
    let detail = `Tool invoke failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        detail = `${detail}: ${payload.detail}`;
      }
    } catch {
      // Ignore JSON parse errors and return base message.
    }
    throw new Error(detail);
  }

  return (await response.json()) as ToolInvokeResponse;
}

function pickIngestOptions(
  value: unknown
): Record<string, unknown> {
  const payload = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const output: Record<string, unknown> = {};
  const passthroughKeys = [
    "collection",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "index_to_qdrant",
  ];
  for (const key of passthroughKeys) {
    if (payload[key] !== undefined) {
      output[key] = payload[key];
    }
  }
  return output;
}

export function registerDocsIpcHandlers(): void {
  ipcMain.removeHandler("docs:list");
  ipcMain.removeHandler("docs:importPdf");
  ipcMain.removeHandler("docs:delete");

  ipcMain.handle("docs:list", async (_event, rawInput?: unknown) => {
    const payload = rawInput && typeof rawInput === "object"
      ? (rawInput as Record<string, unknown>)
      : {};
    const input: Record<string, unknown> = {};
    if (typeof payload.status === "string" && payload.status.trim()) {
      input.status = payload.status.trim();
    }
    return invokeTool("rag.list_docs", input);
  });

  ipcMain.handle("docs:importPdf", async (_event, rawOptions?: unknown) => {
    const selected = await dialog.showOpenDialog({
      title: "Import PDF documents",
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "PDF Documents", extensions: ["pdf"] }],
    });

    if (selected.canceled || selected.filePaths.length === 0) {
      return {
        ok: true,
        cancelled: true,
        results: [] as Array<Record<string, unknown>>,
      };
    }

    const ingestOptions = pickIngestOptions(rawOptions);
    const results: Array<Record<string, unknown>> = [];
    for (const candidatePath of selected.filePaths) {
      const resolvedPath = path.resolve(candidatePath);
      if (path.extname(resolvedPath).toLowerCase() !== ".pdf") {
        continue;
      }
      try {
        const metadata = await stat(resolvedPath);
        if (!metadata.isFile()) {
          continue;
        }
      } catch {
        continue;
      }

      try {
        const run = await invokeTool("rag.ingest_pdf", {
          ...ingestOptions,
          file_path: resolvedPath,
        });
        results.push({
          file_path: resolvedPath,
          ...run,
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown ingest error.";
        results.push({
          file_path: resolvedPath,
          run_id: "",
          tool_id: "rag.ingest_pdf",
          status: "error",
          output: {
            ok: false,
            error: {
              message,
            },
          },
          steps: [],
        });
      }
    }

    return {
      ok: results.every((item) => String(item.status || "error") === "ok"),
      cancelled: false,
      results,
    };
  });

  ipcMain.handle("docs:delete", async (_event, rawInput?: unknown) => {
    const payload = rawInput && typeof rawInput === "object"
      ? (rawInput as Record<string, unknown>)
      : {};
    const input: Record<string, unknown> = {};
    if (typeof payload.doc_id === "string" && payload.doc_id.trim()) {
      input.doc_id = payload.doc_id.trim();
    }
    if (typeof payload.collection === "string" && payload.collection.trim()) {
      input.collection = payload.collection.trim();
    }
    if (payload.delete_from_qdrant !== undefined) {
      input.delete_from_qdrant = payload.delete_from_qdrant;
    }
    return invokeTool("rag.delete_doc", input);
  });
}
