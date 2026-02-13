import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";

type LocalDoc = {
  id: string;
  filename: string;
  stored_path: string;
  bytes: number;
  sha256: string;
  created_at: string;
  updated_at: string;
  status: string;
  ingested_at?: string;
  error_message?: string;
  collection: string;
  chunk_count?: number;
};

type ToolEnvelope = {
  ok?: boolean;
  data?: unknown;
  error?: { message?: string };
};

type ToolRun = {
  run_id?: string;
  status?: string;
  output?: ToolEnvelope;
};

type ImportResult = ToolRun & { file_path?: string };

type ImportResponse = {
  ok?: boolean;
  cancelled?: boolean;
  results?: ImportResult[];
};

const formatBytes = (value: number): string => {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

const statusBadgeClass = (status: string): string => {
  const normalized = status.trim().toLowerCase();
  if (normalized === "ingested") {
    return "border-emerald-400/40 bg-emerald-500/10 text-emerald-100";
  }
  if (normalized === "ingesting") {
    return "border-sky-400/40 bg-sky-500/10 text-sky-100";
  }
  if (normalized === "error") {
    return "border-rose-400/40 bg-rose-500/10 text-rose-100";
  }
  return "border-white/10 bg-white/5 text-secondary";
};

function unwrapToolRun(run: ToolRun): { ok: boolean; data?: unknown; message?: string } {
  const output = run.output;
  if (!output || typeof output !== "object") {
    return { ok: false, message: "Tool response was empty." };
  }

  const envelopeOk = Boolean(output.ok);
  if (!envelopeOk) {
    return {
      ok: false,
      message: output.error?.message ?? "Tool invocation failed.",
    };
  }

  return { ok: true, data: output.data };
}

export default function LocalDocsPage() {
  const [docs, setDocs] = useState<LocalDoc[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isImporting, setIsImporting] = useState<boolean>(false);
  const [deletingDocId, setDeletingDocId] = useState<string>("");
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  const backendAvailable = Boolean(window.ipcRenderer?.invoke);

  const loadDocs = useCallback(async () => {
    if (!backendAvailable) {
      setError("IPC bridge unavailable. Restart the desktop app.");
      return;
    }

    setIsLoading(true);
    setError("");
    try {
      const run = (await window.ipcRenderer.invoke("docs:list", {})) as ToolRun;
      const parsed = unwrapToolRun(run);
      if (!parsed.ok) {
        setDocs([]);
        setError(parsed.message ?? "Unable to list local docs.");
        return;
      }

      const payload = parsed.data as { ok?: boolean; docs?: LocalDoc[]; error?: { message?: string } };
      if (!payload || payload.ok === false) {
        setDocs([]);
        setError(payload?.error?.message ?? "Unable to list local docs.");
        return;
      }

      const nextDocs = Array.isArray(payload.docs) ? payload.docs : [];
      setDocs(nextDocs);
      setMessage(`Loaded ${nextDocs.length} document(s).`);
    } catch (invokeError) {
      const detail =
        invokeError instanceof Error ? invokeError.message : "Unable to list local docs.";
      setError(detail);
      setDocs([]);
    } finally {
      setIsLoading(false);
    }
  }, [backendAvailable]);

  const importPdfs = useCallback(async () => {
    if (!backendAvailable) {
      setError("IPC bridge unavailable. Restart the desktop app.");
      return;
    }

    setIsImporting(true);
    setMessage("");
    setError("");

    try {
      const result = (await window.ipcRenderer.invoke("docs:importPdf", {})) as ImportResponse;
      if (result.cancelled) {
        setMessage("PDF import cancelled.");
        return;
      }

      const runs = Array.isArray(result.results) ? result.results : [];
      const successCount = runs.filter((run) => unwrapToolRun(run).ok).length;
      const failureCount = runs.length - successCount;
      if (failureCount > 0) {
        const firstFailure = runs.find((run) => !unwrapToolRun(run).ok);
        setError(
          unwrapToolRun(firstFailure ?? {}).message ??
            `${failureCount} PDF(s) failed to ingest.`
        );
      }
      setMessage(`Ingested ${successCount}/${runs.length} PDF(s).`);
      await loadDocs();
    } catch (invokeError) {
      const detail =
        invokeError instanceof Error ? invokeError.message : "Unable to import PDFs.";
      setError(detail);
    } finally {
      setIsImporting(false);
    }
  }, [backendAvailable, loadDocs]);

  const deleteDoc = useCallback(
    async (docId: string) => {
      if (!backendAvailable) {
        setError("IPC bridge unavailable. Restart the desktop app.");
        return;
      }

      setDeletingDocId(docId);
      setError("");
      setMessage("");
      try {
        const run = (await window.ipcRenderer.invoke("docs:delete", {
          doc_id: docId,
        })) as ToolRun;
        const parsed = unwrapToolRun(run);
        if (!parsed.ok) {
          setError(parsed.message ?? "Unable to delete document.");
          return;
        }
        const payload = parsed.data as {
          ok?: boolean;
          deleted?: boolean;
          error?: { message?: string };
        };
        if (!payload?.ok) {
          setError(payload?.error?.message ?? "Unable to delete document.");
          return;
        }
        setMessage(payload.deleted ? "Document deleted." : "Document not found.");
        await loadDocs();
      } catch (invokeError) {
        const detail =
          invokeError instanceof Error ? invokeError.message : "Unable to delete document.";
        setError(detail);
      } finally {
        setDeletingDocId("");
      }
    },
    [backendAvailable, loadDocs]
  );

  const sortedDocs = useMemo(() => {
    return [...docs].sort((left, right) =>
      (right.updated_at || "").localeCompare(left.updated_at || "")
    );
  }, [docs]);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  return (
    <div className="bg-panel section-base pr-64 relative flex h-screen min-h-0 flex-col">
      <div className="section-hero pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="section-header text-5xl">Local Docs</h1>
            <p className="section-framer text-secondary">
              Ingest local PDFs through `rag.ingest_pdf` and manage document lifecycle.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              className="h-9 rounded-full border border-subtle bg-transparent px-4 text-sm text-secondary hover:text-primary"
              onClick={() => void loadDocs()}
              disabled={isLoading || isImporting}
            >
              {isLoading ? "Refreshing..." : "Refresh"}
            </Button>
            <Button
              type="button"
              className="h-9 rounded-full border border-sky-400/30 bg-sky-500/15 px-4 text-sm text-sky-100 hover:bg-sky-500/20"
              onClick={() => void importPdfs()}
              disabled={isImporting}
            >
              {isImporting ? "Importing..." : "Import PDFs"}
            </Button>
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-8">
        {message ? (
          <div className="mb-4 rounded-xl border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-100">
            {message}
          </div>
        ) : null}
        {error ? (
          <div className="mb-4 rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-2 text-sm text-rose-100">
            {error}
          </div>
        ) : null}

        {!backendAvailable ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            IPC bridge unavailable. Restart the desktop app.
          </div>
        ) : sortedDocs.length === 0 ? (
          <div className="rounded-2xl border border-subtle bg-[#0b0b10] px-6 py-8 text-sm text-secondary">
            No local docs yet. Use <code>Import PDFs</code> to ingest files into RAG.
          </div>
        ) : (
          <div className="space-y-3">
            {sortedDocs.map((doc) => (
              <div
                key={doc.id}
                className="rounded-2xl border border-subtle bg-[#0b0b10] p-4"
              >
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-base font-semibold text-primary">{doc.filename}</div>
                    <div className="text-xs text-secondary">
                      {doc.id} • {formatBytes(doc.bytes)} • {doc.collection}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className={`border ${statusBadgeClass(doc.status)}`}>
                      {doc.status}
                    </Badge>
                    <Button
                      type="button"
                      className="h-8 rounded-full border border-rose-400/30 bg-rose-500/10 px-3 text-xs text-rose-100 hover:bg-rose-500/20 disabled:opacity-60"
                      onClick={() => void deleteDoc(doc.id)}
                      disabled={deletingDocId === doc.id}
                    >
                      {deletingDocId === doc.id ? "Deleting..." : "Delete"}
                    </Button>
                  </div>
                </div>

                <div className="grid gap-2 text-xs text-secondary md:grid-cols-2">
                  <div>Chunks: {Number(doc.chunk_count ?? 0)}</div>
                  <div>Ingested: {doc.ingested_at || "—"}</div>
                  <div className="md:col-span-2 break-all">Stored: {doc.stored_path}</div>
                  {doc.error_message ? (
                    <div className="md:col-span-2 text-rose-200">
                      Error: {doc.error_message}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
