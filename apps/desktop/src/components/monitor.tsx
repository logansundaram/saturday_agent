import { useEffect, useMemo, useState } from "react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

export default function Monitor() {
  const [open, setOpen] = useState(true);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [ipcAvailable, setIpcAvailable] = useState(true);

  useEffect(() => {
    if (!window.system?.subscribe) {
      setIpcAvailable(false);
      return;
    }

    const unsubscribe = window.system.subscribe((payload) => {
      setMetrics(payload);
    });

    return () => {
      unsubscribe?.();
    };
  }, []);

  const updatedAt = metrics?.timestamp
    ? new Date(metrics.timestamp).toLocaleTimeString()
    : "Waiting for metrics";

  const cpuUsage = metrics?.cpu.usage;
  const memoryUsage = metrics?.memory.usage;
  const gpuUsage = metrics?.gpu?.usage;

  const cpuBar = usageColor(cpuUsage);
  const memBar = usageColor(memoryUsage);
  const gpuBar = usageColor(gpuUsage);

  const backendStatus = metrics?.backend;

  const backendBadges = useMemo(() => {
    if (!backendStatus) {
      return null;
    }

    return (
      <div className="flex flex-wrap gap-2">
        <Badge className={statusBadgeClass(backendStatus.api)}>
          API {backendStatus.api}
        </Badge>
        <Badge className={statusBadgeClass(backendStatus.ollama)}>
          Ollama {backendStatus.ollama}
        </Badge>
      </div>
    );
  }, [backendStatus]);

  return (
    <div className="fixed right-0 top-0 z-40 h-full w-64 border-l border-subtle bg-root">
      <div className="flex items-center justify-between border-b border-subtle px-3 py-2">
        <div>
          <div className="text-sm text-primary">System Monitor</div>
          <div className="text-[10px] text-secondary">{updatedAt}</div>
        </div>
        <Button
          type="button"
          className="h-7 w-7 rounded-full border border-subtle bg-transparent text-sm text-secondary hover:text-primary"
          onClick={() => setOpen((prev) => !prev)}
        >
          {open ? "–" : "+"}
        </Button>
      </div>

      {open ? (
        <div className="h-[calc(100%-44px)] overflow-y-auto px-3 py-3">
          {!ipcAvailable ? (
            <div className="mb-3 rounded-xl border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-100">
              IPC bridge unavailable. Restart the app to reconnect system metrics.
            </div>
          ) : null}

          <div className="space-y-3">
            <section className="rounded-xl border border-subtle bg-[#0b0b10] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-secondary">
                  CPU
                </span>
                <span className="text-sm text-primary">
                  {formatPercent(cpuUsage)}
                </span>
              </div>
              <div className="mt-1 text-sm text-primary">
                {metrics?.cpu.model ?? "—"}
              </div>
              <div className="text-[11px] text-secondary">
                {metrics?.cpu.cores ? `${metrics.cpu.cores} cores` : "—"}
              </div>
              <ProgressBar value={cpuUsage} colorClass={cpuBar} />
            </section>

            <section className="rounded-xl border border-subtle bg-[#0b0b10] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-secondary">
                  Memory
                </span>
                <span className="text-sm text-primary">
                  {formatPercent(memoryUsage)}
                </span>
              </div>
              <div className="mt-1 text-sm text-primary">
                {formatBytes(metrics?.memory.used)} / {formatBytes(metrics?.memory.total)}
              </div>
              <div className="text-[11px] text-secondary">Used / Total</div>
              <ProgressBar value={memoryUsage} colorClass={memBar} />
            </section>

            <section className="rounded-xl border border-subtle bg-[#0b0b10] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-secondary">
                  GPU
                </span>
                <span className="text-sm text-primary">
                  {formatPercent(gpuUsage)}
                </span>
              </div>
              <div className="mt-1 text-sm text-primary">
                {metrics?.gpu?.model ?? "N/A"}
              </div>
              <div className="text-[11px] text-secondary">
                {metrics?.gpu
                  ? `${formatBytes(metrics.gpu.vramUsed)} / ${formatBytes(
                      metrics.gpu.vramTotal
                    )}`
                  : "VRAM N/A"}
              </div>
              <ProgressBar value={gpuUsage} colorClass={gpuBar} />
            </section>

            <section className="rounded-xl border border-subtle bg-[#0b0b10] p-3">
              <div className="text-[11px] uppercase tracking-wide text-secondary">
                System
              </div>
              <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-secondary">
                    Platform
                  </div>
                  <div className="text-primary">
                    {metrics?.system.platform ?? "—"}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-secondary">
                    Arch
                  </div>
                  <div className="text-primary">{metrics?.system.arch ?? "—"}</div>
                </div>
                <div className="col-span-2">
                  <div className="text-[10px] uppercase tracking-wide text-secondary">
                    Uptime
                  </div>
                  <div className="text-primary">
                    {formatUptime(metrics?.system.uptime)}
                  </div>
                </div>
              </div>
            </section>

            <section className="rounded-xl border border-subtle bg-[#0b0b10] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-secondary">
                  Backend
                </span>
                {backendBadges}
              </div>
              <div className="mt-2 text-sm text-primary">
                {backendStatus?.model ? `Model: ${backendStatus.model}` : "Model: N/A"}
              </div>
              <div className="text-[11px] text-secondary">
                FastAPI & Ollama health checks
              </div>
            </section>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ProgressBar({
  value,
  colorClass,
}: {
  value: number | undefined;
  colorClass: string;
}) {
  const width = typeof value === "number" ? Math.min(100, Math.max(0, value)) : 0;
  return (
    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#1b1b1f]">
      <div
        className={`h-full transition-all ${colorClass}`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function formatPercent(value?: number): string {
  if (value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `${Math.round(value)}%`;
}

function formatBytes(value?: number): string {
  if (value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  const gb = 1024 * 1024 * 1024;
  const mb = 1024 * 1024;
  if (value >= gb) {
    return `${(value / gb).toFixed(1)} GB`;
  }
  if (value >= mb) {
    return `${Math.round(value / mb)} MB`;
  }
  return `${Math.round(value)} B`;
}

function formatUptime(value?: number): string {
  if (value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  const seconds = Math.max(0, value);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

function usageColor(value?: number): string {
  if (value === undefined) {
    return "bg-[#2a2638]";
  }
  if (value >= 85) {
    return "bg-[#d4af37]";
  }
  if (value >= 65) {
    return "bg-[#8b5cf6]";
  }
  return "bg-[#5b21b6]";
}

function statusBadgeClass(status: "ok" | "down"): string {
  return status === "ok"
    ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
    : "border-rose-400/40 bg-rose-500/10 text-rose-200";
}
