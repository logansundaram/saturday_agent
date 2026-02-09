import { BrowserWindow } from "electron";
import siModule from "systeminformation";

export type SystemMetrics = {
  cpu: { model: string; usage: number; cores: number };
  memory: { total: number; used: number; usage: number };
  gpu?: { model: string; vramTotal?: number; vramUsed?: number; usage?: number } | null;
  system: { platform: "win" | "mac" | "linux"; arch: string; uptime: number };
  backend: {
    api: "ok" | "down";
    ollama: "ok" | "down";
    model?: string;
  };
  timestamp: number;
};

const POLL_INTERVAL_MS = Number(process.env.SYSTEM_METRICS_INTERVAL_MS ?? 1000);
const API_BASE_URL = process.env.SATURDAY_API_URL ?? "http://localhost:8000";
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";

const si =
  typeof (siModule as unknown as { cpu?: unknown }).cpu === "function"
    ? (siModule as typeof siModule)
    : ((siModule as unknown as { default?: typeof siModule }).default ?? siModule);

let intervalId: NodeJS.Timeout | null = null;
let inFlight = false;

export function startMetricsPolling(): void {
  if (intervalId) {
    return;
  }

  void pollOnce();
  intervalId = setInterval(() => {
    void pollOnce();
  }, POLL_INTERVAL_MS);
}

export function stopMetricsPolling(): void {
  if (!intervalId) {
    return;
  }
  clearInterval(intervalId);
  intervalId = null;
}

async function pollOnce(): Promise<void> {
  if (inFlight) {
    return;
  }
  inFlight = true;

  try {
    const metrics = await collectMetrics();
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send("system:metrics", metrics);
      }
    }
  } catch {
    // Best-effort: ignore individual polling failures.
  } finally {
    inFlight = false;
  }
}

async function collectMetrics(): Promise<SystemMetrics> {
  const [cpuInfo, loadInfo, memInfo, graphicsInfo, osInfo, timeInfo, backend] =
    await Promise.all([
      safe(() => si.cpu(), { brand: "CPU", cores: 0, physicalCores: 0 } as Awaited<ReturnType<typeof si.cpu>>),
      safe(() => si.currentLoad(), { currentLoad: 0 } as Awaited<ReturnType<typeof si.currentLoad>>),
      safe(() => si.mem(), { total: 0, used: 0, active: 0 } as Awaited<ReturnType<typeof si.mem>>),
      safe(() => si.graphics(), { controllers: [] } as Awaited<ReturnType<typeof si.graphics>>),
      safe(() => si.osInfo(), { platform: process.platform, arch: process.arch } as Awaited<ReturnType<typeof si.osInfo>>),
      safe(() => si.time(), { uptime: 0 } as Awaited<ReturnType<typeof si.time>>),
      safe(() => fetchBackendHealth(), { api: "down", ollama: "down" } as SystemMetrics["backend"]),
    ]);

  const cpuUsage = clampPercent(loadInfo.currentLoad ?? 0);
  const memoryUsed = memInfo.used ?? memInfo.active ?? 0;
  const memoryUsage = memInfo.total ? (memoryUsed / memInfo.total) * 100 : 0;

  const gpu = normalizeGpu(graphicsInfo);
  const platform = normalizePlatform(osInfo.platform);

  return {
    cpu: {
      model: cpuInfo.brand || `${cpuInfo.manufacturer} ${cpuInfo.speed} GHz` || "CPU",
      usage: cpuUsage,
      cores: cpuInfo.cores || cpuInfo.physicalCores || 0,
    },
    memory: {
      total: memInfo.total ?? 0,
      used: memoryUsed,
      usage: clampPercent(memoryUsage),
    },
    gpu,
    system: {
      platform,
      arch: osInfo.arch || process.arch,
      uptime: timeInfo.uptime ?? 0,
    },
    backend,
    timestamp: Date.now(),
  };
}

function normalizeGpu(graphicsInfo: Awaited<ReturnType<typeof si.graphics>>):
  | SystemMetrics["gpu"]
  | null {
  const controller = graphicsInfo.controllers?.[0];
  if (!controller) {
    return null;
  }

  const record = controller as Record<string, unknown>;
  const vramTotalRaw = getNumber(record.vram) ?? getNumber(record.memoryTotal);
  const vramUsedRaw = getNumber(record.vramUsed) ?? getNumber(record.memoryUsed);
  const utilization = getNumber(record.utilizationGpu);

  const vramTotal = normalizeBytes(vramTotalRaw);
  const vramUsed = normalizeBytes(vramUsedRaw);

  const usage =
    utilization !== undefined
      ? clampPercent(utilization)
      : vramTotal && vramUsed
      ? clampPercent((vramUsed / vramTotal) * 100)
      : undefined;

  return {
    model: controller.model || controller.vendor || "GPU",
    vramTotal,
    vramUsed,
    usage,
  };
}

async function fetchBackendHealth(): Promise<SystemMetrics["backend"]> {
  const [apiResult, ollamaResult] = await Promise.allSettled([
    fetchJsonWithTimeout(`${API_BASE_URL}/health`, 1200),
    fetchJsonWithTimeout(`${OLLAMA_BASE_URL}/api/tags`, 1200),
  ]);

  let apiStatus: "ok" | "down" = "down";
  let ollamaStatus: "ok" | "down" = "down";
  let model: string | undefined;

  if (apiResult.status === "fulfilled" && apiResult.value.ok) {
    apiStatus = "ok";
    const data = apiResult.value.data as
      | { model_default?: string; ollama?: "ok" | "down" }
      | undefined;
    if (data?.model_default) {
      model = data.model_default;
    }
    if (data?.ollama === "ok") {
      ollamaStatus = "ok";
    }
  }

  if (ollamaResult.status === "fulfilled" && ollamaResult.value.ok) {
    ollamaStatus = "ok";
    const data = ollamaResult.value.data as
      | { models?: Array<{ name?: string }> }
      | undefined;
    const firstModel = data?.models?.[0]?.name;
    if (!model && firstModel) {
      model = firstModel;
    }
  }

  return {
    api: apiStatus,
    ollama: ollamaStatus,
    model,
  };
}

async function fetchJsonWithTimeout(
  url: string,
  timeoutMs: number
): Promise<{ ok: boolean; status: number; data?: unknown }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { signal: controller.signal });
    let data: unknown = undefined;
    if (response.ok) {
      try {
        data = await response.json();
      } catch {
        data = undefined;
      }
    }
    return { ok: response.ok, status: response.status, data };
  } finally {
    clearTimeout(timeoutId);
  }
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 10) / 10));
}

function getNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeBytes(value?: number): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value < 1024 * 1024) {
    return Math.round(value * 1024 * 1024);
  }
  return Math.round(value);
}

async function safe<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await fn();
  } catch {
    return fallback;
  }
}

function normalizePlatform(platform: string | undefined): "win" | "mac" | "linux" {
  if (platform === "darwin") {
    return "mac";
  }
  if (platform === "win32") {
    return "win";
  }
  return "linux";
}
