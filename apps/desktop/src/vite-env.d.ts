/// <reference types="vite/client" />

type SystemMetrics = {
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

interface Window {
  system?: {
    subscribe: (callback: (metrics: SystemMetrics) => void) => () => void;
  };
}
