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

type QdrantStatus = {
  running: boolean;
  port: number | null;
  storagePath: string;
  pid?: number;
  error?: string;
  lastHealthCheckAt?: string;
};

interface Window {
  system?: {
    subscribe: (callback: (metrics: SystemMetrics) => void) => () => void;
  };
  qdrant?: {
    status: () => Promise<QdrantStatus>;
    restart: () => Promise<QdrantStatus>;
    stop: () => Promise<QdrantStatus>;
    subscribe: (callback: (status: QdrantStatus) => void) => () => void;
  };
}
