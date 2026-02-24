import { app } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream, existsSync, type WriteStream } from "node:fs";
import { chmod, mkdir } from "node:fs/promises";
import net from "node:net";
import path from "node:path";

const DEFAULT_QDRANT_PORT = 6333;
const DEFAULT_QDRANT_GRPC_PORT = 6334;
const READINESS_TIMEOUT_MS = 10_000;
const READINESS_POLL_MS = 500;
const HEALTH_INTERVAL_MS = 5_000;
const PROCESS_STOP_TIMEOUT_MS = 2_500;

export type QdrantStatus = {
  running: boolean;
  port: number | null;
  storagePath: string;
  pid?: number;
  error?: string;
  lastHealthCheckAt?: string;
};

type StatusListener = (status: QdrantStatus) => void;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function platformFolderName(): string {
  if (process.platform === "darwin" && process.arch === "arm64") {
    return "macos-arm64";
  }
  if (process.platform === "darwin" && process.arch === "x64") {
    return "macos-x64";
  }
  if (process.platform === "win32" && process.arch === "x64") {
    return "win-x64";
  }
  if (process.platform === "linux" && process.arch === "x64") {
    return "linux-x64";
  }
  throw new Error(
    `Unsupported platform/arch for embedded Qdrant: ${process.platform}/${process.arch}`
  );
}

export class QdrantManager {
  private processRef: ChildProcess | null = null;
  private logStream: WriteStream | null = null;
  private startPromise: Promise<QdrantStatus> | null = null;
  private stopPromise: Promise<QdrantStatus> | null = null;
  private healthInterval: NodeJS.Timeout | null = null;
  private listeners = new Set<StatusListener>();
  private stoppingProcess = false;

  private status: QdrantStatus = {
    running: false,
    port: null,
    storagePath: this.resolveStoragePath(),
  };

  subscribe(listener: StatusListener): () => void {
    this.listeners.add(listener);
    listener(this.getStatus());
    return () => {
      this.listeners.delete(listener);
    };
  }

  getStatus(): QdrantStatus {
    return { ...this.status };
  }

  getUrl(): string | null {
    if (!this.status.port) {
      return null;
    }
    return `http://127.0.0.1:${this.status.port}`;
  }

  setStatusError(error?: string): void {
    this.patchStatus({ error: error?.trim() || undefined });
  }

  async start(): Promise<QdrantStatus> {
    if (this.status.running && this.processRef) {
      return this.getStatus();
    }
    if (this.startPromise) {
      return this.startPromise;
    }
    if (this.stopPromise) {
      await this.stopPromise;
    }

    this.startPromise = this.startInternal()
      .catch((error: unknown) => {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to start embedded Qdrant.";
        this.patchStatus({
          running: false,
          pid: undefined,
          error: message,
          port: this.status.port,
        });
        return this.getStatus();
      })
      .finally(() => {
        this.startPromise = null;
      });

    return this.startPromise;
  }

  async stop(): Promise<QdrantStatus> {
    if (this.stopPromise) {
      return this.stopPromise;
    }

    this.stopPromise = this.stopInternal().finally(() => {
      this.stopPromise = null;
    });

    return this.stopPromise;
  }

  async restart(): Promise<QdrantStatus> {
    await this.stop();
    return this.start();
  }

  private async startInternal(): Promise<QdrantStatus> {
    const storagePath = this.resolveStoragePath();
    const logPath = this.resolveLogPath();
    await mkdir(storagePath, { recursive: true });
    await mkdir(path.dirname(logPath), { recursive: true });

    const binaryPath = this.resolveBinaryPath();
    if (!existsSync(binaryPath)) {
      throw new Error(
        `Embedded Qdrant binary not found at '${binaryPath}'. ` +
          "Bundle resources/qdrant binaries before starting the desktop app."
      );
    }
    await this.ensureExecutable(binaryPath);

    const port = await this.choosePort();
    const grpcPort = await this.chooseGrpcPort(port);
    this.patchStatus({
      running: false,
      port,
      storagePath,
      pid: undefined,
      error: undefined,
      lastHealthCheckAt: undefined,
    });

    const child = spawn(binaryPath, [], {
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        QDRANT__STORAGE__STORAGE_PATH: storagePath,
        QDRANT__SERVICE__HTTP_PORT: String(port),
        QDRANT__SERVICE__GRPC_PORT: String(grpcPort),
      },
    });
    this.processRef = child;
    this.patchStatus({ pid: child.pid ?? undefined });
    this.attachLogging(child, logPath);
    this.attachChildLifecycle(child);

    const ready = await this.waitForReady(child, port, READINESS_TIMEOUT_MS);
    if (!ready) {
      this.stoppingProcess = true;
      try {
        await this.terminateChild(child);
      } finally {
        this.stoppingProcess = false;
      }
      if (this.processRef === child) {
        this.processRef = null;
      }
      const earlyError = this.status.error?.trim();
      if (earlyError) {
        throw new Error(earlyError);
      }
      throw new Error(
        `Embedded Qdrant did not become ready within ${READINESS_TIMEOUT_MS}ms.`
      );
    }

    this.patchStatus({
      running: true,
      error: undefined,
      lastHealthCheckAt: new Date().toISOString(),
    });
    this.startHealthChecks();
    return this.getStatus();
  }

  private async stopInternal(): Promise<QdrantStatus> {
    this.stopHealthChecks();
    const child = this.processRef;
    this.processRef = null;

    if (child) {
      this.stoppingProcess = true;
      try {
        await this.terminateChild(child);
      } finally {
        this.stoppingProcess = false;
      }
    }

    if (this.logStream) {
      this.logStream.end();
      this.logStream = null;
    }

    this.patchStatus({
      running: false,
      pid: undefined,
      port: null,
      lastHealthCheckAt: undefined,
    });
    return this.getStatus();
  }

  private startHealthChecks(): void {
    this.stopHealthChecks();
    this.healthInterval = setInterval(() => {
      void this.refreshHealth();
    }, HEALTH_INTERVAL_MS);
  }

  private stopHealthChecks(): void {
    if (!this.healthInterval) {
      return;
    }
    clearInterval(this.healthInterval);
    this.healthInterval = null;
  }

  private async refreshHealth(): Promise<void> {
    const port = this.status.port;
    if (!port || !this.processRef) {
      return;
    }
    const now = new Date().toISOString();
    const healthy = await this.probeCollections(port);
    if (!healthy && this.status.running) {
      this.patchStatus({
        lastHealthCheckAt: now,
        error: "Embedded Qdrant health probe failed.",
      });
      return;
    }
    this.patchStatus({ lastHealthCheckAt: now });
  }

  private async waitForReady(
    child: ChildProcess,
    port: number,
    timeoutMs: number
  ): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) {
        return false;
      }
      const now = new Date().toISOString();
      this.patchStatus({ lastHealthCheckAt: now });
      if (await this.probeCollections(port)) {
        return true;
      }
      await delay(READINESS_POLL_MS);
    }
    return false;
  }

  private async probeCollections(port: number): Promise<boolean> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1_200);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/collections`, {
        signal: controller.signal,
      });
      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timeout);
    }
  }

  private async terminateChild(child: ChildProcess): Promise<void> {
    if (child.exitCode !== null) {
      return;
    }

    child.kill("SIGTERM");
    const exitedAfterTerm = await this.waitForExit(child, PROCESS_STOP_TIMEOUT_MS);
    if (exitedAfterTerm) {
      return;
    }

    child.kill("SIGKILL");
    await this.waitForExit(child, PROCESS_STOP_TIMEOUT_MS);
  }

  private waitForExit(
    child: ChildProcess,
    timeoutMs: number
  ): Promise<boolean> {
    return new Promise((resolve) => {
      let resolved = false;
      const timeout = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          resolve(false);
        }
      }, timeoutMs);

      child.once("exit", () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          resolve(true);
        }
      });
    });
  }

  private attachLogging(
    child: ChildProcess,
    logPath: string
  ): void {
    if (this.logStream) {
      this.logStream.end();
      this.logStream = null;
    }
    this.logStream = createWriteStream(logPath, { flags: "a" });

    const logLine = (line: string) => {
      const timestamp = new Date().toISOString();
      this.logStream?.write(`[${timestamp}] ${line}\n`);
    };

    child.stdout?.on("data", (chunk: Buffer | string) => {
      logLine(String(chunk).trimEnd());
    });
    child.stderr?.on("data", (chunk: Buffer | string) => {
      logLine(String(chunk).trimEnd());
    });
    child.on("error", (error) => {
      logLine(`process-error: ${error.message}`);
    });
  }

  private attachChildLifecycle(child: ChildProcess): void {
    child.once("error", (error) => {
      this.patchStatus({
        running: false,
        error: `Embedded Qdrant process error: ${error.message}`,
      });
    });

    child.once("exit", (code, signal) => {
      const wasStopping = this.stoppingProcess;
      if (this.processRef === child) {
        this.processRef = null;
      }
      this.stopHealthChecks();
      this.patchStatus({
        running: false,
        pid: undefined,
        port: null,
        error:
          wasStopping
            ? this.status.error
            : this.status.error ??
          `Embedded Qdrant exited (code=${code ?? "null"}, signal=${
            signal ?? "null"
          }).`,
      });
    });
  }

  private async choosePort(): Promise<number> {
    if (await this.isPortAvailable(DEFAULT_QDRANT_PORT)) {
      return DEFAULT_QDRANT_PORT;
    }
    return this.findEphemeralPort();
  }

  private async chooseGrpcPort(httpPort: number): Promise<number> {
    const preferred = httpPort === DEFAULT_QDRANT_PORT ? DEFAULT_QDRANT_GRPC_PORT : httpPort + 1;
    if (preferred !== httpPort && (await this.isPortAvailable(preferred))) {
      return preferred;
    }

    const fallback = await this.findEphemeralPort();
    if (fallback === httpPort) {
      return this.findEphemeralPort();
    }
    return fallback;
  }

  private async isPortAvailable(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once("error", () => {
        resolve(false);
      });
      server.once("listening", () => {
        server.close(() => {
          resolve(true);
        });
      });
      server.listen(port, "127.0.0.1");
    });
  }

  private async findEphemeralPort(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.once("error", (error) => {
        reject(error);
      });
      server.listen(0, "127.0.0.1", () => {
        const address = server.address();
        if (!address || typeof address === "string") {
          server.close(() => {
            reject(new Error("Unable to allocate a free local port for Qdrant."));
          });
          return;
        }
        const selectedPort = address.port;
        server.close((error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve(selectedPort);
        });
      });
    });
  }

  private resolveStoragePath(): string {
    const userData = app.isReady()
      ? app.getPath("userData")
      : path.resolve(process.cwd(), ".saturday", "desktop");
    return path.join(userData, "qdrant", "storage");
  }

  private resolveLogPath(): string {
    const userData = app.getPath("userData");
    return path.join(userData, "logs", "qdrant.log");
  }

  private resolveBinaryPath(): string {
    const platformFolder = platformFolderName();
    const baseDir = app.isPackaged
      ? process.resourcesPath
      : path.resolve(process.env.APP_ROOT ?? path.join(__dirname, "..", ".."), "resources");

    const rawBinary = path.join(baseDir, "qdrant", platformFolder, "qdrant");
    if (process.platform !== "win32") {
      return rawBinary;
    }
    if (existsSync(rawBinary)) {
      return rawBinary;
    }
    return `${rawBinary}.exe`;
  }

  private async ensureExecutable(binaryPath: string): Promise<void> {
    if (process.platform === "win32") {
      return;
    }
    try {
      await chmod(binaryPath, 0o755);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown chmod failure.";
      throw new Error(`Failed to make Qdrant binary executable: ${message}`);
    }
  }

  private patchStatus(partial: Partial<QdrantStatus>): void {
    this.status = {
      ...this.status,
      ...partial,
      storagePath: partial.storagePath ?? this.status.storagePath ?? this.resolveStoragePath(),
    };
    this.emitStatus();
  }

  private emitStatus(): void {
    const snapshot = this.getStatus();
    for (const listener of this.listeners) {
      listener(snapshot);
    }
  }
}
