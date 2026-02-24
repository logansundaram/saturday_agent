import { app, BrowserWindow } from "electron";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { registerDocsIpcHandlers } from "./ipc/api";
import { registerQdrantIpcHandlers } from "./ipc/qdrant";
import { startMetricsPolling, stopMetricsPolling } from "./ipc/systemMetrics";
import {
  QdrantManager,
  type QdrantStatus,
} from "./services/QdrantManager";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// The built directory structure
//
// ├─┬─┬ dist
// │ │ └── index.html
// │ │
// │ ├─┬ dist-electron
// │ │ ├── main.js
// │ │ └── preload.mjs
// │
process.env.APP_ROOT = path.join(__dirname, "..");

// 🚧 Use ['ENV_NAME'] avoid vite:define plugin - Vite@2.x
export const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
export const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
export const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, "public")
  : RENDERER_DIST;

const API_BASE_URL =
  process.env.SATURDAY_API_URL ??
  process.env.VITE_API_BASE_URL ??
  "http://localhost:8000";

const qdrantManager = new QdrantManager();
let teardownQdrantIpcHandlers: (() => void) | null = null;
let win: BrowserWindow | null = null;
let quitting = false;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function configureApiQdrant(status: QdrantStatus): Promise<void> {
  const qdrantUrl = status.port ? `http://127.0.0.1:${status.port}` : null;
  if (!qdrantUrl) {
    return;
  }

  const qdrantConfigPath = path.join(
    app.getPath("userData"),
    "qdrant",
    "qdrant.json"
  );
  try {
    await mkdir(path.dirname(qdrantConfigPath), { recursive: true });
    await writeFile(
      qdrantConfigPath,
      JSON.stringify(
        {
          url: qdrantUrl,
          updated_at: new Date().toISOString(),
        },
        null,
        2
      ),
      "utf-8"
    );
  } catch (error) {
    const detail =
      error instanceof Error ? error.message : "Unable to write qdrant.json.";
    qdrantManager.setStatusError(`Failed to persist Qdrant runtime config: ${detail}`);
  }

  let lastError = "";
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    try {
      const response = await fetch(`${API_BASE_URL}/internal/qdrant/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: qdrantUrl }),
      });
      if (response.ok) {
        qdrantManager.setStatusError(undefined);
        return;
      }

      let detail = `API returned ${response.status}`;
      let normalizedDetail = "";
      try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string" && payload.detail.trim()) {
          detail = payload.detail;
          normalizedDetail = payload.detail.trim().toLowerCase();
        }
      } catch {
        // Ignore parsing failures and keep base status.
      }

      if (response.status === 404 && normalizedDetail === "not found") {
        qdrantManager.setStatusError(undefined);
        return;
      }
      lastError = `Failed to configure API Qdrant URL: ${detail}`;
    } catch (error) {
      lastError =
        error instanceof Error
          ? `Failed to configure API Qdrant URL: ${error.message}`
          : "Failed to configure API Qdrant URL.";
    }
    await delay(500);
  }

  qdrantManager.setStatusError(lastError);
}

function createWindow(): void {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "electron-vite.svg"),
    titleBarStyle: "hidden",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
    },
  });

  // Test active push message to Renderer-process.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", new Date().toLocaleString());
  });

  if (VITE_DEV_SERVER_URL) {
    void win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    // win.loadFile('dist/index.html')
    void win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }

  startMetricsPolling();
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  stopMetricsPolling();
  if (process.platform !== 'darwin') {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("before-quit", (event) => {
  stopMetricsPolling();
  if (quitting) {
    return;
  }

  quitting = true;
  event.preventDefault();
  void qdrantManager.stop().finally(() => {
    teardownQdrantIpcHandlers?.();
    teardownQdrantIpcHandlers = null;
    app.quit();
  });
});

app.whenReady().then(async () => {
  registerDocsIpcHandlers({
    getQdrantUrl: () => qdrantManager.getUrl(),
  });
  teardownQdrantIpcHandlers = registerQdrantIpcHandlers(qdrantManager, {
    onRunning: configureApiQdrant,
  });

  const status = await qdrantManager.start();
  if (status.running) {
    await configureApiQdrant(status);
  }

  createWindow();
});
