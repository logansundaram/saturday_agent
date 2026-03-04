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
import {
  registerUsageTraceIpc,
  traceDesktopEvent,
  writeSmokeReport,
} from "./usageTrace";

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
const SMOKE_MODE = process.env.SATURDAY_SMOKE === "1";
const SKIP_QDRANT = process.env.SATURDAY_SKIP_QDRANT === "1" || SMOKE_MODE;

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

async function bodyIncludes(windowRef: BrowserWindow, expected: string): Promise<boolean> {
  try {
    const bodyText = await windowRef.webContents.executeJavaScript(
      "document.body?.innerText ?? ''",
      true
    );
    return String(bodyText ?? "").includes(expected);
  } catch {
    return false;
  }
}

async function waitForBodyText(
  windowRef: BrowserWindow,
  expected: string,
  timeoutMs: number = 8_000
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await bodyIncludes(windowRef, expected)) {
      return true;
    }
    await delay(250);
  }
  return false;
}

async function navigateSmoke(windowRef: BrowserWindow, page: string): Promise<void> {
  traceDesktopEvent("desktop.smoke.navigate", { page });
  await windowRef.webContents.executeJavaScript(
    `window.dispatchEvent(new CustomEvent("dashboard:navigate", { detail: ${JSON.stringify({
      page,
    })} }));`,
    true
  );
  await delay(700);
}

async function runSmokeScenario(windowRef: BrowserWindow): Promise<void> {
  const assertions: Array<{ name: string; passed: boolean; detail: string }> = [];
  const record = (name: string, passed: boolean, detail: string) => {
    assertions.push({ name, passed, detail });
    traceDesktopEvent("desktop.smoke.assertion", { name, passed, detail });
  };

  record("main_window_created", true, "Electron main window created.");
  const chatReady = await waitForBodyText(windowRef, "Chat");
  record(
    "chat_page_renders",
    chatReady,
    chatReady ? "Chat heading rendered." : "Chat heading did not render."
  );

  await navigateSmoke(windowRef, "tools");
  const toolsReady =
    (await waitForBodyText(windowRef, "Smoke Echo Tool", 12_000)) ||
    (await waitForBodyText(windowRef, "Web Search", 3_000));
  record(
    "tools_list_renders",
    toolsReady,
    toolsReady ? "At least one tool card rendered." : "No tool card rendered."
  );

  await navigateSmoke(windowRef, "builder");
  const builderReady = await waitForBodyText(windowRef, "Builder");
  record(
    "builder_page_renders",
    builderReady,
    builderReady ? "Builder page rendered." : "Builder page did not render."
  );

  await navigateSmoke(windowRef, "inspect");
  const inspectReady = await waitForBodyText(windowRef, "Inspect");
  record(
    "inspect_page_renders",
    inspectReady,
    inspectReady ? "Inspect page rendered." : "Inspect page did not render."
  );

  const ok = assertions.every((item) => item.passed);
  traceDesktopEvent("desktop.smoke.complete", { ok, assertions });
  await writeSmokeReport({
    ok,
    assertions,
    completed_at: new Date().toISOString(),
  });
  app.exit(ok ? 0 : 1);
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
  traceDesktopEvent("desktop.window.created", {
    smoke_mode: SMOKE_MODE,
    api_base_url: API_BASE_URL,
  });

  // Test active push message to Renderer-process.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", new Date().toLocaleString());
    if (SMOKE_MODE && win) {
      void delay(1200).then(() => runSmokeScenario(win as BrowserWindow));
    }
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
  registerUsageTraceIpc();
  registerDocsIpcHandlers({
    getQdrantUrl: () => qdrantManager.getUrl(),
  });

  if (!SKIP_QDRANT) {
    teardownQdrantIpcHandlers = registerQdrantIpcHandlers(qdrantManager, {
      onRunning: configureApiQdrant,
    });

    const status = await qdrantManager.start();
    if (status.running) {
      await configureApiQdrant(status);
    }
  } else {
    traceDesktopEvent("desktop.qdrant.skipped", {
      reason: SMOKE_MODE ? "smoke_mode" : "env_flag",
    });
  }

  createWindow();
});
