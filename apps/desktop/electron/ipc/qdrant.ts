import { BrowserWindow, ipcMain } from "electron";

import type { QdrantStatus } from "../services/QdrantManager";
import { QdrantManager } from "../services/QdrantManager";

const SUBSCRIBE_CHANNEL = "qdrant:subscribe";
const STATUS_CHANNEL = "qdrant:status";
const STOP_CHANNEL = "qdrant:stop";
const RESTART_CHANNEL = "qdrant:restart";
const THROTTLE_MS = 750;

type QdrantIpcOptions = {
  onRunning?: (status: QdrantStatus) => Promise<void> | void;
};

export function registerQdrantIpcHandlers(
  manager: QdrantManager,
  options: QdrantIpcOptions = {}
): () => void {
  ipcMain.removeHandler(STATUS_CHANNEL);
  ipcMain.removeHandler(STOP_CHANNEL);
  ipcMain.removeHandler(RESTART_CHANNEL);

  let pendingStatus: QdrantStatus | null = null;
  let throttleTimer: NodeJS.Timeout | null = null;

  const flush = () => {
    if (!pendingStatus) {
      return;
    }
    const payload = pendingStatus;
    pendingStatus = null;
    for (const windowRef of BrowserWindow.getAllWindows()) {
      if (!windowRef.isDestroyed()) {
        windowRef.webContents.send(SUBSCRIBE_CHANNEL, payload);
      }
    }
  };

  const schedule = (status: QdrantStatus) => {
    pendingStatus = status;
    if (throttleTimer) {
      return;
    }
    throttleTimer = setTimeout(() => {
      throttleTimer = null;
      flush();
    }, THROTTLE_MS);
  };

  const unsubscribe = manager.subscribe((status) => {
    schedule(status);
  });

  ipcMain.handle(STATUS_CHANNEL, async () => manager.getStatus());
  ipcMain.handle(STOP_CHANNEL, async () => manager.stop());
  ipcMain.handle(RESTART_CHANNEL, async () => {
    const status = await manager.restart();
    if (status.running && options.onRunning) {
      await options.onRunning(status);
    }
    return manager.getStatus();
  });

  return () => {
    unsubscribe();
    ipcMain.removeHandler(STATUS_CHANNEL);
    ipcMain.removeHandler(STOP_CHANNEL);
    ipcMain.removeHandler(RESTART_CHANNEL);
    if (throttleTimer) {
      clearTimeout(throttleTimer);
      throttleTimer = null;
    }
  };
}
