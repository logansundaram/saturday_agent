import { contextBridge, ipcRenderer } from "electron";

type QdrantStatus = {
  running: boolean;
  port: number | null;
  storagePath: string;
  pid?: number;
  error?: string;
  lastHealthCheckAt?: string;
};

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
  on(...args: Parameters<typeof ipcRenderer.on>) {
    const [channel, listener] = args
    return ipcRenderer.on(channel, (event, ...args) => listener(event, ...args))
  },
  off(...args: Parameters<typeof ipcRenderer.off>) {
    const [channel, ...omit] = args
    return ipcRenderer.off(channel, ...omit)
  },
  send(...args: Parameters<typeof ipcRenderer.send>) {
    const [channel, ...omit] = args
    return ipcRenderer.send(channel, ...omit)
  },
  invoke(...args: Parameters<typeof ipcRenderer.invoke>) {
    const [channel, ...omit] = args
    return ipcRenderer.invoke(channel, ...omit)
  },

  // You can expose other APTs you need here.
  // ...
});

contextBridge.exposeInMainWorld('system', {
  subscribe(callback: (payload: unknown) => void) {
    const listener = (_event: unknown, payload: unknown) => {
      callback(payload)
    }
    ipcRenderer.on('system:metrics', listener)
    return () => ipcRenderer.off('system:metrics', listener)
  },
});

contextBridge.exposeInMainWorld("qdrant", {
  status(): Promise<QdrantStatus> {
    return ipcRenderer.invoke("qdrant:status") as Promise<QdrantStatus>;
  },
  restart(): Promise<QdrantStatus> {
    return ipcRenderer.invoke("qdrant:restart") as Promise<QdrantStatus>;
  },
  stop(): Promise<QdrantStatus> {
    return ipcRenderer.invoke("qdrant:stop") as Promise<QdrantStatus>;
  },
  subscribe(callback: (status: QdrantStatus) => void) {
    const listener = (_event: unknown, payload: QdrantStatus) => {
      callback(payload);
    };
    ipcRenderer.on("qdrant:subscribe", listener);
    return () => ipcRenderer.off("qdrant:subscribe", listener);
  },
});
