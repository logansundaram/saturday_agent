"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("ipcRenderer", {
  on(...args) {
    const [channel, listener] = args;
    return electron.ipcRenderer.on(channel, (event, ...args2) => listener(event, ...args2));
  },
  off(...args) {
    const [channel, ...omit] = args;
    return electron.ipcRenderer.off(channel, ...omit);
  },
  send(...args) {
    const [channel, ...omit] = args;
    return electron.ipcRenderer.send(channel, ...omit);
  },
  invoke(...args) {
    const [channel, ...omit] = args;
    return electron.ipcRenderer.invoke(channel, ...omit);
  }
  // You can expose other APTs you need here.
  // ...
});
electron.contextBridge.exposeInMainWorld("system", {
  subscribe(callback) {
    const listener = (_event, payload) => {
      callback(payload);
    };
    electron.ipcRenderer.on("system:metrics", listener);
    return () => electron.ipcRenderer.off("system:metrics", listener);
  }
});
electron.contextBridge.exposeInMainWorld("qdrant", {
  status() {
    return electron.ipcRenderer.invoke("qdrant:status");
  },
  restart() {
    return electron.ipcRenderer.invoke("qdrant:restart");
  },
  stop() {
    return electron.ipcRenderer.invoke("qdrant:stop");
  },
  subscribe(callback) {
    const listener = (_event, payload) => {
      callback(payload);
    };
    electron.ipcRenderer.on("qdrant:subscribe", listener);
    return () => electron.ipcRenderer.off("qdrant:subscribe", listener);
  }
});
