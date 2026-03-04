type TracePayload = Record<string, unknown>;

const TRACE_ENABLED = import.meta.env.VITE_USAGE_TRACE === "1";

export function traceRendererRoute(route: string, payload: TracePayload = {}): void {
  if (!TRACE_ENABLED) {
    return;
  }
  window.ipcRenderer?.send("usage-trace:event", {
    event_type: "renderer.route_mount",
    route,
    ...payload,
  });
}
