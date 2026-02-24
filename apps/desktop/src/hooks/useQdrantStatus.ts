import { useEffect, useMemo, useState } from "react";

const FALLBACK_STORAGE_PATH = "";

type HookState = {
  running: boolean;
  port: number | null;
  error?: string;
  storagePath: string;
  loading: boolean;
};

export type QdrantHookStatus = HookState & {
  statusLabel: "running" | "starting" | "error" | "stopped";
};

const INITIAL_STATE: HookState = {
  running: false,
  port: null,
  error: undefined,
  storagePath: FALLBACK_STORAGE_PATH,
  loading: true,
};

export function useQdrantStatus(): QdrantHookStatus {
  const [state, setState] = useState<HookState>(INITIAL_STATE);

  useEffect(() => {
    if (!window.qdrant) {
      setState({
        running: false,
        port: null,
        error: "Qdrant IPC bridge is unavailable.",
        storagePath: FALLBACK_STORAGE_PATH,
        loading: false,
      });
      return;
    }

    let mounted = true;

    const update = (next: QdrantStatus) => {
      if (!mounted) {
        return;
      }
      setState({
        running: Boolean(next.running),
        port: typeof next.port === "number" ? next.port : null,
        error: next.error,
        storagePath: next.storagePath || FALLBACK_STORAGE_PATH,
        loading: false,
      });
    };

    void window.qdrant
      .status()
      .then(update)
      .catch((error: unknown) => {
        if (!mounted) {
          return;
        }
        setState({
          running: false,
          port: null,
          error:
            error instanceof Error
              ? error.message
              : "Unable to read Qdrant status.",
          storagePath: FALLBACK_STORAGE_PATH,
          loading: false,
        });
      });

    const unsubscribe = window.qdrant.subscribe(update);
    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  return useMemo(() => {
    let statusLabel: QdrantHookStatus["statusLabel"] = "stopped";
    if (state.loading) {
      statusLabel = "starting";
    } else if (state.error) {
      statusLabel = "error";
    } else if (state.running) {
      statusLabel = "running";
    }
    return {
      ...state,
      statusLabel,
    };
  }, [state]);
}
