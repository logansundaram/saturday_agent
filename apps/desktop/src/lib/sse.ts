export type StreamSSEOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
};

function resolveTimeout(timeoutMs?: number): number | null {
  if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs)) {
    return null;
  }
  if (timeoutMs <= 0) {
    return null;
  }
  return timeoutMs;
}

function parseSSEFrame<T>(rawFrame: string): T | null {
  const frame = rawFrame.replace(/\r/g, "").trim();
  if (!frame) {
    return null;
  }

  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (eventName !== "message" || dataLines.length === 0) {
    return null;
  }

  const payload = dataLines.join("\n").trim();
  if (!payload) {
    return null;
  }

  try {
    return JSON.parse(payload) as T;
  } catch {
    throw new Error("Received malformed stream payload.");
  }
}

export async function streamSSE<T>(
  url: string,
  body: unknown,
  onEvent: (event: T) => void,
  options?: StreamSSEOptions
): Promise<void> {
  const controller = new AbortController();
  const timeoutMs = resolveTimeout(options?.timeoutMs);
  const timeoutId =
    timeoutMs === null
      ? null
      : window.setTimeout(() => controller.abort(), timeoutMs);

  const upstreamSignal = options?.signal;
  const abortHandler = () => controller.abort();
  if (upstreamSignal) {
    if (upstreamSignal.aborted) {
      controller.abort();
    } else {
      upstreamSignal.addEventListener("abort", abortHandler, { once: true });
    }
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers ?? {}),
      },
      body: JSON.stringify(body ?? {}),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string" && payload.detail.trim()) {
          detail = `${detail}: ${payload.detail}`;
        }
      } catch {
        const raw = await response.text();
        if (raw.trim()) {
          detail = `${detail}: ${raw.trim()}`;
        }
      }
      throw new Error(detail);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("text/event-stream")) {
      throw new Error("Expected SSE response from backend.");
    }

    if (!response.body) {
      throw new Error("Streaming response body is empty.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    let doneReading = false;
    while (!doneReading) {
      const { value, done } = await reader.read();
      doneReading = done;
      if (doneReading) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseSSEFrame<T>(frame);
        if (parsed !== null) {
          onEvent(parsed);
        }
        boundary = buffer.indexOf("\n\n");
      }
    }

    buffer += decoder.decode();
    const trailing = parseSSEFrame<T>(buffer);
    if (trailing !== null) {
      onEvent(trailing);
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      if (upstreamSignal?.aborted) {
        throw new Error("The request was canceled.");
      }
      throw new Error("The request timed out. Please try again.");
    }
    if (error instanceof Error) {
      throw new Error(error.message || "Unable to stream response.");
    }
    throw new Error("Unable to stream response.");
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    if (upstreamSignal) {
      upstreamSignal.removeEventListener("abort", abortHandler);
    }
  }
}
