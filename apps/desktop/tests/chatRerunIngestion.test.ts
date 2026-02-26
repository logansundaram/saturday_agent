import assert from "node:assert/strict";
import test from "node:test";
import {
  extractRerunOutputText,
  extractRerunPromptCandidate,
  mapRunLogStepsToTimeline,
  normalizePromptText,
  resolveSourcePromptUserMessageId,
} from "../src/components/chat/rerunIngestion.ts";
import {
  appendMessage,
  createThread,
  loadState,
  updateMessage,
} from "../src/lib/chatStore.ts";

function installWindowStorageMock(): () => void {
  const store = new Map<string, string>();
  const localStorage: Storage = {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.has(key) ? store.get(key)! : null;
    },
    key(index: number) {
      return Array.from(store.keys())[index] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
  };

  const originalWindow = (globalThis as { window?: unknown }).window;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    writable: true,
    value: {
      localStorage,
    },
  });

  return () => {
    if (originalWindow === undefined) {
      Reflect.deleteProperty(globalThis, "window");
      return;
    }
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      writable: true,
      value: originalWindow,
    });
  };
}

test("prompt extraction prefers forked state task", () => {
  const prompt = extractRerunPromptCandidate({
    run_id: "run-1",
    status: "running",
    forked_from_state_json: {
      task: "Forked task wins",
      messages: [{ role: "user", content: "older task" }],
    },
    payload: {
      message: "payload message",
      input: { task: "payload task" },
    },
  });

  assert.equal(prompt, "Forked task wins");
});

test("prompt extraction falls back through payload fields", () => {
  const prompt = extractRerunPromptCandidate({
    run_id: "run-2",
    status: "running",
    payload: {
      message: "Payload message fallback",
      input: {
        messages: [{ role: "user", content: "message history fallback" }],
      },
    },
  });

  assert.equal(prompt, "Payload message fallback");
});

test("step mapping normalizes statuses for chat timeline", () => {
  const mapped = mapRunLogStepsToTimeline([
    { step_index: 2, name: "tool_call", status: "error" },
    { step_index: 0, name: "ingest", status: "success" },
    { step_index: 1, name: "planner", status: "skipped" },
    { step_index: 3, name: "finish", status: "running" },
  ]);

  assert.equal(mapped.length, 4);
  assert.deepEqual(
    mapped.map((item) => item.status),
    ["ok", "ok", "error", "running"]
  );
  assert.deepEqual(
    mapped.map((item) => item.step_index),
    [0, 1, 2, 3]
  );
});

test("output extraction supports output_text, answer, and error fallback", () => {
  const directOutput = extractRerunOutputText({
    run_id: "run-a",
    status: "ok",
    result: { output_text: "Direct output text" },
  });
  assert.equal(directOutput, "Direct output text");

  const answerOutput = extractRerunOutputText({
    run_id: "run-b",
    status: "ok",
    result: { output: { answer: "Answer field output" } },
  });
  assert.equal(answerOutput, "Answer field output");

  const errorFallback = extractRerunOutputText({
    run_id: "run-c",
    status: "failed",
    result: {},
  });
  assert.equal(errorFallback, "Error: Run ended with status failed.");
});

test("source prompt resolution returns user message paired with source run", () => {
  const userId = "u-source";
  const assistantId = "a-source";
  const resolved = resolveSourcePromptUserMessageId({
    sourceRunId: "run-source",
    runMetaByMessageId: {},
    messages: [
      {
        id: userId,
        role: "user",
        content: "original prompt",
        createdAt: 1,
      },
      {
        id: assistantId,
        role: "assistant",
        content: "original answer",
        createdAt: 2,
        runId: "run-source",
      },
      {
        id: "u-later",
        role: "user",
        content: "later prompt",
        createdAt: 3,
      },
    ],
  });

  assert.equal(resolved, userId);
});

test("source prompt resolution supports run meta fallback", () => {
  const resolved = resolveSourcePromptUserMessageId({
    sourceRunId: "run-meta",
    runMetaByMessageId: {
      "a-meta": {
        runId: "run-meta",
        status: "ok",
        steps: [],
      },
    },
    messages: [
      {
        id: "u-meta",
        role: "user",
        content: "prompt from meta pair",
        createdAt: 1,
      },
      {
        id: "a-meta",
        role: "assistant",
        content: "assistant text",
        createdAt: 2,
      },
    ],
  });

  assert.equal(resolved, "u-meta");
});

test("source prompt resolution returns null when source pair is not found", () => {
  const resolved = resolveSourcePromptUserMessageId({
    sourceRunId: "run-missing",
    runMetaByMessageId: {},
    messages: [
      {
        id: "u1",
        role: "user",
        content: "prompt",
        createdAt: 1,
      },
      {
        id: "a1",
        role: "assistant",
        content: "answer",
        createdAt: 2,
        runId: "run-other",
      },
    ],
  });

  assert.equal(resolved, null);
});

test("prompt normalization uses trimmed comparison semantics", () => {
  assert.equal(normalizePromptText("  hello world  "), "hello world");
});

test("updateMessage patches only the target message and preserves order", () => {
  const restoreWindow = installWindowStorageMock();
  try {
    const threadId = createThread();
    appendMessage(threadId, {
      id: "m1",
      role: "assistant",
      content: "first",
      createdAt: 1,
      runId: "run-1",
    });
    appendMessage(threadId, {
      id: "m2",
      role: "assistant",
      content: "second",
      createdAt: 2,
      runId: "run-2",
    });

    updateMessage(threadId, "m1", {
      content: "first updated",
      modelId: "model-x",
    });

    const state = loadState();
    const messages = state.messagesByThread[threadId] ?? [];
    assert.deepEqual(
      messages.map((message) => message.id),
      ["m1", "m2"]
    );
    assert.equal(messages[0].content, "first updated");
    assert.equal(messages[0].modelId, "model-x");
    assert.equal(messages[1].content, "second");
    assert.equal(messages[1].modelId, undefined);
  } finally {
    restoreWindow();
  }
});
