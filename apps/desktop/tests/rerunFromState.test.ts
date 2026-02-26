import assert from "node:assert/strict";
import test from "node:test";
import { buildRerunFromStateRequest } from "../src/components/inspect/rerunFromState.ts";

test("rerun-from-state request builder rejects invalid JSON", () => {
  const built = buildRerunFromStateRequest({
    stepIndex: 3,
    editorText: "{bad json}",
    resume: "next",
  });

  assert.equal(built.request, null);
  assert.ok(built.parseError);
});

test("rerun-from-state request builder requires object payload", () => {
  const built = buildRerunFromStateRequest({
    stepIndex: 4,
    editorText: "[]",
    resume: "next",
  });

  assert.equal(built.request, null);
  assert.equal(built.parseError, "State JSON must be a JSON object.");
});

test("rerun-from-state request builder serializes valid payload", () => {
  const built = buildRerunFromStateRequest({
    stepIndex: 6,
    editorText: "{\"answer\":\"patched\"}",
    resume: "next",
  });

  assert.equal(built.parseError, null);
  assert.deepEqual(built.request, {
    step_index: 6,
    state_json: { answer: "patched" },
    resume: "next",
  });
});
