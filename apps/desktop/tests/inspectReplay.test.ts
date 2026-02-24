import assert from "node:assert/strict";
import test from "node:test";
import {
  buildReplayRequestFromEditor,
  hasReplayDiagnosticErrors,
  normalizeReplayDiagnostics,
} from "../src/components/inspect/stateDiff.ts";

test("replay request serialization supports overlay mode", () => {
  const built = buildReplayRequestFromEditor({
    fromStepId: "step_10",
    patchText: "{\"answer\":\"patched\"}",
    patchMode: "overlay",
    sandbox: true,
    baseState: "post",
    replayThisStep: false,
  });

  assert.equal(built.parseError, null);
  assert.ok(built.request);
  assert.equal(built.request?.from_step_id, "step_10");
  assert.equal(built.request?.patch_mode, "overlay");
  assert.deepEqual(built.request?.state_patch, { answer: "patched" });
  assert.equal(built.request?.sandbox, true);
  assert.equal(built.request?.base_state, "post");
  assert.equal(built.request?.replay_this_step, false);
});

test("replay request serialization supports replace mode", () => {
  const built = buildReplayRequestFromEditor({
    fromStepId: "step_11",
    patchText: "{\"task\":\"new\"}",
    patchMode: "replace",
    sandbox: false,
    baseState: "pre",
    replayThisStep: true,
  });

  assert.equal(built.parseError, null);
  assert.ok(built.request);
  assert.equal(built.request?.patch_mode, "replace");
  assert.deepEqual(built.request?.state_patch, { task: "new" });
  assert.equal(built.request?.base_state, "pre");
  assert.equal(built.request?.replay_this_step, true);
});

test("replay request serialization supports jsonpatch mode", () => {
  const built = buildReplayRequestFromEditor({
    fromStepId: "step_12",
    patchText: '[{"op":"replace","path":"/answer","value":"ok"}]',
    patchMode: "jsonpatch",
    sandbox: false,
    baseState: "post",
    replayThisStep: false,
  });

  assert.equal(built.parseError, null);
  assert.ok(Array.isArray(built.request?.state_patch));
  assert.equal((built.request?.state_patch as Array<unknown>).length, 1);
});

test("diagnostic normalization is resilient to malformed values", () => {
  const diagnostics = normalizeReplayDiagnostics([
    null,
    { code: "TYPE_MISMATCH", severity: "warning", message: "Type mismatch", path: "state.foo" },
    { message: "missing code and severity" },
    "unexpected",
  ]);

  assert.equal(diagnostics.length, 4);
  assert.equal(diagnostics[0].code, "UNKNOWN");
  assert.equal(diagnostics[1].severity, "warning");
  assert.equal(diagnostics[2].severity, "error");
  assert.equal(diagnostics[3].code, "UNKNOWN");
  assert.equal(hasReplayDiagnosticErrors(diagnostics), true);
});
