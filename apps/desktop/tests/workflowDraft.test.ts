import assert from "node:assert/strict";
import test from "node:test";
import {
  createInitialWorkflowSpec,
  normalizeDraftSpec,
  validateDraftLocally,
} from "../src/components/builder/workflowDraft.ts";

test("workflow spec serialize/deserialize roundtrip", () => {
  const draft = createInitialWorkflowSpec();
  const serialized = JSON.stringify(draft);
  const deserialized = JSON.parse(serialized);
  const normalized = normalizeDraftSpec(deserialized);

  assert.equal(normalized.workflow_id, draft.workflow_id);
  assert.equal(normalized.nodes.length, draft.nodes.length);
  assert.equal(normalized.edges.length, draft.edges.length);
  assert.ok(normalized.nodes.every((node) => node.id && node.type));
});

test("validator reports duplicate node ids", () => {
  const spec = createInitialWorkflowSpec();
  const duplicateNode = { ...spec.nodes[0] };
  spec.nodes.push(duplicateNode);

  const diagnostics = validateDraftLocally(spec, []);
  assert.ok(diagnostics.some((item) => item.code === "DUPLICATE_NODE_ID"));
});

test("validator reports missing finalize node", () => {
  const spec = createInitialWorkflowSpec();
  spec.nodes = spec.nodes.filter((node) => node.type !== "finalize");
  spec.edges = [];

  const diagnostics = validateDraftLocally(spec, []);
  assert.ok(diagnostics.some((item) => item.code === "MISSING_FINALIZE_NODE"));
});

test("validator flags missing tool references", () => {
  const spec = createInitialWorkflowSpec();
  spec.nodes = [
    {
      id: "tool_1",
      type: "tool",
      label: "Tool",
      reads: ["task"],
      writes: ["artifacts"],
      config: {
        tool_name: "tool.custom.missing",
        args_map: { query: "task" },
        output_key: "artifacts",
      },
      position: { x: 100, y: 100 },
    },
    {
      id: "finalize_1",
      type: "finalize",
      label: "Finalize",
      reads: ["task"],
      writes: ["answer"],
      config: { response_template: "{{task}}", output_key: "answer" },
      position: { x: 320, y: 100 },
    },
  ];
  spec.edges = [{ id: "edge_1", from: "tool_1", to: "finalize_1", label: "always" }];

  const diagnostics = validateDraftLocally(spec, []);
  assert.ok(diagnostics.some((item) => item.code === "TOOL_NOT_FOUND"));
  assert.ok(diagnostics.some((item) => item.severity === "error"));
});
