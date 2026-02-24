import type {
  ConditionalNodeConfig,
  EdgeSpec,
  NodeSpec,
  NodeType,
  ToolNodeConfig,
  ValidationDiagnostic,
  VerifyNodeConfig,
  WorkflowSpec,
} from "@saturday/shared/workflow";

const BASE_KEYS = new Set([
  "task",
  "context",
  "messages",
  "plan",
  "answer",
  "artifacts",
  "verify_ok",
  "verify_notes",
  "retry_count",
]);

const VALID_CONDITIONAL_OPERATORS = new Set([
  "equals",
  "contains",
  "gt",
  "lt",
  "exists",
  "not_exists",
  "in",
]);

export function slugify(value: string, fallback: string): string {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || fallback;
}

export function normalizeKeyList(raw: string): string[] {
  return String(raw || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function defaultNodePosition(index: number): { x: number; y: number } {
  const columns = 4;
  return {
    x: 120 + (index % columns) * 240,
    y: 80 + Math.floor(index / columns) * 180,
  };
}

export function defaultNodeForType(
  type: NodeType,
  nodeId: string,
  index: number
): NodeSpec {
  const base: NodeSpec = {
    id: nodeId,
    type,
    label: `${type.toUpperCase()} ${nodeId}`,
    reads: [],
    writes: [],
    config: {},
    position: defaultNodePosition(index),
  };

  if (type === "llm") {
    return {
      ...base,
      reads: ["task"],
      writes: ["answer"],
      config: {
        prompt_template: "Answer the task: {{task}}",
        output_key: "answer",
      },
    };
  }
  if (type === "tool") {
    return {
      ...base,
      reads: ["task"],
      writes: ["artifacts"],
      config: {
        tool_name: "",
        args_map: {
          query: "task",
        },
        output_key: "artifacts",
      },
    };
  }
  if (type === "conditional") {
    return {
      ...base,
      reads: ["verify_ok"],
      writes: [],
      config: {
        expression: "verify_ok == True",
      },
    };
  }
  if (type === "verify") {
    return {
      ...base,
      reads: ["answer"],
      writes: ["verify_ok", "verify_notes"],
      config: {
        mode: "rule",
        expression: "answer is not None",
        output_key: "verify_ok",
      },
    };
  }
  return {
    ...base,
    reads: ["answer"],
    writes: ["answer"],
    config: {
      response_template: "{{answer}}",
      output_key: "answer",
    },
  };
}

export function getNextNodeId(nodes: NodeSpec[], type: NodeType): string {
  let maxValue = 0;
  for (const node of nodes) {
    if (node.type !== type) {
      continue;
    }
    const match = String(node.id).match(new RegExp(`^${type}_(\\d+)$`));
    if (!match) {
      continue;
    }
    const value = Number.parseInt(match[1], 10);
    if (Number.isFinite(value)) {
      maxValue = Math.max(maxValue, value);
    }
  }
  return `${type}_${maxValue + 1}`;
}

export function getNextEdgeId(edges: EdgeSpec[]): string {
  let maxValue = 0;
  for (const edge of edges) {
    const match = String(edge.id || "").match(/^edge_(\d+)$/);
    if (!match) {
      continue;
    }
    const value = Number.parseInt(match[1], 10);
    if (Number.isFinite(value)) {
      maxValue = Math.max(maxValue, value);
    }
  }
  return `edge_${maxValue + 1}`;
}

function ensureNode(node: NodeSpec, index: number): NodeSpec {
  return {
    ...node,
    label: String(node.label || node.id || "").trim(),
    reads: Array.isArray(node.reads) ? node.reads.filter(Boolean) : [],
    writes: Array.isArray(node.writes) ? node.writes.filter(Boolean) : [],
    config: node.config && typeof node.config === "object" ? node.config : {},
    position: node.position || defaultNodePosition(index),
  };
}

export function normalizeDraftSpec(spec: WorkflowSpec): WorkflowSpec {
  const nodes = Array.isArray(spec.nodes)
    ? spec.nodes.map((node, index) => ensureNode(node, index))
    : [];
  const edges = Array.isArray(spec.edges)
    ? spec.edges.map((edge, index) => ({
        id: String(edge.id || `edge_${index + 1}`),
        from: String(edge.from || ""),
        to: String(edge.to || ""),
        label: String(edge.label || "always"),
      }))
    : [];
  return {
    workflow_id: spec.workflow_id ? String(spec.workflow_id) : undefined,
    name: String(spec.name || "").trim(),
    description: String(spec.description || ""),
    allow_cycles: Boolean(spec.allow_cycles),
    state_schema: Array.isArray(spec.state_schema)
      ? spec.state_schema.map((item) => ({
          key: String(item.key || "").trim(),
          type: item.type,
          description: String(item.description || ""),
          required: Boolean(item.required),
        }))
      : [],
    nodes,
    edges,
    metadata:
      spec.metadata && typeof spec.metadata === "object" ? spec.metadata : {},
  };
}

export function createInitialWorkflowSpec(): WorkflowSpec {
  const llmNode = defaultNodeForType("llm", "llm_1", 0);
  const finalizeNode = defaultNodeForType("finalize", "finalize_1", 1);
  return {
    workflow_id: "workflow.custom.untitled",
    name: "Untitled Workflow",
    description: "",
    allow_cycles: false,
    state_schema: [
      { key: "answer", type: "string", description: "Final answer", required: false },
      {
        key: "verify_ok",
        type: "bool",
        description: "Verification status",
        required: false,
      },
      {
        key: "verify_notes",
        type: "string",
        description: "Verification notes",
        required: false,
      },
      {
        key: "artifacts",
        type: "json",
        description: "Tool and node outputs",
        required: false,
      },
    ],
    nodes: [llmNode, finalizeNode],
    edges: [
      {
        id: "edge_1",
        from: llmNode.id,
        to: finalizeNode.id,
        label: "always",
      },
    ],
    metadata: { enabled: true },
  };
}

function collectExprIdentifiers(expression: string): Set<string> {
  const refs = new Set<string>();
  const tokens = String(expression || "").match(/[a-zA-Z_][a-zA-Z0-9_.]*/g) || [];
  for (const token of tokens) {
    if (token === "and" || token === "or" || token === "not") {
      continue;
    }
    if (token === "True" || token === "False" || token === "None") {
      continue;
    }
    refs.add(token);
  }
  return refs;
}

function detectCycle(nodes: NodeSpec[], edges: EdgeSpec[]): string[] | null {
  const adjacency = new Map<string, string[]>();
  for (const node of nodes) {
    adjacency.set(node.id, []);
  }
  for (const edge of edges) {
    if (!adjacency.has(edge.from) || !adjacency.has(edge.to)) {
      continue;
    }
    adjacency.get(edge.from)?.push(edge.to);
  }

  const state = new Map<string, number>();
  const parent = new Map<string, string | null>();
  for (const node of nodes) {
    state.set(node.id, 0);
    parent.set(node.id, null);
  }

  const walk = (nodeId: string): string[] | null => {
    state.set(nodeId, 1);
    for (const nextId of adjacency.get(nodeId) || []) {
      const nextState = state.get(nextId) || 0;
      if (nextState === 0) {
        parent.set(nextId, nodeId);
        const path = walk(nextId);
        if (path) {
          return path;
        }
      } else if (nextState === 1) {
        const path: string[] = [nextId];
        let current = nodeId;
        while (current && current !== nextId) {
          path.push(current);
          current = parent.get(current) || "";
        }
        path.push(nextId);
        path.reverse();
        return path;
      }
    }
    state.set(nodeId, 2);
    return null;
  };

  for (const node of nodes) {
    if ((state.get(node.id) || 0) !== 0) {
      continue;
    }
    const cyclePath = walk(node.id);
    if (cyclePath) {
      return cyclePath;
    }
  }
  return null;
}

function reachableFromEntry(nodes: NodeSpec[], edges: EdgeSpec[]): Set<string> {
  const indegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();
  for (const node of nodes) {
    indegree.set(node.id, 0);
    adjacency.set(node.id, []);
  }
  for (const edge of edges) {
    if (!indegree.has(edge.from) || !indegree.has(edge.to)) {
      continue;
    }
    indegree.set(edge.to, (indegree.get(edge.to) || 0) + 1);
    adjacency.get(edge.from)?.push(edge.to);
  }
  const queue: string[] = [];
  for (const [nodeId, count] of indegree.entries()) {
    if (count === 0) {
      queue.push(nodeId);
    }
  }

  const visited = new Set<string>();
  while (queue.length > 0) {
    const nextId = queue.shift();
    if (!nextId || visited.has(nextId)) {
      continue;
    }
    visited.add(nextId);
    for (const target of adjacency.get(nextId) || []) {
      if (!visited.has(target)) {
        queue.push(target);
      }
    }
  }
  return visited;
}

export function validateDraftLocally(
  spec: WorkflowSpec,
  availableTools: string[]
): ValidationDiagnostic[] {
  const diagnostics: ValidationDiagnostic[] = [];
  const normalized = normalizeDraftSpec(spec);
  const nodeIds = new Set<string>();
  const nodeMap = new Map<string, NodeSpec>();
  const declaredStateKeys = new Set(
    normalized.state_schema.map((item) => String(item.key || "").trim()).filter(Boolean)
  );
  const toolSet = new Set(availableTools.map((item) => String(item).trim()).filter(Boolean));

  for (const node of normalized.nodes) {
    const nodeId = String(node.id || "").trim();
    if (!nodeId) {
      diagnostics.push({
        code: "SCHEMA_INVALID",
        severity: "error",
        message: "Node id is required.",
      });
      continue;
    }
    if (nodeIds.has(nodeId)) {
      diagnostics.push({
        code: "DUPLICATE_NODE_ID",
        severity: "error",
        message: `Duplicate node id '${nodeId}'.`,
        node_id: nodeId,
      });
    }
    nodeIds.add(nodeId);
    nodeMap.set(nodeId, node);

    for (const writeKey of node.writes || []) {
      if (!declaredStateKeys.has(writeKey)) {
        diagnostics.push({
          code: "STATE_KEY_WRITE_UNDECLARED",
          severity: "error",
          message: `Node '${nodeId}' writes undeclared key '${writeKey}'.`,
          node_id: nodeId,
        });
      }
    }

    if (node.type === "tool") {
      const toolConfig = (node.config || {}) as ToolNodeConfig;
      const toolName = String(toolConfig.tool_name || "").trim();
      if (!toolName) {
        diagnostics.push({
          code: "NODE_CONFIG_INVALID",
          severity: "error",
          message: `Tool node '${nodeId}' requires tool_name.`,
          node_id: nodeId,
        });
      } else if (!toolSet.has(toolName)) {
        diagnostics.push({
          code: "TOOL_NOT_FOUND",
          severity: "error",
          message: `Tool node '${nodeId}' references missing tool '${toolName}'.`,
          node_id: nodeId,
        });
      }
    }

    if (node.type === "conditional") {
      const config = (node.config || {}) as ConditionalNodeConfig;
      const expression = String(config.expression || "").trim();
      if (!expression) {
        const operator = String(config.operator || "").trim();
        const hasLegacy = String(config.field || "").trim() && operator;
        if (!hasLegacy || !VALID_CONDITIONAL_OPERATORS.has(operator)) {
          diagnostics.push({
            code: "CONDITIONAL_EXPRESSION_INVALID",
            severity: "error",
            message: `Conditional node '${nodeId}' requires expression or valid legacy rule.`,
            node_id: nodeId,
          });
        }
      } else {
        for (const ref of collectExprIdentifiers(expression)) {
          const root = ref.split(".")[0];
          if (!declaredStateKeys.has(ref) && !declaredStateKeys.has(root) && !BASE_KEYS.has(root)) {
            diagnostics.push({
              code: "CONDITIONAL_EXPRESSION_KEY_UNKNOWN",
              severity: "error",
              message: `Conditional '${nodeId}' references unknown key '${ref}'.`,
              node_id: nodeId,
            });
          }
        }
      }
    }

    if (node.type === "verify") {
      const config = (node.config || {}) as VerifyNodeConfig;
      const mode = String(config.mode || "rule").toLowerCase();
      if (mode !== "rule" && mode !== "llm") {
        diagnostics.push({
          code: "VERIFY_CONFIG_INVALID",
          severity: "error",
          message: `Verify node '${nodeId}' mode must be rule or llm.`,
          node_id: nodeId,
        });
      }
      if (mode === "rule" && !String(config.expression || "").trim()) {
        diagnostics.push({
          code: "VERIFY_CONFIG_INVALID",
          severity: "error",
          message: `Verify node '${nodeId}' requires expression in rule mode.`,
          node_id: nodeId,
        });
      }
      if (mode === "llm" && !String(config.prompt_template || "").trim()) {
        diagnostics.push({
          code: "VERIFY_CONFIG_INVALID",
          severity: "error",
          message: `Verify node '${nodeId}' requires prompt_template in llm mode.`,
          node_id: nodeId,
        });
      }
    }
  }

  const finalizeNodes = normalized.nodes.filter((node) => node.type === "finalize");
  if (finalizeNodes.length === 0) {
    diagnostics.push({
      code: "MISSING_FINALIZE_NODE",
      severity: "error",
      message: "Workflow requires at least one Finalize node.",
    });
  }

  for (const edge of normalized.edges) {
    if (!nodeMap.has(edge.from)) {
      diagnostics.push({
        code: "EDGE_NODE_MISSING",
        severity: "error",
        message: `Edge '${edge.id}' source '${edge.from}' does not exist.`,
        edge_id: edge.id,
      });
    }
    if (!nodeMap.has(edge.to)) {
      diagnostics.push({
        code: "EDGE_NODE_MISSING",
        severity: "error",
        message: `Edge '${edge.id}' target '${edge.to}' does not exist.`,
        edge_id: edge.id,
      });
    }
  }

  const cyclePath = detectCycle(normalized.nodes, normalized.edges);
  if (cyclePath && !normalized.allow_cycles) {
    diagnostics.push({
      code: "CYCLE_DETECTED",
      severity: "error",
      message: `Cycle detected: ${cyclePath.join(" -> ")}.`,
    });
  }

  const reachable = reachableFromEntry(normalized.nodes, normalized.edges);
  for (const node of normalized.nodes) {
    if (!reachable.has(node.id)) {
      diagnostics.push({
        code: "UNREACHABLE_NODE",
        severity: "error",
        message: `Node '${node.id}' is unreachable.`,
        node_id: node.id,
      });
    }
  }
  if (finalizeNodes.length > 0 && !finalizeNodes.some((node) => reachable.has(node.id))) {
    diagnostics.push({
      code: "FINALIZE_UNREACHABLE",
      severity: "error",
      message: "No finalize node is reachable from entry points.",
    });
  }

  if (!cyclePath) {
    const indegree = new Map<string, number>();
    const adjacency = new Map<string, string[]>();
    for (const node of normalized.nodes) {
      indegree.set(node.id, 0);
      adjacency.set(node.id, []);
    }
    for (const edge of normalized.edges) {
      if (!indegree.has(edge.from) || !indegree.has(edge.to)) {
        continue;
      }
      indegree.set(edge.to, (indegree.get(edge.to) || 0) + 1);
      adjacency.get(edge.from)?.push(edge.to);
    }
    const predecessors = new Map<string, Set<string>>();
    for (const node of normalized.nodes) {
      predecessors.set(node.id, new Set());
    }
    for (const edge of normalized.edges) {
      if (predecessors.has(edge.to)) {
        predecessors.get(edge.to)?.add(edge.from);
      }
    }
    const queue = normalized.nodes
      .map((node) => node.id)
      .filter((nodeId) => (indegree.get(nodeId) || 0) === 0);
    const availableByNode = new Map<string, Set<string>>();

    while (queue.length > 0) {
      const nodeId = queue.shift();
      if (!nodeId) {
        continue;
      }
      const baseAvailable = new Set<string>([...BASE_KEYS, ...declaredStateKeys]);
      for (const pred of predecessors.get(nodeId) || new Set<string>()) {
        const predKeys = availableByNode.get(pred);
        if (!predKeys) {
          continue;
        }
        for (const key of predKeys) {
          baseAvailable.add(key);
        }
      }

      const node = nodeMap.get(nodeId);
      if (node) {
        for (const key of node.reads || []) {
          if (!baseAvailable.has(key)) {
            diagnostics.push({
              code: "STATE_READ_NOT_AVAILABLE",
              severity: "error",
              message: `Node '${nodeId}' reads '${key}' before it is available.`,
              node_id: nodeId,
            });
          }
        }
        for (const key of node.writes || []) {
          baseAvailable.add(key);
        }
      }

      availableByNode.set(nodeId, baseAvailable);
      for (const next of adjacency.get(nodeId) || []) {
        indegree.set(next, (indegree.get(next) || 0) - 1);
        if ((indegree.get(next) || 0) === 0) {
          queue.push(next);
        }
      }
    }
  }

  return diagnostics;
}
