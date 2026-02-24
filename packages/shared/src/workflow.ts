export type StateValueType = "string" | "number" | "bool" | "json";

export type NodeType = "llm" | "tool" | "conditional" | "verify" | "finalize";

export type DiagnosticSeverity = "error" | "warning" | "info";

export type ValidationDiagnosticCode =
  | "SCHEMA_INVALID"
  | "DUPLICATE_NODE_ID"
  | "MISSING_FINALIZE_NODE"
  | "EDGE_NODE_MISSING"
  | "CYCLE_DETECTED"
  | "UNREACHABLE_NODE"
  | "FINALIZE_UNREACHABLE"
  | "STATE_KEY_UNKNOWN"
  | "STATE_KEY_WRITE_UNDECLARED"
  | "STATE_READ_NOT_AVAILABLE"
  | "TOOL_NOT_FOUND"
  | "CONDITIONAL_EXPRESSION_INVALID"
  | "CONDITIONAL_EXPRESSION_KEY_UNKNOWN"
  | "VERIFY_CONFIG_INVALID"
  | "NODE_CONFIG_INVALID";

export type ValidationDiagnostic = {
  code: ValidationDiagnosticCode;
  severity: DiagnosticSeverity;
  message: string;
  node_id?: string;
  edge_id?: string;
  path?: string;
};

export type StateKeySpec = {
  key: string;
  type: StateValueType;
  description?: string;
  required?: boolean;
};

export type LLMNodeConfig = {
  prompt_template: string;
  system_prompt?: string;
  model?: string;
  temperature?: number;
  output_key?: string;
};

export type ToolNodeConfig = {
  tool_name: string;
  args_map?: Record<string, unknown>;
  output_key?: string;
};

export type ConditionalNodeConfig = {
  expression?: string;
  field?: string;
  operator?: "equals" | "contains" | "gt" | "lt" | "exists" | "not_exists" | "in";
  value?: unknown;
};

export type VerifyNodeConfig = {
  mode?: "rule" | "llm";
  expression?: string;
  prompt_template?: string;
  output_key?: string;
  fail_message?: string;
};

export type FinalizeNodeConfig = {
  response_template?: string;
  output_key?: string;
};

export type NodeConfigByType = {
  llm: LLMNodeConfig;
  tool: ToolNodeConfig;
  conditional: ConditionalNodeConfig;
  verify: VerifyNodeConfig;
  finalize: FinalizeNodeConfig;
};

export type NodeSpec<T extends NodeType = NodeType> = {
  id: string;
  type: T;
  label?: string;
  reads: string[];
  writes: string[];
  config: NodeConfigByType[T];
  position?: {
    x: number;
    y: number;
  };
};

export type EdgeSpec = {
  id: string;
  from: string;
  to: string;
  label?: string;
};

export type WorkflowSpec = {
  workflow_id?: string;
  name: string;
  description?: string;
  allow_cycles?: boolean;
  state_schema: StateKeySpec[];
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  metadata?: Record<string, unknown>;
};

export type WorkflowCompileResult = {
  valid: boolean;
  workflow_spec: WorkflowSpec;
  compiled: Record<string, unknown>;
  diagnostics: ValidationDiagnostic[];
};

export type ToolRegistryEntry = {
  id: string;
  name: string;
  enabled: boolean;
  type: string;
  kind: string;
  description?: string;
};

export const NODE_TYPE_OPTIONS: NodeType[] = [
  "llm",
  "tool",
  "conditional",
  "verify",
  "finalize",
];

export const DIAGNOSTIC_ERROR_SEVERITY: DiagnosticSeverity = "error";
