export type ToolImplementationKind =
  | "builtin"
  | "python_module"
  | "http"
  | "prompt";

export type ToolSource = "builtin" | "custom" | string;

export type JsonSchema = Record<string, unknown>;

export type ToolSpec = {
  tool_id: string;
  name: string;
  description: string;
  input_schema: JsonSchema;
  output_schema: JsonSchema;
  implementation_kind: ToolImplementationKind;
  implementation_ref: string;
  enabled: boolean;
  version: number;
  deleted_at?: string | null;
  source: ToolSource;
  created_at?: string | null;
  updated_at?: string | null;
  deprecated_reason?: string | null;
  kind?: string;
  type?: string;
  config?: Record<string, unknown>;
};

export type ToolRegistryEntry = ToolSpec & {
  id: string;
};
