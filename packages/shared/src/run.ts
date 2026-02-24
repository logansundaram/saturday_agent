export type ReplayDiagnosticSeverity = "error" | "warning" | "info";

export type RunMode = "normal" | "replay";

export type Run = {
  run_id: string;
  kind?: string;
  status: string;
  workflow_id?: string;
  workflow_version_id?: string | null;
  workflow_type?: string | null;
  model_id?: string | null;
  tool_ids?: string[];
  started_at?: string | null;
  ended_at?: string | null;
  payload?: unknown;
  result?: unknown;
  sandbox_mode?: boolean;
  parent_run_id?: string | null;
  parent_step_id?: string | null;
  forked_from_state_json?: unknown;
  fork_patch_json?: unknown;
  fork_reason?: string | null;
  resume_from_node_id?: string | null;
  mode?: RunMode | string;
};

export type ToolCall = {
  tool_call_id: string;
  tool_name: string;
  status: string;
  approved_bool?: boolean | null;
  started_at?: string;
  finished_at?: string;
  args?: Record<string, unknown>;
  result?: unknown;
  error?: unknown;
};

export type RunStepSummary = {
  step_id: string;
  step_index: number;
  name: string;
  node_id?: string | null;
  node_type?: string | null;
  status: string;
  started_at: string;
  ended_at: string;
  summary?: string | null;
  replayable: boolean;
  replay_disabled_reason?: string | null;
};

export type RunStepDetail = {
  step_id: string;
  step_index: number;
  name: string;
  node_id?: string | null;
  node_type?: string | null;
  status: string;
  started_at: string;
  ended_at: string;
  summary?: string | null;
  input?: unknown;
  output?: unknown;
  error?: unknown;
  pre_state?: unknown;
  post_state?: unknown;
  tool_calls: ToolCall[];
  replayable: boolean;
  replay_disabled_reason?: string | null;
};

export type ReplayPatchMode = "overlay" | "replace" | "jsonpatch";
export type ReplayBaseState = "pre" | "post";

export type ReplayRequest = {
  from_step_id: string;
  state_patch?: unknown;
  patch_mode?: ReplayPatchMode;
  sandbox?: boolean;
  base_state?: ReplayBaseState;
  replay_this_step?: boolean;
};

export type ReplayDiagnostic = {
  code: string;
  severity: ReplayDiagnosticSeverity;
  message: string;
  path?: string | null;
  expected?: string | null;
  actual?: string | null;
};

export type ReplayResponse = {
  new_run_id?: string | null;
  diagnostics: ReplayDiagnostic[];
  fork_start_state?: Record<string, unknown> | null;
  resume_node_id?: string | null;
};

export type ReplayDryRunResponse = ReplayResponse;
