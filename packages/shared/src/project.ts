export type ProjectSummary = {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  chat_count?: number;
  document_count?: number;
  last_run_id?: string | null;
  last_run_at?: string | null;
};

export type ProjectChat = {
  id: string;
  project_id: string;
  name: string;
  created_at: string;
  last_run_id?: string | null;
  last_run_status?: string | null;
  last_run_at?: string | null;
};

export type ProjectDocument = {
  id: string;
  project_id: string;
  filename: string;
  filepath: string;
  embedding_model: string;
  created_at: string;
  updated_at: string;
  status: string;
  chunk_count?: number;
  error_message?: string | null;
};

export type ProjectGroundTruth = {
  project_id: string;
  content: string;
  updated_at: string;
  used_in_last_run?: boolean;
  last_run_id?: string | null;
  last_run_at?: string | null;
};

export type ProjectToolBinding = {
  id: string;
  tool_name: string;
  name: string;
  description: string;
  enabled: boolean;
  version: number;
  kind?: string;
  type?: string;
  source?: string;
  project_binding_id?: string | null;
  bound?: boolean;
};

export type ProjectWorkflow = {
  id?: string | null;
  project_id?: string | null;
  workflow_spec: Record<string, unknown>;
  compiled?: Record<string, unknown>;
  diagnostics?: Array<Record<string, unknown>>;
  valid?: boolean;
  is_default?: boolean;
  created_at?: string | null;
};

export type ProjectDetail = ProjectSummary & {
  chats: ProjectChat[];
  documents: ProjectDocument[];
  ground_truth: ProjectGroundTruth;
  tools: ProjectToolBinding[];
  workflow?: ProjectWorkflow | null;
};
