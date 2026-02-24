from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


NodeType = Literal["llm", "tool", "conditional", "verify", "finalize"]
StateValueType = Literal["string", "number", "bool", "json"]
DiagnosticSeverity = Literal["error", "warning", "info"]


class ValidationDiagnosticModel(BaseModel):
    code: str
    severity: DiagnosticSeverity = "error"
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    path: Optional[str] = None


class StateKeySpecModel(BaseModel):
    key: str
    type: StateValueType
    description: str = ""
    required: bool = False


class NodeSpecModel(BaseModel):
    id: str
    type: NodeType
    label: str = ""
    reads: List[str] = Field(default_factory=list)
    writes: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[Dict[str, float]] = None


class EdgeSpecModel(BaseModel):
    id: str = ""
    from_node: str = Field(alias="from")
    to: str
    label: str = "always"


class WorkflowSpecModel(BaseModel):
    workflow_id: Optional[str] = None
    name: str
    description: str = ""
    allow_cycles: bool = False
    state_schema: List[StateKeySpecModel] = Field(default_factory=list)
    nodes: List[NodeSpecModel] = Field(default_factory=list)
    edges: List[EdgeSpecModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowCompileRequestV2(BaseModel):
    workflow_spec: WorkflowSpecModel


class WorkflowCompileResponseV2(BaseModel):
    valid: bool
    workflow_spec: Dict[str, Any]
    compiled: Dict[str, Any]
    diagnostics: List[ValidationDiagnosticModel] = Field(default_factory=list)


class WorkflowVersionSummaryModel(BaseModel):
    version_id: str
    workflow_id: str
    version_num: int
    created_at: str
    created_by: str


class WorkflowDetailResponseModel(BaseModel):
    workflow_id: str
    name: str
    description: str = ""
    enabled: bool = True
    latest_version: Optional[Dict[str, Any]] = None
    versions: List[WorkflowVersionSummaryModel] = Field(default_factory=list)


class CreateWorkflowVersionRequest(BaseModel):
    workflow_spec: WorkflowSpecModel
    created_by: str = "builder"


class WorkflowRunRequestV2(BaseModel):
    workflow_version_id: Optional[str] = None
    workflow_id: Optional[str] = None
    draft_spec: Optional[WorkflowSpecModel] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    sandbox_mode: bool = False
    created_by: str = "builder"

    @model_validator(mode="after")
    def _validate_selector(self) -> "WorkflowRunRequestV2":
        selectors = [
            bool(str(self.workflow_version_id or "").strip()),
            bool(str(self.workflow_id or "").strip()),
            self.draft_spec is not None,
        ]
        if sum(1 for item in selectors if item) != 1:
            raise ValueError(
                "Exactly one of workflow_version_id, workflow_id, or draft_spec must be provided."
            )
        return self


class WorkflowRunResponseV2(BaseModel):
    run_id: str
    status: str
    workflow_id: str
    workflow_version_id: str
    workflow_version_num: int
    sandbox_mode: bool = False
    diagnostics: List[ValidationDiagnosticModel] = Field(default_factory=list)
    output: Optional[Dict[str, Any]] = None


class PendingToolCallModel(BaseModel):
    tool_call_id: str
    run_id: str
    step_id: Optional[int] = None
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    approved_bool: Optional[bool] = None
    started_at: str = ""
    status: str = "pending"


class PendingToolCallsResponse(BaseModel):
    run_id: str
    pending: List[PendingToolCallModel] = Field(default_factory=list)


class ToolCallApprovalRequest(BaseModel):
    approved: bool


class ToolCallApprovalResponse(BaseModel):
    run_id: str
    tool_call_id: str
    approved: bool
    status: str
