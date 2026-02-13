from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    name: str
    status: str = "ok"
    summary: str = ""
    timestamp: str = Field(default_factory=lambda: utc_now_iso())


class StepSummary(TypedDict):
    name: str
    status: str
    summary: str
    timestamp: str


class ToolCallRecord(TypedDict):
    tool_id: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    status: str


class ToolResultRecord(TypedDict):
    tool_id: str
    status: str
    output: Dict[str, Any]


class ToolHttpConfig(TypedDict, total=False):
    url: str
    method: str
    headers: Dict[str, str]
    timeout_ms: int


class ToolPythonConfig(TypedDict, total=False):
    code: str
    timeout_ms: int
    allowed_imports: List[str]


class ToolPromptConfig(TypedDict, total=False):
    prompt_template: str
    system_prompt: str
    temperature: float
    timeout_ms: int


ToolConfig = ToolHttpConfig | ToolPythonConfig | ToolPromptConfig


class ToolDefinition(TypedDict, total=False):
    id: str
    name: str
    kind: str
    type: str
    description: str
    enabled: bool
    source: str
    config: ToolConfig
    input_schema: Any
    output_schema: Any
    created_at: str
    updated_at: str


class WorkflowNode(TypedDict, total=False):
    id: str
    type: str
    config: Dict[str, Any]


class WorkflowEdge(TypedDict, total=False):
    from_: str
    to: str
    condition: str


class WorkflowGraph(TypedDict, total=False):
    nodes: List[WorkflowNode]
    edges: List[Dict[str, Any]]


class WorkflowDefinition(TypedDict, total=False):
    id: str
    name: str
    title: str
    description: str
    enabled: bool
    source: str
    type: str
    graph: WorkflowGraph
    created_at: str
    updated_at: str


class WorkflowState(TypedDict):
    task: str
    context: Dict[str, Any]
    messages: List[Dict[str, str]]
    artifact_ids: List[str]
    vision_model_id: Optional[str]
    embedding_model: Optional[str]
    retrieval_collection: Optional[str]
    plan: Optional[str]
    answer: Optional[str]
    artifacts: Dict[str, Any]
    retrieval: Dict[str, Any]
    citations: List[Dict[str, Any]]
    tool_calls: List[ToolCallRecord]
    tool_results: List[ToolResultRecord]
    tool_defs: List[ToolDefinition]
    workflow_defs: List[WorkflowDefinition]
    trace: List[StepSummary]
    verify_ok: Optional[bool]
    verify_notes: Optional[str]
    retry_count: int
    model: Optional[str]
    options: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    normalized: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", "user") or "user").lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = str(item.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_string_list(raw_items: Any) -> List[str]:
    if not isinstance(raw_items, list):
        return []
    normalized: List[str] = []
    for item in raw_items:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized


def build_initial_state(
    *,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    default_model: Optional[str] = None,
) -> WorkflowState:
    context_map = dict(context or {})
    options_map = dict(options or {})
    artifact_ids = normalize_string_list(context_map.get("artifact_ids"))
    raw_vision_model = context_map.get("vision_model_id")
    if raw_vision_model is None:
        raw_vision_model = options_map.get("vision_model_id")
    vision_model_id = str(raw_vision_model).strip() if raw_vision_model else None
    raw_embedding_model = context_map.get("embedding_model")
    if raw_embedding_model is None:
        raw_embedding_model = options_map.get("embedding_model")
    embedding_model = str(raw_embedding_model).strip() if raw_embedding_model else None

    raw_retrieval_collection = context_map.get("retrieval_collection")
    if raw_retrieval_collection is None:
        raw_retrieval_collection = options_map.get("retrieval_collection")
    retrieval_collection = (
        str(raw_retrieval_collection).strip() if raw_retrieval_collection else None
    )

    raw_tool_defs = context_map.get("tool_defs")
    tool_defs = (
        [dict(item) for item in raw_tool_defs if isinstance(item, Mapping)]
        if isinstance(raw_tool_defs, list)
        else []
    )

    raw_workflow_defs = context_map.get("workflow_defs")
    workflow_defs = (
        [dict(item) for item in raw_workflow_defs if isinstance(item, Mapping)]
        if isinstance(raw_workflow_defs, list)
        else []
    )

    return WorkflowState(
        task=str(task or ""),
        context=context_map,
        messages=normalize_messages(messages or []),
        artifact_ids=artifact_ids,
        vision_model_id=vision_model_id,
        embedding_model=embedding_model,
        retrieval_collection=retrieval_collection,
        plan=None,
        answer=None,
        artifacts={},
        retrieval={},
        citations=[],
        tool_calls=[],
        tool_results=[],
        tool_defs=tool_defs,
        workflow_defs=workflow_defs,
        trace=[],
        verify_ok=None,
        verify_notes=None,
        retry_count=0,
        model=str(model) if model else default_model,
        options=options_map,
    )


def append_trace(
    state: WorkflowState,
    *,
    name: str,
    summary: str,
    status: str = "ok",
) -> List[StepSummary]:
    current = list(state.get("trace") or [])
    current.append(
        StepSummary(
            name=name,
            status=status,
            summary=summary,
            timestamp=utc_now_iso(),
        )
    )
    return current
