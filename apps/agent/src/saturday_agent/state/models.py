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


class WorkflowState(TypedDict):
    task: str
    context: Dict[str, Any]
    messages: List[Dict[str, str]]
    artifact_ids: List[str]
    vision_model_id: Optional[str]
    plan: Optional[str]
    answer: Optional[str]
    artifacts: Dict[str, Any]
    tool_calls: List[ToolCallRecord]
    tool_results: List[ToolResultRecord]
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

    return WorkflowState(
        task=str(task or ""),
        context=context_map,
        messages=normalize_messages(messages or []),
        artifact_ids=artifact_ids,
        vision_model_id=vision_model_id,
        plan=None,
        answer=None,
        artifacts={},
        tool_calls=[],
        tool_results=[],
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
