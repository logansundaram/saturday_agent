from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from pydantic import BaseModel, Field


class StepEvent(BaseModel):
    name: str
    status: str
    started_at: str
    ended_at: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


StepEmitter = Callable[[StepEvent], None]


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _state_snapshot(state: Mapping[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages")
    tool_calls = state.get("tool_calls")
    tool_results = state.get("tool_results")
    return {
        "task": state.get("task", ""),
        "plan": state.get("plan"),
        "answer": state.get("answer"),
        "retry_count": int(state.get("retry_count") or 0),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "tool_result_count": len(tool_results) if isinstance(tool_results, list) else 0,
    }


def instrument_node(
    *,
    name: str,
    node_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
    step_emitter: Optional[StepEmitter] = None,
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    def wrapped(state: Mapping[str, Any]) -> Dict[str, Any]:
        started_dt = datetime.now(timezone.utc)
        started_at = _to_iso(started_dt)
        input_snapshot = _state_snapshot(state)

        try:
            output = node_fn(state)
            ended_at = _to_iso(datetime.now(timezone.utc))
            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=name,
                        status="ok",
                        started_at=started_at,
                        ended_at=ended_at,
                        input=input_snapshot,
                        output=dict(output or {}),
                    )
                )
            return output
        except Exception as exc:
            ended_at = _to_iso(datetime.now(timezone.utc))
            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=name,
                        status="error",
                        started_at=started_at,
                        ended_at=ended_at,
                        input=input_snapshot,
                        output={"error": str(exc)},
                    )
                )
            raise

    return wrapped
