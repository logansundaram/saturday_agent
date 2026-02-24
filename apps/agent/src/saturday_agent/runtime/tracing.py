from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, Field


class StepEvent(BaseModel):
    name: str
    status: str
    phase: Literal["start", "end"] = "end"
    label: Optional[str] = None
    started_at: str
    ended_at: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None


StepEmitter = Callable[[StepEvent], None]


class ReplayControl:
    def __init__(
        self,
        *,
        replay_mode: bool = False,
        resume_node_id: Optional[str] = None,
    ) -> None:
        self.replay_mode = bool(replay_mode)
        self.resume_node_id = self._normalize_node_id(str(resume_node_id or ""))
        self._resume_reached = not self.replay_mode or not bool(self.resume_node_id)
        self._resume_matched = self._resume_reached

    @staticmethod
    def _normalize_node_id(value: str) -> str:
        text = str(value or "").strip()
        if text.startswith("node."):
            text = text.replace("node.", "", 1)
        elif text.startswith("tool."):
            text = text.replace("tool.", "", 1)
        text = re.sub(r"\s+", " ", text)
        return text

    def should_skip(self, node_name: str) -> bool:
        if not self.replay_mode:
            return False
        if self._resume_reached:
            return False
        normalized = self._normalize_node_id(node_name)
        if normalized == self.resume_node_id:
            self._resume_reached = True
            self._resume_matched = True
            return False
        return True

    @property
    def resume_matched(self) -> bool:
        return self._resume_matched


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=True))
    except Exception:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)


def _state_snapshot(state: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    return _json_safe(dict(state))


def _merge_state_update(
    *,
    pre_state: Dict[str, Any],
    update: Dict[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(pre_state)
    for key, value in update.items():
        merged[str(key)] = _json_safe(value)
    return merged


def instrument_node(
    *,
    name: str,
    node_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
    step_emitter: Optional[StepEmitter] = None,
    replay_control: Optional[ReplayControl] = None,
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    def wrapped(state: Mapping[str, Any]) -> Dict[str, Any]:
        started_dt = datetime.now(timezone.utc)
        started_at = _to_iso(started_dt)
        pre_state = _state_snapshot(state)
        input_snapshot = copy.deepcopy(pre_state)

        if replay_control and replay_control.should_skip(name):
            skipped_at = _to_iso(datetime.now(timezone.utc))
            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=name,
                        status="skipped",
                        phase="end",
                        label=name.replace("_", " "),
                        started_at=started_at,
                        ended_at=skipped_at,
                        input=input_snapshot,
                        output={},
                        pre_state=copy.deepcopy(pre_state),
                        post_state=copy.deepcopy(pre_state),
                    )
                )
            return {}

        if step_emitter:
            step_emitter(
                StepEvent(
                    name=name,
                    status="ok",
                    phase="start",
                    label=name.replace("_", " "),
                    started_at=started_at,
                    ended_at=started_at,
                    input=input_snapshot,
                    output={},
                    pre_state=copy.deepcopy(pre_state),
                    post_state=None,
                )
            )

        try:
            output = node_fn(state)
            output_map = _json_safe(dict(output or {})) if isinstance(output, dict) else {}
            post_state = _merge_state_update(pre_state=pre_state, update=output_map)
            ended_at = _to_iso(datetime.now(timezone.utc))
            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=name,
                        status="ok",
                        phase="end",
                        label=name.replace("_", " "),
                        started_at=started_at,
                        ended_at=ended_at,
                        input=input_snapshot,
                        output=copy.deepcopy(output_map),
                        pre_state=copy.deepcopy(pre_state),
                        post_state=copy.deepcopy(post_state),
                    )
                )
            return output_map
        except Exception as exc:
            ended_at = _to_iso(datetime.now(timezone.utc))
            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=name,
                        status="error",
                        phase="end",
                        label=name.replace("_", " "),
                        started_at=started_at,
                        ended_at=ended_at,
                        input=input_snapshot,
                        output={"error": str(exc)},
                        pre_state=copy.deepcopy(pre_state),
                        post_state=copy.deepcopy(pre_state),
                    )
                )
            raise

    return wrapped
