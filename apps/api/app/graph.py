from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import db

try:
    from saturday_agent.runtime.registry import WorkflowRegistry
    from saturday_agent.runtime.tracing import StepEvent
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    agent_src = repo_root / "apps/agent/src"
    if str(agent_src) not in sys.path:
        sys.path.append(str(agent_src))
    from saturday_agent.runtime.registry import WorkflowRegistry
    from saturday_agent.runtime.tracing import StepEvent


_REGISTRY = WorkflowRegistry()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    return {}


class StepRecorder:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.steps: List[Dict[str, Any]] = []
        self._step_index = 0
        self._node_events = 0

    @property
    def node_events(self) -> int:
        return self._node_events

    def log_ingest(self, input_data: Dict[str, Any], workflow_id: str) -> None:
        started_at = _to_iso(_now_utc())
        ended_at = _to_iso(_now_utc())
        self._log(
            name="ingest_input",
            status="ok",
            input_data=input_data,
            output_data={"workflow_id": workflow_id},
            started_at=started_at,
            ended_at=ended_at,
        )

    def emit(self, event: StepEvent) -> None:
        payload = _model_dump(event)
        started_at = str(payload.get("started_at") or _to_iso(_now_utc()))
        ended_at = str(payload.get("ended_at") or _to_iso(_now_utc()))
        self._node_events += 1
        self._log(
            name=str(payload.get("name") or "node"),
            status=str(payload.get("status") or "ok"),
            input_data=payload.get("input") if isinstance(payload.get("input"), dict) else {},
            output_data=payload.get("output") if isinstance(payload.get("output"), dict) else {},
            started_at=started_at,
            ended_at=ended_at,
        )

    def log_runtime_error(self, workflow_id: str, error: str) -> None:
        started_at = _to_iso(_now_utc())
        ended_at = _to_iso(_now_utc())
        self._log(
            name="runtime_error",
            status="error",
            input_data={"workflow_id": workflow_id},
            output_data={"error": error},
            started_at=started_at,
            ended_at=ended_at,
        )

    def _log(
        self,
        *,
        name: str,
        status: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        started_at: str,
        ended_at: str,
    ) -> None:
        db.add_step(
            run_id=self.run_id,
            step_index=self._step_index,
            name=name,
            input_json=_json_dumps(input_data),
            output_json=_json_dumps(output_data),
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        )
        self.steps.append(
            {
                "step_index": self._step_index,
                "name": name,
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
                "input": input_data,
                "output": output_data,
            }
        )
        self._step_index += 1


def list_workflows() -> List[Dict[str, str]]:
    return _REGISTRY.list_workflows()


def list_models() -> Dict[str, Any]:
    payload = _REGISTRY.list_models()
    if not isinstance(payload, dict):
        return {"models": [], "default_model": "", "ollama_status": "down_or_empty"}
    return payload


def list_tools() -> List[Dict[str, Any]]:
    return _REGISTRY.list_tools()


def route_workflow_type(
    *,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    workflow_type: Optional[str] = None,
) -> str:
    return _REGISTRY.route_workflow_type(
        task=task,
        context=context,
        workflow_type=workflow_type,
    )


def workflow_id_for_type(workflow_type: str) -> str:
    return _REGISTRY.get_workflow_id_for_type(workflow_type)


def compile_workflow(
    *,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    workflow_type: Optional[str] = None,
) -> Dict[str, Any]:
    return _REGISTRY.compile_workflow(
        task=task,
        context=context,
        workflow_type=workflow_type,
    )


def run_workflow(workflow_id: str, input: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    started_at = _now_utc()

    db.create_run(
        run_id,
        "workflow_run",
        _json_dumps({"workflow_id": workflow_id, "input": input}),
        _to_iso(started_at),
    )

    recorder = StepRecorder(run_id)
    recorder.log_ingest(input_data=input, workflow_id=workflow_id)

    status = "ok"
    workflow_type = ""
    output: Dict[str, Any] = {}

    try:
        result = _REGISTRY.run_workflow(
            workflow_id=workflow_id,
            input=input,
            step_emitter=recorder.emit,
        )
        workflow_type = str(result.get("workflow_type") or "")
        maybe_output = result.get("output")
        output = maybe_output if isinstance(maybe_output, dict) else {}
    except Exception as exc:
        status = "error"
        output = {"error": str(exc)}

        try:
            workflow_type = _REGISTRY.get_workflow(workflow_id).workflow_type
        except Exception:
            workflow_type = ""

        if recorder.node_events == 0:
            recorder.log_runtime_error(workflow_id=workflow_id, error=str(exc))

    ended_at = _now_utc()
    result_payload = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow_type": workflow_type,
        "status": status,
        "output": output,
        "steps": recorder.steps,
    }

    db.finish_run(
        run_id,
        status=status,
        ended_at=_to_iso(ended_at),
        result_json=_json_dumps(result_payload),
    )

    return result_payload


def run_chat_workflow(
    *,
    workflow_id: str,
    model_id: str,
    tool_ids: List[str],
    message: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    chat_context = dict(context or {})
    if thread_id:
        chat_context["thread_id"] = thread_id

    workflow_input: Dict[str, Any] = {
        "task": message,
        "messages": [{"role": "user", "content": message}],
        "context": chat_context,
        "model": model_id,
        "tool_ids": tool_ids,
    }

    workflow_result = run_workflow(workflow_id=workflow_id, input=workflow_input)
    status = str(workflow_result.get("status", "error"))
    output = workflow_result.get("output")
    output_text = ""
    if isinstance(output, dict):
        if status == "ok":
            output_text = str(output.get("answer", ""))
        else:
            output_text = str(output.get("error", "Workflow execution failed"))

    raw_steps = workflow_result.get("steps")
    steps: List[Dict[str, str]] = []
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            steps.append(
                {
                    "name": str(step.get("name", "")),
                    "status": str(step.get("status", "ok")),
                    "started_at": str(step.get("started_at", "")),
                    "ended_at": str(step.get("ended_at", "")),
                }
            )

    return {
        "run_id": str(workflow_result.get("run_id", "")),
        "workflow_id": workflow_id,
        "model_id": model_id,
        "tool_ids": tool_ids,
        "status": status,
        "output_text": output_text,
        "steps": steps,
    }
