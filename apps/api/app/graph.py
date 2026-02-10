from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import db

try:
    from saturday_agent.llms.vision_registry import VisionModelRegistry
    from saturday_agent.runtime.registry import WorkflowRegistry
    from saturday_agent.runtime.tracing import StepEvent
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    agent_src = repo_root / "apps/agent/src"
    if str(agent_src) not in sys.path:
        sys.path.append(str(agent_src))
    from saturday_agent.llms.vision_registry import VisionModelRegistry
    from saturday_agent.runtime.registry import WorkflowRegistry
    from saturday_agent.runtime.tracing import StepEvent


_REGISTRY = WorkflowRegistry()
_VISION_REGISTRY = VisionModelRegistry(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    default_model=(
        os.getenv("VITE_VISION_DEFAULT_MODEL")
        or os.getenv("VISION_DEFAULT_MODEL")
        or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ),
    timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "30")),
    allowlist_raw=os.getenv("VISION_MODEL_ALLOWLIST", ""),
)


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


def list_text_models() -> Dict[str, Any]:
    return list_models()


def list_vision_models() -> Dict[str, Any]:
    payload = _VISION_REGISTRY.list_models_payload()
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
    vision_model_id: Optional[str] = None,
    artifact_ids: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    chat_context = dict(context or {})
    if thread_id:
        chat_context["thread_id"] = thread_id

    normalized_artifact_ids = [
        str(item).strip()
        for item in (artifact_ids or [])
        if str(item).strip()
    ]
    if normalized_artifact_ids:
        chat_context["artifact_ids"] = normalized_artifact_ids

    selected_vision_model = str(vision_model_id or "").strip()
    if selected_vision_model:
        chat_context["vision_model_id"] = selected_vision_model

    resolved_tool_ids = [str(item).strip() for item in tool_ids if str(item).strip()]
    if not normalized_artifact_ids or not selected_vision_model:
        resolved_tool_ids = [tool_id for tool_id in resolved_tool_ids if tool_id != "vision.analyze"]
    if (
        normalized_artifact_ids
        and selected_vision_model
        and workflow_id in {"moderate.v1", "complex.v1"}
        and "vision.analyze" not in resolved_tool_ids
    ):
        resolved_tool_ids.insert(0, "vision.analyze")

    if normalized_artifact_ids and selected_vision_model:
        raw_tool_inputs = chat_context.get("tool_inputs")
        tool_inputs = dict(raw_tool_inputs) if isinstance(raw_tool_inputs, dict) else {}
        existing_vision_input = tool_inputs.get("vision.analyze")
        vision_tool_input = (
            dict(existing_vision_input) if isinstance(existing_vision_input, dict) else {}
        )
        if not str(vision_tool_input.get("artifact_id") or "").strip():
            if workflow_id == "moderate.v1" and len(normalized_artifact_ids) > 1:
                vision_tool_input["artifact_id"] = ",".join(normalized_artifact_ids)
            else:
                vision_tool_input["artifact_id"] = normalized_artifact_ids[0]
        if not str(vision_tool_input.get("prompt") or "").strip():
            vision_tool_input["prompt"] = str(message or "Analyze the attached image.")
        vision_tool_input["vision_model_id"] = selected_vision_model
        tool_inputs["vision.analyze"] = vision_tool_input
        chat_context["tool_inputs"] = tool_inputs

    workflow_input: Dict[str, Any] = {
        "task": message,
        "messages": [{"role": "user", "content": message}],
        "context": chat_context,
        "model": model_id,
        "tool_ids": resolved_tool_ids,
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
        "tool_ids": resolved_tool_ids,
        "status": status,
        "output_text": output_text,
        "steps": steps,
    }
