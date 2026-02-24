from __future__ import annotations

import copy
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app import db
from app.services import qdrant_client as qdrant_client_service

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
    timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "0")),
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


def _normalize_tool_definitions(raw_tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        tool_id = str(item.get("id") or "").strip()
        if not tool_id:
            continue
        normalized.append(dict(item))
    return normalized


def _normalize_workflow_definitions(raw_workflows: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_workflows, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_workflows:
        if not isinstance(item, dict):
            continue
        workflow_id = str(item.get("id") or "").strip()
        if not workflow_id:
            continue
        normalized.append(dict(item))
    return normalized


def _merge_definitions(
    base_items: List[Dict[str, Any]],
    override_items: List[Dict[str, Any]],
    *,
    key: str,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in base_items:
        item_id = str(item.get(key) or "").strip()
        if not item_id:
            continue
        merged[item_id] = dict(item)
    for item in override_items:
        item_id = str(item.get(key) or "").strip()
        if not item_id:
            continue
        merged[item_id] = dict(item)
    return list(merged.values())


def _runtime_registry(
    *,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
    workflow_defs: Optional[List[Dict[str, Any]]] = None,
) -> WorkflowRegistry:
    normalized_tools = _normalize_tool_definitions(tool_defs or [])
    normalized_workflows = _normalize_workflow_definitions(workflow_defs or [])
    if not normalized_tools and not normalized_workflows:
        return _REGISTRY
    return WorkflowRegistry(
        dynamic_tool_definitions=normalized_tools,
        dynamic_workflow_definitions=normalized_workflows,
    )


def _runtime_qdrant_url() -> Optional[str]:
    try:
        return qdrant_client_service.get_qdrant_url()
    except Exception:
        return None


def _inject_runtime_qdrant_context(context: Dict[str, Any]) -> Dict[str, Any]:
    context_map = dict(context or {})
    existing = str(context_map.get("qdrant_url") or "").strip()
    if existing:
        return context_map

    runtime_url = str(_runtime_qdrant_url() or "").strip()
    if runtime_url:
        context_map["qdrant_url"] = runtime_url
    return context_map


def _inject_runtime_qdrant_input(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(input_payload or {})
    raw_context = payload.get("context")
    context_map = dict(raw_context) if isinstance(raw_context, dict) else {}
    payload["context"] = _inject_runtime_qdrant_context(context_map)
    return payload


def _inject_runtime_qdrant_state(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(state_payload or {})
    raw_context = payload.get("context")
    context_map = dict(raw_context) if isinstance(raw_context, dict) else {}
    payload["context"] = _inject_runtime_qdrant_context(context_map)
    return payload


def _clip_summary(value: str, *, limit: int = 140) -> str:
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized:
        return "Completed step."
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _first_line(value: Any) -> str:
    text = ""
    if isinstance(value, dict):
        text = str(value.get("message") or value.get("error") or "")
    else:
        text = str(value or "")
    return text.splitlines()[0].strip()


def _step_label(name: str) -> str:
    if name.startswith("tool."):
        tool_id = name.replace("tool.", "", 1).strip()
        return f"tool {tool_id}".strip()
    if name.startswith("node."):
        node_id = name.replace("node.", "", 1).strip()
        return f"node {node_id}".strip()
    return name.replace("_", " ").strip() or "step"


def _count_plan_items(plan_text: str) -> int:
    lines = [line.strip() for line in str(plan_text or "").splitlines() if line.strip()]
    bullets = [
        line
        for line in lines
        if re.match(r"^(\d+[.)]|[-*•])\s+", line)
    ]
    if bullets:
        return len(bullets)
    return len(lines)


def _summary_for_step(
    *,
    name: str,
    status: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
) -> str:
    normalized_status = str(status or "ok").lower()
    if normalized_status == "skipped":
        return _clip_summary(f"Skipped {_step_label(name)} during replay.")

    if normalized_status not in {"ok", "success", "completed"}:
        raw_error = (
            output_data.get("error")
            if isinstance(output_data, dict)
            else None
        )
        if raw_error is None and isinstance(output_data, dict):
            response = output_data.get("response")
            if isinstance(response, dict):
                raw_error = response.get("error")
        error_text = _first_line(raw_error) or "step failed"
        return _clip_summary(f"Failed in {_step_label(name)}: {error_text}")

    if name == "plan":
        plan_text = str(output_data.get("plan") or "")
        item_count = _count_plan_items(plan_text)
        return _clip_summary(f"Created a {item_count}-step plan.")

    if name == "rag_retrieve":
        retrieval = output_data.get("retrieval") if isinstance(output_data.get("retrieval"), dict) else {}
        results = retrieval.get("results") if isinstance(retrieval.get("results"), list) else []
        return _clip_summary(f"Retrieved {len(results)} relevant passages.")

    if name == "decide_tools":
        planned_calls = output_data.get("tool_calls") if isinstance(output_data.get("tool_calls"), list) else []
        return _clip_summary(f"Selected {len(planned_calls)} tools to run.")

    if name == "execute_tools":
        results = output_data.get("tool_results") if isinstance(output_data.get("tool_results"), list) else []
        ok_count = sum(
            1
            for item in results
            if isinstance(item, dict) and str(item.get("status") or "ok").lower() == "ok"
        )
        if len(results) == 0:
            return _clip_summary("No tools were executed.")
        if ok_count == len(results):
            return _clip_summary(f"Ran {ok_count} tools successfully.")
        return _clip_summary(f"Ran {len(results)} tools with {len(results) - ok_count} errors.")

    if name in {"llm_answer", "llm_execute", "llm_synthesize"}:
        return _clip_summary("Generated response content.")

    if name == "build_messages":
        return _clip_summary("Prepared conversation messages.")

    if name == "verify":
        verify_ok = bool(output_data.get("verify_ok"))
        return _clip_summary("Verification passed." if verify_ok else "Verification requested changes.")

    if name == "repair":
        retry_count = int(output_data.get("retry_count") or 0)
        return _clip_summary(f"Updated plan for retry {retry_count}.")

    if name == "finalize":
        return _clip_summary("Prepared final response.")

    if name == "ingest_input":
        return _clip_summary("Captured request input.")

    if name.startswith("tool."):
        tool_id = str(
            input_data.get("tool_id")
            or output_data.get("tool_id")
            or name.replace("tool.", "", 1)
        ).strip()
        response = output_data.get("response")
        response_data = response if isinstance(response, dict) else {}
        if tool_id == "rag.retrieve":
            payload = response_data.get("data") if isinstance(response_data.get("data"), dict) else {}
            results = payload.get("results") if isinstance(payload.get("results"), list) else []
            return _clip_summary(f"Retrieved {len(results)} relevant passages.")
        return _clip_summary(f"Ran {tool_id} successfully.")

    return _clip_summary(f"Completed {_step_label(name)}.")


def _tool_input_summary(tool_id: str, payload: Dict[str, Any]) -> str:
    if tool_id == "rag.retrieve":
        query = str(payload.get("query") or "").strip()
        return _clip_summary(f"Retrieval query: {query or 'unspecified'}")
    if tool_id == "vision.analyze":
        artifact_id = str(payload.get("artifact_id") or "").strip()
        if artifact_id:
            return _clip_summary(f"Analyzing artifact {artifact_id}.")
        return _clip_summary("Analyzing attached image input.")
    query = str(payload.get("query") or "").strip()
    if query:
        return _clip_summary(f"Input query: {query}")
    return _clip_summary(f"Running {tool_id} with {len(payload)} input fields.")


def _tool_output_summary(tool_id: str, output_data: Dict[str, Any], status: str) -> str:
    if str(status or "ok").lower() != "ok":
        response = output_data.get("response")
        error_text = _first_line(response if isinstance(response, dict) else output_data)
        return _clip_summary(error_text or f"{tool_id} failed.")

    response = output_data.get("response")
    response_data = response if isinstance(response, dict) else {}
    if tool_id == "rag.retrieve":
        payload = response_data.get("data") if isinstance(response_data.get("data"), dict) else {}
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        return _clip_summary(f"Retrieved {len(results)} relevant passages.")
    return _clip_summary(f"{tool_id} completed successfully.")


class StepRecorder:
    def __init__(
        self,
        run_id: str,
        *,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.steps: List[Dict[str, Any]] = []
        self._step_index = 0
        self._node_events = 0
        self._pending_steps: Dict[str, Dict[str, Any]] = {}
        self._event_sink = event_sink
        self._state_tracker = (
            copy.deepcopy(initial_state) if isinstance(initial_state, dict) else {}
        )
        self._state_seeded = isinstance(initial_state, dict)

    @property
    def node_events(self) -> int:
        return self._node_events

    def _current_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state_tracker)

    def _seed_state_if_needed(self, candidate: Dict[str, Any]) -> None:
        if self._state_seeded:
            return
        if isinstance(candidate, dict):
            self._state_tracker = copy.deepcopy(candidate)
            self._state_seeded = True

    def _derive_post_state(
        self,
        *,
        pre_state: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str,
        name: str,
    ) -> Dict[str, Any]:
        normalized_status = self._normalize_storage_status(status)
        if normalized_status in {"error", "skipped"}:
            return copy.deepcopy(pre_state)
        if name.startswith("tool."):
            return copy.deepcopy(pre_state)

        merged = copy.deepcopy(pre_state)
        for key, value in output_data.items():
            merged[str(key)] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _normalize_storage_status(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"ok", "success", "completed"}:
            return "success"
        if normalized in {"pending", "running", "skipped"}:
            return normalized
        return "error"

    @staticmethod
    def _stream_status(storage_status: str) -> str:
        return "error" if storage_status == "error" else "ok"

    @staticmethod
    def _step_identity(name: str) -> tuple[str, str]:
        raw_name = str(name or "").strip()
        if raw_name in {"ingest_input", "runtime_error"}:
            return raw_name, "system"
        if raw_name.startswith("tool."):
            return raw_name.replace("tool.", "", 1), "tool"
        if raw_name.startswith("node."):
            return raw_name.replace("node.", "", 1), "node"
        return raw_name, "node"

    def log_ingest(self, input_data: Dict[str, Any], workflow_id: str) -> None:
        self._seed_state_if_needed(input_data if isinstance(input_data, dict) else {})
        started_at = _to_iso(_now_utc())
        ended_at = _to_iso(_now_utc())
        pre_state = self._current_state()
        post_state = self._current_state()
        self._log(
            step_index=self._step_index,
            name="ingest_input",
            status="ok",
            input_data=input_data,
            output_data={"workflow_id": workflow_id},
            started_at=started_at,
            ended_at=ended_at,
            pre_state=pre_state,
            post_state=post_state,
            error_json=None,
            summary=_summary_for_step(
                name="ingest_input",
                status="ok",
                input_data=input_data,
                output_data={"workflow_id": workflow_id},
            ),
            label="ingest input",
        )
        self._step_index += 1

    def emit(self, event: StepEvent) -> None:
        payload = _model_dump(event)
        phase = str(payload.get("phase") or "end").lower()
        started_at = str(payload.get("started_at") or _to_iso(_now_utc()))
        ended_at = str(payload.get("ended_at") or _to_iso(_now_utc()))
        name = str(payload.get("name") or "node")
        status = str(payload.get("status") or "ok")
        label = str(payload.get("label") or _step_label(name))

        input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else {}
        output_payload = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        payload_pre_state = (
            copy.deepcopy(payload.get("pre_state"))
            if isinstance(payload.get("pre_state"), dict)
            else None
        )
        payload_post_state = (
            copy.deepcopy(payload.get("post_state"))
            if isinstance(payload.get("post_state"), dict)
            else None
        )

        if name.startswith("tool."):
            tool_id = str(
                input_payload.get("tool_id")
                or output_payload.get("tool_id")
                or name.replace("tool.", "", 1)
            ).strip()
            raw_tool_input = input_payload.get("payload")
            if not isinstance(raw_tool_input, dict):
                maybe_input = input_payload.get("input")
                raw_tool_input = maybe_input if isinstance(maybe_input, dict) else {}
            raw_tool_output = output_payload.get("response")
            if raw_tool_output is None:
                raw_tool_output = output_payload.get("output")

            input_payload = {
                "tool_id": tool_id,
                "payload": raw_tool_input if isinstance(raw_tool_input, dict) else {},
            }
            output_payload = {
                "tool_id": tool_id,
                "response": raw_tool_output,
            }

        key = f"{name}|{started_at}"
        if phase == "start":
            step_index = self._step_index
            self._step_index += 1
            self._pending_steps[key] = {
                "step_index": step_index,
                "label": label,
                "name": name,
                "started_at": started_at,
                "input": input_payload,
                "pre_state": payload_pre_state if isinstance(payload_pre_state, dict) else self._current_state(),
            }
            if self._event_sink:
                self._event_sink(
                    {
                        "type": "step_start",
                        "run_id": self.run_id,
                        "step_index": step_index,
                        "name": name,
                        "label": label,
                        "started_at": started_at,
                    }
                )
                if name.startswith("tool."):
                    tool_id = str(input_payload.get("tool_id") or name.replace("tool.", "", 1))
                    tool_input = (
                        input_payload.get("payload")
                        if isinstance(input_payload.get("payload"), dict)
                        else {}
                    )
                    self._event_sink(
                        {
                            "type": "tool_call",
                            "run_id": self.run_id,
                            "step_index": step_index,
                            "tool_id": tool_id,
                            "input_summary": _tool_input_summary(tool_id, tool_input),
                        }
                    )
            return

        pending = self._pending_steps.pop(key, None)
        pre_state: Dict[str, Any]
        if pending:
            step_index = int(pending.get("step_index") or 0)
            label = str(pending.get("label") or label)
            pre_state = (
                copy.deepcopy(pending.get("pre_state"))
                if isinstance(pending.get("pre_state"), dict)
                else self._current_state()
            )
        else:
            step_index = self._step_index
            self._step_index += 1
            pre_state = (
                copy.deepcopy(payload_pre_state)
                if isinstance(payload_pre_state, dict)
                else self._current_state()
            )

        post_state = (
            copy.deepcopy(payload_post_state)
            if isinstance(payload_post_state, dict)
            else self._derive_post_state(
                pre_state=pre_state,
                output_data=output_payload if isinstance(output_payload, dict) else {},
                status=status,
                name=name,
            )
        )
        self._state_tracker = copy.deepcopy(post_state)

        summary = _summary_for_step(
            name=name,
            status=status,
            input_data=input_payload if isinstance(input_payload, dict) else {},
            output_data=output_payload if isinstance(output_payload, dict) else {},
        )
        stored_status = self._normalize_storage_status(status)
        error_json: Optional[str] = None
        if stored_status == "error":
            raw_error = output_payload.get("error") if isinstance(output_payload, dict) else None
            error_json = _json_dumps(
                {
                    "message": _first_line(raw_error) or "Step failed.",
                    "raw": raw_error,
                }
            )

        self._node_events += 1
        step_id = self._log(
            step_index=step_index,
            name=name,
            status=stored_status,
            input_data=input_payload if isinstance(input_payload, dict) else {},
            output_data=output_payload if isinstance(output_payload, dict) else {},
            started_at=started_at,
            ended_at=ended_at,
            pre_state=pre_state,
            post_state=post_state,
            error_json=error_json,
            summary=summary,
            label=label,
        )

        if name.startswith("tool."):
            tool_id = str(output_payload.get("tool_id") or name.replace("tool.", "", 1))
            tool_args = (
                input_payload.get("payload")
                if isinstance(input_payload.get("payload"), dict)
                else {}
            )
            tool_response = output_payload.get("response")
            call_row = db.create_tool_call(
                run_id=self.run_id,
                step_id=step_id,
                tool_name=tool_id,
                args_json=_json_dumps(tool_args),
                started_at=started_at,
                approved_bool=True,
                status="completed" if stored_status == "success" else "error",
            )
            db.finish_tool_call(
                tool_call_id=str(call_row.get("tool_call_id") or ""),
                status="completed" if stored_status == "success" else "error",
                finished_at=ended_at,
                result_json=_json_dumps(tool_response),
                error_json=(
                    None
                    if stored_status == "success"
                    else _json_dumps({"message": _first_line(tool_response) or "Tool execution failed."})
                ),
            )

        if self._event_sink:
            self._event_sink(
                {
                    "type": "step_end",
                    "run_id": self.run_id,
                    "step_index": step_index,
                    "name": name,
                    "status": self._stream_status(stored_status),
                    "ended_at": ended_at,
                    "summary": summary,
                    "meta": {"label": label, "storage_status": stored_status},
                }
            )
            if name.startswith("tool."):
                tool_id = str(output_payload.get("tool_id") or name.replace("tool.", "", 1))
                self._event_sink(
                    {
                        "type": "tool_result",
                        "run_id": self.run_id,
                        "step_index": step_index,
                        "tool_id": tool_id,
                        "status": self._stream_status(stored_status),
                        "output_summary": _tool_output_summary(tool_id, output_payload, status),
                    }
                )

    def log_runtime_error(self, workflow_id: str, error: str) -> None:
        started_at = _to_iso(_now_utc())
        ended_at = _to_iso(_now_utc())
        pre_state = self._current_state()
        post_state = self._current_state()
        summary = _summary_for_step(
            name="runtime_error",
            status="error",
            input_data={"workflow_id": workflow_id},
            output_data={"error": error},
        )
        self._log(
            step_index=self._step_index,
            name="runtime_error",
            status="error",
            input_data={"workflow_id": workflow_id},
            output_data={"error": error},
            started_at=started_at,
            ended_at=ended_at,
            pre_state=pre_state,
            post_state=post_state,
            error_json=_json_dumps({"message": error}),
            summary=summary,
            label="runtime error",
        )
        self._step_index += 1

    def _log(
        self,
        *,
        step_index: int,
        name: str,
        status: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        started_at: str,
        ended_at: str,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        summary: str,
        label: str,
        error_json: Optional[str] = None,
    ) -> int:
        node_id, node_type = self._step_identity(name)
        step_id = db.add_step(
            run_id=self.run_id,
            step_index=step_index,
            name=name,
            input_json=_json_dumps(input_data),
            output_json=_json_dumps(output_data),
            summary=summary,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            node_id=node_id,
            node_type=node_type,
            error_json=error_json,
            pre_state_json=_json_dumps(pre_state),
            post_state_json=_json_dumps(post_state),
        )
        self.steps.append(
            {
                "step_id": step_id,
                "external_step_id": db.format_step_id(step_id),
                "step_index": step_index,
                "name": name,
                "node_id": node_id,
                "node_type": node_type,
                "label": label,
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
                "summary": summary,
                "input": input_data,
                "output": output_data,
                "pre_state": pre_state,
                "post_state": post_state,
            }
        )
        return step_id


def list_workflows() -> List[Dict[str, Any]]:
    user_workflows = db.list_workflows()
    registry = _runtime_registry(workflow_defs=user_workflows)
    return registry.list_workflows()


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
    user_tools = db.list_tools()
    registry = _runtime_registry(tool_defs=user_tools)
    return registry.list_tools()


def invoke_tool(
    *,
    tool_id: str,
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    started_at = _now_utc()

    db_tools = _normalize_tool_definitions(db.list_tools())
    merged_tool_defs = _merge_definitions(
        db_tools,
        _normalize_tool_definitions(tool_defs or []),
        key="id",
    )

    runtime_registry = _runtime_registry(tool_defs=merged_tool_defs)
    resolved_context = _inject_runtime_qdrant_context(dict(context or {}))
    payload = {
        "tool_id": str(tool_id or ""),
        "input": dict(tool_input or {}),
        "context": resolved_context,
    }

    db.create_run(
        run_id,
        "tool_invoke",
        _json_dumps(payload),
        _to_iso(started_at),
    )

    recorder = StepRecorder(run_id)
    recorder.log_ingest(input_data=payload, workflow_id=f"tool:{tool_id}")

    status = "ok"
    tool_output: Dict[str, Any] = {}

    step_started = _to_iso(_now_utc())
    recorder.emit(
        StepEvent(
            name=f"tool.{tool_id}",
            status="ok",
            phase="start",
            label=f"tool {tool_id}",
            started_at=step_started,
            ended_at=step_started,
            input={"tool_id": tool_id, "input": dict(tool_input or {})},
            output={},
        )
    )
    try:
        raw_output = runtime_registry.invoke_tool(
            tool_id=str(tool_id or ""),
            tool_input=dict(tool_input or {}),
            context=resolved_context,
        )
        tool_output = raw_output if isinstance(raw_output, dict) else {"value": raw_output}
        if not bool(tool_output.get("ok", True)):
            status = "error"
    except Exception as exc:
        status = "error"
        tool_output = {
            "ok": False,
            "error": {
                "kind": "runtime",
                "message": str(exc),
            },
        }
    step_ended = _to_iso(_now_utc())

    recorder.emit(
        StepEvent(
            name=f"tool.{tool_id}",
            status=status,
            phase="end",
            label=f"tool {tool_id}",
            started_at=step_started,
            ended_at=step_ended,
            input={"tool_id": tool_id, "input": dict(tool_input or {})},
            output={"tool_id": tool_id, "output": tool_output},
        )
    )

    ended_at = _to_iso(_now_utc())
    result_payload = {
        "run_id": run_id,
        "tool_id": str(tool_id or ""),
        "status": status,
        "output": tool_output,
        "steps": recorder.steps,
    }
    db.finish_run(
        run_id,
        status=status,
        ended_at=ended_at,
        result_json=_json_dumps(result_payload),
    )
    return result_payload


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


def _execute_workflow_run(
    *,
    workflow_id: str,
    input: Dict[str, Any],
    recorder: StepRecorder,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
    workflow_defs: Optional[List[Dict[str, Any]]] = None,
    tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    replay_mode: bool = False,
    resume_node_id: Optional[str] = None,
) -> Dict[str, Any]:
    db_tools = _normalize_tool_definitions(db.list_tools())
    db_workflows = _normalize_workflow_definitions(db.list_workflows())
    merged_tool_defs = _merge_definitions(
        db_tools,
        _normalize_tool_definitions(tool_defs or []),
        key="id",
    )
    merged_workflow_defs = _merge_definitions(
        db_workflows,
        _normalize_workflow_definitions(workflow_defs or []),
        key="id",
    )

    runtime_registry = _runtime_registry(
        tool_defs=merged_tool_defs,
        workflow_defs=merged_workflow_defs,
    )

    status = "ok"
    workflow_type = ""
    output: Dict[str, Any] = {}

    try:
        result = runtime_registry.run_workflow(
            workflow_id=workflow_id,
            input=input,
            step_emitter=recorder.emit,
            tool_call_gate=tool_call_gate,
            initial_state=initial_state,
            replay_mode=replay_mode,
            resume_node_id=resume_node_id,
        )
        workflow_type = str(result.get("workflow_type") or "")
        maybe_output = result.get("output")
        output = maybe_output if isinstance(maybe_output, dict) else {}
    except Exception as exc:
        error_text = str(exc)
        status = "rejected" if "rejected" in error_text.lower() else "error"
        output = {"error": error_text}

        try:
            workflow_type = runtime_registry.get_workflow(workflow_id).workflow_type
        except Exception:
            workflow_type = ""

        if recorder.node_events == 0:
            recorder.log_runtime_error(workflow_id=workflow_id, error=str(exc))

    return {
        "status": status,
        "workflow_type": workflow_type,
        "output": output,
    }


def run_workflow(
    workflow_id: str,
    input: Dict[str, Any],
    *,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
    workflow_defs: Optional[List[Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    workflow_version_id: Optional[str] = None,
    sandbox_mode: bool = False,
    tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    replay_mode: bool = False,
    resume_node_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    parent_step_id: Optional[str] = None,
    forked_from_state_json: Optional[str] = None,
    fork_patch_json: Optional[str] = None,
    fork_reason: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_run_id = str(run_id or uuid.uuid4())
    started_at = _now_utc()
    workflow_input = _inject_runtime_qdrant_input(dict(input or {}))
    resolved_initial_state = (
        _inject_runtime_qdrant_state(dict(initial_state))
        if isinstance(initial_state, dict)
        else None
    )
    run_mode = "replay" if replay_mode else "normal"
    run_payload = {"workflow_id": workflow_id, "input": workflow_input}
    if replay_mode:
        run_payload["replay"] = {
            "resume_node_id": str(resume_node_id or ""),
            "has_initial_state": bool(isinstance(resolved_initial_state, dict)),
        }

    db.create_run(
        resolved_run_id,
        "workflow_run",
        _json_dumps(run_payload),
        _to_iso(started_at),
        workflow_version_id=workflow_version_id,
        sandbox_mode=sandbox_mode,
        parent_run_id=parent_run_id,
        parent_step_id=parent_step_id,
        forked_from_state_json=forked_from_state_json,
        fork_patch_json=fork_patch_json,
        fork_reason=fork_reason,
        resume_from_node_id=resume_node_id,
        mode=run_mode,
    )

    recorder = StepRecorder(resolved_run_id, initial_state=resolved_initial_state)
    recorder.log_ingest(input_data=workflow_input, workflow_id=workflow_id)
    execution = _execute_workflow_run(
        workflow_id=workflow_id,
        input=workflow_input,
        recorder=recorder,
        tool_defs=tool_defs,
        workflow_defs=workflow_defs,
        tool_call_gate=tool_call_gate,
        initial_state=resolved_initial_state,
        replay_mode=replay_mode,
        resume_node_id=resume_node_id,
    )
    status = str(execution.get("status") or "error")
    workflow_type = str(execution.get("workflow_type") or "")
    output = execution.get("output") if isinstance(execution.get("output"), dict) else {}

    ended_at = _now_utc()
    result_payload = {
        "run_id": resolved_run_id,
        "workflow_id": workflow_id,
        "workflow_type": workflow_type,
        "workflow_version_id": workflow_version_id,
        "status": status,
        "sandbox_mode": sandbox_mode,
        "mode": run_mode,
        "parent_run_id": parent_run_id,
        "parent_step_id": parent_step_id,
        "resume_from_node_id": resume_node_id,
        "output": output,
        "steps": recorder.steps,
    }

    db.finish_run(
        resolved_run_id,
        status=status,
        ended_at=_to_iso(ended_at),
        result_json=_json_dumps(result_payload),
        error=str(output.get("error") or "") if status != "ok" else None,
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
    prepared = _prepare_chat_execution(
        workflow_id=workflow_id,
        model_id=model_id,
        tool_ids=tool_ids,
        message=message,
        vision_model_id=vision_model_id,
        artifact_ids=artifact_ids,
        context=context,
        thread_id=thread_id,
    )
    if not bool(prepared.get("ok")):
        return dict(prepared.get("response") or {})

    workflow_input = dict(prepared.get("workflow_input") or {})
    db_tool_defs = prepared.get("db_tool_defs")
    db_workflow_defs = prepared.get("db_workflow_defs")
    workflow_result = run_workflow(
        workflow_id=workflow_id,
        input=workflow_input,
        tool_defs=db_tool_defs if isinstance(db_tool_defs, list) else None,
        workflow_defs=db_workflow_defs if isinstance(db_workflow_defs, list) else None,
    )
    return _chat_response_from_workflow_result(
        workflow_result=workflow_result,
        workflow_id=workflow_id,
        model_id=model_id,
        resolved_tool_ids=list(prepared.get("resolved_tool_ids") or []),
    )


def run_chat_workflow_stream(
    *,
    workflow_id: str,
    model_id: str,
    tool_ids: List[str],
    message: str,
    emit_event: Callable[[Dict[str, Any]], None],
    vision_model_id: Optional[str] = None,
    artifact_ids: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    started_at = _to_iso(_now_utc())

    base_payload = {
        "workflow_id": workflow_id,
        "model_id": model_id,
        "tool_ids": list(tool_ids or []),
        "message": message,
        "vision_model_id": vision_model_id,
        "artifact_ids": list(artifact_ids or []),
        "context": dict(context or {}),
        "thread_id": thread_id,
        "stream": True,
    }
    db.create_run(
        run_id,
        "chat_stream",
        _json_dumps(base_payload),
        started_at,
    )

    requested_tool_ids = [str(item).strip() for item in tool_ids if str(item).strip()]

    emit_event(
        {
            "type": "run_start",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "model_id": model_id,
            "tool_ids": requested_tool_ids,
            "started_at": started_at,
        }
    )
    resolved_tool_ids = list(requested_tool_ids)
    recorder: Optional[StepRecorder] = None

    try:
        token_emitter = lambda token: emit_event({"type": "token", "run_id": run_id, "text": str(token)})
        prepared = _prepare_chat_execution(
            workflow_id=workflow_id,
            model_id=model_id,
            tool_ids=tool_ids,
            message=message,
            vision_model_id=vision_model_id,
            artifact_ids=artifact_ids,
            context=context,
            thread_id=thread_id,
            token_emitter=token_emitter,
        )
        resolved_tool_ids = list(prepared.get("resolved_tool_ids") or requested_tool_ids)

        if not bool(prepared.get("ok")):
            response_payload = dict(prepared.get("response") or {})
            error_text = str(response_payload.get("output_text") or "Workflow execution failed.")
            ended_at = _to_iso(_now_utc())
            emit_event({"type": "error", "run_id": run_id, "message": error_text})
            emit_event(
                {
                    "type": "final",
                    "run_id": run_id,
                    "status": "error",
                    "output_text": error_text,
                    "ended_at": ended_at,
                }
            )
            result_payload = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "model_id": model_id,
                "tool_ids": resolved_tool_ids,
                "status": "error",
                "output_text": error_text,
                "steps": [],
            }
            db.finish_run(
                run_id,
                status="error",
                ended_at=ended_at,
                result_json=_json_dumps(result_payload),
            )
            return result_payload

        workflow_input = dict(prepared.get("workflow_input") or {})
        recorder = StepRecorder(run_id, event_sink=emit_event)
        recorder.log_ingest(input_data=workflow_input, workflow_id=workflow_id)

        execution = _execute_workflow_run(
            workflow_id=workflow_id,
            input=workflow_input,
            recorder=recorder,
            tool_defs=prepared.get("db_tool_defs") if isinstance(prepared.get("db_tool_defs"), list) else None,
            workflow_defs=prepared.get("db_workflow_defs") if isinstance(prepared.get("db_workflow_defs"), list) else None,
        )
        status = str(execution.get("status") or "error")
        output = execution.get("output") if isinstance(execution.get("output"), dict) else {}
        output_text = (
            str(output.get("answer") or "")
            if status == "ok"
            else str(output.get("error") or "Workflow execution failed.")
        )
        ended_at = _to_iso(_now_utc())

        if status != "ok":
            emit_event({"type": "error", "run_id": run_id, "message": output_text})

        emit_event(
            {
                "type": "final",
                "run_id": run_id,
                "status": "ok" if status == "ok" else "error",
                "output_text": output_text,
                "ended_at": ended_at,
            }
        )

        result_payload = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "model_id": model_id,
            "tool_ids": resolved_tool_ids,
            "status": status,
            "output_text": output_text,
            "steps": _normalize_chat_steps(recorder.steps),
        }
        db.finish_run(
            run_id,
            status=status,
            ended_at=ended_at,
            result_json=_json_dumps(result_payload),
        )
        return result_payload
    except Exception as exc:
        error_text = _first_line(str(exc)) or "Workflow execution failed."
        ended_at = _to_iso(_now_utc())
        emit_event({"type": "error", "run_id": run_id, "message": error_text})
        emit_event(
            {
                "type": "final",
                "run_id": run_id,
                "status": "error",
                "output_text": error_text,
                "ended_at": ended_at,
            }
        )
        result_payload = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "model_id": model_id,
            "tool_ids": resolved_tool_ids,
            "status": "error",
            "output_text": error_text,
            "steps": _normalize_chat_steps(recorder.steps if recorder else []),
        }
        db.finish_run(
            run_id,
            status="error",
            ended_at=ended_at,
            result_json=_json_dumps(result_payload),
        )
        return result_payload


def _chat_error_response(
    *,
    workflow_id: str,
    model_id: str,
    output_text: str,
) -> Dict[str, Any]:
    return {
        "run_id": "",
        "workflow_id": workflow_id,
        "model_id": model_id,
        "tool_ids": [],
        "status": "error",
        "output_text": output_text,
        "steps": [],
    }


def _normalize_chat_steps(raw_steps: Any) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not isinstance(raw_steps, list):
        return steps
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        raw_status = str(step.get("status", "ok")).lower()
        normalized_status = "error" if raw_status in {"error", "failed", "rejected"} else "ok"
        steps.append(
            {
                "name": str(step.get("name", "")),
                "status": normalized_status,
                "started_at": str(step.get("started_at", "")),
                "ended_at": str(step.get("ended_at", "")),
                "summary": str(step.get("summary", "")),
            }
        )
    return steps


def _chat_response_from_workflow_result(
    *,
    workflow_result: Dict[str, Any],
    workflow_id: str,
    model_id: str,
    resolved_tool_ids: List[str],
) -> Dict[str, Any]:
    status = str(workflow_result.get("status", "error"))
    output = workflow_result.get("output")
    output_text = ""
    if isinstance(output, dict):
        if status == "ok":
            output_text = str(output.get("answer", ""))
        else:
            output_text = str(output.get("error", "Workflow execution failed"))

    return {
        "run_id": str(workflow_result.get("run_id", "")),
        "workflow_id": workflow_id,
        "model_id": model_id,
        "tool_ids": resolved_tool_ids,
        "status": status,
        "output_text": output_text,
        "steps": _normalize_chat_steps(workflow_result.get("steps")),
    }


def _prepare_chat_execution(
    *,
    workflow_id: str,
    model_id: str,
    tool_ids: List[str],
    message: str,
    vision_model_id: Optional[str] = None,
    artifact_ids: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    token_emitter: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    chat_context = _inject_runtime_qdrant_context(dict(context or {}))
    if thread_id:
        chat_context["thread_id"] = thread_id
    if token_emitter is not None:
        chat_context["_token_emitter"] = token_emitter

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

    workflows_map = {
        str(item.get("id") or ""): item
        for item in list_workflows()
        if isinstance(item, dict)
    }
    selected_workflow = workflows_map.get(workflow_id)
    if not isinstance(selected_workflow, dict):
        return {
            "ok": False,
            "resolved_tool_ids": [],
            "response": _chat_error_response(
                workflow_id=workflow_id,
                model_id=model_id,
                output_text=f"Unknown workflow_id: {workflow_id}",
            ),
        }
    if str(selected_workflow.get("status") or "").lower() == "disabled":
        return {
            "ok": False,
            "resolved_tool_ids": [],
            "response": _chat_error_response(
                workflow_id=workflow_id,
                model_id=model_id,
                output_text=f"Workflow '{workflow_id}' is disabled.",
            ),
        }

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

    available_tools = {str(tool.get("id") or ""): tool for tool in list_tools()}
    selected_tool_defs: List[Dict[str, Any]] = []
    deduped_tool_ids: List[str] = []
    seen_tool_ids: set[str] = set()
    for tool_id in resolved_tool_ids:
        normalized_tool_id = str(tool_id).strip()
        if not normalized_tool_id or normalized_tool_id in seen_tool_ids:
            continue
        seen_tool_ids.add(normalized_tool_id)

        tool_meta = available_tools.get(normalized_tool_id)
        if not isinstance(tool_meta, dict):
            continue
        if not bool(tool_meta.get("enabled", False)):
            continue

        deduped_tool_ids.append(normalized_tool_id)
        selected_tool_defs.append(dict(tool_meta))
    resolved_tool_ids = deduped_tool_ids

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

    if resolved_tool_ids:
        raw_tool_inputs = chat_context.get("tool_inputs")
        tool_inputs = dict(raw_tool_inputs) if isinstance(raw_tool_inputs, dict) else {}
        for tool_id in resolved_tool_ids:
            existing_input = tool_inputs.get(tool_id)
            tool_input = dict(existing_input) if isinstance(existing_input, dict) else {}
            if not str(tool_input.get("query") or "").strip():
                tool_input["query"] = str(message or "")
            tool_inputs[tool_id] = tool_input
        chat_context["tool_inputs"] = tool_inputs
        chat_context["tool_defs"] = selected_tool_defs

    db_tool_defs = db.list_tools()
    db_workflow_defs = db.list_workflows()
    workflow_input: Dict[str, Any] = {
        "task": message,
        "messages": [{"role": "user", "content": message}],
        "context": chat_context,
        "model": model_id,
        "tool_ids": resolved_tool_ids,
        "tool_defs": selected_tool_defs,
        "workflow_defs": db_workflow_defs,
    }
    return {
        "ok": True,
        "resolved_tool_ids": resolved_tool_ids,
        "selected_tool_defs": selected_tool_defs,
        "db_tool_defs": db_tool_defs,
        "db_workflow_defs": db_workflow_defs,
        "workflow_input": workflow_input,
    }
