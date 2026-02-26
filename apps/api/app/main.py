from __future__ import annotations

import asyncio
import copy
import ipaddress
import json
import os
import re
import socket
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app import artifacts, db, graph, streaming
from app.services import qdrant_client as qdrant_client_service
from app.workflows import service as workflow_service
from app.workflows.validation import BASE_STATE_KEYS

try:
    from dotenv import load_dotenv

    _MAIN_FILE = Path(__file__).resolve()
    _API_DIR = _MAIN_FILE.parents[1]
    _REPO_ROOT = _MAIN_FILE.parents[3]

    for _dotenv_path in (
        _REPO_ROOT / ".local.env",
        _API_DIR / ".local.env",
        _REPO_ROOT / ".env",
        _API_DIR / ".env",
    ):
        if _dotenv_path.exists():
            load_dotenv(dotenv_path=_dotenv_path, override=False)
except Exception:
    pass

WorkflowType = Literal["simple", "moderate", "complex"]
ToolType = Literal["http", "python", "prompt"]
ToolKind = Literal["local", "external"]


class ChatRunRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    workflow_id: str
    model_id: str
    tool_ids: List[str] = Field(default_factory=list)
    vision_model_id: Optional[str] = None
    artifact_ids: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    stream: bool = False


class ChatRunStep(BaseModel):
    name: str
    status: Literal["ok", "error"]
    started_at: str
    ended_at: str
    summary: Optional[str] = None


class ChatRunResponse(BaseModel):
    run_id: str
    workflow_id: str
    model_id: str
    tool_ids: List[str]
    output_text: str
    steps: List[ChatRunStep]


class HealthResponse(BaseModel):
    api: str
    ollama: Literal["ok", "down"]
    ollama_base_url: str
    model_default: str


class QdrantConfigRequest(BaseModel):
    url: str


class QdrantConfigResponse(BaseModel):
    ok: bool
    url: str


class RagHealthResponse(BaseModel):
    qdrantReachable: bool
    url: Optional[str] = None
    error: Optional[str] = None


class WorkflowSummary(BaseModel):
    id: str
    title: str
    name: Optional[str] = None
    description: str
    type: str
    version: str
    status: str
    source: str = "builtin"
    enabled: bool = True


class WorkflowsResponse(BaseModel):
    workflows: List[WorkflowSummary]


class ModelSummary(BaseModel):
    id: str
    name: str
    source: str = "ollama"
    status: str = "installed"


class ModelsResponse(BaseModel):
    models: List[ModelSummary]
    default_model: str
    ollama_status: Optional[str] = None


class ArtifactUploadResponse(BaseModel):
    artifact_id: str
    mime: str
    size: int
    sha256: str


class ToolSummary(BaseModel):
    id: str
    name: str
    kind: str = "external"
    type: str = "http"
    description: str
    enabled: bool
    source: str = "builtin"
    config: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Any = None
    output_schema: Any = None
    created_at: str
    updated_at: str


class ToolsResponse(BaseModel):
    tools: List[ToolSummary]


class ToolInvokeRequest(BaseModel):
    tool_id: str
    input: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None


class ToolInvokeResponse(BaseModel):
    run_id: str
    tool_id: str
    status: str
    output: Any = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)


class CreateToolRequest(BaseModel):
    name: str
    id: Optional[str] = None
    kind: ToolKind = "external"
    description: str = ""
    type: ToolType
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    input_schema: Any = None
    output_schema: Any = None


class UpdateToolRequest(BaseModel):
    enabled: bool


class WorkflowNodeSchema(BaseModel):
    id: str
    type: Literal["start", "llm", "tool", "condition", "end"]
    config: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEdgeSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(alias="from")
    to: str
    condition: Literal["always", "true", "false"] = "always"


class WorkflowGraphSchema(BaseModel):
    nodes: List[WorkflowNodeSchema] = Field(default_factory=list)
    edges: List[WorkflowEdgeSchema] = Field(default_factory=list)


class WorkflowDefinitionResponse(BaseModel):
    id: str
    name: str
    title: Optional[str] = None
    description: str
    enabled: bool
    source: str = "custom"
    type: str = "custom"
    graph: Dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    created_at: str
    updated_at: str


class CreateWorkflowRequest(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    enabled: bool = True
    graph: WorkflowGraphSchema


class UpdateWorkflowRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    graph: Optional[WorkflowGraphSchema] = None


class WorkflowCompileRequest(BaseModel):
    workflow_spec: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    workflow_type: Optional[str] = None


class WorkflowCompileResponse(BaseModel):
    valid: bool = True
    workflow_id: str = ""
    workflow_type: str = ""
    graph: Dict[str, Any] = Field(default_factory=dict)
    workflow_spec: Dict[str, Any] = Field(default_factory=dict)
    compiled: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowRunRequest(BaseModel):
    workflow_version_id: Optional[str] = None
    workflow_id: Optional[str] = None
    draft_spec: Optional[Dict[str, Any]] = None
    input: Dict[str, Any]
    sandbox_mode: bool = False
    created_by: str = "builder"


class WorkflowRunResponse(BaseModel):
    run_id: str
    workflow_id: str
    workflow_type: str = "custom"
    workflow_version_id: str = ""
    workflow_version_num: int = 0
    status: str
    sandbox_mode: bool = False
    output: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)


class CreateWorkflowVersionRequest(BaseModel):
    workflow_spec: Dict[str, Any]
    created_by: str = "builder"


class RunStepError(BaseModel):
    message: str
    stack: Optional[str] = None


class RunLogStep(BaseModel):
    step_id: str
    step_index: int
    name: str
    status: str
    started_at: str
    ended_at: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    summary: Optional[str] = None
    input: Any = None
    output: Any = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[RunStepError] = None


class RunLogsResponse(BaseModel):
    run_id: str
    steps: List[RunLogStep]


class RunDetailResponse(BaseModel):
    run_id: str
    kind: Literal["chat", "workflow"]
    status: str
    workflow_id: str
    workflow_version_id: Optional[str] = None
    workflow_type: Optional[str] = None
    model_id: Optional[str] = None
    tool_ids: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    payload: Any = None
    result: Any = None
    sandbox_mode: bool = False
    parent_run_id: Optional[str] = None
    parent_step_id: Optional[str] = None
    forked_from_state_json: Any = None
    fork_patch_json: Any = None
    fork_reason: Optional[str] = None
    resume_from_node_id: Optional[str] = None
    mode: str = "normal"


class RunSnapshot(BaseModel):
    step_index: int
    timestamp: str
    state: Dict[str, Any]


class RunStateResponse(BaseModel):
    run_id: str
    snapshots: List[RunSnapshot]
    derived: bool = False


class ReplayDiagnostic(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    path: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None


class RunStepSummary(BaseModel):
    step_id: str
    step_index: int
    name: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    status: str
    started_at: str
    ended_at: str
    summary: Optional[str] = None
    replayable: bool = False
    replay_disabled_reason: Optional[str] = None


class RunStepsResponse(BaseModel):
    run_id: str
    steps: List[RunStepSummary] = Field(default_factory=list)


class RunStepDetailModel(BaseModel):
    step_id: str
    step_index: int
    name: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    status: str
    started_at: str
    ended_at: str
    summary: Optional[str] = None
    input: Any = None
    output: Any = None
    error: Any = None
    pre_state: Any = None
    post_state: Any = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    replayable: bool = False
    replay_disabled_reason: Optional[str] = None


class RunStepDetailResponse(BaseModel):
    run_id: str
    step: RunStepDetailModel


class RunReplayRequest(BaseModel):
    from_step_id: str
    state_patch: Any = Field(default_factory=dict)
    patch_mode: Literal["overlay", "replace", "jsonpatch"] = "overlay"
    sandbox: Optional[bool] = None
    base_state: Literal["pre", "post"] = "post"
    replay_this_step: bool = False


class RunReplayResponse(BaseModel):
    new_run_id: Optional[str] = None
    diagnostics: List[ReplayDiagnostic] = Field(default_factory=list)
    fork_start_state: Optional[Dict[str, Any]] = None
    resume_node_id: Optional[str] = None


class RunRerunFromStateRequest(BaseModel):
    step_index: int
    state_json: Dict[str, Any] = Field(default_factory=dict)
    resume: Literal["next", "same"] = "next"
    sandbox: Optional[bool] = None


class RunRerunFromStateResponse(BaseModel):
    new_run_id: Optional[str] = None
    diagnostics: List[ReplayDiagnostic] = Field(default_factory=list)


class PendingToolCallsResponse(BaseModel):
    run_id: str
    pending: List[Dict[str, Any]] = Field(default_factory=list)


class ToolCallApprovalRequest(BaseModel):
    approved: bool


class ToolCallApprovalResponse(BaseModel):
    run_id: str
    tool_call_id: str
    approved: bool
    status: str


def _extract_boundary(content_type: str) -> bytes:
    if not content_type:
        raise ValueError("Missing Content-Type header.")
    pieces = [piece.strip() for piece in content_type.split(";")]
    if not pieces or pieces[0].lower() != "multipart/form-data":
        raise ValueError("Content-Type must be multipart/form-data.")

    boundary_value = ""
    for piece in pieces[1:]:
        if piece.lower().startswith("boundary="):
            boundary_value = piece.split("=", 1)[1].strip().strip('"')
            break
    if not boundary_value:
        raise ValueError("Missing multipart boundary.")
    return boundary_value.encode("utf-8")


def _parse_content_disposition(value: str) -> Tuple[str, str]:
    name_match = re.search(r'name="([^"]+)"', value)
    filename_match = re.search(r'filename="([^"]*)"', value)
    field_name = name_match.group(1) if name_match else ""
    filename = filename_match.group(1) if filename_match else ""
    return field_name, filename


def _parse_multipart_file(
    body: bytes,
    content_type: str,
    *,
    field_name: str = "file",
) -> Tuple[str, str, bytes]:
    boundary = _extract_boundary(content_type)
    delimiter = b"--" + boundary
    chunks = body.split(delimiter)

    for chunk in chunks:
        if not chunk:
            continue
        part = chunk
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part in {b"--", b"--\r\n"}:
            continue
        if part.endswith(b"--\r\n"):
            part = part[:-4]
        elif part.endswith(b"--"):
            part = part[:-2]

        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue

        headers_blob = part[:header_end].decode("latin-1")
        content = part[header_end + 4 :]
        if content.endswith(b"\r\n"):
            content = content[:-2]

        headers: Dict[str, str] = {}
        for line in headers_blob.split("\r\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        disposition = headers.get("content-disposition", "")
        current_field_name, filename = _parse_content_disposition(disposition)
        if current_field_name != field_name:
            continue
        if not filename:
            raise ValueError(f"Multipart field '{field_name}' is missing filename.")

        mime = headers.get("content-type", "application/octet-stream").strip().lower()
        return filename, mime, content

    raise ValueError(f"Multipart field '{field_name}' was not found.")


def _json_loads(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_kind(raw_kind: Optional[str]) -> Literal["chat", "workflow"]:
    kind = str(raw_kind or "").lower()
    if "chat" in kind:
        return "chat"
    return "workflow"


def _coerce_step_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ok", "success", "completed"}:
        return "success"
    if normalized in {"pending", "running", "skipped", "error"}:
        return normalized
    return "error"


def _is_success_step_status(value: Any) -> bool:
    return _coerce_step_status(value) == "success"


def _extract_run_metadata(payload: Any, result: Any) -> Dict[str, Any]:
    payload_obj = payload if isinstance(payload, dict) else {}
    result_obj = result if isinstance(result, dict) else {}

    input_obj = payload_obj.get("input")
    input_obj = input_obj if isinstance(input_obj, dict) else {}

    output_obj = result_obj.get("output")
    output_obj = output_obj if isinstance(output_obj, dict) else {}

    artifacts_obj = output_obj.get("artifacts")
    artifacts_obj = artifacts_obj if isinstance(artifacts_obj, dict) else {}

    workflow_id = payload_obj.get("workflow_id") or result_obj.get("workflow_id")
    workflow_type = result_obj.get("workflow_type") or artifacts_obj.get("workflow_type")
    model_id = (
        payload_obj.get("model_id")
        or input_obj.get("model")
        or input_obj.get("model_id")
        or result_obj.get("model_id")
    )

    raw_tool_ids = (
        payload_obj.get("tool_ids")
        or input_obj.get("tool_ids")
        or result_obj.get("tool_ids")
        or output_obj.get("selected_tool_ids")
    )
    tool_ids: Optional[List[str]] = None
    if isinstance(raw_tool_ids, list):
        tool_ids = [str(item) for item in raw_tool_ids if str(item).strip()]

    return {
        "workflow_id": str(workflow_id) if workflow_id is not None else None,
        "workflow_type": str(workflow_type) if workflow_type is not None else None,
        "model_id": str(model_id) if model_id is not None else None,
        "tool_ids": tool_ids,
    }


def _step_error(step_output: Any, status: str) -> Optional[Dict[str, str]]:
    if _coerce_step_status(status) != "error":
        return None
    if isinstance(step_output, dict):
        raw_error = step_output.get("error")
        if isinstance(raw_error, dict):
            message = str(raw_error.get("message") or raw_error.get("detail") or "Step failed.")
            stack = raw_error.get("stack")
            payload: Dict[str, str] = {"message": message}
            if isinstance(stack, str) and stack.strip():
                payload["stack"] = stack
            return payload
        if isinstance(raw_error, str) and raw_error.strip():
            return {"message": raw_error}
    return {"message": "Step failed."}


def _step_replay_disabled_reason(step: Dict[str, Any]) -> Optional[str]:
    status = _coerce_step_status(step.get("status"))
    if status != "success":
        return "Step must be completed successfully before replay."
    return None


def _safe_state(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {"value": value}


def _derive_state_snapshots(payload: Any, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload_obj = payload if isinstance(payload, dict) else {}
    base_state = payload_obj.get("input")
    if not isinstance(base_state, dict):
        base_state = payload_obj if isinstance(payload_obj, dict) else {}

    current_state: Dict[str, Any] = _safe_state(base_state)
    snapshots: List[Dict[str, Any]] = []

    for index, step in enumerate(steps):
        step_output = step.get("output")
        if isinstance(step_output, dict):
            for key, value in step_output.items():
                current_state[key] = copy.deepcopy(value)

        snapshots.append(
            {
                "step_index": int(step.get("step_index") if step.get("step_index") is not None else index),
                "timestamp": str(step.get("ended_at") or step.get("started_at") or ""),
                "state": copy.deepcopy(current_state),
            }
        )

    return snapshots


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = {str(key): copy.deepcopy(value) for key, value in base.items()}
        for key, value in overlay.items():
            key_text = str(key)
            if key_text in merged:
                merged[key_text] = _deep_merge(merged[key_text], value)
            else:
                merged[key_text] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(overlay)


def _json_pointer_segments(path: str) -> List[str]:
    if path == "":
        return []
    if not str(path).startswith("/"):
        raise ValueError("JSONPatch path must start with '/'.")
    raw = str(path).split("/")[1:]
    return [segment.replace("~1", "/").replace("~0", "~") for segment in raw]


def _resolve_json_pointer_parent(document: Any, path: str) -> Tuple[Any, str]:
    segments = _json_pointer_segments(path)
    if not segments:
        return None, ""
    current = document
    for segment in segments[:-1]:
        if isinstance(current, dict):
            if segment not in current:
                current[segment] = {}
            current = current.get(segment)
            continue
        if isinstance(current, list):
            if segment == "-":
                raise ValueError("'-' is only valid in final JSONPatch segment.")
            try:
                index = int(segment)
            except ValueError as exc:
                raise ValueError(f"Invalid array index '{segment}'.") from exc
            if index < 0 or index >= len(current):
                raise ValueError(f"Array index '{segment}' is out of bounds.")
            current = current[index]
            continue
        raise ValueError("JSONPatch path traverses a non-container value.")
    return current, segments[-1]


def _apply_jsonpatch(document: Any, operations: Any) -> Tuple[Any, List[ReplayDiagnostic]]:
    diagnostics: List[ReplayDiagnostic] = []
    current = copy.deepcopy(document)
    if not isinstance(operations, list):
        diagnostics.append(
            ReplayDiagnostic(
                code="PATCH_INVALID",
                severity="error",
                message="jsonpatch mode requires a list of operations.",
                path="$",
            )
        )
        return current, diagnostics

    for index, operation in enumerate(operations):
        op_path = f"state_patch[{index}]"
        if not isinstance(operation, dict):
            diagnostics.append(
                ReplayDiagnostic(
                    code="PATCH_INVALID",
                    severity="error",
                    message="JSONPatch operation must be an object.",
                    path=op_path,
                )
            )
            continue
        op_name = str(operation.get("op") or "").strip().lower()
        pointer = str(operation.get("path") or "")
        if op_name not in {"add", "remove", "replace"}:
            diagnostics.append(
                ReplayDiagnostic(
                    code="PATCH_UNSUPPORTED_OP",
                    severity="error",
                    message=f"Unsupported jsonpatch op '{op_name}'.",
                    path=op_path,
                )
            )
            continue
        try:
            if pointer == "":
                if op_name == "remove":
                    diagnostics.append(
                        ReplayDiagnostic(
                            code="PATCH_INVALID",
                            severity="error",
                            message="Cannot remove the root path.",
                            path=op_path,
                        )
                    )
                    continue
                current = copy.deepcopy(operation.get("value"))
                continue

            parent, segment = _resolve_json_pointer_parent(current, pointer)
            if parent is None:
                diagnostics.append(
                    ReplayDiagnostic(
                        code="PATCH_INVALID",
                        severity="error",
                        message="Invalid root-level patch operation.",
                        path=op_path,
                    )
                )
                continue

            if isinstance(parent, dict):
                if op_name == "remove":
                    if segment not in parent:
                        raise ValueError(f"Key '{segment}' does not exist for remove.")
                    parent.pop(segment, None)
                else:
                    parent[segment] = copy.deepcopy(operation.get("value"))
                continue

            if isinstance(parent, list):
                if segment == "-":
                    if op_name == "remove":
                        raise ValueError("'-' is not valid for remove.")
                    parent.append(copy.deepcopy(operation.get("value")))
                    continue
                try:
                    idx = int(segment)
                except ValueError as exc:
                    raise ValueError(f"Invalid array index '{segment}'.") from exc
                if op_name == "add":
                    if idx < 0 or idx > len(parent):
                        raise ValueError(f"Array index '{segment}' is out of bounds for add.")
                    parent.insert(idx, copy.deepcopy(operation.get("value")))
                elif op_name == "replace":
                    if idx < 0 or idx >= len(parent):
                        raise ValueError(f"Array index '{segment}' is out of bounds for replace.")
                    parent[idx] = copy.deepcopy(operation.get("value"))
                else:
                    if idx < 0 or idx >= len(parent):
                        raise ValueError(f"Array index '{segment}' is out of bounds for remove.")
                    parent.pop(idx)
                continue

            raise ValueError("JSONPatch parent path is not addressable.")
        except ValueError as exc:
            diagnostics.append(
                ReplayDiagnostic(
                    code="PATCH_INVALID",
                    severity="error",
                    message=str(exc),
                    path=op_path,
                )
            )

    return current, diagnostics


def _apply_state_patch(
    *,
    base_state: Any,
    patch: Any,
    patch_mode: str,
) -> Tuple[Any, List[ReplayDiagnostic]]:
    mode = str(patch_mode or "overlay").strip().lower()
    if mode == "replace":
        if not isinstance(patch, dict):
            return base_state, [
                ReplayDiagnostic(
                    code="PATCH_INVALID",
                    severity="error",
                    message="replace mode requires a full object state payload.",
                    path="state_patch",
                )
            ]
        return copy.deepcopy(patch), []

    if mode == "overlay":
        if not isinstance(base_state, dict):
            return base_state, [
                ReplayDiagnostic(
                    code="STATE_INVALID",
                    severity="error",
                    message="Base replay state must be an object.",
                    path="base_state",
                )
            ]
        if not isinstance(patch, dict):
            return base_state, [
                ReplayDiagnostic(
                    code="PATCH_INVALID",
                    severity="error",
                    message="overlay mode requires an object payload.",
                    path="state_patch",
                )
            ]
        return _deep_merge(base_state, patch), []

    if mode == "jsonpatch":
        return _apply_jsonpatch(base_state, patch)

    return base_state, [
        ReplayDiagnostic(
            code="PATCH_INVALID_MODE",
            severity="error",
            message=f"Unsupported patch_mode '{patch_mode}'.",
            path="patch_mode",
        )
    ]


def _python_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _matches_state_type(expected_type: str, value: Any) -> bool:
    normalized = str(expected_type or "json").strip().lower()
    if normalized == "json":
        return True
    if normalized == "string":
        return isinstance(value, str)
    if normalized == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized == "bool":
        return isinstance(value, bool)
    return True


def _validate_state_schema(
    *,
    state: Any,
    workflow_spec: Dict[str, Any],
) -> List[ReplayDiagnostic]:
    diagnostics: List[ReplayDiagnostic] = []
    if not isinstance(state, dict):
        diagnostics.append(
            ReplayDiagnostic(
                code="STATE_INVALID",
                severity="error",
                message="Replay state must be a JSON object.",
                path="$",
            )
        )
        return diagnostics

    schema_items = workflow_spec.get("state_schema")
    schema_list = schema_items if isinstance(schema_items, list) else []
    declared_types: Dict[str, str] = {}
    required_keys: set[str] = set()
    for item in schema_list:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        declared_types[key] = str(item.get("type") or "json").strip().lower()
        if bool(item.get("required", False)):
            required_keys.add(key)

    metadata = workflow_spec.get("metadata") if isinstance(workflow_spec.get("metadata"), dict) else {}
    allow_additional = bool(metadata.get("allow_additional", False))
    allowed_keys = set(BASE_STATE_KEYS.keys()) | set(declared_types.keys())
    for key in state.keys():
        key_text = str(key)
        if not allow_additional and key_text not in allowed_keys:
            diagnostics.append(
                ReplayDiagnostic(
                    code="UNKNOWN_KEY",
                    severity="error",
                    message=f"Unknown state key '{key_text}'.",
                    path=f"state.{key_text}",
                )
            )

    for required_key in sorted(required_keys):
        if required_key not in state:
            diagnostics.append(
                ReplayDiagnostic(
                    code="REQUIRED_KEY_MISSING",
                    severity="error",
                    message=f"Required state key '{required_key}' is missing.",
                    path=f"state.{required_key}",
                )
            )

    for key, expected in declared_types.items():
        if key not in state:
            continue
        value = state.get(key)
        if _matches_state_type(expected, value):
            continue
        diagnostics.append(
            ReplayDiagnostic(
                code="TYPE_MISMATCH",
                severity="error",
                message=f"State key '{key}' expected '{expected}' but got '{_python_type_name(value)}'.",
                path=f"state.{key}",
                expected=expected,
                actual=_python_type_name(value),
            )
        )

    return diagnostics


def _extract_resume_node(step: Dict[str, Any]) -> Optional[str]:
    node_id = str(step.get("node_id") or "").strip()
    node_type = str(step.get("node_type") or "").strip().lower()
    if node_id and node_type in {"node", "tool"}:
        return node_id
    name = str(step.get("name") or "").strip()
    if name.startswith("node."):
        return name.replace("node.", "", 1)
    if name.startswith("tool."):
        return name.replace("tool.", "", 1)
    if name in {"ingest_input", "runtime_error"}:
        return None
    if name:
        return name
    return None


def _has_replay_errors(diagnostics: List[ReplayDiagnostic]) -> bool:
    return any(str(item.severity) == "error" for item in diagnostics)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SandboxApprovalManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}
        self._decisions: Dict[str, bool] = {}

    def request_decision(self, run_id: str, payload: Dict[str, Any]) -> bool:
        tool_name = str(payload.get("tool_id") or payload.get("tool_name") or "").strip()
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
        started_at = str(payload.get("started_at") or _now_utc_iso())

        call_row = db.create_tool_call(
            run_id=run_id,
            step_id=None,
            tool_name=tool_name or "tool",
            args_json=json.dumps(tool_input, default=str),
            started_at=started_at,
            approved_bool=None,
            status="pending",
        )
        tool_call_id = str(call_row.get("tool_call_id") or "")
        if not tool_call_id:
            raise RuntimeError("Failed to persist pending tool call.")

        event = threading.Event()
        with self._lock:
            self._events[tool_call_id] = event

        event.wait(timeout=24 * 60 * 60)
        with self._lock:
            approved = self._decisions.pop(tool_call_id, False)
            self._events.pop(tool_call_id, None)

        db.update_tool_call_approval(tool_call_id, approved)
        if not approved:
            db.finish_tool_call(
                tool_call_id=tool_call_id,
                status="rejected",
                finished_at=_now_utc_iso(),
                result_json=None,
                error_json=json.dumps({"message": "Tool call rejected by user."}, default=str),
            )
        return bool(approved)

    def submit_decision(self, run_id: str, tool_call_id: str, approved: bool) -> bool:
        call_row = db.get_tool_call(tool_call_id)
        if call_row is None:
            return False
        if str(call_row.get("run_id") or "") != run_id:
            return False
        if str(call_row.get("status") or "") not in {"pending", "approved", "rejected"}:
            return False

        with self._lock:
            self._decisions[tool_call_id] = bool(approved)
            event = self._events.get(tool_call_id)

        db.update_tool_call_approval(tool_call_id, approved)
        if event:
            event.set()
        else:
            if not approved:
                db.finish_tool_call(
                    tool_call_id=tool_call_id,
                    status="rejected",
                    finished_at=_now_utc_iso(),
                    result_json=None,
                    error_json=json.dumps({"message": "Tool call rejected by user."}, default=str),
                )
            else:
                db.finish_tool_call(
                    tool_call_id=tool_call_id,
                    status="approved",
                    finished_at=_now_utc_iso(),
                    result_json=None,
                    error_json=None,
                )
        return True


_SANDBOX_APPROVALS = SandboxApprovalManager()


def _normalize_tool_id_fragment(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "custom_tool"


def _generate_tool_id(name: str, existing_ids: set[str]) -> str:
    base = f"tool.custom.{_normalize_tool_id_fragment(name)}"
    if base not in existing_ids:
        return base

    index = 2
    while f"{base}_{index}" in existing_ids:
        index += 1
    return f"{base}_{index}"


def _generate_workflow_id(name: str, existing_ids: set[str]) -> str:
    base = f"workflow.custom.{_normalize_tool_id_fragment(name)}"
    if base not in existing_ids:
        return base

    index = 2
    while f"{base}_{index}" in existing_ids:
        index += 1
    return f"{base}_{index}"


def _is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _is_loopback_host(hostname: str) -> bool:
    value = str(hostname or "").strip().lower()
    if not value:
        return False
    if value in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip_value = ipaddress.ip_address(value)
        return bool(ip_value.is_loopback)
    except ValueError:
        pass

    try:
        resolved = socket.getaddrinfo(value, None)
    except socket.gaierror:
        return False

    loopback_only = True
    for _, _, _, _, sockaddr in resolved:
        ip_text = str(sockaddr[0] or "")
        try:
            if not ipaddress.ip_address(ip_text).is_loopback:
                loopback_only = False
                break
        except ValueError:
            loopback_only = False
            break
    return loopback_only


def _is_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    return _is_loopback_host(client_host)


def _normalize_local_qdrant_url(raw_url: str) -> str:
    parsed = urlparse(str(raw_url or "").strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Qdrant URL must use http or https.")
    if not parsed.hostname:
        raise ValueError("Qdrant URL must include a host.")
    if not _is_loopback_host(parsed.hostname):
        raise ValueError("Qdrant URL host must resolve to localhost/loopback.")

    netloc = parsed.netloc or parsed.hostname
    if not netloc:
        raise ValueError("Qdrant URL must include host:port.")
    return f"{parsed.scheme.lower()}://{netloc}".rstrip("/")


def _normalize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in headers.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[key_text] = str(value)
    return normalized


def _is_valid_tool_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_.-]+", value))


def _is_valid_workflow_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_.-]+", value))


def _normalize_tool_config(tool_type: ToolType, config: Dict[str, Any]) -> Dict[str, Any]:
    if tool_type == "http":
        url = str(config.get("url") or "").strip()
        if not _is_valid_http_url(url):
            raise ValueError("Tool URL must be a valid http(s) URL.")

        method = str(config.get("method") or "POST").strip().upper()
        if method not in {"GET", "POST"}:
            method = "POST"

        timeout_raw = config.get("timeout_ms")
        timeout_ms = 8000
        if isinstance(timeout_raw, (int, float)):
            timeout_ms = int(timeout_raw)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0.")

        headers = config.get("headers")
        normalized_headers = _normalize_headers(headers if isinstance(headers, dict) else {})

        return {
            "url": url,
            "method": method,
            "headers": normalized_headers,
            "timeout_ms": timeout_ms,
        }

    if tool_type == "python":
        code = str(config.get("code") or "")
        if not code.strip():
            raise ValueError("Python tool requires non-empty config.code.")

        timeout_raw = config.get("timeout_ms")
        timeout_ms = 5000
        if isinstance(timeout_raw, (int, float)):
            timeout_ms = int(timeout_raw)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0.")

        allowed_imports_raw = config.get("allowed_imports")
        allowed_imports: List[str] = []
        if isinstance(allowed_imports_raw, list):
            for item in allowed_imports_raw:
                token = str(item).strip()
                if token:
                    allowed_imports.append(token)

        return {
            "code": code,
            "timeout_ms": timeout_ms,
            "allowed_imports": sorted(set(allowed_imports)),
        }

    prompt_template = str(config.get("prompt_template") or "")
    if not prompt_template.strip():
        raise ValueError("Prompt tool requires non-empty config.prompt_template.")

    timeout_raw = config.get("timeout_ms")
    timeout_ms = 30000
    if isinstance(timeout_raw, (int, float)):
        timeout_ms = int(timeout_raw)
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be greater than 0.")

    temperature_raw = config.get("temperature")
    temperature = 0.2
    if isinstance(temperature_raw, (int, float)):
        temperature = float(temperature_raw)

    return {
        "prompt_template": prompt_template,
        "system_prompt": str(config.get("system_prompt") or ""),
        "temperature": temperature,
        "timeout_ms": timeout_ms,
    }


def _validate_workflow_graph(graph_payload: WorkflowGraphSchema) -> Dict[str, Any]:
    nodes = graph_payload.nodes
    edges = graph_payload.edges

    if not nodes:
        raise ValueError("Workflow graph requires at least one node.")

    node_ids: set[str] = set()
    normalized_nodes: List[Dict[str, Any]] = []
    start_count = 0
    end_count = 0

    for node in nodes:
        node_id = node.id.strip()
        if not node_id:
            raise ValueError("Workflow node id is required.")
        if node_id in node_ids:
            raise ValueError(f"Duplicate node id '{node_id}'.")
        node_ids.add(node_id)

        node_type = str(node.type).strip().lower()
        config = dict(node.config or {})

        if node_type == "start":
            start_count += 1
        elif node_type == "end":
            end_count += 1
        elif node_type == "llm":
            prompt_template = str(config.get("prompt_template") or "")
            if not prompt_template.strip():
                raise ValueError(f"Node '{node_id}' (llm) requires config.prompt_template.")
            if "output_key" in config:
                config["output_key"] = str(config.get("output_key") or "").strip()
        elif node_type == "tool":
            tool_id = str(config.get("tool_id") or "")
            if not tool_id.strip():
                raise ValueError(f"Node '{node_id}' (tool) requires config.tool_id.")
            input_map = config.get("input_map")
            if input_map is not None and not isinstance(input_map, dict):
                raise ValueError(f"Node '{node_id}' input_map must be an object.")
            config["tool_id"] = tool_id.strip()
            config["input_map"] = input_map if isinstance(input_map, dict) else {}
            if "output_key" in config:
                config["output_key"] = str(config.get("output_key") or "").strip()
        elif node_type == "condition":
            field = str(config.get("field") or "").strip()
            operator = str(config.get("operator") or "").strip().lower()
            if not field:
                raise ValueError(f"Node '{node_id}' (condition) requires config.field.")
            if operator not in {"equals", "contains", "gt", "lt", "exists", "not_exists", "in"}:
                raise ValueError(
                    f"Node '{node_id}' has invalid condition operator '{operator}'."
                )
            config["field"] = field
            config["operator"] = operator
        else:
            raise ValueError(f"Unsupported node type '{node_type}'.")

        normalized_nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "config": config,
            }
        )

    if start_count != 1:
        raise ValueError("Workflow graph must include exactly one start node.")
    if end_count != 1:
        raise ValueError("Workflow graph must include exactly one end node.")

    normalized_edges: List[Dict[str, Any]] = []
    outgoing: Dict[str, List[Dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        from_node = edge.from_node.strip()
        to_node = edge.to.strip()
        condition = str(edge.condition or "always").strip().lower()

        if from_node not in node_ids:
            raise ValueError(f"Edge references unknown from node '{from_node}'.")
        if to_node not in node_ids:
            raise ValueError(f"Edge references unknown to node '{to_node}'.")
        if condition not in {"always", "true", "false"}:
            raise ValueError(f"Edge condition must be always|true|false, got '{condition}'.")

        record = {"from": from_node, "to": to_node, "condition": condition}
        normalized_edges.append(record)
        outgoing[from_node].append(record)

    node_type_by_id = {node["id"]: node["type"] for node in normalized_nodes}
    start_node_id = next(node["id"] for node in normalized_nodes if node["type"] == "start")
    end_node_id = next(node["id"] for node in normalized_nodes if node["type"] == "end")

    if len(outgoing[start_node_id]) == 0:
        raise ValueError("Start node must have at least one outgoing edge.")

    for node_id, node_type in node_type_by_id.items():
        outgoing_edges = outgoing.get(node_id, [])
        if node_type == "end":
            if outgoing_edges:
                raise ValueError("End node cannot have outgoing edges.")
            continue

        if node_type == "condition":
            conditions = {str(edge.get("condition") or "always") for edge in outgoing_edges}
            if "true" not in conditions or "false" not in conditions:
                raise ValueError(
                    f"Condition node '{node_id}' must have both true and false edges."
                )
            continue

        if len(outgoing_edges) == 0:
            raise ValueError(f"Node '{node_id}' must have at least one outgoing edge.")

        for edge in outgoing_edges:
            if str(edge.get("condition") or "always") != "always":
                raise ValueError(
                    f"Only condition nodes may use true/false edges (node '{node_id}')."
                )

    if not any(edge["to"] == end_node_id for edge in normalized_edges):
        raise ValueError("Workflow graph must route to the end node.")

    return {
        "nodes": normalized_nodes,
        "edges": normalized_edges,
    }


def _normalize_workflow_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = str(item.get("id") or "")
    title = str(item.get("title") or item.get("name") or workflow_id)
    workflow_type = str(item.get("type") or "custom")
    version = str(item.get("version") or workflow_id.split(".")[-1] or "v1")
    status = str(item.get("status") or "available")
    source = str(item.get("source") or "builtin")
    enabled = bool(item.get("enabled", status != "disabled"))
    return {
        "id": workflow_id,
        "title": title,
        "name": title,
        "description": str(item.get("description") or ""),
        "type": workflow_type,
        "version": version,
        "status": status,
        "source": source,
        "enabled": enabled,
    }


def _resolve_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    env_value = os.getenv("SATURDAY_DB_PATH")
    if env_value:
        env_path = Path(env_value)
        if env_path.is_absolute():
            return env_path
        if env_path.parts and env_path.parts[0] == "apps":
            return repo_root / env_path
        return Path.cwd() / env_path
    return repo_root / "apps/api/saturday.db"


def _resolve_httpx_timeout(timeout_seconds: float) -> float | None:
    try:
        timeout_value = float(timeout_seconds)
    except (TypeError, ValueError):
        return None

    if timeout_value <= 0:
        return None
    return timeout_value


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "0"))
DB_PATH = _resolve_db_path()

app = FastAPI(title="Saturday API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    db.init_db(str(DB_PATH))


@app.post("/artifacts/upload", response_model=ArtifactUploadResponse)
async def upload_artifact(request: Request) -> ArtifactUploadResponse:
    try:
        body = await request.body()
        filename, mime, content = _parse_multipart_file(
            body=body,
            content_type=str(request.headers.get("content-type") or ""),
            field_name="file",
        )
        payload = artifacts.save_bytes(
            filename=filename,
            mime=mime,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return ArtifactUploadResponse(
        artifact_id=str(payload.get("artifact_id") or ""),
        mime=str(payload.get("mime") or "application/octet-stream"),
        size=int(payload.get("size") or 0),
        sha256=str(payload.get("sha256") or ""),
    )


@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str) -> Response:
    try:
        payload = artifacts.read_artifact(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Artifact read failed: {exc}") from exc

    return Response(
        content=payload.get("bytes", b""),
        media_type=str(payload.get("mime") or "application/octet-stream"),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_status: Literal["ok", "down"] = "down"
    try:
        async with httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL, timeout=_resolve_httpx_timeout(OLLAMA_TIMEOUT)
        ) as client:
            resp = await client.get("/api/tags")
        if resp.status_code == 200:
            ollama_status = "ok"
    except httpx.HTTPError:
        ollama_status = "down"

    return HealthResponse(
        api="ok",
        ollama=ollama_status,
        ollama_base_url=OLLAMA_BASE_URL,
        model_default=OLLAMA_MODEL_DEFAULT,
    )


@app.post("/internal/qdrant/config", response_model=QdrantConfigResponse)
async def configure_qdrant_runtime(
    request: Request,
    payload: QdrantConfigRequest,
) -> QdrantConfigResponse:
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="Only localhost clients may configure Qdrant runtime.")

    try:
        normalized_url = _normalize_local_qdrant_url(payload.url)
        stored_url = qdrant_client_service.set_qdrant_url(normalized_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store Qdrant URL: {exc}") from exc

    return QdrantConfigResponse(ok=True, url=stored_url)


@app.get("/rag/health", response_model=RagHealthResponse)
async def rag_health() -> RagHealthResponse:
    url = qdrant_client_service.get_qdrant_url()
    reachable, error = qdrant_client_service.is_qdrant_reachable(url)
    return RagHealthResponse(
        qdrantReachable=reachable,
        url=url,
        error=error if not reachable else None,
    )


@app.post("/chat", response_model=ChatRunResponse)
async def chat(request: ChatRunRequest) -> ChatRunResponse:
    result = graph.run_chat_workflow(
        workflow_id=request.workflow_id,
        model_id=request.model_id,
        tool_ids=request.tool_ids,
        message=request.message,
        vision_model_id=request.vision_model_id,
        artifact_ids=request.artifact_ids,
        context=request.context,
        thread_id=request.thread_id,
    )
    if str(result.get("status", "error")) != "ok":
        detail = "LangGraph workflow execution failed"
        output_text = result.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            detail = output_text
        raise HTTPException(status_code=503, detail=detail)
    return ChatRunResponse(**result)


@app.post("/chat/stream")
async def chat_stream(request: ChatRunRequest) -> StreamingResponse:
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stream_run_id = ""
    final_sent = False

    def emit_event(payload: Dict[str, Any]) -> None:
        nonlocal stream_run_id, final_sent
        event_payload = dict(payload)
        maybe_run_id = str(event_payload.get("run_id") or "").strip()
        if maybe_run_id:
            stream_run_id = maybe_run_id
        if str(event_payload.get("type") or "") == "final":
            final_sent = True
        loop.call_soon_threadsafe(queue.put_nowait, event_payload)

    def run_stream() -> None:
        try:
            graph.run_chat_workflow_stream(
                workflow_id=request.workflow_id,
                model_id=request.model_id,
                tool_ids=request.tool_ids,
                message=request.message,
                emit_event=emit_event,
                vision_model_id=request.vision_model_id,
                artifact_ids=request.artifact_ids,
                context=request.context,
                thread_id=request.thread_id,
            )
        except Exception as exc:
            message = str(exc) or "Workflow execution failed."
            emit_event(
                {
                    "type": "error",
                    "run_id": stream_run_id,
                    "message": message,
                }
            )
            if not final_sent:
                emit_event(
                    {
                        "type": "final",
                        "run_id": stream_run_id,
                        "status": "error",
                        "output_text": message,
                        "ended_at": _now_utc_iso(),
                    }
                )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(asyncio.to_thread(run_stream))

    async def stream_events():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield streaming.encode_sse_message(event)

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/models", response_model=ModelsResponse)
@app.get("/models/text", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    payload = graph.list_text_models()
    models_payload = payload.get("models")
    default_model = str(payload.get("default_model") or OLLAMA_MODEL_DEFAULT)
    ollama_status = payload.get("ollama_status")

    if not isinstance(models_payload, list):
        models_payload = []

    model_items = [ModelSummary(**item) for item in models_payload if isinstance(item, dict)]
    return ModelsResponse(
        models=model_items,
        default_model=default_model,
        ollama_status=str(ollama_status) if ollama_status is not None else None,
    )


@app.get("/models/vision", response_model=ModelsResponse)
async def vision_models() -> ModelsResponse:
    payload = graph.list_vision_models()
    models_payload = payload.get("models")
    default_model = str(payload.get("default_model") or "")
    ollama_status = payload.get("ollama_status")

    if not isinstance(models_payload, list):
        models_payload = []

    model_items = [ModelSummary(**item) for item in models_payload if isinstance(item, dict)]
    return ModelsResponse(
        models=model_items,
        default_model=default_model,
        ollama_status=str(ollama_status) if ollama_status is not None else None,
    )


@app.get("/workflows", response_model=WorkflowsResponse)
async def workflows() -> WorkflowsResponse:
    payload = graph.list_workflows()
    items = [
        WorkflowSummary(**_normalize_workflow_summary(item))
        for item in payload
        if isinstance(item, dict)
    ]
    return WorkflowsResponse(workflows=items)


@app.post("/builder/workflows", response_model=WorkflowDefinitionResponse)
async def create_or_update_builder_workflow(
    request: CreateWorkflowRequest,
) -> WorkflowDefinitionResponse:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workflow name is required.")

    try:
        normalized_graph = _validate_workflow_graph(request.graph)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing_workflows = graph.list_workflows()
    existing_ids = {str(item.get("id") or "").strip() for item in existing_workflows}
    builtin_ids = {
        str(item.get("id") or "").strip()
        for item in existing_workflows
        if str(item.get("source") or "builtin") == "builtin"
    }

    requested_id = str(request.id or "").strip()
    if requested_id and not _is_valid_workflow_id(requested_id):
        raise HTTPException(
            status_code=400,
            detail="Workflow id may only contain letters, numbers, '.', '_' and '-'.",
        )

    workflow_id = requested_id or _generate_workflow_id(name, existing_ids)
    if workflow_id in builtin_ids:
        raise HTTPException(status_code=409, detail=f"Workflow id '{workflow_id}' is reserved.")

    existing_custom = db.get_workflow(workflow_id)
    now = _now_utc_iso()
    created_at = str(existing_custom.get("created_at") or now) if existing_custom else now

    record: Dict[str, Any] = {
        "id": workflow_id,
        "name": name,
        "description": request.description.strip(),
        "enabled": bool(request.enabled),
        "graph": normalized_graph,
        "created_at": created_at,
        "updated_at": now,
    }
    db.upsert_workflow(record)

    persisted = db.get_workflow(workflow_id)
    if not persisted:
        raise HTTPException(status_code=500, detail="Workflow was not persisted.")
    return WorkflowDefinitionResponse(**persisted)


@app.get("/builder/workflows/{workflow_id}", response_model=WorkflowDefinitionResponse)
async def get_builder_workflow(workflow_id: str) -> WorkflowDefinitionResponse:
    payload = db.get_workflow(workflow_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return WorkflowDefinitionResponse(**payload)


@app.patch("/builder/workflows/{workflow_id}", response_model=WorkflowDefinitionResponse)
async def update_builder_workflow(
    workflow_id: str,
    request: UpdateWorkflowRequest,
) -> WorkflowDefinitionResponse:
    existing = db.get_workflow(workflow_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    name = str(existing.get("name") or workflow_id)
    if request.name is not None:
        next_name = request.name.strip()
        if not next_name:
            raise HTTPException(status_code=400, detail="Workflow name cannot be empty.")
        name = next_name

    description = str(existing.get("description") or "")
    if request.description is not None:
        description = request.description.strip()

    enabled = bool(existing.get("enabled", True))
    if request.enabled is not None:
        enabled = bool(request.enabled)

    next_graph = existing.get("graph") if isinstance(existing.get("graph"), dict) else {"nodes": [], "edges": []}
    if request.graph is not None:
        try:
            next_graph = _validate_workflow_graph(request.graph)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    record: Dict[str, Any] = {
        "id": workflow_id,
        "name": name,
        "description": description,
        "enabled": enabled,
        "graph": next_graph,
        "created_at": str(existing.get("created_at") or _now_utc_iso()),
        "updated_at": _now_utc_iso(),
    }
    db.upsert_workflow(record)

    payload = db.get_workflow(workflow_id)
    if not payload:
        raise HTTPException(status_code=500, detail="Workflow update did not persist.")
    return WorkflowDefinitionResponse(**payload)


@app.get("/tools", response_model=ToolsResponse)
async def tools() -> ToolsResponse:
    payload = graph.list_tools()
    return ToolsResponse(tools=[ToolSummary(**item) for item in payload])


@app.post("/tools/invoke", response_model=ToolInvokeResponse)
async def invoke_tool(request: ToolInvokeRequest) -> ToolInvokeResponse:
    tool_id = str(request.tool_id or "").strip()
    if not tool_id:
        raise HTTPException(status_code=400, detail="tool_id is required.")
    try:
        result = graph.invoke_tool(
            tool_id=tool_id,
            tool_input=dict(request.input or {}),
            context=dict(request.context or {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tool invocation failed: {exc}") from exc
    return ToolInvokeResponse(**result)


@app.post("/builder/tools", response_model=ToolSummary)
async def create_builder_tool(request: CreateToolRequest) -> ToolSummary:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tool name is required.")

    existing_tools = graph.list_tools()
    existing_ids = {str(item.get("id") or "").strip() for item in existing_tools}

    requested_id = str(request.id or "").strip()
    if requested_id and not _is_valid_tool_id(requested_id):
        raise HTTPException(
            status_code=400,
            detail="Tool id may only contain letters, numbers, '.', '_' and '-'.",
        )
    tool_id = requested_id or _generate_tool_id(name, existing_ids)
    if tool_id in existing_ids:
        raise HTTPException(status_code=409, detail=f"Tool id '{tool_id}' already exists.")

    try:
        normalized_config = _normalize_tool_config(request.type, request.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    now = _now_utc_iso()
    record: Dict[str, Any] = {
        "id": tool_id,
        "name": name,
        "kind": request.kind,
        "type": request.type,
        "description": request.description.strip(),
        "enabled": bool(request.enabled),
        "config": normalized_config,
        "input_schema": request.input_schema,
        "output_schema": request.output_schema,
        "created_at": now,
        "updated_at": now,
    }
    db.upsert_tool(record)

    created = db.get_tool(tool_id)
    if not created:
        raise HTTPException(status_code=500, detail="Tool was not persisted.")
    return ToolSummary(**created)


@app.post("/tools", response_model=ToolSummary)
async def create_tool_v2(request: CreateToolRequest) -> ToolSummary:
    return await create_builder_tool(request)


@app.patch("/tools/{tool_id}", response_model=ToolSummary)
async def update_tool(tool_id: str, request: UpdateToolRequest) -> ToolSummary:
    existing = db.get_tool(tool_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Tool not found or not editable.",
        )

    updated = db.set_tool_enabled(tool_id, bool(request.enabled))
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update tool.")

    payload = db.get_tool(tool_id)
    if not payload:
        raise HTTPException(status_code=500, detail="Tool update did not persist.")
    return ToolSummary(**payload)


@app.post("/workflow/compile", response_model=WorkflowCompileResponse)
async def workflow_compile(
    request: WorkflowCompileRequest,
) -> WorkflowCompileResponse:
    if isinstance(request.workflow_spec, dict):
        compile_payload = workflow_service.compile_workflow_spec(
            workflow_spec=request.workflow_spec
        )
        normalized = (
            compile_payload.get("workflow_spec")
            if isinstance(compile_payload.get("workflow_spec"), dict)
            else {}
        )
        compiled = (
            compile_payload.get("compiled")
            if isinstance(compile_payload.get("compiled"), dict)
            else {}
        )
        return WorkflowCompileResponse(
            valid=bool(compile_payload.get("valid", False)),
            workflow_id=str(normalized.get("workflow_id") or ""),
            workflow_type="custom",
            graph=dict(compiled.get("runtime_graph") or {}),
            workflow_spec=normalized,
            compiled=compiled,
            diagnostics=[
                dict(item)
                for item in list(compile_payload.get("diagnostics") or [])
                if isinstance(item, dict)
            ],
        )

    task = str(request.task or "").strip()
    if not task:
        raise HTTPException(
            status_code=400,
            detail="workflow_spec or task is required for /workflow/compile.",
        )

    result = graph.compile_workflow(
        task=task,
        context=request.context,
        workflow_type=request.workflow_type,
    )
    return WorkflowCompileResponse(
        valid=True,
        workflow_id=str(result.get("workflow_id") or ""),
        workflow_type=str(result.get("workflow_type") or ""),
        graph=dict(result.get("graph") or {}),
        workflow_spec={},
        compiled={"runtime_graph": dict(result.get("graph") or {})},
        diagnostics=[],
    )


@app.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(request: WorkflowRunRequest) -> WorkflowRunResponse:
    selector_count = sum(
        1
        for item in (
            bool(str(request.workflow_version_id or "").strip()),
            bool(str(request.workflow_id or "").strip()),
            isinstance(request.draft_spec, dict),
        )
        if item
    )
    if selector_count != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly one of workflow_version_id, workflow_id, or draft_spec must be provided."
            ),
        )

    diagnostics: List[Dict[str, Any]] = []
    workflow_defs_override: List[Dict[str, Any]] = []
    resolved_workflow_id = ""
    resolved_version_id = ""
    resolved_version_num = 0

    if str(request.workflow_version_id or "").strip():
        version = db.get_workflow_version(str(request.workflow_version_id or "").strip())
        if version is None:
            raise HTTPException(status_code=404, detail="Workflow version not found.")
        workflow_info = version.get("workflow") if isinstance(version.get("workflow"), dict) else {}
        compiled = version.get("compiled") if isinstance(version.get("compiled"), dict) else {}
        runtime_graph = compiled.get("runtime_graph") if isinstance(compiled.get("runtime_graph"), dict) else {}
        resolved_workflow_id = str(version.get("workflow_id") or "")
        resolved_version_id = str(version.get("version_id") or "")
        resolved_version_num = int(version.get("version_num") or 0)
        workflow_defs_override.append(
            {
                "id": resolved_workflow_id,
                "name": str(workflow_info.get("name") or resolved_workflow_id),
                "title": str(workflow_info.get("name") or resolved_workflow_id),
                "description": str(workflow_info.get("description") or ""),
                "type": "custom",
                "source": "custom",
                "enabled": True,
                "graph": runtime_graph,
                "compiled": compiled,
                "spec": dict(version.get("spec") or {}),
            }
        )
    elif str(request.workflow_id or "").strip():
        latest_version = db.get_latest_workflow_version(str(request.workflow_id or "").strip())
        if latest_version is None:
            raise HTTPException(status_code=404, detail="Workflow not found.")
        workflow_info = (
            latest_version.get("workflow")
            if isinstance(latest_version.get("workflow"), dict)
            else {}
        )
        compiled = (
            latest_version.get("compiled")
            if isinstance(latest_version.get("compiled"), dict)
            else {}
        )
        runtime_graph = (
            compiled.get("runtime_graph")
            if isinstance(compiled.get("runtime_graph"), dict)
            else {}
        )
        resolved_workflow_id = str(latest_version.get("workflow_id") or "")
        resolved_version_id = str(latest_version.get("version_id") or "")
        resolved_version_num = int(latest_version.get("version_num") or 0)
        workflow_defs_override.append(
            {
                "id": resolved_workflow_id,
                "name": str(workflow_info.get("name") or resolved_workflow_id),
                "title": str(workflow_info.get("name") or resolved_workflow_id),
                "description": str(workflow_info.get("description") or ""),
                "type": "custom",
                "source": "custom",
                "enabled": True,
                "graph": runtime_graph,
                "compiled": compiled,
                "spec": dict(latest_version.get("spec") or {}),
            }
        )
    else:
        draft_payload = workflow_service.compile_draft_to_runtime_defs(
            workflow_spec=dict(request.draft_spec or {})
        )
        compile_payload = (
            draft_payload.get("compile")
            if isinstance(draft_payload.get("compile"), dict)
            else {}
        )
        diagnostics = [
            dict(item)
            for item in list(compile_payload.get("diagnostics") or [])
            if isinstance(item, dict)
        ]
        if not bool(compile_payload.get("valid", False)):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Draft workflow failed validation.",
                    "diagnostics": diagnostics,
                },
            )
        workflow_def = (
            draft_payload.get("workflow_def")
            if isinstance(draft_payload.get("workflow_def"), dict)
            else {}
        )
        resolved_workflow_id = str(workflow_def.get("id") or f"draft.workflow.{uuid.uuid4()}")
        resolved_version_id = f"draft:{resolved_workflow_id}"
        resolved_version_num = 0
        workflow_defs_override.append(workflow_def)

    run_input = dict(request.input or {})
    resolved_run_id = str(uuid.uuid4())

    if request.sandbox_mode:
        def _tool_gate(payload: Dict[str, Any]) -> bool:
            return _SANDBOX_APPROVALS.request_decision(
                run_id=resolved_run_id,
                payload=payload,
            )

        def _run_async() -> None:
            graph.run_workflow(
                workflow_id=resolved_workflow_id,
                input=run_input,
                workflow_defs=workflow_defs_override,
                run_id=resolved_run_id,
                workflow_version_id=resolved_version_id,
                sandbox_mode=True,
                tool_call_gate=_tool_gate,
            )

        threading.Thread(target=_run_async, daemon=True).start()
        return WorkflowRunResponse(
            run_id=resolved_run_id,
            workflow_id=resolved_workflow_id,
            workflow_version_id=resolved_version_id,
            workflow_version_num=resolved_version_num,
            status="running",
            sandbox_mode=True,
            diagnostics=diagnostics,
            output={},
            steps=[],
        )

    result = graph.run_workflow(
        workflow_id=resolved_workflow_id,
        input=run_input,
        workflow_defs=workflow_defs_override,
        run_id=resolved_run_id,
        workflow_version_id=resolved_version_id,
        sandbox_mode=False,
    )
    return WorkflowRunResponse(
        run_id=str(result.get("run_id") or resolved_run_id),
        workflow_id=str(result.get("workflow_id") or resolved_workflow_id),
        workflow_type=str(result.get("workflow_type") or "custom"),
        workflow_version_id=resolved_version_id,
        workflow_version_num=resolved_version_num,
        status=str(result.get("status") or "error"),
        sandbox_mode=False,
        output=result.get("output") if isinstance(result.get("output"), dict) else {},
        steps=list(result.get("steps") or []),
        diagnostics=diagnostics,
    )


@app.get("/workflow/{workflow_id}")
async def workflow_detail_latest(workflow_id: str) -> Dict[str, Any]:
    try:
        return workflow_service.get_latest_workflow_payload(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/workflow/{workflow_id}/versions")
async def workflow_versions(workflow_id: str) -> Dict[str, Any]:
    versions = workflow_service.get_workflow_versions_payload(workflow_id)
    return {"workflow_id": workflow_id, "versions": versions}


@app.post("/workflow/{workflow_id}/versions")
async def workflow_create_version(
    workflow_id: str,
    request: CreateWorkflowVersionRequest,
) -> Dict[str, Any]:
    try:
        created = workflow_service.create_workflow_version(
            workflow_id=workflow_id,
            workflow_spec=dict(request.workflow_spec or {}),
            created_by=request.created_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return created


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
async def run_detail(run_id: str) -> RunDetailResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    payload = _json_loads(run_row.get("payload_json"))
    result = _json_loads(run_row.get("result_json"))
    metadata = _extract_run_metadata(payload=payload, result=result)

    return RunDetailResponse(
        run_id=str(run_row.get("run_id") or ""),
        kind=_normalize_kind(run_row.get("kind")),
        status=str(run_row.get("status") or ""),
        workflow_id=str(metadata.get("workflow_id") or ""),
        workflow_version_id=str(run_row.get("workflow_version_id") or "") or None,
        workflow_type=metadata.get("workflow_type"),
        model_id=metadata.get("model_id"),
        tool_ids=metadata.get("tool_ids") or [],
        started_at=run_row.get("started_at"),
        ended_at=run_row.get("ended_at"),
        payload=payload,
        result=result,
        sandbox_mode=bool(run_row.get("sandbox_mode")),
        parent_run_id=str(run_row.get("parent_run_id") or "") or None,
        parent_step_id=str(run_row.get("parent_step_id") or "") or None,
        forked_from_state_json=_json_loads(run_row.get("forked_from_state_json")),
        fork_patch_json=_json_loads(run_row.get("fork_patch_json")),
        fork_reason=str(run_row.get("fork_reason") or "") or None,
        resume_from_node_id=str(run_row.get("resume_from_node_id") or "") or None,
        mode=str(run_row.get("mode") or "normal"),
    )


@app.get("/runs/{run_id}/logs", response_model=RunLogsResponse)
async def run_logs(run_id: str) -> RunLogsResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    step_rows = db.read_steps(run_id)
    tool_call_rows = db.list_tool_calls(run_id)
    tool_calls_by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in tool_call_rows:
        step_id = row.get("step_id")
        if step_id is None:
            continue
        try:
            args = _json_loads(row.get("args_json"))
        except Exception:
            args = {}
        try:
            result_payload = _json_loads(row.get("result_json"))
        except Exception:
            result_payload = None
        try:
            error_payload = _json_loads(row.get("error_json"))
        except Exception:
            error_payload = None

        tool_calls_by_step[int(step_id)].append(
            {
                "tool_call_id": str(row.get("tool_call_id") or ""),
                "tool_name": str(row.get("tool_name") or ""),
                "status": str(row.get("status") or "pending"),
                "approved_bool": row.get("approved_bool"),
                "started_at": str(row.get("started_at") or ""),
                "finished_at": str(row.get("finished_at") or ""),
                "args": args if isinstance(args, dict) else {},
                "result": result_payload,
                "error": error_payload,
            }
        )

    steps_payload: List[RunLogStep] = []
    for index, step in enumerate(step_rows):
        status = _coerce_step_status(step.get("status"))
        step_output = _json_loads(step.get("output_json"))
        step_error_payload = _json_loads(step.get("error_json"))
        step_id_value = step.get("step_id")
        if step_id_value is None:
            step_id_value = step.get("id")
        external_step_id = db.format_step_id(step_id_value)
        step_tool_calls = (
            tool_calls_by_step.get(int(step_id_value))
            if step_id_value is not None
            else []
        )
        error_payload_source = (
            step_error_payload
            if step_error_payload is not None
            else step_output
        )
        steps_payload.append(
            RunLogStep(
                step_id=external_step_id,
                step_index=int(step.get("step_index") if step.get("step_index") is not None else index),
                name=str(step.get("name") or ""),
                status=status,
                started_at=str(step.get("started_at") or ""),
                ended_at=str(step.get("ended_at") or ""),
                node_id=str(step.get("node_id") or "") or None,
                node_type=str(step.get("node_type") or "") or None,
                summary=str(step.get("summary") or ""),
                input=_json_loads(step.get("input_json")),
                output=step_output,
                tool_calls=step_tool_calls if isinstance(step_tool_calls, list) else [],
                error=_step_error(error_payload_source, status),
            )
        )

    return RunLogsResponse(run_id=run_id, steps=steps_payload)


@app.get("/runs/{run_id}/steps", response_model=RunStepsResponse)
async def run_steps(run_id: str) -> RunStepsResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    step_rows = db.read_steps(run_id)
    payload: List[RunStepSummary] = []
    for index, step in enumerate(step_rows):
        step_id_value = step.get("step_id")
        if step_id_value is None:
            step_id_value = step.get("id")
        node_id = str(step.get("node_id") or "").strip() or _extract_resume_node(step)
        node_type = str(step.get("node_type") or "").strip() or None
        status = _coerce_step_status(step.get("status"))
        replay_disabled_reason = _step_replay_disabled_reason({"status": status})
        payload.append(
            RunStepSummary(
                step_id=db.format_step_id(step_id_value),
                step_index=int(step.get("step_index") if step.get("step_index") is not None else index),
                name=str(step.get("name") or ""),
                node_id=node_id or None,
                node_type=node_type,
                status=status,
                started_at=str(step.get("started_at") or ""),
                ended_at=str(step.get("ended_at") or ""),
                summary=str(step.get("summary") or ""),
                replayable=replay_disabled_reason is None,
                replay_disabled_reason=replay_disabled_reason,
            )
        )
    return RunStepsResponse(run_id=run_id, steps=payload)


@app.get("/runs/{run_id}/steps/{step_id}", response_model=RunStepDetailResponse)
async def run_step_detail(run_id: str, step_id: str) -> RunStepDetailResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    step = db.get_step(run_id, step_id)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    tool_calls = []
    for row in db.list_tool_calls_for_step(run_id, step.get("step_id")):
        args_payload = _json_loads(row.get("args_json"))
        result_payload = _json_loads(row.get("result_json"))
        error_payload = _json_loads(row.get("error_json"))
        tool_calls.append(
            {
                "tool_call_id": str(row.get("tool_call_id") or ""),
                "tool_name": str(row.get("tool_name") or ""),
                "status": str(row.get("status") or "pending"),
                "approved_bool": row.get("approved_bool"),
                "started_at": str(row.get("started_at") or ""),
                "finished_at": str(row.get("finished_at") or ""),
                "args": args_payload if isinstance(args_payload, dict) else {},
                "result": result_payload,
                "error": error_payload,
            }
        )

    status = _coerce_step_status(step.get("status"))
    replay_disabled_reason = _step_replay_disabled_reason({"status": status})
    output_payload = _json_loads(step.get("output_json"))
    error_payload = _json_loads(step.get("error_json"))
    return RunStepDetailResponse(
        run_id=run_id,
        step=RunStepDetailModel(
            step_id=db.format_step_id(step.get("step_id")),
            step_index=int(step.get("step_index") if step.get("step_index") is not None else 0),
            name=str(step.get("name") or ""),
            node_id=str(step.get("node_id") or "") or _extract_resume_node(step),
            node_type=str(step.get("node_type") or "") or None,
            status=status,
            started_at=str(step.get("started_at") or ""),
            ended_at=str(step.get("ended_at") or ""),
            summary=str(step.get("summary") or ""),
            input=_json_loads(step.get("input_json")),
            output=output_payload,
            error=error_payload if error_payload is not None else _step_error(output_payload, status),
            pre_state=_json_loads(step.get("pre_state_json")),
            post_state=_json_loads(step.get("post_state_json")),
            tool_calls=tool_calls,
            replayable=replay_disabled_reason is None,
            replay_disabled_reason=replay_disabled_reason,
        ),
    )


def _prepare_replay_context(
    *,
    run_id: str,
    request: RunReplayRequest,
) -> Dict[str, Any]:
    source_run = db.read_run(run_id)
    if not source_run:
        raise HTTPException(status_code=404, detail="Run not found")

    diagnostics: List[ReplayDiagnostic] = []
    workflow_version_id = str(source_run.get("workflow_version_id") or "").strip()
    if not workflow_version_id:
        diagnostics.append(
            ReplayDiagnostic(
                code="WORKFLOW_VERSION_MISSING",
                severity="error",
                message="Run does not have a workflow_version_id and cannot be replayed.",
                path="run.workflow_version_id",
            )
        )
        return {
            "diagnostics": diagnostics,
            "source_run": source_run,
            "version": None,
            "from_step": None,
            "fork_start_state": None,
            "resume_node_id": None,
        }

    version = db.get_workflow_version(workflow_version_id)
    if version is None:
        diagnostics.append(
            ReplayDiagnostic(
                code="WORKFLOW_VERSION_NOT_FOUND",
                severity="error",
                message="Referenced workflow version no longer exists.",
                path="run.workflow_version_id",
            )
        )

    from_step = db.get_step(run_id, request.from_step_id)
    if from_step is None:
        diagnostics.append(
            ReplayDiagnostic(
                code="STEP_NOT_FOUND",
                severity="error",
                message=f"Step '{request.from_step_id}' was not found in this run.",
                path="from_step_id",
            )
        )

    if from_step is not None and not _is_success_step_status(from_step.get("status")):
        diagnostics.append(
            ReplayDiagnostic(
                code="STEP_NOT_REPLAYABLE",
                severity="error",
                message="Replay source step must be completed successfully.",
                path="from_step_id",
            )
        )

    if _has_replay_errors(diagnostics):
        return {
            "diagnostics": diagnostics,
            "source_run": source_run,
            "version": version,
            "from_step": from_step,
            "fork_start_state": None,
            "resume_node_id": None,
        }

    base_source = "pre_state_json" if request.base_state == "pre" else "post_state_json"
    base_state = _json_loads(from_step.get(base_source))
    if not isinstance(base_state, dict):
        diagnostics.append(
            ReplayDiagnostic(
                code="BASE_STATE_MISSING",
                severity="error",
                message=f"Selected step is missing {request.base_state} state snapshot.",
                path=base_source,
            )
        )
        base_state = {}

    ordered_steps = db.read_steps(run_id)
    from_step_internal = db.parse_step_id(from_step.get("step_id"))
    resume_node_id: Optional[str] = None
    if bool(request.replay_this_step):
        resume_node_id = _extract_resume_node(from_step)
    else:
        found_anchor = False
        for candidate in ordered_steps:
            candidate_internal = db.parse_step_id(candidate.get("step_id") or candidate.get("id"))
            if from_step_internal is not None and candidate_internal == from_step_internal:
                found_anchor = True
                continue
            if not found_anchor:
                continue
            maybe_node = _extract_resume_node(candidate)
            if maybe_node:
                resume_node_id = maybe_node
                break
    if not resume_node_id:
        diagnostics.append(
            ReplayDiagnostic(
                code="RESUME_NODE_NOT_FOUND",
                severity="error",
                message="Unable to determine a resume node from the selected step.",
                path="from_step_id",
            )
        )

    fork_start_state, patch_diagnostics = _apply_state_patch(
        base_state=base_state,
        patch=request.state_patch,
        patch_mode=request.patch_mode,
    )
    diagnostics.extend(patch_diagnostics)

    workflow_spec = version.get("spec") if isinstance(version, dict) and isinstance(version.get("spec"), dict) else {}
    diagnostics.extend(
        _validate_state_schema(
            state=fork_start_state,
            workflow_spec=workflow_spec,
        )
    )
    return {
        "diagnostics": diagnostics,
        "source_run": source_run,
        "version": version,
        "from_step": from_step,
        "fork_start_state": fork_start_state if isinstance(fork_start_state, dict) else None,
        "resume_node_id": resume_node_id,
    }


def _resolve_step_id_by_index(run_id: str, step_index: int) -> Optional[str]:
    target_step_index = int(step_index)
    for row in db.read_steps(run_id):
        candidate_raw = row.get("step_index")
        if candidate_raw is None:
            continue
        try:
            candidate_step_index = int(candidate_raw)
        except (TypeError, ValueError):
            continue
        if candidate_step_index != target_step_index:
            continue
        step_id_value = row.get("step_id")
        if step_id_value is None:
            step_id_value = row.get("id")
        resolved = db.format_step_id(step_id_value)
        if resolved:
            return resolved
    return None


_PROMPT_EDIT_KEYS: Tuple[str, ...] = ("task", "message", "messages")
_LLM_STEP_HINTS: Tuple[str, ...] = ("llm", "build_messages")
_KNOWN_LLM_STEP_NAMES: Set[str] = {
    "llm_answer",
    "llm_execute",
    "llm_synthesize",
    "build_messages",
}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _did_prompt_fields_change(
    *,
    original_state: Dict[str, Any],
    edited_state: Dict[str, Any],
) -> bool:
    for key in _PROMPT_EDIT_KEYS:
        if _canonical_json(original_state.get(key)) != _canonical_json(edited_state.get(key)):
            return True
    return False


def _step_sort_key(step: Dict[str, Any]) -> Tuple[int, int]:
    step_index_raw = step.get("step_index")
    try:
        step_index = int(step_index_raw)
    except (TypeError, ValueError):
        step_index = 10**9

    internal_step_id = db.parse_step_id(step.get("step_id") or step.get("id"))
    if internal_step_id is None:
        internal_step_id = 10**9
    return (step_index, int(internal_step_id))


def _is_llm_related_step(step: Dict[str, Any]) -> bool:
    node_id = str(step.get("node_id") or "").strip().lower()
    node_type = str(step.get("node_type") or "").strip().lower()
    name = str(step.get("name") or "").strip().lower()

    if node_type == "llm":
        return True
    if name in _KNOWN_LLM_STEP_NAMES or node_id in _KNOWN_LLM_STEP_NAMES:
        return True
    return any(hint in node_id or hint in name for hint in _LLM_STEP_HINTS)


def _find_previous_successful_step_id(
    ordered_steps: List[Dict[str, Any]],
    before_index: int,
) -> Optional[str]:
    for pointer in range(before_index - 1, -1, -1):
        candidate = ordered_steps[pointer]
        if not _is_success_step_status(candidate.get("status")):
            continue
        step_id_value = candidate.get("step_id")
        if step_id_value is None:
            step_id_value = candidate.get("id")
        resolved = db.format_step_id(step_id_value)
        if resolved:
            return resolved
    return None


def _resolve_prompt_rewind_anchor_step_id(run_id: str) -> Optional[str]:
    ordered_steps = sorted(db.read_steps(run_id), key=_step_sort_key)
    if not ordered_steps:
        return None

    for index, step in enumerate(ordered_steps):
        if not _is_llm_related_step(step):
            continue
        previous_step_id = _find_previous_successful_step_id(ordered_steps, index)
        if previous_step_id:
            return previous_step_id
        break

    for index, step in enumerate(ordered_steps):
        if str(step.get("node_type") or "").strip().lower() == "system":
            continue
        if not _is_success_step_status(step.get("status")):
            continue
        previous_step_id = _find_previous_successful_step_id(ordered_steps, index)
        if previous_step_id:
            return previous_step_id
        break

    first_step = ordered_steps[0]
    if _is_success_step_status(first_step.get("status")):
        first_step_id = first_step.get("step_id")
        if first_step_id is None:
            first_step_id = first_step.get("id")
        resolved = db.format_step_id(first_step_id)
        if resolved:
            return resolved

    for step in ordered_steps:
        if not _is_success_step_status(step.get("status")):
            continue
        step_id_value = step.get("step_id")
        if step_id_value is None:
            step_id_value = step.get("id")
        resolved = db.format_step_id(step_id_value)
        if resolved:
            return resolved
    return None


def _resolve_resume_node_for_step(
    *,
    run_id: str,
    from_step: Dict[str, Any],
    replay_this_step: bool,
) -> Optional[str]:
    ordered_steps = db.read_steps(run_id)
    from_step_internal = db.parse_step_id(from_step.get("step_id"))

    if replay_this_step:
        return _extract_resume_node(from_step)

    found_anchor = False
    for candidate in ordered_steps:
        candidate_internal = db.parse_step_id(candidate.get("step_id") or candidate.get("id"))
        if from_step_internal is not None and candidate_internal == from_step_internal:
            found_anchor = True
            continue
        if not found_anchor:
            continue
        maybe_node = _extract_resume_node(candidate)
        if maybe_node:
            return maybe_node
    return None


def _build_run_input_from_source_run(source_run: Dict[str, Any]) -> Dict[str, Any]:
    payload = _json_loads(source_run.get("payload_json"))
    payload_obj = payload if isinstance(payload, dict) else {}
    source_input = payload_obj.get("input")
    if isinstance(source_input, dict):
        return copy.deepcopy(source_input)

    run_input: Dict[str, Any] = {}
    raw_task = str(payload_obj.get("task") or "").strip()
    raw_message = str(payload_obj.get("message") or "").strip()
    if raw_task:
        run_input["task"] = raw_task
    elif raw_message:
        run_input["task"] = raw_message

    raw_messages = payload_obj.get("messages")
    if isinstance(raw_messages, list):
        run_input["messages"] = copy.deepcopy(raw_messages)

    raw_context = payload_obj.get("context")
    if isinstance(raw_context, dict):
        run_input["context"] = copy.deepcopy(raw_context)

    raw_model_id = str(payload_obj.get("model_id") or "").strip()
    if raw_model_id:
        run_input["model"] = raw_model_id

    raw_tool_ids = payload_obj.get("tool_ids")
    if isinstance(raw_tool_ids, list):
        run_input["tool_ids"] = [str(item) for item in raw_tool_ids if str(item).strip()]

    raw_options = payload_obj.get("options")
    if isinstance(raw_options, dict):
        run_input["options"] = copy.deepcopy(raw_options)

    return run_input


@app.post("/runs/{run_id}/replay/dry_run", response_model=RunReplayResponse)
async def run_replay_dry_run(run_id: str, request: RunReplayRequest) -> RunReplayResponse:
    prepared = _prepare_replay_context(run_id=run_id, request=request)
    diagnostics = list(prepared.get("diagnostics") or [])
    return RunReplayResponse(
        new_run_id=None,
        diagnostics=diagnostics,
        fork_start_state=prepared.get("fork_start_state"),
        resume_node_id=str(prepared.get("resume_node_id") or "") or None,
    )


@app.post("/runs/{run_id}/replay", response_model=RunReplayResponse)
async def run_replay(run_id: str, request: RunReplayRequest) -> RunReplayResponse:
    prepared = _prepare_replay_context(run_id=run_id, request=request)
    diagnostics = list(prepared.get("diagnostics") or [])
    if _has_replay_errors(diagnostics):
        return RunReplayResponse(
            new_run_id=None,
            diagnostics=diagnostics,
            fork_start_state=prepared.get("fork_start_state"),
            resume_node_id=str(prepared.get("resume_node_id") or "") or None,
        )

    source_run = prepared.get("source_run") if isinstance(prepared.get("source_run"), dict) else {}
    version = prepared.get("version") if isinstance(prepared.get("version"), dict) else {}
    from_step = prepared.get("from_step") if isinstance(prepared.get("from_step"), dict) else {}
    resume_node_id = str(prepared.get("resume_node_id") or "").strip()
    fork_start_state = prepared.get("fork_start_state")
    if not isinstance(fork_start_state, dict):
        return RunReplayResponse(
            new_run_id=None,
            diagnostics=[
                ReplayDiagnostic(
                    code="FORK_STATE_INVALID",
                    severity="error",
                    message="Fork start state is invalid.",
                    path="fork_start_state",
                )
            ],
            fork_start_state=None,
            resume_node_id=resume_node_id or None,
        )

    payload = _json_loads(source_run.get("payload_json"))
    payload_obj = payload if isinstance(payload, dict) else {}
    source_input = payload_obj.get("input")
    run_input = dict(source_input) if isinstance(source_input, dict) else {}

    workflow_info = version.get("workflow") if isinstance(version.get("workflow"), dict) else {}
    compiled = version.get("compiled") if isinstance(version.get("compiled"), dict) else {}
    runtime_graph = compiled.get("runtime_graph") if isinstance(compiled.get("runtime_graph"), dict) else {}
    workflow_id = str(version.get("workflow_id") or "")
    workflow_defs_override = [
        {
            "id": workflow_id,
            "name": str(workflow_info.get("name") or workflow_id),
            "title": str(workflow_info.get("name") or workflow_id),
            "description": str(workflow_info.get("description") or ""),
            "type": "custom",
            "source": "custom",
            "enabled": True,
            "graph": runtime_graph,
            "compiled": compiled,
            "spec": dict(version.get("spec") or {}),
        }
    ]

    replay_run_id = str(uuid.uuid4())
    sandbox_mode = (
        bool(request.sandbox)
        if request.sandbox is not None
        else bool(source_run.get("sandbox_mode"))
    )
    parent_step_id = db.format_step_id(from_step.get("step_id"))
    fork_reason = "manual_state_edit" if bool(request.state_patch) else "replay"

    if sandbox_mode:
        def _tool_gate(payload: Dict[str, Any]) -> bool:
            return _SANDBOX_APPROVALS.request_decision(
                run_id=replay_run_id,
                payload=payload,
            )

        def _run_async() -> None:
            graph.run_workflow(
                workflow_id=workflow_id,
                input=run_input,
                workflow_defs=workflow_defs_override,
                run_id=replay_run_id,
                workflow_version_id=str(version.get("version_id") or ""),
                sandbox_mode=True,
                tool_call_gate=_tool_gate,
                initial_state=fork_start_state,
                replay_mode=True,
                resume_node_id=resume_node_id,
                parent_run_id=run_id,
                parent_step_id=parent_step_id,
                forked_from_state_json=json.dumps(fork_start_state, default=str, ensure_ascii=True),
                fork_patch_json=json.dumps(request.state_patch, default=str, ensure_ascii=True),
                fork_reason=fork_reason,
            )

        threading.Thread(target=_run_async, daemon=True).start()
        return RunReplayResponse(
            new_run_id=replay_run_id,
            diagnostics=diagnostics,
            fork_start_state=fork_start_state,
            resume_node_id=resume_node_id or None,
        )

    graph.run_workflow(
        workflow_id=workflow_id,
        input=run_input,
        workflow_defs=workflow_defs_override,
        run_id=replay_run_id,
        workflow_version_id=str(version.get("version_id") or ""),
        sandbox_mode=False,
        initial_state=fork_start_state,
        replay_mode=True,
        resume_node_id=resume_node_id,
        parent_run_id=run_id,
        parent_step_id=parent_step_id,
        forked_from_state_json=json.dumps(fork_start_state, default=str, ensure_ascii=True),
        fork_patch_json=json.dumps(request.state_patch, default=str, ensure_ascii=True),
        fork_reason=fork_reason,
    )
    return RunReplayResponse(
        new_run_id=replay_run_id,
        diagnostics=diagnostics,
        fork_start_state=fork_start_state,
        resume_node_id=resume_node_id or None,
    )


@app.post("/runs/{run_id}/rerun_from_state", response_model=RunRerunFromStateResponse)
async def run_rerun_from_state(
    run_id: str,
    request: RunRerunFromStateRequest,
) -> RunRerunFromStateResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    selected_step_id = _resolve_step_id_by_index(run_id, request.step_index)
    if not selected_step_id:
        return RunRerunFromStateResponse(
            new_run_id=None,
            diagnostics=[
                ReplayDiagnostic(
                    code="STEP_INDEX_OUT_OF_RANGE",
                    severity="error",
                    message=f"Step index {request.step_index} was not found in this run.",
                    path="step_index",
                )
            ],
        )

    selected_step = db.get_step(run_id, selected_step_id)
    if selected_step is None:
        return RunRerunFromStateResponse(
            new_run_id=None,
            diagnostics=[
                ReplayDiagnostic(
                    code="STEP_NOT_FOUND",
                    severity="error",
                    message=f"Step '{selected_step_id}' was not found in this run.",
                    path="step_index",
                )
            ],
        )

    if not _is_success_step_status(selected_step.get("status")):
        return RunRerunFromStateResponse(
            new_run_id=None,
            diagnostics=[
                ReplayDiagnostic(
                    code="STEP_NOT_REPLAYABLE",
                    severity="error",
                    message="Replay source step must be completed successfully.",
                    path="step_index",
                )
            ],
        )

    edited_state = copy.deepcopy(dict(request.state_json or {}))
    selected_post_state = _json_loads(selected_step.get("post_state_json"))
    selected_post_state_obj = (
        selected_post_state if isinstance(selected_post_state, dict) else {}
    )
    prompt_changed = _did_prompt_fields_change(
        original_state=selected_post_state_obj,
        edited_state=edited_state,
    )

    effective_from_step_id = selected_step_id
    replay_this_step = request.resume == "same"
    diagnostics: List[ReplayDiagnostic] = []
    if prompt_changed:
        rewind_step_id = _resolve_prompt_rewind_anchor_step_id(run_id)
        if rewind_step_id:
            effective_from_step_id = rewind_step_id
            replay_this_step = False
            diagnostics.append(
                ReplayDiagnostic(
                    code="PROMPT_CHANGE_REWIND_APPLIED",
                    severity="info",
                    message=(
                        "Prompt fields changed. Replay start was rewound to regenerate LLM output."
                    ),
                    path="step_index",
                )
            )

    source_workflow_version_id = str(run_row.get("workflow_version_id") or "").strip()
    if not source_workflow_version_id:
        from_step = db.get_step(run_id, effective_from_step_id)
        if from_step is None:
            return RunRerunFromStateResponse(
                new_run_id=None,
                diagnostics=[
                    ReplayDiagnostic(
                        code="STEP_NOT_FOUND",
                        severity="error",
                        message=f"Step '{effective_from_step_id}' was not found in this run.",
                        path="step_index",
                    )
                ],
            )

        if not _is_success_step_status(from_step.get("status")):
            return RunRerunFromStateResponse(
                new_run_id=None,
                diagnostics=[
                    ReplayDiagnostic(
                        code="STEP_NOT_REPLAYABLE",
                        severity="error",
                        message="Replay source step must be completed successfully.",
                        path="step_index",
                    )
                ],
            )

        resume_node_id = _resolve_resume_node_for_step(
            run_id=run_id,
            from_step=from_step,
            replay_this_step=replay_this_step,
        )
        if not resume_node_id and not replay_this_step:
            # If "next" points past the terminal node, fallback to replaying the selected step.
            replay_this_step = True
            resume_node_id = _resolve_resume_node_for_step(
                run_id=run_id,
                from_step=from_step,
                replay_this_step=True,
            )
        if not resume_node_id:
            return RunRerunFromStateResponse(
                new_run_id=None,
                diagnostics=[
                    ReplayDiagnostic(
                        code="RESUME_NODE_NOT_FOUND",
                        severity="error",
                        message="Unable to determine a resume node from the selected step.",
                        path="step_index",
                    )
                ],
            )

        payload = _json_loads(run_row.get("payload_json"))
        result = _json_loads(run_row.get("result_json"))
        metadata = _extract_run_metadata(payload=payload, result=result)
        workflow_id = str(metadata.get("workflow_id") or "").strip()
        if not workflow_id:
            return RunRerunFromStateResponse(
                new_run_id=None,
                diagnostics=[
                    ReplayDiagnostic(
                        code="WORKFLOW_ID_MISSING",
                        severity="error",
                        message="Run does not have a workflow_id and cannot be replayed.",
                        path="run.workflow_id",
                    )
                ],
            )

        replay_run_id = str(uuid.uuid4())
        sandbox_mode = (
            bool(request.sandbox)
            if request.sandbox is not None
            else bool(run_row.get("sandbox_mode"))
        )
        parent_step_id = db.format_step_id(from_step.get("step_id"))
        fork_start_state = edited_state
        fork_state_json = json.dumps(fork_start_state, default=str, ensure_ascii=True)
        fork_patch_json = json.dumps(edited_state, default=str, ensure_ascii=True)
        fork_reason = "manual_state_edit" if bool(edited_state) else "replay"
        run_input = _build_run_input_from_source_run(run_row)

        if sandbox_mode:
            def _tool_gate(payload: Dict[str, Any]) -> bool:
                return _SANDBOX_APPROVALS.request_decision(
                    run_id=replay_run_id,
                    payload=payload,
                )

            def _run_async() -> None:
                graph.run_workflow(
                    workflow_id=workflow_id,
                    input=run_input,
                    run_id=replay_run_id,
                    sandbox_mode=True,
                    tool_call_gate=_tool_gate,
                    initial_state=fork_start_state,
                    replay_mode=True,
                    resume_node_id=resume_node_id,
                    parent_run_id=run_id,
                    parent_step_id=parent_step_id,
                    forked_from_state_json=fork_state_json,
                    fork_patch_json=fork_patch_json,
                    fork_reason=fork_reason,
                )

            threading.Thread(target=_run_async, daemon=True).start()
            return RunRerunFromStateResponse(
                new_run_id=replay_run_id,
                diagnostics=diagnostics,
            )

        graph.run_workflow(
            workflow_id=workflow_id,
            input=run_input,
            run_id=replay_run_id,
            sandbox_mode=False,
            initial_state=fork_start_state,
            replay_mode=True,
            resume_node_id=resume_node_id,
            parent_run_id=run_id,
            parent_step_id=parent_step_id,
            forked_from_state_json=fork_state_json,
            fork_patch_json=fork_patch_json,
            fork_reason=fork_reason,
        )
        return RunRerunFromStateResponse(
            new_run_id=replay_run_id,
            diagnostics=diagnostics,
        )

    replay_request = RunReplayRequest(
        from_step_id=effective_from_step_id,
        state_patch=edited_state,
        patch_mode="replace",
        sandbox=request.sandbox,
        base_state="post",
        replay_this_step=replay_this_step,
    )
    replay_response = await run_replay(run_id, replay_request)
    if (
        not replay_this_step
        and replay_response.new_run_id is None
        and any(item.code == "RESUME_NODE_NOT_FOUND" for item in replay_response.diagnostics)
    ):
        replay_request.replay_this_step = True
        replay_response = await run_replay(run_id, replay_request)
    return RunRerunFromStateResponse(
        new_run_id=replay_response.new_run_id,
        diagnostics=[*diagnostics, *(list(replay_response.diagnostics or []))],
    )


@app.get("/runs/{run_id}/pending_tool_calls", response_model=PendingToolCallsResponse)
async def run_pending_tool_calls(run_id: str) -> PendingToolCallsResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    pending_rows = db.list_pending_tool_calls(run_id)
    pending: List[Dict[str, Any]] = []
    for row in pending_rows:
        args_payload = _json_loads(row.get("args_json"))
        pending.append(
            {
                "tool_call_id": str(row.get("tool_call_id") or ""),
                "run_id": str(row.get("run_id") or run_id),
                "step_id": row.get("step_id"),
                "tool_name": str(row.get("tool_name") or ""),
                "args": args_payload if isinstance(args_payload, dict) else {},
                "approved_bool": row.get("approved_bool"),
                "started_at": str(row.get("started_at") or ""),
                "status": str(row.get("status") or "pending"),
            }
        )
    return PendingToolCallsResponse(run_id=run_id, pending=pending)


@app.post(
    "/runs/{run_id}/tool_calls/{tool_call_id}/approve",
    response_model=ToolCallApprovalResponse,
)
async def approve_tool_call(
    run_id: str,
    tool_call_id: str,
    request: ToolCallApprovalRequest,
) -> ToolCallApprovalResponse:
    success = _SANDBOX_APPROVALS.submit_decision(
        run_id=run_id,
        tool_call_id=tool_call_id,
        approved=bool(request.approved),
    )
    if not success:
        raise HTTPException(status_code=404, detail="Pending tool call not found.")

    return ToolCallApprovalResponse(
        run_id=run_id,
        tool_call_id=tool_call_id,
        approved=bool(request.approved),
        status="approved" if request.approved else "rejected",
    )


@app.get("/runs/{run_id}/state", response_model=RunStateResponse)
async def run_state(run_id: str) -> RunStateResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    snapshots_rows = db.read_state_snapshots(run_id)
    if snapshots_rows:
        snapshots: List[RunSnapshot] = []
        for index, snapshot in enumerate(snapshots_rows):
            parsed_state = _json_loads(snapshot.get("state_json"))
            snapshots.append(
                RunSnapshot(
                    step_index=int(
                        snapshot.get("step_index")
                        if snapshot.get("step_index") is not None
                        else index
                    ),
                    timestamp=str(snapshot.get("timestamp") or ""),
                    state=_safe_state(parsed_state),
                )
            )
        return RunStateResponse(run_id=run_id, snapshots=snapshots, derived=False)

    payload = _json_loads(run_row.get("payload_json"))
    step_rows = db.read_steps(run_id)
    parsed_steps: List[Dict[str, Any]] = []
    for index, step in enumerate(step_rows):
        parsed_steps.append(
            {
                "step_index": int(step.get("step_index") if step.get("step_index") is not None else index),
                "started_at": str(step.get("started_at") or ""),
                "ended_at": str(step.get("ended_at") or ""),
                "output": _json_loads(step.get("output_json")),
            }
        )

    derived_snapshots = _derive_state_snapshots(payload=payload, steps=parsed_steps)
    return RunStateResponse(
        run_id=run_id,
        snapshots=[RunSnapshot(**snapshot) for snapshot in derived_snapshots],
        derived=True,
    )


# curl -X GET http://localhost:8000/models
# curl -X GET http://localhost:8000/workflows
#
# curl -X POST http://localhost:8000/builder/tools \
#   -H "Content-Type: application/json" \
#   -d '{"name":"Acme Search","kind":"external","description":"Query Acme API","type":"http","config":{"url":"https://api.example.com/search","method":"POST","headers":{"Authorization":"Bearer token"},"timeout_ms":8000},"enabled":true}'
#
# curl -X GET http://localhost:8000/tools
#
# curl -X PATCH http://localhost:8000/tools/tool.custom.acme_search \
#   -H "Content-Type: application/json" \
#   -d '{"enabled":false}'
#
# curl -X POST http://localhost:8000/builder/workflows \
#   -H "Content-Type: application/json" \
#   -d '{"name":"Customer Triage","description":"Route customer queries","enabled":true,"graph":{"nodes":[{"id":"start","type":"start","config":{}},{"id":"llm_1","type":"llm","config":{"prompt_template":"Classify: {{query}}","output_key":"classification"}},{"id":"end","type":"end","config":{"response_template":"Result: {{artifacts.classification}}"}}],"edges":[{"from":"start","to":"llm_1","condition":"always"},{"from":"llm_1","to":"end","condition":"always"}]}}'
#
# curl -X GET http://localhost:8000/builder/workflows/workflow.custom.customer_triage
#
# curl -X PATCH http://localhost:8000/builder/workflows/workflow.custom.customer_triage \
#   -H "Content-Type: application/json" \
#   -d '{"enabled":false}'
#
# curl -X POST http://localhost:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"message":"Find recent alerts","workflow_id":"complex.v1","model_id":"llama3.1:8b","tool_ids":["tool.custom.acme_search"]}'
#
# curl -N -X POST http://localhost:8000/chat/stream \
#   -H "Content-Type: application/json" \
#   -d '{"message":"Summarize our release notes","workflow_id":"simple.v1","model_id":"llama3.1:8b","tool_ids":[],"stream":true}'
#
# curl -X POST http://localhost:8000/workflow/compile \
#   -H "Content-Type: application/json" \
#   -d '{"task":"Summarize architecture changes","workflow_type":"complex","context":{"repo":"saturday_agent"}}'
#
# curl -X POST http://localhost:8000/workflow/run \
#   -H "Content-Type: application/json" \
#   -d '{"workflow_id":"workflow.custom.customer_triage","input":{"task":"classify this","context":{}}}'
#
# curl -X GET http://localhost:8000/runs/<run_id>
# curl -X GET http://localhost:8000/runs/<run_id>/logs
# curl -X GET http://localhost:8000/runs/<run_id>/state
