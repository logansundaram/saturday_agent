from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app import artifacts, db, graph, streaming

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
    task: str
    context: Optional[Dict[str, Any]] = None
    workflow_type: Optional[WorkflowType] = None


class WorkflowCompileResponse(BaseModel):
    workflow_id: str
    workflow_type: str
    graph: Dict[str, Any]


class WorkflowRunRequest(BaseModel):
    workflow_id: str
    input: Dict[str, Any]


class WorkflowRunResponse(BaseModel):
    run_id: str
    workflow_id: str
    workflow_type: str
    status: str
    output: Dict[str, Any]
    steps: List[Dict[str, Any]]


class RunStepError(BaseModel):
    message: str
    stack: Optional[str] = None


class RunLogStep(BaseModel):
    step_index: int
    name: str
    status: Literal["ok", "error"]
    started_at: str
    ended_at: str
    summary: Optional[str] = None
    input: Any = None
    output: Any = None
    error: Optional[RunStepError] = None


class RunLogsResponse(BaseModel):
    run_id: str
    steps: List[RunLogStep]


class RunDetailResponse(BaseModel):
    run_id: str
    kind: Literal["chat", "workflow"]
    status: str
    workflow_id: str
    workflow_type: Optional[str] = None
    model_id: Optional[str] = None
    tool_ids: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    payload: Any = None
    result: Any = None


class RunSnapshot(BaseModel):
    step_index: int
    timestamp: str
    state: Dict[str, Any]


class RunStateResponse(BaseModel):
    run_id: str
    snapshots: List[RunSnapshot]
    derived: bool = False


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


def _coerce_step_status(value: Any) -> Literal["ok", "error"]:
    return "ok" if str(value or "").lower() == "ok" else "error"


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


def _step_error(step_output: Any, status: Literal["ok", "error"]) -> Optional[Dict[str, str]]:
    if status != "error":
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


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    result = graph.compile_workflow(
        task=request.task,
        context=request.context,
        workflow_type=request.workflow_type,
    )
    return WorkflowCompileResponse(**result)


@app.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(request: WorkflowRunRequest) -> WorkflowRunResponse:
    result = graph.run_workflow(
        workflow_id=request.workflow_id,
        input=request.input,
    )
    return WorkflowRunResponse(**result)


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
        workflow_type=metadata.get("workflow_type"),
        model_id=metadata.get("model_id"),
        tool_ids=metadata.get("tool_ids") or [],
        started_at=run_row.get("started_at"),
        ended_at=run_row.get("ended_at"),
        payload=payload,
        result=result,
    )


@app.get("/runs/{run_id}/logs", response_model=RunLogsResponse)
async def run_logs(run_id: str) -> RunLogsResponse:
    run_row = db.read_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    step_rows = db.read_steps(run_id)
    steps_payload: List[RunLogStep] = []
    for index, step in enumerate(step_rows):
        status = _coerce_step_status(step.get("status"))
        step_output = _json_loads(step.get("output_json"))
        steps_payload.append(
            RunLogStep(
                step_index=int(step.get("step_index") if step.get("step_index") is not None else index),
                name=str(step.get("name") or ""),
                status=status,
                started_at=str(step.get("started_at") or ""),
                ended_at=str(step.get("ended_at") or ""),
                summary=str(step.get("summary") or ""),
                input=_json_loads(step.get("input_json")),
                output=step_output,
                error=_step_error(step_output, status),
            )
        )

    return RunLogsResponse(run_id=run_id, steps=steps_payload)


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
