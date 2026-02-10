from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import artifacts, db, graph

try:
    from dotenv import load_dotenv

    _MAIN_FILE = Path(__file__).resolve()
    _API_DIR = _MAIN_FILE.parents[1]
    _REPO_ROOT = _MAIN_FILE.parents[3]

    # Load non-committed local env first, then standard .env files.
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


class ChatRunRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    workflow_id: str
    model_id: str
    tool_ids: List[str] = Field(default_factory=list)
    vision_model_id: Optional[str] = None
    artifact_ids: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class ChatRunStep(BaseModel):
    name: str
    status: Literal["ok", "error"]
    started_at: str
    ended_at: str


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
    description: str
    type: str
    version: str
    status: str


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


class ToolSummary(BaseModel):
    id: str
    name: str
    kind: str = "local"
    description: str = ""
    enabled: bool


class ToolsResponse(BaseModel):
    tools: List[ToolSummary]


class WorkflowCompileRequest(BaseModel):
    task: str
    context: Optional[Dict[str, Any]] = None
    workflow_type: Optional[WorkflowType] = None


class WorkflowGraph(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowCompileResponse(BaseModel):
    workflow_id: str
    workflow_type: WorkflowType
    graph: WorkflowGraph


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


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))
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
            base_url=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT
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
    return WorkflowsResponse(workflows=[WorkflowSummary(**item) for item in payload])


@app.get("/tools", response_model=ToolsResponse)
async def tools() -> ToolsResponse:
    payload = graph.list_tools()
    return ToolsResponse(tools=[ToolSummary(**item) for item in payload])


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
# curl -X GET http://localhost:8000/tools
# curl -X POST http://localhost:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"message":"Summarize this repo","workflow_id":"moderate.v1","model_id":"llama3.1:8b","tool_ids":["filesystem.read"],"context":{"tone":"concise"}}'
#
# curl -X POST http://localhost:8000/workflow/compile \
#   -H "Content-Type: application/json" \
#   -d '{"task":"Summarize architecture changes","workflow_type":"complex","context":{"repo":"saturday_agent"}}'
#
# curl -X POST http://localhost:8000/workflow/compile \
#   -H "Content-Type: application/json" \
#   -d '{"task":"Say hello in one sentence"}'
#
# curl -X POST http://localhost:8000/workflow/run \
#   -H "Content-Type: application/json" \
#   -d '{"workflow_id":"moderate.v1","input":{"task":"Explain this repo"}}'
#
# curl -X GET http://localhost:8000/runs/<run_id>
# curl -X GET http://localhost:8000/runs/<run_id>/logs
# curl -X GET http://localhost:8000/runs/<run_id>/state
