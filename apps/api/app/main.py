from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import db, graph

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


def _json_loads(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
async def models() -> ModelsResponse:
    payload = graph.list_models()
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


@app.get("/runs/{run_id}/logs")
async def run_logs(run_id: str) -> Dict[str, Any]:
    run_row = db.get_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    steps = db.list_steps(run_id)
    steps_payload = []
    for step in steps:
        steps_payload.append(
            {
                "id": step.get("id"),
                "run_id": step.get("run_id"),
                "step_index": step.get("step_index"),
                "name": step.get("name"),
                "status": step.get("status"),
                "input": _json_loads(step.get("input_json")),
                "output": _json_loads(step.get("output_json")),
                "started_at": step.get("started_at"),
                "ended_at": step.get("ended_at"),
            }
        )

    return {
        "run": {
            "run_id": run_row.get("run_id"),
            "kind": run_row.get("kind"),
            "status": run_row.get("status"),
            "payload": _json_loads(run_row.get("payload_json")),
            "result": _json_loads(run_row.get("result_json")),
            "started_at": run_row.get("started_at"),
            "ended_at": run_row.get("ended_at"),
        },
        "steps": steps_payload,
    }


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
# curl -X GET http://localhost:8000/runs/<run_id>/logs
