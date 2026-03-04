from __future__ import annotations

import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app import db, graph
from app.services import qdrant_client as qdrant_client_service
from app.workflows import service as workflow_service

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from saturday_agent.rag.embeddings import DEFAULT_OLLAMA_EMBED_MODEL
    from saturday_agent.rag.ingest.chunking import chunk_text
    from saturday_agent.rag.ingest.pdf_extract import PdfExtractionError, extract_pdf_text
    from saturday_agent.rag.ingest.qdrant_index import index_chunks_to_qdrant
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    agent_src = repo_root / "apps/agent/src"
    if str(agent_src) not in sys.path:
        sys.path.append(str(agent_src))
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from saturday_agent.rag.embeddings import DEFAULT_OLLAMA_EMBED_MODEL
    from saturday_agent.rag.ingest.chunking import chunk_text
    from saturday_agent.rag.ingest.pdf_extract import PdfExtractionError, extract_pdf_text
    from saturday_agent.rag.ingest.qdrant_index import index_chunks_to_qdrant


PROJECT_DOCS_ROOT = Path(__file__).resolve().parents[2] / "data" / "project_docs"
DEFAULT_PROJECT_WORKFLOW_ID_SUFFIX = "default"
MAX_RETRIEVAL_RESULTS = 5


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_collection_name(project_id: str) -> str:
    return f"project_{project_id}"


def project_runtime_workflow_id(project_id: str) -> str:
    return f"project.{project_id}.workflow.{DEFAULT_PROJECT_WORKFLOW_ID_SUFFIX}"


def list_projects_payload() -> List[dict]:
    return db.list_projects()


def create_project_payload(*, name: str, description: str = "") -> dict:
    if not str(name or "").strip():
        raise ValueError("Project name is required.")
    return db.create_project(name=name, description=description)


def get_project_payload(project_id: str) -> dict:
    payload = db.get_project_detail(project_id)
    if payload is None:
        raise ValueError("Project not found.")
    payload["tools"] = list_project_tools_payload(project_id)
    workflow_payload = get_project_workflow_payload(project_id)
    payload["workflow"] = workflow_payload
    return payload


def delete_project_payload(project_id: str) -> bool:
    existing = db.get_project_detail(project_id)
    if existing is None:
        raise ValueError("Project not found.")

    for document in existing.get("documents") or []:
        if not isinstance(document, dict):
            continue
        filepath = Path(str(document.get("filepath") or "")).expanduser()
        doc_dir = filepath.parent
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
    return db.delete_project(project_id)


def create_project_chat_payload(*, project_id: str, name: str) -> dict:
    return db.create_project_chat(project_id=project_id, name=name)


def list_project_documents_payload(project_id: str) -> List[dict]:
    _require_project(project_id)
    return db.list_project_documents(project_id)


def ingest_project_document_payload(
    *,
    project_id: str,
    filename: str,
    content: bytes,
    embedding_model: Optional[str] = None,
) -> dict:
    _require_project(project_id)

    safe_name = Path(str(filename or "").strip() or f"upload-{uuid.uuid4().hex}.txt").name
    resolved_embedding_model = str(
        embedding_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
    ).strip()
    if not resolved_embedding_model:
        raise ValueError("An embedding model is required for project document ingest.")

    doc_record = db.create_project_document(
        project_id=project_id,
        filename=safe_name,
        filepath="",
        embedding_model=resolved_embedding_model,
        status="ingesting",
    )
    doc_id = str(doc_record.get("id") or "")
    if not doc_id:
        raise RuntimeError("Project document id was not generated.")

    target_dir = PROJECT_DOCS_ROOT / project_id / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_path = (target_dir / safe_name).resolve()
    stored_path.write_bytes(bytes(content or b""))
    db.update_project_document(
        project_id=project_id,
        doc_id=doc_id,
        filepath=str(stored_path),
    )

    try:
        extracted_text = _extract_document_text(filename=safe_name, stored_path=stored_path)
        chunks = chunk_text(extracted_text, chunk_size=900, overlap=150)
        if not chunks:
            raise ValueError("No non-empty chunks were produced from the uploaded document.")

        qdrant_url = _runtime_qdrant_url()
        index_chunks_to_qdrant(
            chunks,
            doc_id=doc_id,
            filename=safe_name,
            sha256=_sha256_for(stored_path),
            source_path=str(stored_path),
            collection=project_collection_name(project_id),
            embeddings_model=resolved_embedding_model,
            qdrant_url=qdrant_url or None,
        )
        updated = db.update_project_document(
            project_id=project_id,
            doc_id=doc_id,
            status="indexed",
            chunk_count=len(chunks),
            error_message="",
            filepath=str(stored_path),
            embedding_model=resolved_embedding_model,
        )
        if updated is None:
            raise RuntimeError("Project document metadata update failed.")
        return updated
    except Exception as exc:
        db.update_project_document(
            project_id=project_id,
            doc_id=doc_id,
            status="error",
            error_message=str(exc),
            filepath=str(stored_path),
            embedding_model=resolved_embedding_model,
        )
        raise


def delete_project_document_payload(*, project_id: str, doc_id: str) -> dict:
    document = db.get_project_document(project_id=project_id, doc_id=doc_id)
    if document is None:
        raise ValueError("Project document not found.")

    qdrant_url = _runtime_qdrant_url()
    if qdrant_url:
        client = QdrantClient(url=qdrant_url)
        doc_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_id",
                    match=qdrant_models.MatchValue(value=doc_id),
                )
            ]
        )
        selector = qdrant_models.FilterSelector(filter=doc_filter)
        try:
            client.delete(
                collection_name=project_collection_name(project_id),
                points_selector=selector,
                wait=True,
            )
        except TypeError:
            client.delete(
                collection_name=project_collection_name(project_id),
                points_selector=doc_filter,
                wait=True,
            )

    filepath = Path(str(document.get("filepath") or "")).expanduser()
    doc_dir = filepath.parent
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    deleted = db.delete_project_document(project_id, doc_id)
    if deleted is None:
        raise RuntimeError("Project document delete failed.")
    return deleted


def get_project_ground_truth_payload(project_id: str) -> dict:
    return db.get_project_ground_truth(project_id)


def update_project_ground_truth_payload(*, project_id: str, content: str) -> dict:
    return db.upsert_project_ground_truth(project_id=project_id, content=content)


def list_project_tools_payload(project_id: str) -> List[dict]:
    _require_project(project_id)
    available_tools = {
        str(item.get("id") or ""): dict(item)
        for item in graph.list_tools()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    bindings_by_tool = {
        str(item.get("tool_name") or ""): dict(item)
        for item in db.list_project_tools(project_id)
        if isinstance(item, dict)
    }

    payload: List[dict] = []
    for tool_id in sorted(available_tools.keys(), key=lambda item: available_tools[item].get("name", item)):
        tool = dict(available_tools[tool_id])
        binding = bindings_by_tool.get(tool_id)
        tool["project_binding_id"] = str(binding.get("id") or "") if isinstance(binding, dict) else None
        tool["tool_name"] = tool_id
        tool["bound"] = binding is not None
        if binding is not None:
            tool["enabled"] = bool(binding.get("enabled", tool.get("enabled", False)))
            tool["version"] = int(binding.get("version") or tool.get("version") or 1)
        payload.append(tool)
    return payload


def replace_project_tools_payload(
    *,
    project_id: str,
    bindings: Sequence[Dict[str, Any]],
) -> List[dict]:
    available_tools = {
        str(item.get("id") or ""): dict(item)
        for item in graph.list_tools()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    normalized: List[Dict[str, Any]] = []
    for item in bindings:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or item.get("id") or "").strip()
        if not tool_name:
            continue
        if tool_name not in available_tools:
            raise ValueError(f"Unknown tool '{tool_name}'.")
        base_tool = available_tools[tool_name]
        normalized.append(
            {
                "id": str(item.get("id") or item.get("project_binding_id") or uuid.uuid4()),
                "tool_name": tool_name,
                "enabled": bool(item.get("enabled", base_tool.get("enabled", True))),
                "version": int(item.get("version") or base_tool.get("version") or 1),
            }
        )
    db.replace_project_tools(project_id, normalized)
    return list_project_tools_payload(project_id)


def get_project_workflow_payload(project_id: str) -> Optional[dict]:
    _require_project(project_id)
    stored = db.get_default_project_workflow(project_id)
    if stored is None:
        return None
    normalized_spec = _normalize_project_workflow_spec(
        project_id=project_id,
        workflow_spec=stored.get("workflow_spec") if isinstance(stored.get("workflow_spec"), dict) else {},
    )
    compile_payload = workflow_service.compile_workflow_spec(workflow_spec=normalized_spec)
    return {
        **stored,
        "workflow_spec": normalized_spec,
        "compiled": dict(compile_payload.get("compiled") or {}),
        "diagnostics": list(compile_payload.get("diagnostics") or []),
        "valid": bool(compile_payload.get("valid", False)),
    }


def save_project_workflow_payload(
    *,
    project_id: str,
    workflow_spec: Dict[str, Any],
) -> dict:
    _require_project(project_id)
    normalized_spec = _normalize_project_workflow_spec(
        project_id=project_id,
        workflow_spec=dict(workflow_spec or {}),
    )
    compile_payload = workflow_service.compile_workflow_spec(workflow_spec=normalized_spec)
    diagnostics = list(compile_payload.get("diagnostics") or [])
    has_errors = any(str(item.get("severity") or "") == "error" for item in diagnostics)
    if has_errors:
        return {
            "valid": False,
            "workflow_spec": normalized_spec,
            "compiled": dict(compile_payload.get("compiled") or {}),
            "diagnostics": diagnostics,
        }

    stored = db.create_project_workflow(
        project_id=project_id,
        workflow_spec=normalized_spec,
        is_default=True,
    )
    return {
        **stored,
        "valid": True,
        "workflow_spec": normalized_spec,
        "compiled": dict(compile_payload.get("compiled") or {}),
        "diagnostics": diagnostics,
    }


def run_project_chat_workflow(
    *,
    project_id: str,
    chat_id: str,
    message: str,
    workflow_id: str,
    model_id: str,
    tool_ids: Sequence[str],
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    vision_model_id: Optional[str] = None,
    artifact_ids: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _execute_project_chat_run(
        project_id=project_id,
        chat_id=chat_id,
        message=message,
        workflow_id=workflow_id,
        model_id=model_id,
        tool_ids=tool_ids,
        messages=messages,
        vision_model_id=vision_model_id,
        artifact_ids=artifact_ids,
        context=context,
        emit_event=None,
    )


def run_project_chat_workflow_stream(
    *,
    project_id: str,
    chat_id: str,
    message: str,
    workflow_id: str,
    model_id: str,
    tool_ids: Sequence[str],
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    vision_model_id: Optional[str] = None,
    artifact_ids: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    emit_event: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    return _execute_project_chat_run(
        project_id=project_id,
        chat_id=chat_id,
        message=message,
        workflow_id=workflow_id,
        model_id=model_id,
        tool_ids=tool_ids,
        messages=messages,
        vision_model_id=vision_model_id,
        artifact_ids=artifact_ids,
        context=context,
        emit_event=emit_event,
    )


def _execute_project_chat_run(
    *,
    project_id: str,
    chat_id: str,
    message: str,
    workflow_id: str,
    model_id: str,
    tool_ids: Sequence[str],
    messages: Optional[Sequence[Dict[str, Any]]],
    vision_model_id: Optional[str],
    artifact_ids: Optional[Sequence[str]],
    context: Optional[Dict[str, Any]],
    emit_event: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    project = _require_project(project_id)
    chat = db.get_project_chat(project_id=project_id, chat_id=chat_id)
    if chat is None:
        raise ValueError("Project chat not found.")

    execution = _resolve_project_execution(
        project_id=project_id,
        requested_workflow_id=workflow_id,
        message=message,
        requested_tool_ids=tool_ids,
        model_id=model_id,
        messages=messages,
        vision_model_id=vision_model_id,
        artifact_ids=artifact_ids,
        context=context,
        project=project,
        chat=chat,
    )

    run_id = str(uuid.uuid4())
    started_at = graph._to_iso(graph._now_utc())
    base_payload = {
        "project_id": project_id,
        "chat_id": chat_id,
        "workflow_id": execution["workflow_id"],
        "workflow_source": execution["workflow_source"],
        "project_workflow_id": execution.get("project_workflow_id"),
        "model_id": model_id,
        "tool_ids": execution["resolved_tool_ids"],
        "message": message,
        "messages": execution["request_messages"],
        "vision_model_id": vision_model_id,
        "artifact_ids": list(execution["artifact_ids"]),
        "context": execution["request_context"],
    }

    db.create_run(
        run_id,
        "project_chat_stream" if emit_event else "project_chat",
        graph._json_dumps(base_payload),
        started_at,
        workflow_id=execution["workflow_id"],
        project_id=project_id,
        chat_id=chat_id,
    )

    if emit_event:
        emit_event(
            {
                "type": "run_start",
                "run_id": run_id,
                "workflow_id": execution["workflow_id"],
                "model_id": model_id,
                "tool_ids": list(execution["resolved_tool_ids"]),
                "started_at": started_at,
            }
        )

    recorder = graph.StepRecorder(run_id, event_sink=emit_event)
    recorder.log_ingest(
        input_data={
            "project_id": project_id,
            "chat_id": chat_id,
            "message": message,
            "workflow_id": execution["workflow_id"],
            "tool_ids": list(execution["resolved_tool_ids"]),
        },
        workflow_id=execution["workflow_id"],
    )

    _record_project_context_injection(
        recorder=recorder,
        run_id=run_id,
        project=project,
        chat=chat,
        message=message,
        normalized_messages=execution["request_messages"],
        ground_truth=execution["ground_truth"],
    )
    _record_project_retrieval(
        recorder=recorder,
        retrieval_input=execution["retrieval_input"],
        retrieval_output=execution["retrieval_output"],
        status=execution["retrieval_status"],
    )

    workflow_result = graph._execute_workflow_run(
        workflow_id=execution["workflow_id"],
        input=execution["workflow_input"],
        recorder=recorder,
        tool_defs=execution["project_tool_overrides"],
        workflow_defs=execution["workflow_defs_override"],
        initial_state=execution["initial_state"],
    )
    status = str(workflow_result.get("status") or "error")
    output = workflow_result.get("output") if isinstance(workflow_result.get("output"), dict) else {}
    output_text = (
        str(output.get("answer") or "")
        if status == "ok"
        else str(output.get("error") or "Project workflow execution failed.")
    )
    ended_at = graph._to_iso(graph._now_utc())

    if emit_event and status != "ok":
        emit_event({"type": "error", "run_id": run_id, "message": output_text})
    if emit_event:
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
        "project_id": project_id,
        "chat_id": chat_id,
        "workflow_id": execution["workflow_id"],
        "workflow_source": execution["workflow_source"],
        "project_workflow_id": execution.get("project_workflow_id"),
        "model_id": model_id,
        "tool_ids": list(execution["resolved_tool_ids"]),
        "status": status,
        "output_text": output_text,
        "steps": graph._normalize_chat_steps(recorder.steps),
    }
    db.finish_run(
        run_id,
        status=status,
        ended_at=ended_at,
        result_json=graph._json_dumps(result_payload),
        error=str(output.get("error") or "") if status != "ok" else None,
    )
    return result_payload


def _resolve_project_execution(
    *,
    project_id: str,
    requested_workflow_id: str,
    message: str,
    requested_tool_ids: Sequence[str],
    model_id: str,
    messages: Optional[Sequence[Dict[str, Any]]],
    vision_model_id: Optional[str],
    artifact_ids: Optional[Sequence[str]],
    context: Optional[Dict[str, Any]],
    project: Dict[str, Any],
    chat: Dict[str, Any],
) -> Dict[str, Any]:
    workflow_resolution = _resolve_workflow_override(
        project_id=project_id,
        requested_workflow_id=requested_workflow_id,
    )

    request_messages = _normalize_messages(message=message, raw_messages=messages)
    request_context = dict(context or {})
    normalized_artifact_ids = [
        str(item).strip()
        for item in (artifact_ids or [])
        if str(item).strip()
    ]

    tool_catalog = {
        str(item.get("id") or ""): dict(item)
        for item in list_project_tools_payload(project_id)
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    resolved_tool_ids, selected_tool_defs = _resolve_requested_tools(
        requested_tool_ids=requested_tool_ids,
        tool_catalog=tool_catalog,
        normalized_artifact_ids=normalized_artifact_ids,
        vision_model_id=vision_model_id,
        workflow_id=workflow_resolution["workflow_id"],
    )
    resolved_tool_ids = [tool_id for tool_id in resolved_tool_ids if tool_id != "rag.retrieve"]

    selected_embedding_model = _resolve_project_embedding_model(project_id)
    retrieval_context = graph._inject_runtime_qdrant_context(dict(request_context))
    retrieval_context["retrieval_collection"] = project_collection_name(project_id)
    retrieval_context["embedding_model"] = selected_embedding_model
    retrieval_context["project_id"] = project_id
    retrieval_context["project_chat_id"] = chat.get("id")

    retrieval_input = {
        "query": str(message or "").strip(),
        "collection": project_collection_name(project_id),
        "embedding_model": selected_embedding_model,
        "top_k": MAX_RETRIEVAL_RESULTS,
    }
    retrieval_output, retrieval_status = _run_project_retrieval(
        tool_defs=workflow_resolution["project_tool_overrides"],
        tool_input=retrieval_input,
        context=retrieval_context,
    )
    retrieval_data = (
        retrieval_output.get("data")
        if isinstance(retrieval_output.get("data"), dict)
        else {}
    )
    retrieval_results = (
        retrieval_data.get("results")
        if isinstance(retrieval_data.get("results"), list)
        else []
    )
    citations = _format_citations(retrieval_results)

    project_ground_truth = db.get_project_ground_truth(project_id)
    ground_truth_text = str(project_ground_truth.get("content") or "")
    history_text = _render_history_text(request_messages)
    retrieval_text = _render_retrieval_text(retrieval_results)
    system_message = _build_project_system_message(
        project=project,
        ground_truth=ground_truth_text,
        history_text=history_text,
        retrieval_text=retrieval_text,
        latest_prompt=message,
    )

    workflow_context = graph._inject_runtime_qdrant_context(dict(request_context))
    workflow_context.update(
        {
            "project_id": project_id,
            "project_name": str(project.get("name") or ""),
            "project_description": str(project.get("description") or ""),
            "project_chat_id": str(chat.get("id") or ""),
            "project_chat_name": str(chat.get("name") or ""),
            "project_ground_truth": ground_truth_text,
            "project_history_text": history_text,
            "project_retrieval": {
                "results": retrieval_results,
                "collection": project_collection_name(project_id),
                "embedding_model": selected_embedding_model,
                "query": str(message or ""),
            },
            "retrieval_collection": project_collection_name(project_id),
            "embedding_model": selected_embedding_model,
            "rag_enabled": False,
            "workflow_source": workflow_resolution["workflow_source"],
            "project_workflow_id": workflow_resolution.get("project_workflow_id"),
        }
    )
    if normalized_artifact_ids:
            workflow_context["artifact_ids"] = list(normalized_artifact_ids)
    if vision_model_id:
        workflow_context["vision_model_id"] = str(vision_model_id)

    tool_inputs = dict(workflow_context.get("tool_inputs") or {})
    if normalized_artifact_ids and str(vision_model_id or "").strip():
        existing_vision_input = tool_inputs.get("vision.analyze")
        vision_tool_input = (
            dict(existing_vision_input) if isinstance(existing_vision_input, dict) else {}
        )
        if not str(vision_tool_input.get("artifact_id") or "").strip():
            if workflow_resolution["workflow_id"] == "moderate.v1" and len(normalized_artifact_ids) > 1:
                vision_tool_input["artifact_id"] = ",".join(normalized_artifact_ids)
            else:
                vision_tool_input["artifact_id"] = normalized_artifact_ids[0]
        if not str(vision_tool_input.get("prompt") or "").strip():
            vision_tool_input["prompt"] = str(message or "Analyze the attached image.")
        vision_tool_input["vision_model_id"] = str(vision_model_id)
        tool_inputs["vision.analyze"] = vision_tool_input
    for tool_id in resolved_tool_ids:
        existing_input = tool_inputs.get(tool_id)
        tool_input = dict(existing_input) if isinstance(existing_input, dict) else {}
        if not str(tool_input.get("query") or "").strip():
            tool_input["query"] = str(message or "")
        tool_inputs[tool_id] = tool_input
    if tool_inputs:
        workflow_context["tool_inputs"] = tool_inputs
        workflow_context["tool_defs"] = selected_tool_defs

    workflow_messages = [
        {"role": "system", "content": system_message},
        *[item for item in request_messages if str(item.get("role") or "").lower() != "system"],
    ]

    initial_state = {
        "project_ground_truth": ground_truth_text,
        "project_context": {
            "project_id": project_id,
            "chat_id": str(chat.get("id") or ""),
            "history": request_messages,
        },
        "retrieval": {
            "results": retrieval_results,
            "collection": project_collection_name(project_id),
            "embedding_model": selected_embedding_model,
            "query": str(message or ""),
        },
        "citations": citations,
        "artifacts": {
            "project_id": project_id,
            "chat_id": str(chat.get("id") or ""),
            "workflow_source": workflow_resolution["workflow_source"],
            "project_workflow_id": workflow_resolution.get("project_workflow_id"),
        },
    }

    workflow_input = {
        "task": str(message or ""),
        "messages": workflow_messages,
        "context": workflow_context,
        "model": model_id,
        "tool_ids": resolved_tool_ids,
        "tool_defs": selected_tool_defs,
    }

    return {
        "workflow_id": workflow_resolution["workflow_id"],
        "workflow_source": workflow_resolution["workflow_source"],
        "project_workflow_id": workflow_resolution.get("project_workflow_id"),
        "workflow_defs_override": workflow_resolution["workflow_defs_override"],
        "project_tool_overrides": workflow_resolution["project_tool_overrides"],
        "workflow_input": workflow_input,
        "initial_state": initial_state,
        "resolved_tool_ids": resolved_tool_ids,
        "artifact_ids": normalized_artifact_ids,
        "request_messages": request_messages,
        "request_context": request_context,
        "ground_truth": ground_truth_text,
        "retrieval_input": retrieval_input,
        "retrieval_output": retrieval_output,
        "retrieval_status": retrieval_status,
    }


def _resolve_workflow_override(
    *,
    project_id: str,
    requested_workflow_id: str,
) -> Dict[str, Any]:
    project_tool_overrides = _project_tool_overrides(project_id)
    project_workflow = get_project_workflow_payload(project_id)
    if project_workflow is not None:
        if not bool(project_workflow.get("valid", False)):
            raise ValueError("Project workflow failed validation and cannot be executed.")
        workflow_def = workflow_service.compile_draft_to_runtime_defs(
            workflow_spec=dict(project_workflow.get("workflow_spec") or {})
        ).get("workflow_def")
        if not isinstance(workflow_def, dict):
            raise ValueError("Project workflow could not be compiled.")
        return {
            "workflow_id": str(workflow_def.get("id") or project_runtime_workflow_id(project_id)),
            "workflow_source": "project",
            "project_workflow_id": str(project_workflow.get("id") or ""),
            "workflow_defs_override": [workflow_def],
            "project_tool_overrides": project_tool_overrides,
        }

    available_workflows = {
        str(item.get("id") or ""): dict(item)
        for item in graph.list_workflows()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    fallback_workflow_id = str(requested_workflow_id or "").strip() or "complex.v1"
    selected_workflow = available_workflows.get(fallback_workflow_id)
    if not isinstance(selected_workflow, dict):
        raise ValueError(f"Unknown workflow_id: {fallback_workflow_id}")
    if str(selected_workflow.get("status") or "").lower() == "disabled":
        raise ValueError(f"Workflow '{fallback_workflow_id}' is disabled.")
    return {
        "workflow_id": fallback_workflow_id,
        "workflow_source": "global",
        "project_workflow_id": None,
        "workflow_defs_override": [],
        "project_tool_overrides": project_tool_overrides,
    }


def _project_tool_overrides(project_id: str) -> List[dict]:
    available_tools = {
        str(item.get("id") or ""): dict(item)
        for item in graph.list_tools()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    overrides: List[dict] = []
    for binding in db.list_project_tools(project_id):
        if not isinstance(binding, dict):
            continue
        tool_name = str(binding.get("tool_name") or "").strip()
        if not tool_name:
            continue
        base_tool = available_tools.get(tool_name)
        if not isinstance(base_tool, dict):
            continue
        override = dict(base_tool)
        override["enabled"] = bool(binding.get("enabled", override.get("enabled", False)))
        override["version"] = int(binding.get("version") or override.get("version") or 1)
        overrides.append(override)
    return overrides


def _resolve_requested_tools(
    *,
    requested_tool_ids: Sequence[str],
    tool_catalog: Dict[str, Dict[str, Any]],
    normalized_artifact_ids: Sequence[str],
    vision_model_id: Optional[str],
    workflow_id: str,
) -> Tuple[List[str], List[dict]]:
    resolved_tool_ids = [
        str(item).strip() for item in requested_tool_ids if str(item).strip()
    ]
    if not normalized_artifact_ids or not str(vision_model_id or "").strip():
        resolved_tool_ids = [tool_id for tool_id in resolved_tool_ids if tool_id != "vision.analyze"]
    if (
        normalized_artifact_ids
        and str(vision_model_id or "").strip()
        and workflow_id in {"moderate.v1", "complex.v1"}
        and "vision.analyze" not in resolved_tool_ids
    ):
        resolved_tool_ids.insert(0, "vision.analyze")

    deduped_ids: List[str] = []
    selected_defs: List[dict] = []
    seen: set[str] = set()
    for tool_id in resolved_tool_ids:
        if tool_id in seen:
            continue
        seen.add(tool_id)
        tool = tool_catalog.get(tool_id)
        if not isinstance(tool, dict):
            continue
        if not bool(tool.get("enabled", False)):
            continue
        deduped_ids.append(tool_id)
        selected_defs.append(dict(tool))
    return deduped_ids, selected_defs


def _record_project_context_injection(
    *,
    recorder: graph.StepRecorder,
    run_id: str,
    project: Dict[str, Any],
    chat: Dict[str, Any],
    message: str,
    normalized_messages: Sequence[Dict[str, Any]],
    ground_truth: str,
) -> None:
    started_at = graph._to_iso(graph._now_utc())
    recorder.emit(
        graph.StepEvent(
            name="node.project_context_injection",
            status="ok",
            phase="start",
            label="project context injection",
            started_at=started_at,
            ended_at=started_at,
            input={
                "project_id": str(project.get("id") or ""),
                "chat_id": str(chat.get("id") or ""),
                "message": str(message or ""),
            },
            output={},
        )
    )
    ended_at = graph._to_iso(graph._now_utc())
    recorder.emit(
        graph.StepEvent(
            name="node.project_context_injection",
            status="ok",
            phase="end",
            label="project context injection",
            started_at=started_at,
            ended_at=ended_at,
            input={
                "project_id": str(project.get("id") or ""),
                "chat_id": str(chat.get("id") or ""),
            },
            output={
                "project_ground_truth": ground_truth,
                "project_messages": list(normalized_messages),
                "project_message_count": len(list(normalized_messages)),
            },
        )
    )


def _record_project_retrieval(
    *,
    recorder: graph.StepRecorder,
    retrieval_input: Dict[str, Any],
    retrieval_output: Dict[str, Any],
    status: str,
) -> None:
    started_at = graph._to_iso(graph._now_utc())
    recorder.emit(
        graph.StepEvent(
            name="tool.rag.retrieve",
            status="ok",
            phase="start",
            label="tool rag.retrieve",
            started_at=started_at,
            ended_at=started_at,
            input={"tool_id": "rag.retrieve", "input": dict(retrieval_input)},
            output={},
        )
    )
    ended_at = graph._to_iso(graph._now_utc())
    recorder.emit(
        graph.StepEvent(
            name="tool.rag.retrieve",
            status=status,
            phase="end",
            label="tool rag.retrieve",
            started_at=started_at,
            ended_at=ended_at,
            input={"tool_id": "rag.retrieve", "input": dict(retrieval_input)},
            output={"tool_id": "rag.retrieve", "output": dict(retrieval_output)},
        )
    )


def _run_project_retrieval(
    *,
    tool_defs: Sequence[Dict[str, Any]],
    tool_input: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    try:
        runtime_registry = graph._runtime_registry(tool_defs=list(tool_defs or []))
        output = runtime_registry.invoke_tool(
            tool_id="rag.retrieve",
            tool_input=dict(tool_input or {}),
            context=dict(context or {}),
        )
        if not isinstance(output, dict):
            output = {"ok": False, "error": {"message": "Retrieval returned invalid output."}}
    except Exception as exc:
        output = {
            "ok": False,
            "error": {
                "message": str(exc),
                "kind": "runtime",
            },
        }
    status = "ok" if bool(output.get("ok", False)) else "error"
    return output, status


def _normalize_messages(
    *,
    message: str,
    raw_messages: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in raw_messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        normalized.append({"role": role, "content": content})
    if not normalized:
        normalized.append({"role": "user", "content": str(message or "")})
        return normalized
    latest_user = next(
        (
            str(item.get("content") or "")
            for item in reversed(normalized)
            if str(item.get("role") or "").lower() == "user"
        ),
        "",
    )
    if str(latest_user or "").strip() != str(message or "").strip():
        normalized.append({"role": "user", "content": str(message or "")})
    return normalized


def _render_history_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().title()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _render_retrieval_text(results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "unknown").strip() or "unknown"
        chunk_id = item.get("chunk_id")
        text = str(item.get("text") or "").strip()
        header = f"[{index}] {doc_id}"
        if chunk_id not in (None, ""):
            header += f"#{chunk_id}"
        lines.append(header)
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _build_project_system_message(
    *,
    project: Dict[str, Any],
    ground_truth: str,
    history_text: str,
    retrieval_text: str,
    latest_prompt: str,
) -> str:
    sections = [
        f"Project: {str(project.get('name') or '').strip()}",
        f"Description: {str(project.get('description') or '').strip()}",
        "Ground Truth:",
        ground_truth.strip() or "(empty)",
    ]
    if retrieval_text.strip():
        sections.extend(["Retrieved Project Context:", retrieval_text.strip()])
    if history_text.strip():
        sections.extend(["Conversation History:", history_text.strip()])
    sections.extend(["Latest User Prompt:", str(latest_prompt or "").strip()])
    return "\n\n".join(sections).strip()


def _format_citations(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        doc_id = str(item.get("doc_id") or metadata.get("doc_id") or "").strip()
        if not doc_id:
            continue
        citation: Dict[str, Any] = {
            "doc_id": doc_id,
            "score": float(item.get("score") or 0.0),
        }
        chunk_id = item.get("chunk_id") or metadata.get("chunk_id") or metadata.get("chunk_index")
        if chunk_id not in (None, ""):
            citation["chunk_id"] = chunk_id
        source = str(
            metadata.get("source")
            or metadata.get("path")
            or metadata.get("file_path")
            or ""
        ).strip()
        if source:
            citation["source"] = source
        citations.append(citation)
    return citations


def _resolve_project_embedding_model(project_id: str) -> str:
    documents = db.list_project_documents(project_id)
    for item in documents:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("embedding_model") or "").strip()
        if model_id:
            return model_id
    return str(os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)).strip()


def _normalize_project_workflow_spec(
    *,
    project_id: str,
    workflow_spec: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(workflow_spec or {})
    normalized["workflow_id"] = project_runtime_workflow_id(project_id)
    if not str(normalized.get("name") or "").strip():
        normalized["name"] = f"{project_id} Project Workflow"
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    metadata.setdefault("enabled", True)
    metadata["project_id"] = project_id
    normalized["metadata"] = metadata
    return normalized


def _extract_document_text(*, filename: str, stored_path: Path) -> str:
    suffix = stored_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(str(stored_path))

    raw_bytes = stored_path.read_bytes()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="ignore")


def _runtime_qdrant_url() -> Optional[str]:
    try:
        return qdrant_client_service.get_qdrant_url()
    except Exception:
        return None


def _sha256_for(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_project(project_id: str) -> dict:
    project = db.get_project(project_id)
    if project is None:
        raise ValueError("Project not found.")
    return project
