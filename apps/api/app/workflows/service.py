from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from app import db, graph
from app.workflows.validation import validate_and_compile_workflow_spec


def compile_workflow_spec(*, workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
    tool_registry = graph.list_tools()
    result = validate_and_compile_workflow_spec(
        workflow_spec=workflow_spec,
        tool_registry=tool_registry,
    )
    return {
        "valid": result.valid,
        "workflow_spec": result.workflow_spec,
        "compiled": result.compiled,
        "diagnostics": [item.model_dump() for item in result.diagnostics],
    }


def create_workflow_version(
    *,
    workflow_id: str,
    workflow_spec: Mapping[str, Any],
    created_by: str,
) -> Dict[str, Any]:
    compile_payload = compile_workflow_spec(workflow_spec=workflow_spec)
    diagnostics = list(compile_payload.get("diagnostics") or [])
    has_errors = any(str(item.get("severity") or "") == "error" for item in diagnostics)
    if has_errors:
        raise ValueError("Workflow has validation errors and cannot be versioned.")

    normalized = dict(compile_payload.get("workflow_spec") or {})
    name = str(normalized.get("name") or workflow_id)
    description = str(normalized.get("description") or "")
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    enabled = bool(metadata.get("enabled", True))

    created = db.create_workflow_version(
        workflow_id=workflow_id,
        name=name,
        description=description,
        enabled=enabled,
        spec=normalized,
        compiled=dict(compile_payload.get("compiled") or {}),
        created_by=str(created_by or "builder"),
    )
    created["diagnostics"] = diagnostics
    return created


def get_latest_workflow_payload(workflow_id: str) -> Dict[str, Any]:
    workflow = db.get_workflow(workflow_id)
    if workflow is None:
        raise ValueError("Workflow not found.")
    versions = db.list_workflow_versions(workflow_id)
    latest = versions[0] if versions else None
    return {
        "workflow_id": str(workflow.get("id") or workflow_id),
        "name": str(workflow.get("name") or workflow_id),
        "description": str(workflow.get("description") or ""),
        "enabled": bool(workflow.get("enabled", True)),
        "latest_version": latest,
        "versions": [
            {
                "version_id": str(item.get("version_id") or ""),
                "workflow_id": str(item.get("workflow_id") or workflow_id),
                "version_num": int(item.get("version_num") or 0),
                "created_at": str(item.get("created_at") or ""),
                "created_by": str(item.get("created_by") or ""),
            }
            for item in versions
        ],
    }


def get_workflow_versions_payload(workflow_id: str) -> List[Dict[str, Any]]:
    return db.list_workflow_versions(workflow_id)


def compile_draft_to_runtime_defs(workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
    payload = compile_workflow_spec(workflow_spec=workflow_spec)
    normalized = dict(payload.get("workflow_spec") or {})
    compiled = dict(payload.get("compiled") or {})
    runtime_graph = compiled.get("runtime_graph")
    if not isinstance(runtime_graph, dict):
        runtime_graph = {"nodes": [], "edges": []}

    workflow_id = str(normalized.get("workflow_id") or "draft.workflow")
    workflow_name = str(normalized.get("name") or workflow_id)
    description = str(normalized.get("description") or "")

    workflow_def = {
        "id": workflow_id,
        "name": workflow_name,
        "title": workflow_name,
        "description": description,
        "type": "custom",
        "source": "custom",
        "enabled": True,
        "graph": runtime_graph,
        "spec": normalized,
    }

    return {
        "compile": payload,
        "workflow_def": workflow_def,
    }
