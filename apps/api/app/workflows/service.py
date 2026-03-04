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
    normalized = dict(result.workflow_spec or {})
    if normalized.get("version") is None:
        normalized["version"] = 1
    return {
        "valid": result.valid,
        "workflow_spec": normalized,
        "compiled": result.compiled,
        "diagnostics": [item.model_dump() for item in result.diagnostics],
    }


def compile_workflow_request(
    *,
    task: str,
    context: Mapping[str, Any] | None = None,
    workflow_type: str | None = None,
) -> Dict[str, Any]:
    result = graph.compile_workflow(
        task=task,
        context=dict(context or {}),
        workflow_type=workflow_type,
    )
    workflow_id = str(result.get("workflow_id") or "")
    resolved_type = str(result.get("workflow_type") or "")
    runtime_graph = (
        dict(result.get("graph") or {})
        if isinstance(result.get("graph"), dict)
        else {"nodes": [], "edges": []}
    )
    workflow_spec = _runtime_graph_to_workflow_spec(
        workflow_id=workflow_id,
        workflow_type=resolved_type,
        runtime_graph=runtime_graph,
    )
    return {
        "valid": True,
        "workflow_id": workflow_id,
        "workflow_type": resolved_type,
        "workflow_spec": workflow_spec,
        "compiled": {
            "runtime_graph": runtime_graph,
            "entry_nodes": [workflow_spec.get("entry_node")] if workflow_spec.get("entry_node") else [],
        },
        "diagnostics": [],
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


def normalize_run_selector(
    *,
    workflow_version_id: str | None = None,
    workflow_id: str | None = None,
    draft_spec: Mapping[str, Any] | None = None,
    workflow_spec: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_workflow_version_id = str(workflow_version_id or "").strip() or None
    normalized_workflow_id = str(workflow_id or "").strip() or None
    normalized_draft_spec = (
        dict(draft_spec or {})
        if isinstance(draft_spec, Mapping)
        else dict(workflow_spec or {})
        if isinstance(workflow_spec, Mapping)
        else None
    )
    return {
        "workflow_version_id": normalized_workflow_version_id,
        "workflow_id": normalized_workflow_id,
        "draft_spec": normalized_draft_spec,
    }


def _runtime_graph_to_workflow_spec(
    *,
    workflow_id: str,
    workflow_type: str,
    runtime_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_nodes = runtime_graph.get("nodes") if isinstance(runtime_graph.get("nodes"), list) else []
    raw_edges = runtime_graph.get("edges") if isinstance(runtime_graph.get("edges"), list) else []

    nodes: List[Dict[str, Any]] = []
    node_ids: set[str] = set()
    incoming: Dict[str, int] = {}
    tool_refs: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            continue
        node_id = str(raw.get("id") or "").strip()
        node_type = str(raw.get("type") or "").strip().lower()
        if not node_id or node_type in {"start", "end"}:
            continue
        canonical_type = _canonical_node_type(node_id=node_id, node_type=node_type)
        config = dict(raw.get("config") or {})
        tool_name = str(config.get("tool_name") or config.get("tool_id") or "").strip()
        if tool_name:
            tool_refs.add(tool_name)
        node_ids.add(node_id)
        incoming.setdefault(node_id, 0)
        nodes.append(
            {
                "id": node_id,
                "type": canonical_type,
                "label": node_id.replace("_", " ").title(),
                "reads": list(raw.get("reads") or []),
                "writes": list(raw.get("writes") or []),
                "config": config,
            }
        )

    edges: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_edges, start=1):
        if not isinstance(raw, Mapping):
            continue
        from_node = str(raw.get("from") or raw.get("from_node") or "").strip()
        to_node = str(raw.get("to") or "").strip()
        if from_node == "start" or from_node == "START":
            from_node = ""
        if to_node in {"end", "END"} or not to_node:
            continue
        if from_node in {"", "start", "START"}:
            incoming[to_node] = int(incoming.get(to_node, 0))
            continue
        if from_node not in node_ids or to_node not in node_ids:
            continue
        incoming[to_node] = int(incoming.get(to_node, 0)) + 1
        edges.append(
            {
                "id": f"edge_{index}",
                "from": from_node,
                "to": to_node,
                "label": str(raw.get("condition") or raw.get("label") or "always"),
            }
        )

    entry_node = ""
    for node_id in sorted(node_ids):
        if int(incoming.get(node_id, 0)) == 0:
            entry_node = node_id
            break
    terminal_nodes = sorted(
        [
            str(item.get("id") or "")
            for item in nodes
            if str(item.get("type") or "") == "finalize"
        ]
    )
    return {
        "workflow_id": workflow_id or f"{workflow_type or 'workflow'}.compiled",
        "version": 1,
        "name": workflow_id or workflow_type or "Compiled Workflow",
        "description": f"Compiled {workflow_type or 'workflow'} workflow.",
        "allow_cycles": False,
        "state_schema": [],
        "nodes": nodes,
        "edges": edges,
        "entry_node": entry_node or None,
        "terminal_nodes": terminal_nodes,
        "tool_refs": sorted(tool_refs),
        "metadata": {
            "source": "builtin",
            "compiled_from_runtime_graph": True,
            "workflow_type": workflow_type,
        },
    }


def _canonical_node_type(*, node_id: str, node_type: str) -> str:
    normalized_id = str(node_id or "").strip().lower()
    normalized_type = str(node_type or "").strip().lower()
    if normalized_type in {"tool", "verify", "finalize", "conditional"}:
        return normalized_type
    if normalized_id.startswith("verify"):
        return "verify"
    if normalized_id.startswith("final"):
        return "finalize"
    if normalized_id.endswith("tools") or "tool" in normalized_id:
        return "tool"
    return "llm"
