from __future__ import annotations

from typing import Any, Dict, List, Optional

from app import db

try:
    from saturday_agent.tools.registry import ToolRegistry as RuntimeToolRegistry
except ModuleNotFoundError:  # pragma: no cover - import path parity with graph.py
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    agent_src = repo_root / "apps/agent/src"
    if str(agent_src) not in sys.path:
        sys.path.append(str(agent_src))
    from saturday_agent.tools.registry import ToolRegistry as RuntimeToolRegistry


def list_tool_definitions(*, include_deleted: bool = False) -> List[Dict[str, Any]]:
    return db.list_tools(include_deleted=include_deleted)


def get_tool_definition(
    tool_id: str,
    *,
    include_deleted: bool = False,
) -> Optional[Dict[str, Any]]:
    return db.get_tool(tool_id, include_deleted=include_deleted)


def build_runtime_tool_registry(
    *,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
    include_deleted: bool = False,
) -> RuntimeToolRegistry:
    merged_defs = list_tool_definitions(include_deleted=include_deleted)
    if tool_defs:
        for item in tool_defs:
            if not isinstance(item, dict):
                continue
            tool_id = str(item.get("id") or item.get("tool_id") or "").strip()
            if not tool_id:
                continue
            merged_defs = [
                dict(defn)
                for defn in merged_defs
                if str(defn.get("id") or defn.get("tool_id") or "").strip() != tool_id
            ]
            merged_defs.append(dict(item))
    return RuntimeToolRegistry(dynamic_tools=merged_defs)


def list_tools(*, include_deleted: bool = False) -> List[Dict[str, Any]]:
    return build_runtime_tool_registry(include_deleted=include_deleted).list_tools()


def get_tool(tool_id: str, *, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    registry = build_runtime_tool_registry(include_deleted=include_deleted)
    return registry.get_tool(tool_id)


def invoke_tool(
    *,
    tool_id: str,
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tool_defs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    registry = build_runtime_tool_registry(tool_defs=tool_defs)
    return registry.invoke_tool(
        tool_id=tool_id,
        tool_input=tool_input,
        context=context,
    )
