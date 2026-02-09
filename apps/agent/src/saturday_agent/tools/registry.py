from __future__ import annotations

from typing import Any, Callable, Dict, List

from saturday_agent.tools.search_tavily import (
    SEARCH_INPUT_SCHEMA,
    SEARCH_OUTPUT_SCHEMA,
    search_web_tavily,
)

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    """Stub registry for future dynamic tool management."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, ToolHandler] = {}

        self.register_tool(
            tool_id="filesystem.read",
            name="File System Read",
            description="Read, write, and manage files on the local machine.",
            kind="local",
            enabled=True,
        )
        self.register_tool(
            tool_id="workflow.inspect",
            name="Workflow Inspector",
            description="Inspect workflow nodes, edges, and run traces.",
            kind="local",
            enabled=True,
        )
        self.register_tool(
            tool_id="search.web",
            name="Web Search",
            description="Search the web using Tavily",
            kind="external",
            enabled=True,
            input_schema=SEARCH_INPUT_SCHEMA,
            output_schema=SEARCH_OUTPUT_SCHEMA,
            handler=search_web_tavily,
        )

    def register_tool(
        self,
        *,
        tool_id: str,
        name: str,
        description: str,
        kind: str = "local",
        enabled: bool = True,
        input_schema: Dict[str, Any] | None = None,
        output_schema: Dict[str, Any] | None = None,
        handler: ToolHandler | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self._tools[tool_id] = {
            "id": tool_id,
            "name": name,
            "kind": kind,
            "description": description,
            "enabled": enabled,
            "input_schema": dict(input_schema or {}),
            "output_schema": dict(output_schema or {}),
            "metadata": dict(metadata or {}),
        }
        if handler is not None:
            self._handlers[tool_id] = handler

    def list_tools(self) -> List[Dict[str, Any]]:
        ordered = sorted(self._tools.values(), key=lambda item: str(item.get("name", "")))
        return [
            {
                "id": str(tool.get("id", "")),
                "name": str(tool.get("name", "")),
                "kind": str(tool.get("kind", "local")),
                "description": str(tool.get("description", "")),
                "enabled": bool(tool.get("enabled", False)),
                "input_schema": dict(tool.get("input_schema") or {}),
                "output_schema": dict(tool.get("output_schema") or {}),
            }
            for tool in ordered
        ]

    def get_tool(self, tool_id: str) -> Dict[str, Any] | None:
        tool = self._tools.get(tool_id)
        if tool is None:
            return None
        return dict(tool)

    def invoke_tool(self, tool_id: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get_tool(tool_id)
        if tool is None:
            raise ValueError(f"Unknown tool_id: {tool_id}")
        if not bool(tool.get("enabled", False)):
            raise ValueError(f"Tool '{tool_id}' is disabled.")
        if not isinstance(tool_input, dict):
            raise ValueError("Tool input must be an object.")

        handler = self._handlers.get(tool_id)
        if handler is None:
            raise ValueError(f"Tool '{tool_id}' is not executable yet.")
        return handler(dict(tool_input))

    def set_enabled(self, tool_id: str, enabled: bool) -> bool:
        tool = self._tools.get(tool_id)
        if not tool:
            return False
        tool["enabled"] = bool(enabled)
        return True

    def decide_tools(self, *, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        _ = task
        selected = context.get("selected_tool_ids")
        if not isinstance(selected, list):
            return []

        selected_ids = {str(item).strip() for item in selected if str(item).strip()}
        if not selected_ids:
            return []

        planned_calls: List[Dict[str, Any]] = []
        tool_inputs = context.get("tool_inputs")
        tool_inputs_map = tool_inputs if isinstance(tool_inputs, dict) else {}

        for tool in self.list_tools():
            tool_id = str(tool.get("id", ""))
            if tool_id in selected_ids and bool(tool.get("enabled", False)):
                call_input = {}
                raw_input = tool_inputs_map.get(tool_id)
                if isinstance(raw_input, dict):
                    call_input = dict(raw_input)
                if tool_id == "search.web" and not str(call_input.get("query", "")).strip():
                    call_input["query"] = task

                planned_calls.append(
                    {
                        "tool_id": tool_id,
                        "name": str(tool.get("name", tool_id)),
                        "kind": str(tool.get("kind", "local")),
                        "status": "planned",
                        "input": call_input,
                    }
                )
        return planned_calls

    # TODO: Add dynamic tool config persistence for future Tools page.
