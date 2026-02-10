from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

import httpx

from saturday_agent.llms.ollama_chat import extract_assistant_text, ollama_chat
from saturday_agent.state.models import ToolDefinition
from saturday_agent.tools.search_tavily import (
    SEARCH_INPUT_SCHEMA,
    SEARCH_OUTPUT_SCHEMA,
    search_web_tavily,
)
from saturday_agent.tools.vision_ollama import (
    TOOL_DESCRIPTION as VISION_TOOL_DESCRIPTION,
    TOOL_NAME as VISION_TOOL_NAME,
    VISION_ANALYZE_INPUT_SCHEMA,
    VISION_ANALYZE_OUTPUT_SCHEMA,
    analyze_image_ollama,
)

ToolHandler = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]]

DEFAULT_HTTP_TIMEOUT_MS = 8000
DEFAULT_PYTHON_TIMEOUT_MS = 5000
DEFAULT_PROMPT_TIMEOUT_MS = 30000
DEFAULT_PROMPT_TEMPERATURE = 0.2
DEFAULT_ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "typing",
}

_TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


class ToolRegistry:
    """Registry for built-in and user-defined tools."""

    def __init__(self, *, dynamic_tools: Optional[List[Dict[str, Any]]] = None) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, ToolHandler] = {}

        self.register_tool(
            tool_id="filesystem.read",
            name="File System Read",
            description="Read, write, and manage files on the local machine.",
            kind="local",
            type="builtin",
            enabled=True,
            metadata={"source": "builtin"},
        )
        self.register_tool(
            tool_id="workflow.inspect",
            name="Workflow Inspector",
            description="Inspect workflow nodes, edges, and run traces.",
            kind="local",
            type="builtin",
            enabled=True,
            metadata={"source": "builtin"},
        )
        self.register_tool(
            tool_id="search.web",
            name="Web Search",
            description="Search the web using Tavily",
            kind="external",
            type="builtin",
            enabled=True,
            input_schema=SEARCH_INPUT_SCHEMA,
            output_schema=SEARCH_OUTPUT_SCHEMA,
            handler=self._wrap_legacy_handler(search_web_tavily),
            metadata={"source": "builtin"},
        )
        self.register_tool(
            tool_id="vision.analyze",
            name=VISION_TOOL_NAME,
            description=VISION_TOOL_DESCRIPTION,
            kind="local",
            type="builtin",
            enabled=True,
            input_schema=VISION_ANALYZE_INPUT_SCHEMA,
            output_schema=VISION_ANALYZE_OUTPUT_SCHEMA,
            handler=self._wrap_legacy_handler(analyze_image_ollama),
            metadata={"source": "builtin"},
        )
        self.register_dynamic_tools(dynamic_tools or [])

    def register_tool(
        self,
        *,
        tool_id: str,
        name: str,
        description: str,
        kind: str = "local",
        type: str = "builtin",
        enabled: bool = True,
        config: Dict[str, Any] | None = None,
        input_schema: Dict[str, Any] | None = None,
        output_schema: Dict[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        handler: ToolHandler | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self._tools[tool_id] = {
            "id": tool_id,
            "name": name,
            "kind": kind,
            "type": type,
            "description": description,
            "enabled": enabled,
            "config": dict(config or {}),
            "input_schema": dict(input_schema or {}),
            "output_schema": dict(output_schema or {}),
            "created_at": str(created_at or ""),
            "updated_at": str(updated_at or ""),
            "metadata": dict(metadata or {}),
        }
        if handler is not None:
            self._handlers[tool_id] = handler

    def register_dynamic_tools(
        self,
        tool_defs: List[Dict[str, Any]] | List[ToolDefinition],
    ) -> None:
        for raw_tool in tool_defs:
            if not isinstance(raw_tool, dict):
                continue

            tool_id = str(raw_tool.get("id") or "").strip()
            if not tool_id:
                continue

            tool_name = str(raw_tool.get("name") or tool_id).strip() or tool_id
            tool_type = str(raw_tool.get("type") or "http").strip().lower()
            if tool_type not in {"http", "python", "prompt"}:
                continue

            config = dict(raw_tool.get("config") or {})
            handler: ToolHandler
            if tool_type == "http":
                handler = self._build_http_handler(tool_id=tool_id, config=config)
            elif tool_type == "python":
                handler = self._build_python_handler(tool_id=tool_id, config=config)
            else:
                handler = self._build_prompt_handler(tool_id=tool_id, config=config)

            self.register_tool(
                tool_id=tool_id,
                name=tool_name,
                description=str(raw_tool.get("description") or ""),
                kind=str(raw_tool.get("kind") or "external"),
                type=tool_type,
                enabled=bool(raw_tool.get("enabled", True)),
                config=config,
                input_schema=(
                    dict(raw_tool.get("input_schema"))
                    if isinstance(raw_tool.get("input_schema"), dict)
                    else None
                ),
                output_schema=(
                    dict(raw_tool.get("output_schema"))
                    if isinstance(raw_tool.get("output_schema"), dict)
                    else None
                ),
                created_at=str(raw_tool.get("created_at") or ""),
                updated_at=str(raw_tool.get("updated_at") or ""),
                handler=handler,
                metadata={"source": str(raw_tool.get("source") or "custom")},
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        ordered = sorted(self._tools.values(), key=lambda item: str(item.get("name", "")))
        return [
            {
                "id": str(tool.get("id", "")),
                "name": str(tool.get("name", "")),
                "kind": str(tool.get("kind", "local")),
                "type": str(tool.get("type", "builtin")),
                "description": str(tool.get("description", "")),
                "enabled": bool(tool.get("enabled", False)),
                "config": dict(tool.get("config") or {}),
                "input_schema": dict(tool.get("input_schema") or {}),
                "output_schema": dict(tool.get("output_schema") or {}),
                "created_at": str(tool.get("created_at") or ""),
                "updated_at": str(tool.get("updated_at") or ""),
                "source": str((tool.get("metadata") or {}).get("source") or "builtin"),
            }
            for tool in ordered
        ]

    def get_tool(self, tool_id: str) -> Dict[str, Any] | None:
        tool = self._tools.get(tool_id)
        if tool is None:
            return None
        record = dict(tool)
        record["config"] = dict(tool.get("config") or {})
        record["input_schema"] = dict(tool.get("input_schema") or {})
        record["output_schema"] = dict(tool.get("output_schema") or {})
        record["metadata"] = dict(tool.get("metadata") or {})
        return record

    def invoke(
        self,
        tool_id: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

        try:
            response = handler(dict(tool_input), dict(context or {}))
        except Exception as exc:
            return self._error_envelope(
                tool=tool,
                kind="runtime",
                message=str(exc),
                meta={"tool_id": tool_id},
            )

        if self._is_standard_envelope(response):
            return response

        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            err = response.get("error") or {}
            return self._error_envelope(
                tool=tool,
                kind=str(err.get("type") or "runtime"),
                message=str(err.get("message") or "Tool execution failed."),
                meta={"tool_id": tool_id, "raw": response},
            )

        return self._success_envelope(tool=tool, data=response, meta={"tool_id": tool_id})

    def invoke_tool(
        self,
        tool_id: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.invoke(tool_id=tool_id, tool_input=tool_input, context=context)

    def set_enabled(self, tool_id: str, enabled: bool) -> bool:
        tool = self._tools.get(tool_id)
        if not tool:
            return False
        tool["enabled"] = bool(enabled)
        return True

    def decide_tools(self, *, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        normalized_task = str(task or "").strip()
        selected = context.get("selected_tool_ids")
        if not isinstance(selected, list):
            selected = []

        selected_ids = {str(item).strip() for item in selected if str(item).strip()}
        planned_calls: List[Dict[str, Any]] = []
        tool_inputs = context.get("tool_inputs")
        tool_inputs_map = tool_inputs if isinstance(tool_inputs, dict) else {}

        artifact_ids_raw = context.get("artifact_ids")
        artifact_ids = (
            [str(item).strip() for item in artifact_ids_raw if str(item).strip()]
            if isinstance(artifact_ids_raw, list)
            else []
        )
        vision_model_id = str(context.get("vision_model_id") or "").strip()
        vision_prompt = str(context.get("vision_prompt") or normalized_task or "").strip()
        vision_detail = str(context.get("vision_detail") or "").strip().lower()
        per_vision_input = tool_inputs_map.get("vision.analyze")
        per_vision_input_map = (
            dict(per_vision_input) if isinstance(per_vision_input, dict) else {}
        )
        if not vision_prompt:
            vision_prompt = "Analyze the attached image."

        vision_tool = self.get_tool("vision.analyze")
        vision_enabled = bool(vision_tool and vision_tool.get("enabled"))
        if vision_enabled and artifact_ids and vision_model_id:
            for artifact_id in artifact_ids:
                call_input = dict(per_vision_input_map)
                call_input["artifact_id"] = artifact_id
                call_input["prompt"] = str(call_input.get("prompt") or vision_prompt)
                call_input["vision_model_id"] = str(
                    call_input.get("vision_model_id") or vision_model_id
                )
                if vision_detail in {"low", "high"} and "detail" not in call_input:
                    call_input["detail"] = vision_detail

                planned_calls.append(
                    {
                        "tool_id": "vision.analyze",
                        "name": str(vision_tool.get("name") or "Vision Analyze"),
                        "kind": str(vision_tool.get("kind") or "local"),
                        "status": "planned",
                        "input": call_input,
                    }
                )

        for tool in self.list_tools():
            tool_id = str(tool.get("id", ""))
            if tool_id == "vision.analyze":
                continue
            if tool_id in selected_ids and bool(tool.get("enabled", False)):
                call_input = {}
                raw_input = tool_inputs_map.get(tool_id)
                if isinstance(raw_input, dict):
                    call_input = dict(raw_input)
                if not str(call_input.get("query", "")).strip() and normalized_task:
                    call_input["query"] = normalized_task

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

    def _build_http_handler(self, *, tool_id: str, config: Dict[str, Any]) -> ToolHandler:
        normalized_config = self._normalize_http_config(config)

        def _handler(
            tool_input: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = context
            method = str(normalized_config.get("method") or "POST").upper()
            url = str(normalized_config.get("url") or "").strip()
            headers = dict(normalized_config.get("headers") or {})
            timeout_ms = int(normalized_config.get("timeout_ms") or DEFAULT_HTTP_TIMEOUT_MS)
            timeout_seconds = max(timeout_ms, 1) / 1000.0

            request_kwargs: Dict[str, Any] = {
                "headers": headers,
            }
            if method == "GET":
                request_kwargs["params"] = {
                    str(key): value
                    for key, value in (tool_input or {}).items()
                    if value is not None
                }
            else:
                request_kwargs["json"] = dict(tool_input or {})

            try:
                with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                    response = client.request(method, url, **request_kwargs)
            except httpx.TimeoutException as exc:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="timeout",
                    message=str(exc),
                    meta={"tool_id": tool_id, "url": url, "method": method},
                )
            except httpx.RequestError as exc:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="network",
                    message=str(exc),
                    meta={"tool_id": tool_id, "url": url, "method": method},
                )

            try:
                payload_data: Any = response.json()
            except ValueError:
                payload_data = response.text

            response_payload = {
                "status": int(response.status_code),
                "ok": response.status_code < 400,
                "data": payload_data,
                "headers": dict(response.headers),
            }

            if response.status_code >= 400:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="http",
                    message=f"HTTP {response.status_code} from {tool_id}",
                    meta={"tool_id": tool_id, "url": url, "method": method, **response_payload},
                )

            return self._success_envelope(
                tool=self.get_tool(tool_id),
                data=response_payload,
                meta={"tool_id": tool_id, "url": url, "method": method},
            )

        return _handler

    def _build_python_handler(self, *, tool_id: str, config: Dict[str, Any]) -> ToolHandler:
        normalized = self._normalize_python_config(config)
        code = str(normalized.get("code") or "")
        allowed_imports = set(normalized.get("allowed_imports") or DEFAULT_ALLOWED_IMPORTS)
        timeout_ms = int(normalized.get("timeout_ms") or DEFAULT_PYTHON_TIMEOUT_MS)

        parse_error, blocked_imports = self._validate_python_code(
            code=code,
            allowed_imports=allowed_imports,
        )

        def _handler(
            tool_input: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            if parse_error:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="syntax",
                    message=parse_error,
                    meta={"tool_id": tool_id},
                )

            if blocked_imports:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="validation",
                    message=(
                        "Blocked imports detected: " + ", ".join(sorted(blocked_imports))
                    ),
                    meta={"tool_id": tool_id},
                )

            payload = {
                "code": code,
                "input": dict(tool_input or {}),
                "context": dict(context or {}),
            }

            runner = (
                "import json,sys,traceback\n"
                "raw = sys.stdin.read() or '{}'\n"
                "payload = json.loads(raw)\n"
                "namespace = {}\n"
                "code = str(payload.get('code') or '')\n"
                "tool_input = payload.get('input') if isinstance(payload.get('input'), dict) else {}\n"
                "ctx = payload.get('context') if isinstance(payload.get('context'), dict) else {}\n"
                "try:\n"
                "    exec(code, namespace)\n"
                "    run_fn = namespace.get('run')\n"
                "    if not callable(run_fn):\n"
                "        raise ValueError('Python tool must define callable run(input, context)')\n"
                "    result = run_fn(tool_input, ctx)\n"
                "    sys.stdout.write(json.dumps({'ok': True, 'result': result}, default=str))\n"
                "except Exception as exc:\n"
                "    sys.stdout.write(json.dumps({'ok': False, 'error': {'message': str(exc), 'traceback': traceback.format_exc()}}, default=str))\n"
                "    sys.exit(1)\n"
            )

            timeout_seconds = max(timeout_ms, 1) / 1000.0
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", runner],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="timeout",
                    message=f"Python tool timed out after {timeout_ms}ms.",
                    meta={"tool_id": tool_id, "timeout_ms": timeout_ms},
                )
            except Exception as exc:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="runtime",
                    message=str(exc),
                    meta={"tool_id": tool_id},
                )

            output_text = (completed.stdout or "").strip()
            stderr_text = (completed.stderr or "").strip()
            parsed: Dict[str, Any]
            try:
                parsed = json.loads(output_text) if output_text else {}
            except json.JSONDecodeError:
                parsed = {}

            if not parsed and stderr_text:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="runtime",
                    message=stderr_text,
                    meta={"tool_id": tool_id},
                )

            if not bool(parsed.get("ok")):
                err = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
                message = str(err.get("message") or stderr_text or "Python tool failed.")
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="runtime",
                    message=message,
                    meta={
                        "tool_id": tool_id,
                        "traceback": err.get("traceback"),
                        "stderr": stderr_text,
                    },
                )

            return self._success_envelope(
                tool=self.get_tool(tool_id),
                data=parsed.get("result"),
                meta={"tool_id": tool_id, "timeout_ms": timeout_ms},
            )

        return _handler

    def _build_prompt_handler(self, *, tool_id: str, config: Dict[str, Any]) -> ToolHandler:
        normalized = self._normalize_prompt_config(config)

        def _handler(
            tool_input: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            context_map = dict(context or {})
            prompt_template = str(normalized.get("prompt_template") or "")
            system_template = str(normalized.get("system_prompt") or "")
            timeout_ms = int(normalized.get("timeout_ms") or DEFAULT_PROMPT_TIMEOUT_MS)
            temperature = float(normalized.get("temperature") or DEFAULT_PROMPT_TEMPERATURE)

            user_prompt = self._render_template(
                prompt_template,
                tool_input=tool_input,
                context=context_map,
            ).strip()
            if not user_prompt:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="validation",
                    message="Prompt template rendered empty output.",
                    meta={"tool_id": tool_id},
                )

            system_prompt = self._render_template(
                system_template,
                tool_input=tool_input,
                context=context_map,
            ).strip()

            model_id = str(
                context_map.get("model_id")
                or context_map.get("model")
                or context_map.get("selected_model_id")
                or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            ).strip()
            base_url = str(
                context_map.get("ollama_base_url")
                or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ).strip()
            timeout_seconds = max(timeout_ms, 1) / 1000.0

            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            try:
                raw = ollama_chat(
                    messages=messages,
                    model=model_id,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    options={"temperature": temperature},
                )
                text = extract_assistant_text(raw).strip()
            except httpx.TimeoutException as exc:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="timeout",
                    message=str(exc),
                    meta={"tool_id": tool_id, "model_id": model_id},
                )
            except Exception as exc:
                return self._error_envelope(
                    tool=self.get_tool(tool_id),
                    kind="llm",
                    message=str(exc),
                    meta={"tool_id": tool_id, "model_id": model_id},
                )

            return self._success_envelope(
                tool=self.get_tool(tool_id),
                data={"text": text, "raw": raw},
                meta={"tool_id": tool_id, "model_id": model_id},
            )

        return _handler

    @staticmethod
    def _wrap_legacy_handler(
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> ToolHandler:
        def _wrapped(
            tool_input: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = context
            return handler(tool_input)

        return _wrapped

    @staticmethod
    def _is_standard_envelope(value: Dict[str, Any]) -> bool:
        if not isinstance(value, dict):
            return False
        if "ok" not in value or "type" not in value:
            return False
        return isinstance(value.get("ok"), bool)

    def _success_envelope(
        self,
        *,
        tool: Optional[Dict[str, Any]],
        data: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tool_type = str((tool or {}).get("type") or "unknown")
        return {
            "ok": True,
            "type": tool_type,
            "data": data,
            "meta": dict(meta or {}),
        }

    def _error_envelope(
        self,
        *,
        tool: Optional[Dict[str, Any]],
        kind: str,
        message: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tool_type = str((tool or {}).get("type") or "unknown")
        return {
            "ok": False,
            "type": tool_type,
            "error": {
                "message": message,
                "kind": kind,
            },
            "meta": dict(meta or {}),
        }

    @staticmethod
    def _normalize_http_config(config: Dict[str, Any]) -> Dict[str, Any]:
        url = str(config.get("url") or "").strip()
        method = str(config.get("method") or "POST").strip().upper()
        if method not in {"GET", "POST"}:
            method = "POST"

        headers: Dict[str, str] = {}
        raw_headers = config.get("headers")
        if isinstance(raw_headers, dict):
            for key, value in raw_headers.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                headers[key_text] = str(value)

        timeout_ms = DEFAULT_HTTP_TIMEOUT_MS
        raw_timeout = config.get("timeout_ms")
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            timeout_ms = int(raw_timeout)

        return {
            "url": url,
            "method": method,
            "headers": headers,
            "timeout_ms": timeout_ms,
        }

    @staticmethod
    def _normalize_python_config(config: Dict[str, Any]) -> Dict[str, Any]:
        code = str(config.get("code") or "")
        raw_timeout = config.get("timeout_ms")
        timeout_ms = DEFAULT_PYTHON_TIMEOUT_MS
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            timeout_ms = int(raw_timeout)

        allowed_imports_raw = config.get("allowed_imports")
        allowed_imports: List[str] = []
        if isinstance(allowed_imports_raw, list):
            for item in allowed_imports_raw:
                value = str(item or "").strip()
                if value:
                    allowed_imports.append(value)

        return {
            "code": code,
            "timeout_ms": timeout_ms,
            "allowed_imports": allowed_imports or sorted(DEFAULT_ALLOWED_IMPORTS),
        }

    @staticmethod
    def _normalize_prompt_config(config: Dict[str, Any]) -> Dict[str, Any]:
        prompt_template = str(config.get("prompt_template") or "")
        system_prompt = str(config.get("system_prompt") or "")

        raw_timeout = config.get("timeout_ms")
        timeout_ms = DEFAULT_PROMPT_TIMEOUT_MS
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            timeout_ms = int(raw_timeout)

        raw_temperature = config.get("temperature")
        temperature = DEFAULT_PROMPT_TEMPERATURE
        if isinstance(raw_temperature, (int, float)):
            temperature = float(raw_temperature)

        return {
            "prompt_template": prompt_template,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "timeout_ms": timeout_ms,
        }

    @staticmethod
    def _validate_python_code(
        *,
        code: str,
        allowed_imports: set[str],
    ) -> tuple[Optional[str], List[str]]:
        if not code.strip():
            return "Python config.code is required.", []

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Invalid Python syntax: {exc}", []

        blocked: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = str(alias.name or "").split(".")[0]
                    if module_name and module_name not in allowed_imports:
                        blocked.append(module_name)
            elif isinstance(node, ast.ImportFrom):
                module_name = str(node.module or "").split(".")[0]
                if module_name and module_name not in allowed_imports:
                    blocked.append(module_name)

        return None, blocked

    def _render_template(
        self,
        template: str,
        *,
        tool_input: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        if not template:
            return ""

        def _replace(match: re.Match[str]) -> str:
            token = str(match.group(1) or "").strip()
            if not token:
                return ""

            if token == "query":
                return str(tool_input.get("query") or context.get("task") or "")

            if token.startswith("input."):
                value = self._resolve_path(tool_input, token[len("input.") :])
                return "" if value is None else str(value)

            if token.startswith("context."):
                value = self._resolve_path(context, token[len("context.") :])
                return "" if value is None else str(value)

            value = self._resolve_path(tool_input, token)
            if value is None:
                value = self._resolve_path(context, token)
            return "" if value is None else str(value)

        return _TEMPLATE_PATTERN.sub(_replace, template)

    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        current = root
        for part in path.split("."):
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
                continue
            return None
        return current
