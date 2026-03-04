from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, create_model

ToolHandler = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any]
ToolLogger = Callable[[Dict[str, Any]], None]


class ToolSchemaValidationError(ValueError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_type(tool_spec: Dict[str, Any]) -> str:
    return str(tool_spec.get("type") or tool_spec.get("implementation_kind") or "unknown")


def _success_envelope(
    tool_spec: Dict[str, Any],
    data: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "type": _tool_type(tool_spec),
        "data": data,
        "meta": dict(meta or {}),
    }


def _error_envelope(
    tool_spec: Dict[str, Any],
    *,
    kind: str,
    message: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "type": _tool_type(tool_spec),
        "error": {
            "message": message,
            "kind": kind,
        },
        "meta": dict(meta or {}),
    }


def _is_standard_envelope(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("ok"), bool) and "type" in value


def _schema_to_python_type(schema: Dict[str, Any]) -> Any:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), None)
    normalized = str(raw_type or "").strip().lower()
    if normalized == "string":
        return str
    if normalized == "integer":
        return int
    if normalized == "number":
        return float
    if normalized == "boolean":
        return bool
    if normalized == "array":
        return list[Any]
    if normalized == "object":
        return dict[str, Any]
    return Any


def _build_args_schema(tool_spec: Dict[str, Any]) -> Any:
    tool_id = str(tool_spec.get("tool_id") or tool_spec.get("id") or "tool").replace(".", "_")
    schema = tool_spec.get("input_schema")
    if not isinstance(schema, dict):
        schema = {}
    properties = schema.get("properties")
    required = set(schema.get("required") or [])
    fields: Dict[str, Any] = {}
    if isinstance(properties, dict):
        for key, value in properties.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            prop_schema = value if isinstance(value, dict) else {}
            annotation = _schema_to_python_type(prop_schema)
            default = ... if key_text in required else None
            fields[key_text] = (annotation, default)
    return create_model(
        f"{tool_id.title()}Args",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def _validate_schema(
    schema: Dict[str, Any],
    value: Any,
    *,
    path: str,
) -> None:
    if not isinstance(schema, dict) or not schema:
        return
    raw_type = schema.get("type")
    allowed_types = raw_type if isinstance(raw_type, list) else [raw_type] if raw_type else []
    if allowed_types:
        if not any(_matches_type(expected, value) for expected in allowed_types):
            expected = ", ".join(str(item) for item in allowed_types if item)
            raise ToolSchemaValidationError(
                f"{path} expected {expected or 'any'}, got {type(value).__name__}."
            )

    normalized_type = next((str(item) for item in allowed_types if item), "")
    if normalized_type == "object" and isinstance(value, dict):
        properties = schema.get("properties")
        required = schema.get("required") or []
        if isinstance(required, list):
            for key in required:
                key_text = str(key)
                if key_text not in value:
                    raise ToolSchemaValidationError(f"{path}.{key_text} is required.")
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                key_text = str(key)
                if key_text in value and isinstance(child_schema, dict):
                    _validate_schema(child_schema, value[key_text], path=f"{path}.{key_text}")
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            extras = [key for key in value.keys() if str(key) not in properties]
            if extras:
                raise ToolSchemaValidationError(
                    f"{path} includes unsupported keys: {', '.join(sorted(str(key) for key in extras))}."
                )
    if normalized_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item_schema, item, path=f"{path}[{index}]")


def _matches_type(expected: Any, value: Any) -> bool:
    normalized = str(expected or "").strip().lower()
    if not normalized:
        return True
    if normalized == "object":
        return isinstance(value, dict)
    if normalized == "array":
        return isinstance(value, list)
    if normalized == "string":
        return isinstance(value, str)
    if normalized == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if normalized == "boolean":
        return isinstance(value, bool)
    if normalized == "null":
        return value is None
    return True


def _resolve_sync(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - thread handoff
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


class LangGraphToolAdapter:
    def __init__(
        self,
        *,
        tool_spec: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        self.tool_spec = dict(tool_spec)
        self._handler = handler
        self._args_schema = _build_args_schema(self.tool_spec)
        self._langgraph_tool = StructuredTool.from_function(
            func=self._invoke_from_langgraph,
            coroutine=self._ainvoke_from_langgraph,
            name=str(
                self.tool_spec.get("tool_id")
                or self.tool_spec.get("id")
                or self.tool_spec.get("name")
                or "tool"
            ),
            description=str(self.tool_spec.get("description") or ""),
            args_schema=self._args_schema,
            infer_schema=False,
        )

    @property
    def langgraph_tool(self) -> StructuredTool:
        return self._langgraph_tool

    def invoke(
        self,
        tool_input: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        logger: Optional[ToolLogger] = None,
    ) -> Dict[str, Any]:
        return _resolve_sync(
            self.ainvoke(tool_input, context=context, logger=logger)
        )

    async def ainvoke(
        self,
        tool_input: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        logger: Optional[ToolLogger] = None,
    ) -> Dict[str, Any]:
        started_at = _utc_now_iso()
        normalized_input = dict(tool_input or {})
        log_meta: Dict[str, Any] = {
            "tool_id": str(
                self.tool_spec.get("tool_id") or self.tool_spec.get("id") or ""
            ),
        }
        if logger is not None:
            logger(
                {
                    **log_meta,
                    "phase": "start",
                    "input": normalized_input,
                    "ts": started_at,
                }
            )
        try:
            _validate_schema(
                self.tool_spec.get("input_schema")
                if isinstance(self.tool_spec.get("input_schema"), dict)
                else {},
                normalized_input,
                path="input",
            )
            raw_output = await self._run_handler(normalized_input, context=context)
            envelope = self._normalize_output(raw_output)
            if bool(envelope.get("ok", False)):
                _validate_schema(
                    self.tool_spec.get("output_schema")
                    if isinstance(self.tool_spec.get("output_schema"), dict)
                    else {},
                    envelope.get("data"),
                    path="output",
                )
        except ToolSchemaValidationError as exc:
            envelope = _error_envelope(
                self.tool_spec,
                kind="validation",
                message=str(exc),
                meta=log_meta,
            )
        except Exception as exc:
            envelope = _error_envelope(
                self.tool_spec,
                kind="runtime",
                message=str(exc),
                meta=log_meta,
            )
        ended_at = _utc_now_iso()
        if logger is not None:
            logger(
                {
                    **log_meta,
                    "phase": "end",
                    "input": normalized_input,
                    "output": envelope,
                    "ts": ended_at,
                }
            )
        return envelope

    async def _ainvoke_from_langgraph(self, **tool_input: Any) -> Dict[str, Any]:
        return await self.ainvoke(dict(tool_input))

    def _invoke_from_langgraph(self, **tool_input: Any) -> Dict[str, Any]:
        return self.invoke(dict(tool_input))

    async def _run_handler(
        self,
        tool_input: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        maybe_output = self._handler(tool_input, dict(context or {}))
        if inspect.isawaitable(maybe_output):
            return await maybe_output
        return maybe_output

    def _normalize_output(self, value: Any) -> Dict[str, Any]:
        if _is_standard_envelope(value):
            return dict(value)
        if isinstance(value, dict) and isinstance(value.get("error"), dict):
            error_map = value.get("error") or {}
            return _error_envelope(
                self.tool_spec,
                kind=str(error_map.get("kind") or error_map.get("type") or "runtime"),
                message=str(error_map.get("message") or "Tool execution failed."),
                meta={"raw": value},
            )
        return _success_envelope(self.tool_spec, value)


def build_langgraph_tool_adapter(
    *,
    tool_spec: Dict[str, Any],
    handler: ToolHandler,
) -> LangGraphToolAdapter:
    return LangGraphToolAdapter(tool_spec=tool_spec, handler=handler)
