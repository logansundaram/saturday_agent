from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse

from app import db, graph
from app.tools import registry as tool_registry
from app.tools.schemas import CreateToolRequestModel, ToolSpecModel


def list_tools_payload() -> list[dict]:
    return tool_registry.list_tools()


def invoke_tool_payload(
    *,
    tool_id: str,
    tool_input: Dict[str, Any],
    context: Dict[str, Any] | None = None,
) -> dict:
    return graph.invoke_tool(
        tool_id=tool_id,
        tool_input=tool_input,
        context=context,
    )


def create_tool_payload(request: CreateToolRequestModel) -> dict:
    name = request.name.strip()
    if not name:
        raise ValueError("Tool name is required.")

    existing_tools = tool_registry.list_tools(include_deleted=True)
    existing_ids = {str(item.get("id") or "").strip() for item in existing_tools}

    requested_id = str(request.id or "").strip()
    if requested_id and not _is_valid_tool_id(requested_id):
        raise ValueError(
            "Tool id may only contain letters, numbers, '.', '_' and '-'."
        )
    tool_id = requested_id or _generate_tool_id(name, existing_ids)
    if tool_id in existing_ids:
        raise ValueError(f"Tool id '{tool_id}' already exists.")

    normalized_config = _normalize_tool_config(str(request.type), dict(request.config or {}))
    now = _now_utc_iso()
    record = {
        "id": tool_id,
        "tool_id": tool_id,
        "name": name,
        "kind": request.kind,
        "type": request.type,
        "description": request.description.strip(),
        "enabled": bool(request.enabled),
        "config": normalized_config,
        "input_schema": dict(request.input_schema or {}),
        "output_schema": dict(request.output_schema or {}),
        "source": "custom",
        "implementation_kind": _implementation_kind_for_type(str(request.type)),
        "implementation_ref": _implementation_ref_for_tool(
            tool_id=tool_id,
            tool_type=str(request.type),
            config=normalized_config,
        ),
        "created_at": now,
        "updated_at": now,
    }
    db.upsert_tool(record, created_by="builder")
    created = db.get_tool(tool_id, include_deleted=True)
    if not created:
        raise RuntimeError("Tool was not persisted.")
    return created


def update_tool_enabled_payload(tool_id: str, enabled: bool) -> dict:
    existing = db.get_tool(tool_id, include_deleted=True)
    if not existing:
        raise ValueError("Tool not found or not editable.")
    updated = db.set_tool_enabled(tool_id, enabled, created_by="builder")
    if not updated:
        raise RuntimeError("Failed to update tool.")
    payload = db.get_tool(tool_id, include_deleted=True)
    if not payload:
        raise RuntimeError("Tool update did not persist.")
    return payload


def normalize_tool_spec(payload: Dict[str, Any]) -> dict:
    return ToolSpecModel(**payload).model_dump()


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


def _normalize_tool_config(tool_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    if tool_type == "http":
        url = str(config.get("url") or "").strip()
        if not _is_valid_http_url(url):
            raise ValueError("Tool URL must be a valid http(s) URL.")
        method = str(config.get("method") or "POST").strip().upper()
        if method not in {"GET", "POST"}:
            method = "POST"
        timeout_ms = int(config.get("timeout_ms") or 8000)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0.")
        headers = config.get("headers")
        return {
            "url": url,
            "method": method,
            "headers": _normalize_headers(headers if isinstance(headers, dict) else {}),
            "timeout_ms": timeout_ms,
        }

    if tool_type == "python":
        code = str(config.get("code") or "")
        if not code.strip():
            raise ValueError("Python tool requires non-empty config.code.")
        timeout_ms = int(config.get("timeout_ms") or 5000)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0.")
        raw_allowed_imports = config.get("allowed_imports")
        allowed_imports = []
        if isinstance(raw_allowed_imports, list):
            for item in raw_allowed_imports:
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
    timeout_ms = int(config.get("timeout_ms") or 30000)
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be greater than 0.")
    temperature = float(config.get("temperature") or 0.2)
    return {
        "prompt_template": prompt_template,
        "system_prompt": str(config.get("system_prompt") or ""),
        "temperature": temperature,
        "timeout_ms": timeout_ms,
    }


def _implementation_kind_for_type(tool_type: str) -> str:
    if tool_type == "http":
        return "http"
    if tool_type == "prompt":
        return "prompt"
    return "python_module"


def _implementation_ref_for_tool(
    *,
    tool_id: str,
    tool_type: str,
    config: Dict[str, Any],
) -> str:
    if tool_type == "http":
        return str(config.get("url") or f"http://tool/{tool_id}")
    if tool_type == "prompt":
        return f"prompt://{tool_id}"
    return f"inline://{tool_id}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
