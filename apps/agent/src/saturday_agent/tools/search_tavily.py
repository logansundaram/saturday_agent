from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx

TOOL_ID = "search.web"
TOOL_NAME = "search"
TOOL_DESCRIPTION = "Search the web for up-to-date information using Tavily."

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
MAX_CONTENT_CHARS = 1500
REQUEST_TIMEOUT_SECONDS = 20.0

SEARCH_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text."},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        "include_domains": {"type": "array", "items": {"type": "string"}},
        "exclude_domains": {"type": "array", "items": {"type": "string"}},
        "recency_days": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
    "additionalProperties": False,
}

SEARCH_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "content": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["title", "url", "content"],
            },
        },
        "error": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    },
    "required": ["query", "results"],
}


def search_web_tavily(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool_input, dict):
        raise ValueError("search.web input must be an object.")

    query = str(tool_input.get("query", "")).strip()
    if not query:
        raise ValueError("search.web requires a non-empty 'query' field.")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Set it in your environment.")

    max_results = _sanitize_max_results(tool_input.get("max_results"))
    include_domains = _sanitize_domains(tool_input.get("include_domains"))
    exclude_domains = _sanitize_domains(tool_input.get("exclude_domains"))
    recency_days = _sanitize_recency_days(tool_input.get("recency_days"))

    payload: Dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    if recency_days is not None:
        payload["days"] = recency_days

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        return _failure_result(query=query, error_type="http_error", message=str(exc))
    except ValueError as exc:
        return _failure_result(
            query=query,
            error_type="invalid_response",
            message=f"Tavily returned invalid JSON: {exc}",
        )

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return _failure_result(
            query=query,
            error_type="invalid_response",
            message="Tavily response did not include a valid 'results' list.",
        )

    results: List[Dict[str, Any]] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        score = item.get("score")

        if not title and url:
            title = url
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS].rstrip() + "..."

        result: Dict[str, Any] = {
            "title": title,
            "url": url,
            "content": content,
        }
        if isinstance(score, (int, float)):
            result["score"] = float(score)
        results.append(result)

    return {
        "query": query,
        "results": results,
    }


def _sanitize_max_results(raw_value: Any) -> int:
    if raw_value is None:
        return DEFAULT_MAX_RESULTS
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'max_results' must be an integer.") from exc
    return max(1, min(20, value))


def _sanitize_domains(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError("Domain filters must be arrays of strings.")

    normalized: List[str] = []
    for item in raw_value:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized


def _sanitize_recency_days(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'recency_days' must be an integer.") from exc
    if value <= 0:
        raise ValueError("'recency_days' must be greater than 0.")
    return value


def _failure_result(*, query: str, error_type: str, message: str) -> Dict[str, Any]:
    return {
        "query": query,
        "results": [],
        "error": {
            "type": error_type,
            "message": message,
        },
    }


# Example tool invocation payload:
# {"query":"latest CUDA release notes","max_results":5,"include_domains":["nvidia.com"]}
#
# Example tool output:
# {"query":"latest CUDA release notes","results":[{"title":"...","url":"...","content":"...","score":0.98}]}
