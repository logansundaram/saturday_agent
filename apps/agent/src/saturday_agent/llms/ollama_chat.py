from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


def ollama_chat(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    base_url: str,
    timeout_seconds: float,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()


def extract_assistant_text(raw: Dict[str, Any]) -> str:
    message = raw.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return str(content)

    response = raw.get("response")
    if response is None:
        return ""
    return str(response)
