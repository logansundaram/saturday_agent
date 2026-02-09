from __future__ import annotations

from typing import Any, Dict, List

import httpx


class ModelRegistry:
    """Stub model registry for future model management UI."""

    def __init__(self, *, base_url: str, default_model: str, timeout_seconds: float) -> None:
        self._base_url = base_url
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds

    def _fetch_ollama_tags(self) -> tuple[Dict[str, Any], bool]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                return response.json(), True
        except (httpx.HTTPError, ValueError):
            return {}, False

    def list_models(self) -> List[Dict[str, Any]]:
        payload, _ = self._fetch_ollama_tags()
        models = payload.get("models")
        if not isinstance(models, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
            normalized.append(
                {
                    "id": name,
                    "name": name,
                    "source": "ollama",
                    "status": "installed",
                }
            )
        return normalized

    def list_models_payload(self) -> Dict[str, Any]:
        payload, reachable = self._fetch_ollama_tags()
        models = []
        raw_models = payload.get("models")
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("model") or "").strip()
                if not name:
                    continue
                models.append(
                    {
                        "id": name,
                        "name": name,
                        "source": "ollama",
                        "status": "installed",
                    }
                )

        if reachable:
            ollama_status = "ok" if models else "ok_empty"
        else:
            ollama_status = "down"
        return {
            "models": models,
            "default_model": self._default_model,
            "ollama_status": ollama_status,
        }

    def get_default_model(self) -> str:
        return self._default_model

    def set_default_model(self, model: str) -> None:
        self._default_model = str(model)
        # TODO: Persist this choice so model selection survives process restarts.

    # TODO: Add install/download hooks for future Models page.
