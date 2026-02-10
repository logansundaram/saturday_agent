from __future__ import annotations

from typing import Any, Dict, List, Set

import httpx

_VISION_HINTS = (
    "vision",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen2.5vl",
    "gemma3",
    "vl",
)


def _resolve_httpx_timeout(timeout_seconds: float) -> float | None:
    try:
        timeout_value = float(timeout_seconds)
    except (TypeError, ValueError):
        return None

    if timeout_value <= 0:
        return None
    return timeout_value


class VisionModelRegistry:
    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        timeout_seconds: float,
        allowlist_raw: str = "",
    ) -> None:
        self._base_url = base_url
        self._default_model = str(default_model or "").strip()
        self._timeout_seconds = timeout_seconds
        self._allowlist = self._parse_allowlist(allowlist_raw)

    @staticmethod
    def _parse_allowlist(allowlist_raw: str) -> Set[str]:
        return {
            model.strip().lower()
            for model in str(allowlist_raw or "").split(",")
            if model.strip()
        }

    def _fetch_ollama_tags(self) -> tuple[Dict[str, Any], bool]:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=_resolve_httpx_timeout(self._timeout_seconds),
            ) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                return response.json(), True
        except (httpx.HTTPError, ValueError):
            return {}, False

    def _is_vision_candidate(self, model_name: str, raw_item: Dict[str, Any]) -> bool:
        normalized_name = model_name.strip().lower()
        if not normalized_name:
            return False

        if self._allowlist:
            return normalized_name in self._allowlist

        if any(hint in normalized_name for hint in _VISION_HINTS):
            return True

        details = raw_item.get("details")
        if isinstance(details, dict):
            families = details.get("families")
            if isinstance(families, list):
                normalized_families = {str(item).strip().lower() for item in families}
                if {"clip", "vision"} & normalized_families:
                    return True

        capabilities = raw_item.get("capabilities")
        if isinstance(capabilities, list):
            normalized_capabilities = {str(item).strip().lower() for item in capabilities}
            if "vision" in normalized_capabilities:
                return True

        return False

    def list_vision_models(self) -> List[Dict[str, Any]]:
        payload, _ = self._fetch_ollama_tags()
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            return []

        models: List[Dict[str, Any]] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue
            model_name = str(raw_model.get("name") or raw_model.get("model") or "").strip()
            if not model_name:
                continue
            if not self._is_vision_candidate(model_name, raw_model):
                continue
            models.append(
                {
                    "id": model_name,
                    "name": model_name,
                    "source": "ollama",
                    "status": "installed",
                }
            )
        return models

    def default_vision_model(self) -> str:
        if self._default_model:
            return self._default_model
        models = self.list_vision_models()
        if models:
            return str(models[0].get("id") or "")
        return ""

    def list_models_payload(self) -> Dict[str, Any]:
        _, reachable = self._fetch_ollama_tags()
        models = self.list_vision_models()
        if reachable:
            ollama_status = "ok" if models else "ok_empty"
        else:
            ollama_status = "down"
        return {
            "models": models,
            "default_model": self.default_vision_model(),
            "ollama_status": ollama_status,
        }

    def is_vision_model(self, model_name: str) -> bool:
        normalized_name = str(model_name or "").strip().lower()
        if not normalized_name:
            return False
        models = self.list_vision_models()
        return any(str(item.get("id") or "").strip().lower() == normalized_name for item in models)
