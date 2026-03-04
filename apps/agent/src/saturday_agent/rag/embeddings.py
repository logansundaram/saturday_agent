from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from langchain_ollama.embeddings import OllamaEmbeddings
else:
    OllamaEmbeddings = Any

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_MODEL_LOOKUP_TIMEOUT_SECONDS = 5.0


class OllamaEmbeddingsDependencyError(RuntimeError):
    """Raised when the Ollama embeddings integration dependency is missing."""


def _resolve_embeddings_class() -> type[Any]:
    try:
        from langchain_ollama.embeddings import OllamaEmbeddings as _OllamaEmbeddings
    except ModuleNotFoundError as exc:
        raise OllamaEmbeddingsDependencyError(
            "langchain-ollama is required for rag.retrieve. "
            "Install it with `pip install langchain-ollama`."
        ) from exc

    return _OllamaEmbeddings


def _fetch_ollama_model_names(base_url: str) -> list[str]:
    try:
        with httpx.Client(
            base_url=base_url,
            timeout=OLLAMA_MODEL_LOOKUP_TIMEOUT_SECONDS,
        ) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []

    names: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def _model_base_name(model_name: str) -> str:
    return str(model_name or "").strip().split(":", 1)[0].strip().lower()


def _resolve_installed_ollama_model(model_id: str, base_url: str) -> str:
    requested_model = str(model_id or "").strip()
    if not requested_model:
        return requested_model

    available_models = _fetch_ollama_model_names(base_url)
    if not available_models:
        return requested_model

    exact_matches = {
        str(name).strip().lower(): str(name).strip()
        for name in available_models
        if str(name).strip()
    }
    exact = exact_matches.get(requested_model.lower())
    if exact:
        return exact

    requested_base = _model_base_name(requested_model)
    if not requested_base:
        return requested_model

    base_matches = [
        candidate
        for candidate in available_models
        if _model_base_name(candidate) == requested_base
    ]
    if len(base_matches) == 1:
        return base_matches[0]

    return requested_model


def get_ollama_embeddings(model_id: str, base_url: str) -> OllamaEmbeddings:
    resolved_model = str(model_id or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)).strip()
    if not resolved_model:
        raise ValueError("Embedding model is required (input.embedding_model or OLLAMA_EMBED_MODEL).")

    resolved_base_url = str(base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).strip()
    if not resolved_base_url:
        raise ValueError("OLLAMA_BASE_URL must be a non-empty URL.")
    resolved_model = _resolve_installed_ollama_model(resolved_model, resolved_base_url)

    embeddings_class = _resolve_embeddings_class()
    try:
        return embeddings_class(model=resolved_model, base_url=resolved_base_url)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Ollama embeddings model '{resolved_model}' "
            f"at '{resolved_base_url}': {exc}"
        ) from exc
