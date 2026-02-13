from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_ollama.embeddings import OllamaEmbeddings
else:
    OllamaEmbeddings = Any

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


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


def get_ollama_embeddings(model_id: str, base_url: str) -> OllamaEmbeddings:
    resolved_model = str(model_id or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)).strip()
    if not resolved_model:
        raise ValueError("Embedding model is required (input.embedding_model or OLLAMA_EMBED_MODEL).")

    resolved_base_url = str(base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).strip()
    if not resolved_base_url:
        raise ValueError("OLLAMA_BASE_URL must be a non-empty URL.")

    embeddings_class = _resolve_embeddings_class()
    try:
        return embeddings_class(model=resolved_model, base_url=resolved_base_url)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Ollama embeddings model '{resolved_model}' "
            f"at '{resolved_base_url}': {exc}"
        ) from exc
