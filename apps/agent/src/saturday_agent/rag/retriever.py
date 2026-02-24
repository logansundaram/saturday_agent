from __future__ import annotations

import json
import os
from typing import Any, Dict

from qdrant_client.http import models as qdrant_models

from saturday_agent.rag.embeddings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_EMBED_MODEL,
    get_ollama_embeddings,
)
from saturday_agent.rag.qdrant_store import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
    get_vectorstore,
)

MAX_TEXT_CHARS = 1500
MAX_TOP_K = 20


def _to_json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _normalize_top_k(value: Any) -> int:
    if value is None:
        return 5
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer.") from exc
    return max(1, min(parsed, MAX_TOP_K))


def _build_qdrant_filter(filters: Dict[str, Any] | None) -> qdrant_models.Filter | None:
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object that matches Qdrant Filter schema.")

    try:
        if hasattr(qdrant_models.Filter, "model_validate"):
            return qdrant_models.Filter.model_validate(filters)
        if hasattr(qdrant_models.Filter, "parse_obj"):
            return qdrant_models.Filter.parse_obj(filters)
        return qdrant_models.Filter(**filters)
    except Exception as exc:
        raise ValueError(f"Invalid Qdrant filter payload: {exc}") from exc


def _first_non_empty(metadata: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def retrieve(
    query: str,
    *,
    collection: str,
    embeddings_model: str,
    top_k: int,
    filters: Dict[str, Any] | None,
    qdrant_url: str | None = None,
) -> Dict[str, Any]:
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("query is required for rag.retrieve.")

    resolved_collection = str(collection or os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)).strip()
    if not resolved_collection:
        raise ValueError("collection is required (input.collection or QDRANT_COLLECTION).")

    resolved_embedding_model = str(
        embeddings_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
    ).strip()
    if not resolved_embedding_model:
        raise ValueError("embedding_model is required (input.embedding_model or OLLAMA_EMBED_MODEL).")

    resolved_top_k = _normalize_top_k(top_k)
    resolved_qdrant_url = str(
        qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    ).strip()
    if not resolved_qdrant_url:
        raise ValueError(
            "Qdrant URL is required (input.qdrant_url, context.qdrant_url, or QDRANT_URL)."
        )
    ollama_base_url = str(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).strip()

    embeddings = get_ollama_embeddings(resolved_embedding_model, ollama_base_url)
    store = get_vectorstore(resolved_collection, embeddings, resolved_qdrant_url)
    qdrant_filter = _build_qdrant_filter(filters)

    search_kwargs: Dict[str, Any] = {}
    if qdrant_filter is not None:
        search_kwargs["filter"] = qdrant_filter

    raw_results = store.similarity_search_with_score(query_text, k=resolved_top_k, **search_kwargs)

    results: list[Dict[str, Any]] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        document = item[0]
        score_raw = item[1]

        metadata_raw = getattr(document, "metadata", {})
        metadata = _to_json_safe(metadata_raw) if isinstance(metadata_raw, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        text = str(getattr(document, "page_content", "") or "")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS].rstrip() + "..."

        doc_id = _first_non_empty(metadata, ["doc_id", "document_id", "id", "source"])
        source = _first_non_empty(metadata, ["source", "path", "file_path"])
        chunk_id = _first_non_empty(metadata, ["chunk_id", "chunk_index"])

        if not doc_id and source:
            doc_id = source
        if not doc_id:
            doc_id = f"doc_{index + 1}"

        metadata["doc_id"] = doc_id
        if chunk_id:
            metadata["chunk_id"] = chunk_id
        if source:
            metadata["source"] = source

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "score": float(score_raw) if isinstance(score_raw, (int, float)) else 0.0,
            "text": text,
            "metadata": metadata,
        }
        if chunk_id:
            result["chunk_id"] = chunk_id

        results.append(result)

    return {
        "query": query_text,
        "collection": resolved_collection,
        "embedding_model": resolved_embedding_model,
        "top_k": resolved_top_k,
        "results": results,
    }
