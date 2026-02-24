from __future__ import annotations

import os
from typing import Any, Dict, Optional

from saturday_agent.rag.retriever import retrieve

TOOL_ID = "rag.retrieve"
TOOL_NAME = "RAG Retrieve (Qdrant)"
TOOL_DESCRIPTION = "Retrieve relevant document chunks from a Qdrant knowledge base using Ollama embeddings."

DEFAULT_COLLECTION = "saturday_docs"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

RAG_RETRIEVE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Natural language retrieval query."},
        "collection": {
            "type": "string",
            "default": DEFAULT_COLLECTION,
            "description": "Qdrant collection name.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "default": 5,
            "description": "Maximum number of chunks to retrieve.",
        },
        "embedding_model": {
            "type": "string",
            "default": DEFAULT_EMBED_MODEL,
            "description": "Installed Ollama embedding model id.",
        },
        "filters": {
            "type": "object",
            "description": "Optional Qdrant payload filter object.",
        },
        "qdrant_url": {
            "type": "string",
            "description": "Optional Qdrant base URL override.",
        },
        "include_text": {
            "type": "boolean",
            "default": True,
            "description": "If false, returned chunks include empty text fields.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

RAG_RETRIEVE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "collection": {"type": "string"},
        "embedding_model": {"type": "string"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": ["string", "number"]},
                    "score": {"type": "number"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["doc_id", "score", "text", "metadata"],
            },
        },
    },
    "required": ["query", "collection", "embedding_model", "results"],
}


def _to_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _normalize_top_k(value: Any) -> int:
    if value is None:
        return 5
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer.") from exc
    return max(1, parsed)


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "qdrant" in message or "unable to connect" in message or "was not found at" in message:
        return "qdrant"
    if "embedding" in message or "ollama" in message:
        return "embeddings"
    return "config"


def retrieve_qdrant_chunks(
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context_map = dict(context or {})
    payload = dict(tool_input or {})

    try:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("rag.retrieve requires a non-empty 'query' field.")

        collection = str(
            payload.get("collection")
            or context_map.get("retrieval_collection")
            or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        ).strip()
        embedding_model = str(
            payload.get("embedding_model")
            or context_map.get("embedding_model")
            or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        ).strip()
        qdrant_url = str(
            payload.get("qdrant_url")
            or context_map.get("qdrant_url")
            or os.getenv("QDRANT_URL", "")
        ).strip()
        top_k = _normalize_top_k(payload.get("top_k"))

        filters = payload.get("filters")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be an object.")

        include_text = _to_bool(payload.get("include_text"), default=True)

        response = retrieve(
            query,
            collection=collection,
            embeddings_model=embedding_model,
            top_k=top_k,
            filters=filters,
            qdrant_url=qdrant_url or None,
        )

        if not include_text:
            for item in response.get("results", []):
                if isinstance(item, dict):
                    item["text"] = ""

        return response
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "message": str(exc),
                "type": _classify_error(exc),
            },
        }


# Example tool invocation payload:
# {
#   "query": "How do we configure local auth?",
#   "collection": "saturday_docs",
#   "top_k": 5,
#   "embedding_model": "nomic-embed-text",
#   "filters": {"must": [{"key": "source", "match": {"value": "docs/auth.md"}}]},
#   "include_text": true
# }
#
# Example expected output shape:
# {
#   "query": "How do we configure local auth?",
#   "collection": "saturday_docs",
#   "embedding_model": "nomic-embed-text",
#   "results": [
#     {
#       "doc_id": "docs/auth.md",
#       "chunk_id": "17",
#       "score": 0.121,
#       "text": "...",
#       "metadata": {"doc_id": "docs/auth.md", "chunk_id": "17", "source": "docs/auth.md"}
#     }
#   ]
# }
#
# Example workflow source section (generated in synthesis nodes):
# Sources:
# - [1] docs/auth.md#17 (score=0.121)
