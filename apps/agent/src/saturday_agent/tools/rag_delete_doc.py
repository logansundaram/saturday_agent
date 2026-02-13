from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from saturday_agent.rag.ingest.persist import (
    DEFAULT_COLLECTION,
    delete_doc_chunks,
    get_local_doc,
    mark_doc_deleted,
)

TOOL_ID = "rag.delete_doc"
TOOL_NAME = "RAG Delete Local Doc"
TOOL_DESCRIPTION = "Delete a previously ingested local document and its stored chunk records."

RAG_DELETE_DOC_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "collection": {"type": "string", "default": DEFAULT_COLLECTION},
        "delete_from_qdrant": {"type": "boolean", "default": True},
    },
    "required": ["doc_id"],
    "additionalProperties": False,
}

RAG_DELETE_DOC_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "deleted": {"type": "boolean"},
        "error": {"type": "object"},
    },
    "required": ["ok", "deleted"],
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


def _resolve_collection(payload: Dict[str, Any], context_map: Dict[str, Any]) -> str:
    return str(
        payload.get("collection")
        or context_map.get("retrieval_collection")
        or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    ).strip()


def _delete_from_qdrant(*, doc_id: str, collection: str) -> None:
    qdrant_url = str(os.getenv("QDRANT_URL", "http://127.0.0.1:6333")).strip()
    client = QdrantClient(url=qdrant_url)

    doc_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="doc_id",
                match=qdrant_models.MatchValue(value=doc_id),
            )
        ]
    )

    selector = qdrant_models.FilterSelector(filter=doc_filter)
    try:
        client.delete(collection_name=collection, points_selector=selector, wait=True)
    except TypeError:
        client.delete(collection_name=collection, points_selector=doc_filter, wait=True)


def delete_rag_doc(
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(tool_input or {})
    context_map = dict(context or {})

    try:
        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("rag.delete_doc requires 'doc_id'.")

        collection = _resolve_collection(payload, context_map)
        if not collection:
            raise ValueError("collection must be provided or set via QDRANT_COLLECTION.")

        delete_from_qdrant = _to_bool(payload.get("delete_from_qdrant"), default=True)

        existing = get_local_doc(doc_id)
        if not existing:
            return {"ok": True, "deleted": False}

        delete_doc_chunks(doc_id)
        mark_doc_deleted(doc_id)

        stored_path = Path(str(existing.get("stored_path") or "")).expanduser()
        if stored_path.name:
            doc_dir = stored_path.parent
            if doc_dir.exists():
                shutil.rmtree(doc_dir, ignore_errors=False)
            elif stored_path.exists():
                stored_path.unlink()

        qdrant_error: Optional[Exception] = None
        if delete_from_qdrant:
            try:
                _delete_from_qdrant(doc_id=doc_id, collection=collection)
            except Exception as exc:
                qdrant_error = exc

        if qdrant_error is not None:
            return {
                "ok": False,
                "deleted": True,
                "error": {
                    "type": "qdrant",
                    "message": str(qdrant_error),
                },
            }

        return {"ok": True, "deleted": True}
    except Exception as exc:
        error_type = "qdrant" if "qdrant" in str(exc).lower() else "config"
        return {
            "ok": False,
            "deleted": False,
            "error": {
                "type": error_type,
                "message": str(exc),
            },
        }
