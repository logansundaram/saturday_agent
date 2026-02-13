from __future__ import annotations

from typing import Any, Dict, Optional

from saturday_agent.rag.ingest.persist import list_local_docs

TOOL_ID = "rag.list_docs"
TOOL_NAME = "RAG List Local Docs"
TOOL_DESCRIPTION = "List local ingested documents and ingestion status."

RAG_LIST_DOCS_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "description": "Optional status filter (ready|ingesting|ingested|error|deleted|all).",
        }
    },
    "additionalProperties": False,
}

RAG_LIST_DOCS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "docs": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["ok", "docs"],
}


def list_rag_docs(
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ = context
    payload = dict(tool_input or {})
    try:
        status = payload.get("status")
        status_value = str(status).strip() if status is not None else None
        docs = list_local_docs(status=status_value)
        return {"ok": True, "docs": docs}
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "type": "db",
                "message": str(exc),
            },
        }
