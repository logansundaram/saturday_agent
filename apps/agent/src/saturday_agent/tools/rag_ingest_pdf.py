from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from saturday_agent.rag.embeddings import DEFAULT_OLLAMA_EMBED_MODEL
from saturday_agent.rag.ingest.chunking import chunk_text
from saturday_agent.rag.ingest.pdf_extract import PdfExtractionError, extract_pdf_text
from saturday_agent.rag.ingest.persist import (
    DEFAULT_COLLECTION,
    PersistError,
    create_local_doc,
    docs_root_dir,
    replace_doc_chunks,
    set_doc_status,
)
from saturday_agent.rag.ingest.qdrant_index import index_chunks_to_qdrant

TOOL_ID = "rag.ingest_pdf"
TOOL_NAME = "RAG Ingest PDF"
TOOL_DESCRIPTION = (
    "Ingest a local PDF into Saturday local docs storage, chunk it, persist chunk records, "
    "and optionally index to Qdrant."
)

INGEST_PDF_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute file path to a local PDF file.",
        },
        "collection": {
            "type": "string",
            "default": DEFAULT_COLLECTION,
            "description": "Qdrant collection name.",
        },
        "embedding_model": {
            "type": "string",
            "default": DEFAULT_OLLAMA_EMBED_MODEL,
            "description": "Installed Ollama embedding model id.",
        },
        "chunk_size": {
            "type": "integer",
            "default": 900,
            "minimum": 200,
            "description": "Chunk size in characters.",
        },
        "chunk_overlap": {
            "type": "integer",
            "default": 150,
            "minimum": 0,
            "description": "Chunk overlap in characters.",
        },
        "index_to_qdrant": {
            "type": "boolean",
            "default": True,
            "description": "Whether to write chunk vectors into Qdrant.",
        },
    },
    "required": ["file_path"],
    "additionalProperties": False,
}

INGEST_PDF_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "doc": {"type": "object"},
        "chunk_count": {"type": "integer"},
        "indexed": {"type": "object"},
        "error": {"type": "object"},
    },
    "required": ["ok"],
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


def _normalize_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _resolve_collection(payload: Dict[str, Any], context_map: Dict[str, Any]) -> str:
    return str(
        payload.get("collection")
        or context_map.get("retrieval_collection")
        or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    ).strip()


def _resolve_embedding_model(payload: Dict[str, Any], context_map: Dict[str, Any]) -> str:
    return str(
        payload.get("embedding_model")
        or context_map.get("embedding_model")
        or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
    ).strip()


def _is_pdf_file(path: Path) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(5).startswith(b"%PDF-")
    except OSError:
        return False


def _sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, (FileNotFoundError, NotADirectoryError, IsADirectoryError, OSError)):
        return "io"
    if isinstance(exc, PdfExtractionError):
        return "pdf"
    if isinstance(exc, (PersistError, sqlite3.Error)):
        return "db"
    if "qdrant" in message or "collection" in message:
        return "qdrant"
    if "embedding" in message or "ollama" in message:
        return "embeddings"
    return "config"


def ingest_pdf_document(
    tool_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(tool_input or {})
    context_map = dict(context or {})

    doc_id = ""
    doc_record: Optional[Dict[str, Any]] = None
    collection = ""

    try:
        raw_file_path = str(payload.get("file_path") or "").strip()
        if not raw_file_path:
            raise ValueError("rag.ingest_pdf requires 'file_path'.")

        source_path = Path(raw_file_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"PDF file was not found: {source_path}")
        if not _is_pdf_file(source_path):
            raise ValueError(
                f"Expected a PDF file but received '{source_path.name}'. "
                "Choose a '.pdf' file from Local Docs import."
            )

        collection = _resolve_collection(payload, context_map)
        if not collection:
            raise ValueError("collection must be provided or set via QDRANT_COLLECTION.")

        embedding_model = _resolve_embedding_model(payload, context_map)
        if not embedding_model:
            raise ValueError(
                "embedding_model must be provided or set via OLLAMA_EMBED_MODEL."
            )

        chunk_size = _normalize_int(
            payload.get("chunk_size"),
            default=900,
            minimum=200,
            maximum=4000,
        )
        chunk_overlap = _normalize_int(
            payload.get("chunk_overlap"),
            default=150,
            minimum=0,
            maximum=max(chunk_size - 1, 0),
        )
        index_to_qdrant = _to_bool(payload.get("index_to_qdrant"), default=True)

        doc_id = str(uuid.uuid4())
        filename = source_path.name
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        target_dir = docs_root_dir() / doc_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_path = target_dir / filename
        shutil.copy2(str(source_path), str(stored_path))

        bytes_size = int(stored_path.stat().st_size)
        sha256 = _sha256_for(stored_path)
        doc_record = create_local_doc(
            doc_id=doc_id,
            filename=filename,
            stored_path=str(stored_path),
            bytes_size=bytes_size,
            sha256=sha256,
            collection=collection,
            status="ingesting",
        )

        extracted_text = extract_pdf_text(str(stored_path))
        chunks = chunk_text(extracted_text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No non-empty chunks were produced from the PDF text.")

        chunk_count = replace_doc_chunks(
            doc_id=doc_id,
            chunks=chunks,
            base_metadata={
                "doc_id": doc_id,
                "filename": filename,
                "sha256": sha256,
                "source_path": str(stored_path),
                "source": str(stored_path),
                "collection": collection,
            },
        )

        indexed_payload: Optional[Dict[str, Any]] = None
        if index_to_qdrant:
            indexed_payload = index_chunks_to_qdrant(
                chunks,
                doc_id=doc_id,
                filename=filename,
                sha256=sha256,
                source_path=str(stored_path),
                collection=collection,
                embeddings_model=embedding_model,
            )

        doc_record = set_doc_status(
            doc_id,
            status="ingested",
            error_message=None,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            collection=collection,
        ) or doc_record
        if not doc_record:
            raise PersistError(f"Unable to read updated doc row for doc_id={doc_id}")

        response: Dict[str, Any] = {
            "ok": True,
            "doc": doc_record,
            "chunk_count": int(chunk_count),
        }
        if index_to_qdrant:
            response["indexed"] = indexed_payload or {"ok": True, "added": 0}
        return response

    except Exception as exc:
        message = str(exc)
        if doc_id:
            try:
                updated = set_doc_status(
                    doc_id,
                    status="error",
                    error_message=message,
                    ingested_at=None,
                    collection=collection if collection else None,
                )
                if updated:
                    doc_record = updated
            except Exception:
                pass

        error_payload: Dict[str, Any] = {
            "type": _classify_error(exc),
            "message": message,
        }
        response = {
            "ok": False,
            "error": error_payload,
        }
        if doc_record:
            response["doc"] = doc_record
        return response


# Smoke test payload:
# {
#   "file_path": "/Users/Logan/Documents/notes/architecture.pdf",
#   "collection": "saturday_docs",
#   "embedding_model": "nomic-embed-text",
#   "chunk_size": 900,
#   "chunk_overlap": 150,
#   "index_to_qdrant": true
# }
#
# Expected output shape:
# {
#   "ok": true,
#   "doc": {
#     "id": "...",
#     "filename": "architecture.pdf",
#     "bytes": 12345,
#     "sha256": "...",
#     "status": "ingested",
#     "collection": "saturday_docs",
#     "stored_path": "...",
#     "created_at": "...",
#     "updated_at": "...",
#     "ingested_at": "..."
#   },
#   "chunk_count": 17,
#   "indexed": {"ok": true, "added": 17}
# }
#
# Workflow source behavior:
# - `rag.retrieve` returns chunks with doc_id/chunk_id metadata.
# - synthesis nodes append a `Sources:` section from `state.citations`.
