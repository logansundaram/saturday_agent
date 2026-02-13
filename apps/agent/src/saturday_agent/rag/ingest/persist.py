from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

DEFAULT_DATA_DIR = "./data"
DEFAULT_COLLECTION = "saturday_docs"


class PersistError(RuntimeError):
    """Raised when local document metadata cannot be persisted."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "apps").exists() and (parent / "apps" / "api").exists():
            return parent
    return Path.cwd()


def resolve_db_path() -> Path:
    env_value = os.getenv("SATURDAY_DB_PATH", "").strip()
    if env_value:
        candidate = Path(env_value)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "apps":
            return _repo_root() / candidate
        return Path.cwd() / candidate
    return _repo_root() / "apps" / "api" / "saturday.db"


def resolve_data_dir() -> Path:
    value = str(os.getenv("SATURDAY_DATA_DIR", DEFAULT_DATA_DIR)).strip() or DEFAULT_DATA_DIR
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def docs_root_dir() -> Path:
    root = resolve_data_dir() / "docs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _connect() -> sqlite3.Connection:
    db_path = resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def ensure_tables() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_docs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ready',
                ingested_at TEXT NULL,
                error_message TEXT NULL,
                collection TEXT NOT NULL DEFAULT 'saturday_docs'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_doc_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_local_doc_chunks_doc_id_chunk_index
            ON local_doc_chunks (doc_id, chunk_index)
            """
        )
        conn.commit()


def _row_to_doc(row: sqlite3.Row) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(row["id"]),
        "filename": str(row["filename"] or ""),
        "stored_path": str(row["stored_path"] or ""),
        "bytes": int(row["bytes"] or 0),
        "sha256": str(row["sha256"] or ""),
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
        "status": str(row["status"] or ""),
        "collection": str(row["collection"] or DEFAULT_COLLECTION),
    }
    ingested_at = row["ingested_at"]
    error_message = row["error_message"]
    if ingested_at:
        payload["ingested_at"] = str(ingested_at)
    if error_message:
        payload["error_message"] = str(error_message)
    if "chunk_count" in row.keys():
        payload["chunk_count"] = int(row["chunk_count"] or 0)
    return payload


def create_local_doc(
    *,
    doc_id: str,
    filename: str,
    stored_path: str,
    bytes_size: int,
    sha256: str,
    collection: str,
    status: str = "ingesting",
) -> Dict[str, Any]:
    ensure_tables()
    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO local_docs (
                id, filename, stored_path, bytes, sha256, created_at, updated_at,
                status, ingested_at, error_message, collection
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (
                str(doc_id),
                str(filename),
                str(stored_path),
                int(bytes_size),
                str(sha256),
                now,
                now,
                str(status or "ingesting"),
                str(collection or DEFAULT_COLLECTION),
            ),
        )
        conn.commit()
    created = get_local_doc(doc_id)
    if not created:
        raise PersistError(f"Failed to insert local_docs row for doc_id={doc_id}.")
    return created


def set_doc_status(
    doc_id: str,
    *,
    status: str,
    error_message: Optional[str] = None,
    ingested_at: Optional[str] = None,
    collection: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE local_docs
            SET status = ?,
                updated_at = ?,
                error_message = ?,
                ingested_at = ?,
                collection = COALESCE(?, collection)
            WHERE id = ?
            """,
            (
                str(status or ""),
                _utc_now_iso(),
                str(error_message) if error_message else None,
                str(ingested_at) if ingested_at else None,
                str(collection) if collection else None,
                str(doc_id),
            ),
        )
        conn.commit()
    return get_local_doc(doc_id)


def get_local_doc(doc_id: str) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, filename, stored_path, bytes, sha256, created_at, updated_at,
                   status, ingested_at, error_message, collection
            FROM local_docs
            WHERE id = ?
            """,
            (str(doc_id),),
        ).fetchone()
    if row is None:
        return None
    return _row_to_doc(row)


def replace_doc_chunks(
    *,
    doc_id: str,
    chunks: Sequence[str],
    base_metadata: Dict[str, Any],
) -> int:
    ensure_tables()
    created_at = _utc_now_iso()
    with _connect() as conn:
        conn.execute("DELETE FROM local_doc_chunks WHERE doc_id = ?", (str(doc_id),))
        for chunk_index, chunk_text in enumerate(chunks):
            text_value = str(chunk_text or "").strip()
            if not text_value:
                continue
            metadata = dict(base_metadata or {})
            metadata["doc_id"] = str(doc_id)
            metadata["chunk_index"] = int(chunk_index)
            metadata_json = _json_dumps(metadata)
            conn.execute(
                """
                INSERT INTO local_doc_chunks (
                    id, doc_id, chunk_index, text, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    str(doc_id),
                    int(chunk_index),
                    text_value,
                    metadata_json,
                    created_at,
                ),
            )
        conn.commit()

    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) AS count FROM local_doc_chunks WHERE doc_id = ?",
            (str(doc_id),),
        ).fetchone()
    if row is None:
        return 0
    return int(row["count"] or 0)


def delete_doc_chunks(doc_id: str) -> int:
    ensure_tables()
    with _connect() as conn:
        result = conn.execute(
            "DELETE FROM local_doc_chunks WHERE doc_id = ?",
            (str(doc_id),),
        )
        conn.commit()
        return int(result.rowcount or 0)


def mark_doc_deleted(doc_id: str) -> Optional[Dict[str, Any]]:
    return set_doc_status(
        doc_id,
        status="deleted",
        error_message=None,
        ingested_at=None,
    )


def list_local_docs(status: Optional[str] = None) -> list[Dict[str, Any]]:
    ensure_tables()
    status_value = str(status or "").strip().lower()
    params: list[Any] = []
    where_clause = "WHERE d.status != 'deleted'"
    if status_value and status_value != "all":
        where_clause = "WHERE d.status = ?"
        params.append(status_value)

    query = f"""
        SELECT d.id, d.filename, d.stored_path, d.bytes, d.sha256, d.created_at, d.updated_at,
               d.status, d.ingested_at, d.error_message, d.collection,
               (SELECT COUNT(1) FROM local_doc_chunks c WHERE c.doc_id = d.id) AS chunk_count
        FROM local_docs d
        {where_clause}
        ORDER BY d.created_at DESC
    """

    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

    return [_row_to_doc(row) for row in rows]


def list_doc_chunks(doc_id: str) -> list[Dict[str, Any]]:
    ensure_tables()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, doc_id, chunk_index, text, metadata_json, created_at
            FROM local_doc_chunks
            WHERE doc_id = ?
            ORDER BY chunk_index ASC
            """,
            (str(doc_id),),
        ).fetchall()

    payloads: list[Dict[str, Any]] = []
    for row in rows:
        payloads.append(
            {
                "id": str(row["id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": int(row["chunk_index"] or 0),
                "text": str(row["text"] or ""),
                "metadata": _json_loads(row["metadata_json"]),
                "created_at": str(row["created_at"] or ""),
            }
        )
    return payloads
