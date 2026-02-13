from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH: Optional[str] = None


def init_db(db_path: str) -> None:
    """Initialize the SQLite database and create tables if missing."""
    global _DB_PATH
    _DB_PATH = db_path

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                kind TEXT,
                status TEXT,
                payload_json TEXT,
                result_json TEXT,
                started_at TEXT,
                ended_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step_index INTEGER,
                name TEXT,
                status TEXT,
                input_json TEXT,
                output_json TEXT,
                started_at TEXT,
                ended_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step_index INTEGER,
                timestamp TEXT,
                state_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                path TEXT,
                mime TEXT,
                size INTEGER,
                sha256 TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT,
                kind TEXT,
                type TEXT,
                description TEXT,
                enabled INTEGER,
                config_json TEXT,
                input_schema_json TEXT,
                output_schema_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                enabled INTEGER,
                graph_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
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


def create_run(run_id: str, kind: str, payload_json: str, started_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, kind, status, payload_json, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, kind, "running", payload_json, started_at),
        )
        conn.commit()


def finish_run(run_id: str, status: str, ended_at: str, result_json: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?, result_json = ?
            WHERE run_id = ?
            """,
            (status, ended_at, result_json, run_id),
        )
        conn.commit()


def add_step(
    run_id: str,
    step_index: int,
    name: str,
    input_json: str,
    output_json: str,
    status: str,
    started_at: str,
    ended_at: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO steps (
                run_id, step_index, name, status,
                input_json, output_json, started_at, ended_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step_index,
                name,
                status,
                input_json,
                output_json,
                started_at,
                ended_at,
            ),
        )
        conn.commit()


def get_run(run_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, kind, status, payload_json, result_json, started_at, ended_at
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def list_steps(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, step_index, name, status,
                   input_json, output_json, started_at, ended_at
            FROM steps
            WHERE run_id = ?
            ORDER BY step_index ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def read_run(run_id: str) -> Optional[dict]:
    return get_run(run_id)


def read_steps(run_id: str) -> list[dict]:
    return list_steps(run_id)


def read_state_snapshots(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, step_index, timestamp, state_json
            FROM state_snapshots
            WHERE run_id = ?
            ORDER BY step_index ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def add_artifact(
    *,
    artifact_id: str,
    path: str,
    mime: str,
    size: int,
    sha256: str,
    created_at: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, path, mime, size, sha256, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, path, mime, int(size), sha256, created_at),
        )
        conn.commit()


def read_artifact(artifact_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT artifact_id, path, mime, size, sha256, created_at
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def upsert_tool(tool: dict) -> None:
    tool_id = str(tool.get("id") or "").strip()
    if not tool_id:
        raise ValueError("Tool id is required.")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tools (
                id, name, kind, type, description, enabled,
                config_json, input_schema_json, output_schema_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                kind = excluded.kind,
                type = excluded.type,
                description = excluded.description,
                enabled = excluded.enabled,
                config_json = excluded.config_json,
                input_schema_json = excluded.input_schema_json,
                output_schema_json = excluded.output_schema_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                tool_id,
                str(tool.get("name") or tool_id),
                str(tool.get("kind") or "external"),
                str(tool.get("type") or "http"),
                str(tool.get("description") or ""),
                1 if bool(tool.get("enabled", True)) else 0,
                _json_dumps(tool.get("config") if isinstance(tool.get("config"), dict) else {}),
                _json_dumps(tool.get("input_schema")),
                _json_dumps(tool.get("output_schema")),
                str(tool.get("created_at") or ""),
                str(tool.get("updated_at") or ""),
            ),
        )
        conn.commit()


def list_tools() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, kind, type, description, enabled,
                   config_json, input_schema_json, output_schema_json,
                   created_at, updated_at
            FROM tools
            ORDER BY name COLLATE NOCASE ASC, id ASC
            """
        ).fetchall()
        return [_tool_row_to_definition(row) for row in rows]


def get_tool(tool_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, kind, type, description, enabled,
                   config_json, input_schema_json, output_schema_json,
                   created_at, updated_at
            FROM tools
            WHERE id = ?
            """,
            (tool_id,),
        ).fetchone()
        if row is None:
            return None
        return _tool_row_to_definition(row)


def set_tool_enabled(tool_id: str, enabled: bool) -> bool:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE tools
            SET enabled = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (1 if enabled else 0, updated_at, tool_id),
        )
        conn.commit()
        return result.rowcount > 0


def upsert_workflow(workflow: dict) -> None:
    workflow_id = str(workflow.get("id") or "").strip()
    if not workflow_id:
        raise ValueError("Workflow id is required.")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workflows (
                id, name, description, enabled, graph_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                enabled = excluded.enabled,
                graph_json = excluded.graph_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                workflow_id,
                str(workflow.get("name") or workflow_id),
                str(workflow.get("description") or ""),
                1 if bool(workflow.get("enabled", True)) else 0,
                _json_dumps(workflow.get("graph") if isinstance(workflow.get("graph"), dict) else {}),
                str(workflow.get("created_at") or ""),
                str(workflow.get("updated_at") or ""),
            ),
        )
        conn.commit()


def list_workflows() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, enabled, graph_json, created_at, updated_at
            FROM workflows
            ORDER BY name COLLATE NOCASE ASC, id ASC
            """
        ).fetchall()
        return [_workflow_row_to_definition(row) for row in rows]


def get_workflow(workflow_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, description, enabled, graph_json, created_at, updated_at
            FROM workflows
            WHERE id = ?
            """,
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return _workflow_row_to_definition(row)


def set_workflow_enabled(workflow_id: str, enabled: bool) -> bool:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE workflows
            SET enabled = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (1 if enabled else 0, updated_at, workflow_id),
        )
        conn.commit()
        return result.rowcount > 0


def _tool_row_to_definition(row: sqlite3.Row) -> dict:
    input_schema = _json_loads(row["input_schema_json"])
    output_schema = _json_loads(row["output_schema_json"])
    return {
        "id": str(row["id"]),
        "name": str(row["name"] or row["id"]),
        "kind": str(row["kind"] or "external"),
        "type": str(row["type"] or "http"),
        "description": str(row["description"] or ""),
        "enabled": bool(int(row["enabled"] or 0)),
        "config": _json_loads(row["config_json"]) or {},
        "input_schema": input_schema if isinstance(input_schema, (dict, list)) else None,
        "output_schema": output_schema if isinstance(output_schema, (dict, list)) else None,
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def _workflow_row_to_definition(row: sqlite3.Row) -> dict:
    graph = _json_loads(row["graph_json"])
    return {
        "id": str(row["id"]),
        "name": str(row["name"] or row["id"]),
        "title": str(row["name"] or row["id"]),
        "description": str(row["description"] or ""),
        "enabled": bool(int(row["enabled"] or 0)),
        "source": "custom",
        "type": "custom",
        "graph": graph if isinstance(graph, dict) else {"nodes": [], "edges": []},
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def _json_loads(raw: Optional[str]) -> object:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _json_dumps(value: object) -> str:
    return json.dumps(value, default=str)


def _connect() -> sqlite3.Connection:
    if not _DB_PATH:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
