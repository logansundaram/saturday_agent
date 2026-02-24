from __future__ import annotations

import copy
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_DB_PATH: Optional[str] = None


def init_db(db_path: str) -> None:
    """Initialize the SQLite database and apply schema migrations."""
    global _DB_PATH
    _DB_PATH = db_path

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        _apply_migrations(conn)
        conn.commit()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    version_row = conn.execute("PRAGMA user_version").fetchone()
    current_version = int(version_row[0]) if version_row and len(version_row) > 0 else 0

    migrations = [
        _migration_1_legacy_tables,
        _migration_2_workflow_versioning,
        _migration_3_run_steps_and_tool_calls,
        _migration_4_run_replay_and_state_snapshots,
    ]

    while current_version < len(migrations):
        migrations[current_version](conn)
        current_version += 1
        conn.execute(f"PRAGMA user_version = {current_version}")


def _migration_1_legacy_tables(conn: sqlite3.Connection) -> None:
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
            summary TEXT,
            started_at TEXT,
            ended_at TEXT
        )
        """
    )
    _ensure_column(conn, "steps", "summary", "TEXT DEFAULT ''")

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


def _migration_2_workflow_versioning(conn: sqlite3.Connection) -> None:
    if _table_exists(conn, "workflows") and _column_exists(conn, "workflows", "graph_json"):
        if not _table_exists(conn, "workflows_legacy"):
            conn.execute("ALTER TABLE workflows RENAME TO workflows_legacy")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_versions (
            version_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            version_num INTEGER NOT NULL,
            spec_json TEXT NOT NULL,
            compiled_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL DEFAULT 'system',
            FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id),
            UNIQUE (workflow_id, version_num)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_workflow_versions_workflow_id_version_num
        ON workflow_versions (workflow_id, version_num DESC)
        """
    )

    if _table_exists(conn, "workflows_legacy"):
        rows = conn.execute(
            """
            SELECT id, name, description, enabled, graph_json, created_at, updated_at
            FROM workflows_legacy
            ORDER BY name COLLATE NOCASE ASC, id ASC
            """
        ).fetchall()
        for row in rows:
            workflow_id = str(row["id"] or "").strip()
            if not workflow_id:
                continue
            name = str(row["name"] or workflow_id)
            description = str(row["description"] or "")
            enabled = 1 if bool(int(row["enabled"] or 0)) else 0
            created_at = str(row["created_at"] or _now_utc_iso())
            updated_at = str(row["updated_at"] or created_at)
            graph = _json_loads(row["graph_json"])
            graph_map = graph if isinstance(graph, dict) else {"nodes": [], "edges": []}
            spec = _legacy_graph_to_workflow_spec(
                workflow_id=workflow_id,
                name=name,
                description=description,
                enabled=bool(enabled),
                graph=graph_map,
            )
            compiled = _compile_skeleton(spec)

            conn.execute(
                """
                INSERT OR IGNORE INTO workflows (
                    workflow_id, name, description, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (workflow_id, name, description, enabled, created_at, updated_at),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO workflow_versions (
                    version_id, workflow_id, version_num,
                    spec_json, compiled_json, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{workflow_id}:v1",
                    workflow_id,
                    1,
                    _json_dumps(spec),
                    _json_dumps(compiled),
                    created_at,
                    "legacy_migration",
                ),
            )


def _migration_3_run_steps_and_tool_calls(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "runs", "workflow_version_id", "TEXT")
    _ensure_column(conn, "runs", "sandbox_mode", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "runs", "error", "TEXT")
    _ensure_column(conn, "runs", "finished_at", "TEXT")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            node_id TEXT,
            node_type TEXT,
            started_at TEXT,
            finished_at TEXT,
            status TEXT,
            input_json TEXT,
            output_json TEXT,
            error_json TEXT,
            summary TEXT,
            step_index INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            step_id INTEGER,
            tool_name TEXT NOT NULL,
            args_json TEXT,
            result_json TEXT,
            approved_bool INTEGER,
            started_at TEXT,
            finished_at TEXT,
            error_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (step_id) REFERENCES run_steps(step_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_workflow_version_id
        ON runs (workflow_version_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_steps_run_id_step_index
        ON run_steps (run_id, step_index)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id_approval_status
        ON tool_calls (run_id, approved_bool, status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_calls_step_id
        ON tool_calls (step_id)
        """
    )


def _migration_4_run_replay_and_state_snapshots(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "runs", "parent_run_id", "TEXT")
    _ensure_column(conn, "runs", "parent_step_id", "TEXT")
    _ensure_column(conn, "runs", "forked_from_state_json", "TEXT")
    _ensure_column(conn, "runs", "fork_patch_json", "TEXT")
    _ensure_column(conn, "runs", "fork_reason", "TEXT")
    _ensure_column(conn, "runs", "resume_from_node_id", "TEXT")
    _ensure_column(conn, "runs", "mode", "TEXT NOT NULL DEFAULT 'normal'")

    _ensure_column(conn, "run_steps", "pre_state_json", "TEXT")
    _ensure_column(conn, "run_steps", "post_state_json", "TEXT")

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_parent_run_id
        ON runs (parent_run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_steps_run_id_step_id
        ON run_steps (run_id, step_id)
        """
    )
    conn.execute(
        """
        UPDATE runs
        SET mode = 'normal'
        WHERE mode IS NULL OR TRIM(mode) = ''
        """
    )
    _backfill_run_step_states(conn)


def _backfill_run_step_states(conn: sqlite3.Connection) -> None:
    run_rows = conn.execute(
        """
        SELECT run_id, payload_json
        FROM runs
        ORDER BY started_at ASC, run_id ASC
        """
    ).fetchall()

    for run_row in run_rows:
        run_id = str(run_row["run_id"] or "").strip()
        if not run_id:
            continue

        payload = _json_loads(run_row["payload_json"])
        base_state: dict[str, Any] = {}
        if isinstance(payload, dict):
            raw_input = payload.get("input")
            if isinstance(raw_input, dict):
                base_state = copy.deepcopy(raw_input)
            else:
                base_state = copy.deepcopy(payload)

        current_state = copy.deepcopy(base_state)
        step_rows = conn.execute(
            """
            SELECT step_id, node_type, status, output_json, pre_state_json, post_state_json
            FROM run_steps
            WHERE run_id = ?
            ORDER BY step_index ASC, step_id ASC
            """,
            (run_id,),
        ).fetchall()
        for step_row in step_rows:
            step_id = int(step_row["step_id"])
            existing_pre = _json_loads(step_row["pre_state_json"])
            existing_post = _json_loads(step_row["post_state_json"])
            pre_state = (
                copy.deepcopy(existing_pre)
                if isinstance(existing_pre, dict)
                else copy.deepcopy(current_state)
            )
            post_state = (
                copy.deepcopy(existing_post)
                if isinstance(existing_post, dict)
                else _derive_post_state_for_backfill(
                    pre_state=pre_state,
                    node_type=str(step_row["node_type"] or ""),
                    status=str(step_row["status"] or ""),
                    output_payload=_json_loads(step_row["output_json"]),
                )
            )
            conn.execute(
                """
                UPDATE run_steps
                SET pre_state_json = COALESCE(pre_state_json, ?),
                    post_state_json = COALESCE(post_state_json, ?)
                WHERE step_id = ?
                """,
                (_json_dumps(pre_state), _json_dumps(post_state), step_id),
            )
            current_state = copy.deepcopy(post_state)


def _derive_post_state_for_backfill(
    *,
    pre_state: dict[str, Any],
    node_type: str,
    status: str,
    output_payload: Any,
) -> dict[str, Any]:
    normalized_status = str(status or "").strip().lower()
    normalized_type = str(node_type or "").strip().lower()
    if normalized_status == "skipped" or normalized_type == "tool":
        return copy.deepcopy(pre_state)

    if isinstance(output_payload, dict):
        post_state = copy.deepcopy(pre_state)
        for key, value in output_payload.items():
            post_state[str(key)] = copy.deepcopy(value)
        return post_state
    return copy.deepcopy(pre_state)


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------

def create_run(
    run_id: str,
    kind: str,
    payload_json: str,
    started_at: str,
    *,
    workflow_version_id: Optional[str] = None,
    sandbox_mode: bool = False,
    status: str = "running",
    parent_run_id: Optional[str] = None,
    parent_step_id: Optional[str] = None,
    forked_from_state_json: Optional[str] = None,
    fork_patch_json: Optional[str] = None,
    fork_reason: Optional[str] = None,
    resume_from_node_id: Optional[str] = None,
    mode: str = "normal",
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, kind, status, payload_json, started_at,
                workflow_version_id, sandbox_mode,
                parent_run_id, parent_step_id,
                forked_from_state_json, fork_patch_json, fork_reason,
                resume_from_node_id, mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                kind,
                status,
                payload_json,
                started_at,
                workflow_version_id,
                1 if sandbox_mode else 0,
                parent_run_id,
                parent_step_id,
                forked_from_state_json,
                fork_patch_json,
                fork_reason,
                resume_from_node_id,
                str(mode or "normal"),
            ),
        )
        conn.commit()


def finish_run(
    run_id: str,
    status: str,
    ended_at: str,
    result_json: str,
    *,
    error: Optional[str] = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?,
                ended_at = ?,
                finished_at = ?,
                result_json = ?,
                error = ?
            WHERE run_id = ?
            """,
            (status, ended_at, ended_at, result_json, error, run_id),
        )
        conn.commit()


def add_step(
    run_id: str,
    step_index: int,
    name: str,
    input_json: str,
    output_json: str,
    summary: str,
    status: str,
    started_at: str,
    ended_at: str,
    *,
    node_id: Optional[str] = None,
    node_type: Optional[str] = None,
    error_json: Optional[str] = None,
    pre_state_json: Optional[str] = None,
    post_state_json: Optional[str] = None,
) -> int:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO steps (
                run_id, step_index, name, status,
                input_json, output_json, summary, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step_index,
                name,
                status,
                input_json,
                output_json,
                summary,
                started_at,
                ended_at,
            ),
        )

        node_id_final = node_id if node_id is not None else _derive_node_id(name)
        node_type_final = node_type if node_type is not None else _derive_node_type(name)
        cursor = conn.execute(
            """
            INSERT INTO run_steps (
                run_id, node_id, node_type, started_at, finished_at,
                status, input_json, output_json, error_json, summary, step_index,
                pre_state_json, post_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                node_id_final,
                node_type_final,
                started_at,
                ended_at,
                status,
                input_json,
                output_json,
                error_json,
                summary,
                step_index,
                pre_state_json,
                post_state_json,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_run(run_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, kind, status, payload_json, result_json,
                   started_at, ended_at, finished_at,
                   workflow_version_id, sandbox_mode, error,
                   parent_run_id, parent_step_id,
                   forked_from_state_json, fork_patch_json, fork_reason,
                   resume_from_node_id, mode
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["sandbox_mode"] = bool(int(payload.get("sandbox_mode") or 0))
        payload["mode"] = str(payload.get("mode") or "normal")
        return payload


def list_steps(run_id: str) -> list[dict]:
    with _connect() as conn:
        run_steps_rows = conn.execute(
            """
            SELECT step_id, run_id, step_index,
                   node_id, node_type, status,
                   input_json, output_json, error_json,
                   summary, started_at, finished_at,
                   pre_state_json, post_state_json
            FROM run_steps
            WHERE run_id = ?
            ORDER BY step_index ASC, step_id ASC
            """,
            (run_id,),
        ).fetchall()
        if run_steps_rows:
            output: list[dict] = []
            for row in run_steps_rows:
                item = dict(row)
                output.append(
                    {
                        "id": item.get("step_id"),
                        "step_id": item.get("step_id"),
                        "run_id": item.get("run_id"),
                        "step_index": item.get("step_index"),
                        "name": _derive_step_name(
                            node_id=str(item.get("node_id") or ""),
                            node_type=str(item.get("node_type") or ""),
                        ),
                        "node_id": item.get("node_id"),
                        "node_type": item.get("node_type"),
                        "status": item.get("status"),
                        "input_json": item.get("input_json"),
                        "output_json": item.get("output_json"),
                        "error_json": item.get("error_json"),
                        "pre_state_json": item.get("pre_state_json"),
                        "post_state_json": item.get("post_state_json"),
                        "summary": item.get("summary"),
                        "started_at": item.get("started_at"),
                        "ended_at": item.get("finished_at"),
                        "finished_at": item.get("finished_at"),
                        "external_step_id": format_step_id(item.get("step_id")),
                    }
                )
            return output

        rows = conn.execute(
            """
            SELECT id, run_id, step_index, name, status,
                   input_json, output_json, summary, started_at, ended_at
            FROM steps
            WHERE run_id = ?
            ORDER BY step_index ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            {
                "id": row.get("id"),
                "step_id": row.get("id"),
                "run_id": row.get("run_id"),
                "step_index": row.get("step_index"),
                "name": row.get("name"),
                "node_id": _derive_node_id(str(row.get("name") or "")),
                "node_type": _derive_node_type(str(row.get("name") or "")),
                "status": row.get("status"),
                "input_json": row.get("input_json"),
                "output_json": row.get("output_json"),
                "error_json": None,
                "pre_state_json": None,
                "post_state_json": None,
                "summary": row.get("summary"),
                "started_at": row.get("started_at"),
                "ended_at": row.get("ended_at"),
                "finished_at": row.get("ended_at"),
                "external_step_id": format_step_id(row.get("id")),
            }
            for row in rows
        ]


def read_run(run_id: str) -> Optional[dict]:
    return get_run(run_id)


def read_steps(run_id: str) -> list[dict]:
    return list_steps(run_id)


def format_step_id(step_id: Any) -> str:
    try:
        return f"step_{int(step_id)}"
    except Exception:
        return ""


def parse_step_id(step_id: Any) -> Optional[int]:
    if step_id is None:
        return None
    if isinstance(step_id, int):
        return int(step_id)
    raw = str(step_id).strip()
    if not raw:
        return None
    if raw.startswith("step_"):
        raw = raw[len("step_") :]
    try:
        return int(raw)
    except ValueError:
        return None


def get_step(run_id: str, step_id: Any) -> Optional[dict]:
    internal_step_id = parse_step_id(step_id)
    if internal_step_id is None:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT step_id, run_id, step_index,
                   node_id, node_type, status,
                   input_json, output_json, error_json,
                   summary, started_at, finished_at,
                   pre_state_json, post_state_json
            FROM run_steps
            WHERE run_id = ?
              AND step_id = ?
            LIMIT 1
            """,
            (run_id, internal_step_id),
        ).fetchone()
        if row is not None:
            item = dict(row)
            return {
                "id": item.get("step_id"),
                "step_id": item.get("step_id"),
                "external_step_id": format_step_id(item.get("step_id")),
                "run_id": item.get("run_id"),
                "step_index": item.get("step_index"),
                "name": _derive_step_name(
                    node_id=str(item.get("node_id") or ""),
                    node_type=str(item.get("node_type") or ""),
                ),
                "node_id": item.get("node_id"),
                "node_type": item.get("node_type"),
                "status": item.get("status"),
                "input_json": item.get("input_json"),
                "output_json": item.get("output_json"),
                "error_json": item.get("error_json"),
                "pre_state_json": item.get("pre_state_json"),
                "post_state_json": item.get("post_state_json"),
                "summary": item.get("summary"),
                "started_at": item.get("started_at"),
                "ended_at": item.get("finished_at"),
                "finished_at": item.get("finished_at"),
            }

        legacy_row = conn.execute(
            """
            SELECT id, run_id, step_index, name, status,
                   input_json, output_json, summary, started_at, ended_at
            FROM steps
            WHERE run_id = ?
              AND id = ?
            LIMIT 1
            """,
            (run_id, internal_step_id),
        ).fetchone()
        if legacy_row is None:
            return None
        item = dict(legacy_row)
        return {
            "id": item.get("id"),
            "step_id": item.get("id"),
            "external_step_id": format_step_id(item.get("id")),
            "run_id": item.get("run_id"),
            "step_index": item.get("step_index"),
            "name": item.get("name"),
            "node_id": _derive_node_id(str(item.get("name") or "")),
            "node_type": _derive_node_type(str(item.get("name") or "")),
            "status": item.get("status"),
            "input_json": item.get("input_json"),
            "output_json": item.get("output_json"),
            "error_json": None,
            "pre_state_json": None,
            "post_state_json": None,
            "summary": item.get("summary"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
            "finished_at": item.get("ended_at"),
        }


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


def add_state_snapshot(run_id: str, step_index: int, timestamp: str, state_json: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO state_snapshots (run_id, step_index, timestamp, state_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, step_index, timestamp, state_json),
        )
        conn.commit()


def list_tool_calls_for_step(run_id: str, step_id: Any) -> list[dict]:
    internal_step_id = parse_step_id(step_id)
    if internal_step_id is None:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT tool_call_id, run_id, step_id, tool_name,
                   args_json, result_json, approved_bool,
                   started_at, finished_at, error_json, status
            FROM tool_calls
            WHERE run_id = ?
              AND step_id = ?
            ORDER BY started_at ASC, tool_call_id ASC
            """,
            (run_id, internal_step_id),
        ).fetchall()
        return [_tool_call_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Tool call persistence for sandbox / inspector
# ---------------------------------------------------------------------------

def create_tool_call(
    *,
    run_id: str,
    step_id: Optional[int],
    tool_name: str,
    args_json: str,
    started_at: str,
    approved_bool: Optional[bool] = None,
    status: str = "pending",
) -> dict:
    tool_call_id = str(uuid.uuid4())
    approved_value = (
        None if approved_bool is None else (1 if bool(approved_bool) else 0)
    )
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_call_id, run_id, step_id, tool_name,
                args_json, approved_bool, started_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tool_call_id,
                run_id,
                step_id,
                tool_name,
                args_json,
                approved_value,
                started_at,
                status,
            ),
        )
        conn.commit()
    payload = get_tool_call(tool_call_id)
    return payload or {
        "tool_call_id": tool_call_id,
        "run_id": run_id,
        "step_id": step_id,
        "tool_name": tool_name,
        "args_json": args_json,
        "approved_bool": approved_bool,
        "started_at": started_at,
        "status": status,
    }


def update_tool_call_approval(tool_call_id: str, approved: bool) -> bool:
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE tool_calls
            SET approved_bool = ?,
                status = CASE WHEN ? = 1 THEN 'approved' ELSE 'rejected' END
            WHERE tool_call_id = ?
            """,
            (1 if approved else 0, 1 if approved else 0, tool_call_id),
        )
        conn.commit()
        return result.rowcount > 0


def finish_tool_call(
    *,
    tool_call_id: str,
    status: str,
    finished_at: str,
    result_json: Optional[str] = None,
    error_json: Optional[str] = None,
) -> bool:
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE tool_calls
            SET status = ?,
                finished_at = ?,
                result_json = ?,
                error_json = ?
            WHERE tool_call_id = ?
            """,
            (status, finished_at, result_json, error_json, tool_call_id),
        )
        conn.commit()
        return result.rowcount > 0


def list_pending_tool_calls(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT tool_call_id, run_id, step_id, tool_name,
                   args_json, result_json, approved_bool,
                   started_at, finished_at, error_json, status
            FROM tool_calls
            WHERE run_id = ?
              AND status = 'pending'
            ORDER BY started_at ASC, tool_call_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_tool_call_row(row) for row in rows]


def list_tool_calls(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT tool_call_id, run_id, step_id, tool_name,
                   args_json, result_json, approved_bool,
                   started_at, finished_at, error_json, status
            FROM tool_calls
            WHERE run_id = ?
            ORDER BY started_at ASC, tool_call_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_tool_call_row(row) for row in rows]


def get_tool_call(tool_call_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT tool_call_id, run_id, step_id, tool_name,
                   args_json, result_json, approved_bool,
                   started_at, finished_at, error_json, status
            FROM tool_calls
            WHERE tool_call_id = ?
            """,
            (tool_call_id,),
        ).fetchone()
        if row is None:
            return None
        return _tool_call_row(row)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

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
            ) VALUES (?, ?, ?, ?, ?, ?)
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


# ---------------------------------------------------------------------------
# Tool registry persistence
# ---------------------------------------------------------------------------

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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _json_dumps(
                    tool.get("config") if isinstance(tool.get("config"), dict) else {}
                ),
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
    updated_at = _now_utc_iso()
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


# ---------------------------------------------------------------------------
# Workflow versioning persistence
# ---------------------------------------------------------------------------

def create_or_update_workflow(
    *,
    workflow_id: str,
    name: str,
    description: str,
    enabled: bool,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> None:
    created = str(created_at or _now_utc_iso())
    updated = str(updated_at or created)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workflows (
                workflow_id, name, description, enabled, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (workflow_id, name, description, 1 if enabled else 0, created, updated),
        )
        conn.commit()


def create_workflow_version(
    *,
    workflow_id: str,
    name: str,
    description: str,
    enabled: bool,
    spec: dict,
    compiled: dict,
    created_by: str,
) -> dict:
    now = _now_utc_iso()
    spec_map = dict(spec or {})
    compiled_map = dict(compiled or {})
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO workflows (
                workflow_id, name, description, enabled, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (workflow_id, name, description, 1 if enabled else 0, now, now),
        )

        version_row = conn.execute(
            """
            SELECT COALESCE(MAX(version_num), 0) AS max_version
            FROM workflow_versions
            WHERE workflow_id = ?
            """,
            (workflow_id,),
        ).fetchone()
        next_version = int(version_row["max_version"] or 0) + 1
        version_id = str(uuid.uuid4())

        conn.execute(
            """
            INSERT INTO workflow_versions (
                version_id, workflow_id, version_num,
                spec_json, compiled_json, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                workflow_id,
                next_version,
                _json_dumps(spec_map),
                _json_dumps(compiled_map),
                now,
                created_by,
            ),
        )
        conn.commit()

    payload = get_workflow_version(version_id)
    if payload is None:
        raise RuntimeError("Workflow version was not persisted.")
    return payload


def get_workflow(workflow_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT w.workflow_id, w.name, w.description, w.enabled,
                   w.created_at, w.updated_at,
                   v.version_id, v.version_num, v.spec_json, v.compiled_json,
                   v.created_at AS version_created_at,
                   v.created_by
            FROM workflows w
            LEFT JOIN workflow_versions v
              ON v.version_id = (
                   SELECT vv.version_id
                   FROM workflow_versions vv
                   WHERE vv.workflow_id = w.workflow_id
                   ORDER BY vv.version_num DESC
                   LIMIT 1
                 )
            WHERE w.workflow_id = ?
            """,
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return _workflow_row_to_definition(row)


def list_workflows() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT w.workflow_id, w.name, w.description, w.enabled,
                   w.created_at, w.updated_at,
                   v.version_id, v.version_num, v.spec_json, v.compiled_json,
                   v.created_at AS version_created_at,
                   v.created_by
            FROM workflows w
            LEFT JOIN workflow_versions v
              ON v.version_id = (
                   SELECT vv.version_id
                   FROM workflow_versions vv
                   WHERE vv.workflow_id = w.workflow_id
                   ORDER BY vv.version_num DESC
                   LIMIT 1
                 )
            ORDER BY w.name COLLATE NOCASE ASC, w.workflow_id ASC
            """
        ).fetchall()
        return [_workflow_row_to_definition(row) for row in rows]


def get_workflow_version(version_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT v.version_id, v.workflow_id, v.version_num,
                   v.spec_json, v.compiled_json,
                   v.created_at, v.created_by,
                   w.name, w.description, w.enabled,
                   w.created_at AS workflow_created_at,
                   w.updated_at AS workflow_updated_at
            FROM workflow_versions v
            JOIN workflows w
              ON w.workflow_id = v.workflow_id
            WHERE v.version_id = ?
            """,
            (version_id,),
        ).fetchone()
        if row is None:
            return None
        return _workflow_version_row_to_payload(row)


def get_latest_workflow_version(workflow_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT v.version_id, v.workflow_id, v.version_num,
                   v.spec_json, v.compiled_json,
                   v.created_at, v.created_by,
                   w.name, w.description, w.enabled,
                   w.created_at AS workflow_created_at,
                   w.updated_at AS workflow_updated_at
            FROM workflow_versions v
            JOIN workflows w
              ON w.workflow_id = v.workflow_id
            WHERE v.workflow_id = ?
            ORDER BY v.version_num DESC
            LIMIT 1
            """,
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return _workflow_version_row_to_payload(row)


def list_workflow_versions(workflow_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT v.version_id, v.workflow_id, v.version_num,
                   v.spec_json, v.compiled_json,
                   v.created_at, v.created_by,
                   w.name, w.description, w.enabled,
                   w.created_at AS workflow_created_at,
                   w.updated_at AS workflow_updated_at
            FROM workflow_versions v
            JOIN workflows w
              ON w.workflow_id = v.workflow_id
            WHERE v.workflow_id = ?
            ORDER BY v.version_num DESC
            """,
            (workflow_id,),
        ).fetchall()
        return [_workflow_version_row_to_payload(row) for row in rows]


def set_workflow_enabled(workflow_id: str, enabled: bool) -> bool:
    updated_at = _now_utc_iso()
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE workflows
            SET enabled = ?,
                updated_at = ?
            WHERE workflow_id = ?
            """,
            (1 if enabled else 0, updated_at, workflow_id),
        )
        conn.commit()
        return result.rowcount > 0


# Backward-compatible helper for existing builder endpoints.
def upsert_workflow(workflow: dict) -> None:
    workflow_id = str(workflow.get("id") or workflow.get("workflow_id") or "").strip()
    if not workflow_id:
        raise ValueError("Workflow id is required.")

    name = str(workflow.get("name") or workflow_id)
    description = str(workflow.get("description") or "")
    enabled = bool(workflow.get("enabled", True))
    raw_graph = workflow.get("graph")
    graph = raw_graph if isinstance(raw_graph, dict) else {"nodes": [], "edges": []}

    spec = _legacy_graph_to_workflow_spec(
        workflow_id=workflow_id,
        name=name,
        description=description,
        enabled=enabled,
        graph=graph,
    )
    compiled = _compile_skeleton(spec)

    create_workflow_version(
        workflow_id=workflow_id,
        name=name,
        description=description,
        enabled=enabled,
        spec=spec,
        compiled=compiled,
        created_by="legacy_builder",
    )


# ---------------------------------------------------------------------------
# Row conversion helpers
# ---------------------------------------------------------------------------

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
    workflow_id = str(row["workflow_id"] or "")
    spec = _json_loads(row["spec_json"])
    spec_map = spec if isinstance(spec, dict) else {
        "workflow_id": workflow_id,
        "name": str(row["name"] or workflow_id),
        "description": str(row["description"] or ""),
        "allow_cycles": False,
        "state_schema": [],
        "nodes": [],
        "edges": [],
    }

    compiled = _json_loads(row["compiled_json"])
    compiled_map = compiled if isinstance(compiled, dict) else {}
    runtime_graph = compiled_map.get("runtime_graph")
    if not isinstance(runtime_graph, dict):
        runtime_graph = _spec_to_runtime_graph(spec_map)

    return {
        "id": workflow_id,
        "workflow_id": workflow_id,
        "name": str(row["name"] or workflow_id),
        "title": str(row["name"] or workflow_id),
        "description": str(row["description"] or ""),
        "enabled": bool(int(row["enabled"] or 0)),
        "source": "custom",
        "type": "custom",
        "version_id": str(row["version_id"] or ""),
        "version_num": int(row["version_num"] or 0),
        "graph": runtime_graph,
        "spec": spec_map,
        "compiled": compiled_map,
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def _workflow_version_row_to_payload(row: sqlite3.Row) -> dict:
    spec = _json_loads(row["spec_json"])
    compiled = _json_loads(row["compiled_json"])
    return {
        "version_id": str(row["version_id"]),
        "workflow_id": str(row["workflow_id"]),
        "version_num": int(row["version_num"] or 0),
        "spec": spec if isinstance(spec, dict) else {},
        "compiled": compiled if isinstance(compiled, dict) else {},
        "created_at": str(row["created_at"] or ""),
        "created_by": str(row["created_by"] or ""),
        "workflow": {
            "workflow_id": str(row["workflow_id"]),
            "name": str(row["name"] or row["workflow_id"]),
            "description": str(row["description"] or ""),
            "enabled": bool(int(row["enabled"] or 0)),
            "created_at": str(row["workflow_created_at"] or ""),
            "updated_at": str(row["workflow_updated_at"] or ""),
        },
    }


def _tool_call_row(row: sqlite3.Row) -> dict:
    approved_raw = row["approved_bool"]
    approved: Optional[bool]
    if approved_raw is None:
        approved = None
    else:
        approved = bool(int(approved_raw))

    return {
        "tool_call_id": str(row["tool_call_id"]),
        "run_id": str(row["run_id"]),
        "step_id": int(row["step_id"]) if row["step_id"] is not None else None,
        "tool_name": str(row["tool_name"] or ""),
        "args_json": str(row["args_json"] or "{}"),
        "result_json": str(row["result_json"] or "null"),
        "approved_bool": approved,
        "started_at": str(row["started_at"] or ""),
        "finished_at": str(row["finished_at"] or ""),
        "error_json": str(row["error_json"] or "null"),
        "status": str(row["status"] or "pending"),
    }


# ---------------------------------------------------------------------------
# Workflow normalization helpers for migration/compat
# ---------------------------------------------------------------------------

def _legacy_graph_to_workflow_spec(
    *,
    workflow_id: str,
    name: str,
    description: str,
    enabled: bool,
    graph: dict,
) -> dict:
    raw_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    raw_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

    start_nodes: set[str] = set()
    nodes: list[dict] = []
    state_types: dict[str, str] = {}

    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "").strip()
        if not node_id:
            continue
        node_type = str(raw.get("type") or "").strip().lower()
        config = dict(raw.get("config") or {})

        if node_type == "start":
            start_nodes.add(node_id)
            continue
        mapped_type = node_type
        if node_type == "condition":
            mapped_type = "conditional"
        elif node_type == "end":
            mapped_type = "finalize"

        reads: list[str] = []
        writes: list[str] = []

        if mapped_type == "llm":
            out_key = str(config.get("output_key") or f"{node_id}_output").strip() or f"{node_id}_output"
            writes.append(out_key)
            state_types[out_key] = "string"
        elif mapped_type == "tool":
            out_key = str(config.get("output_key") or f"{node_id}_output").strip() or f"{node_id}_output"
            writes.append(out_key)
            state_types[out_key] = "json"
            if "tool_id" in config and "tool_name" not in config:
                config["tool_name"] = str(config.get("tool_id") or "")
        elif mapped_type == "conditional":
            field = str(config.get("field") or "").strip()
            if field:
                reads.append(field)
            expr = _legacy_condition_expression(config)
            if expr:
                config["expression"] = expr
            result_key = f"{node_id}.result"
            writes.append(result_key)
            state_types[result_key] = "bool"
        elif mapped_type == "finalize":
            output_key = str(config.get("output_key") or "final_output").strip() or "final_output"
            writes.append(output_key)
            state_types[output_key] = "string"

        if mapped_type not in {"llm", "tool", "conditional", "verify", "finalize"}:
            continue

        nodes.append(
            {
                "id": node_id,
                "type": mapped_type,
                "label": str(raw.get("label") or node_id),
                "reads": sorted(set(reads)),
                "writes": sorted(set(writes)),
                "config": config,
            }
        )

    edges: list[dict] = []
    edge_index = 1
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        from_node = str(raw.get("from") or raw.get("from_node") or "").strip()
        to_node = str(raw.get("to") or "").strip()
        if not from_node or not to_node:
            continue
        if from_node in start_nodes:
            # Drop explicit start node and let topological roots become entry points.
            continue
        label = str(raw.get("label") or raw.get("condition") or "always").strip() or "always"
        edges.append(
            {
                "id": f"edge_{edge_index}",
                "from": from_node,
                "to": to_node,
                "label": label,
            }
        )
        edge_index += 1

    state_schema = [
        {
            "key": key,
            "type": state_type,
            "required": False,
            "description": "",
        }
        for key, state_type in sorted(state_types.items())
    ]

    return {
        "workflow_id": workflow_id,
        "name": name,
        "description": description,
        "allow_cycles": False,
        "state_schema": state_schema,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "legacy_imported": True,
            "enabled": enabled,
        },
    }


def _legacy_condition_expression(config: dict) -> str:
    field = str(config.get("field") or "").strip()
    operator = str(config.get("operator") or "").strip().lower()
    value = config.get("value")
    if not field or not operator:
        return ""

    if operator == "equals":
        return f"{field} == {repr(value)}"
    if operator == "contains":
        return f"{repr(value)} in {field}"
    if operator == "gt":
        return f"{field} > {repr(value)}"
    if operator == "lt":
        return f"{field} < {repr(value)}"
    if operator == "exists":
        return f"{field} is not None"
    if operator == "not_exists":
        return f"{field} is None"
    if operator == "in":
        return f"{field} in {repr(value)}"
    return ""


def _compile_skeleton(spec: dict) -> dict:
    nodes = spec.get("nodes") if isinstance(spec.get("nodes"), list) else []
    edges = spec.get("edges") if isinstance(spec.get("edges"), list) else []

    normalized_nodes = sorted(
        [dict(item) for item in nodes if isinstance(item, dict)],
        key=lambda item: str(item.get("id") or ""),
    )
    normalized_edges = sorted(
        [dict(item) for item in edges if isinstance(item, dict)],
        key=lambda item: (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("label") or ""),
        ),
    )

    adjacency: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    for node in normalized_nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        adjacency[node_id] = []
        indegree[node_id] = 0

    for edge in normalized_edges:
        from_node = str(edge.get("from") or "")
        to_node = str(edge.get("to") or "")
        if from_node in adjacency and to_node in indegree:
            adjacency[from_node].append(to_node)
            indegree[to_node] = int(indegree.get(to_node, 0)) + 1

    entry_nodes = sorted([node_id for node_id, count in indegree.items() if count == 0])

    normalized = {
        "workflow_spec": spec,
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "entry_nodes": entry_nodes,
        "adjacency": adjacency,
    }
    digest = _hash_json(normalized)

    return {
        "hash": digest,
        "normalized": normalized,
        "runtime_graph": _spec_to_runtime_graph(spec),
    }


def _spec_to_runtime_graph(spec: dict) -> dict:
    raw_nodes = spec.get("nodes") if isinstance(spec.get("nodes"), list) else []
    raw_edges = spec.get("edges") if isinstance(spec.get("edges"), list) else []

    runtime_nodes: list[dict] = [{"id": "start", "type": "start", "config": {}}]
    node_ids: list[str] = []

    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "").strip()
        node_type = str(raw.get("type") or "").strip().lower()
        if not node_id or not node_type:
            continue
        node_ids.append(node_id)

        config = dict(raw.get("config") or {})
        if node_type == "conditional":
            runtime_type = "conditional"
        elif node_type == "finalize":
            runtime_type = "finalize"
        elif node_type == "verify":
            runtime_type = "verify"
        else:
            runtime_type = node_type

        runtime_nodes.append(
            {
                "id": node_id,
                "type": runtime_type,
                "config": config,
                "reads": list(raw.get("reads") or []),
                "writes": list(raw.get("writes") or []),
            }
        )

    runtime_nodes.append({"id": "end", "type": "end", "config": {}})

    edges: list[dict] = []
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}

    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        from_node = str(raw.get("from") or "").strip()
        to_node = str(raw.get("to") or "").strip()
        label = str(raw.get("label") or "always").strip() or "always"
        if from_node not in indegree or to_node not in indegree:
            continue
        edges.append(
            {
                "from": from_node,
                "to": to_node,
                "condition": label,
            }
        )
        indegree[to_node] = int(indegree.get(to_node, 0)) + 1

    entry_nodes = [node_id for node_id, count in indegree.items() if count == 0]
    for node_id in sorted(entry_nodes):
        edges.append({"from": "start", "to": node_id, "condition": "always"})

    finalize_nodes = {
        str(node.get("id") or "")
        for node in runtime_nodes
        if str(node.get("type") or "") == "finalize"
    }
    outgoing: dict[str, int] = {}
    for edge in edges:
        from_node = str(edge.get("from") or "")
        outgoing[from_node] = int(outgoing.get(from_node, 0)) + 1

    for node_id in sorted(finalize_nodes):
        if int(outgoing.get(node_id, 0)) == 0:
            edges.append({"from": node_id, "to": "end", "condition": "always"})

    return {
        "nodes": runtime_nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not _table_exists(conn, table_name):
        return False
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for row in rows:
        if str(row["name"] or "") == column_name:
            return True
    return False


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    ddl_fragment: str,
) -> None:
    if _column_exists(conn, table_name, column_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl_fragment}")


def _derive_node_id(step_name: str) -> str:
    name = str(step_name or "").strip()
    if name.startswith("node."):
        return name.replace("node.", "", 1)
    if name.startswith("tool."):
        return name.replace("tool.", "", 1)
    return name


def _derive_node_type(step_name: str) -> str:
    name = str(step_name or "").strip()
    if name.startswith("tool."):
        return "tool"
    if name.startswith("node."):
        return "node"
    return "system"


def _derive_step_name(*, node_id: str, node_type: str) -> str:
    normalized_type = str(node_type or "").strip().lower()
    normalized_id = str(node_id or "").strip()
    if normalized_type == "tool" and normalized_id:
        return f"tool.{normalized_id}"
    if normalized_type and normalized_id:
        return f"node.{normalized_id}"
    if normalized_id:
        return normalized_id
    return "step"


def _json_loads(raw: Optional[str]) -> object:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _json_dumps(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=True)


def _hash_json(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    import hashlib

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    if not _DB_PATH:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
