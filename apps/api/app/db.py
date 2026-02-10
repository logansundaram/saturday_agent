from __future__ import annotations

import sqlite3
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


def _connect() -> sqlite3.Connection:
    if not _DB_PATH:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
