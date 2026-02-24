from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app import db
from app.workflows.validation import validate_and_compile_workflow_spec


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_spec() -> dict:
    return {
        "workflow_id": "workflow.custom.test_validation",
        "name": "Validation Test Workflow",
        "description": "",
        "allow_cycles": False,
        "state_schema": [
            {"key": "answer", "type": "string", "description": "", "required": False},
            {"key": "artifacts", "type": "json", "description": "", "required": False},
            {"key": "verify_ok", "type": "bool", "description": "", "required": False},
        ],
        "nodes": [
            {
                "id": "tool_1",
                "type": "tool",
                "label": "Tool",
                "reads": ["task"],
                "writes": ["artifacts"],
                "config": {
                    "tool_name": "tool.custom.echo",
                    "args_map": {"query": "task"},
                    "output_key": "artifacts",
                },
                "position": {"x": 100, "y": 100},
            },
            {
                "id": "finalize_1",
                "type": "finalize",
                "label": "Finalize",
                "reads": ["task"],
                "writes": ["answer"],
                "config": {"response_template": "Done {{task}}", "output_key": "answer"},
                "position": {"x": 300, "y": 100},
            },
        ],
        "edges": [
            {
                "id": "edge_1",
                "from": "tool_1",
                "to": "finalize_1",
                "label": "always",
            }
        ],
        "metadata": {"enabled": True},
    }


class WorkflowValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool_registry = [{"id": "tool.custom.echo", "name": "Echo", "enabled": True}]

    def test_valid_spec_compiles(self) -> None:
        result = validate_and_compile_workflow_spec(
            workflow_spec=_base_spec(),
            tool_registry=self.tool_registry,
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.compiled.get("hash"))
        runtime_graph = result.compiled.get("runtime_graph")
        self.assertIsInstance(runtime_graph, dict)
        self.assertGreaterEqual(len(runtime_graph.get("nodes") or []), 2)

    def test_cycle_detection_respects_allow_cycles(self) -> None:
        spec = _base_spec()
        spec["edges"].append(
            {"id": "edge_2", "from": "finalize_1", "to": "tool_1", "label": "always"}
        )

        blocked = validate_and_compile_workflow_spec(
            workflow_spec=spec,
            tool_registry=self.tool_registry,
        )
        blocked_codes = [item.code for item in blocked.diagnostics]
        self.assertIn("CYCLE_DETECTED", blocked_codes)
        self.assertFalse(blocked.valid)

        spec["allow_cycles"] = True
        allowed = validate_and_compile_workflow_spec(
            workflow_spec=spec,
            tool_registry=self.tool_registry,
        )
        allowed_cycle = [
            item for item in allowed.diagnostics if item.code == "CYCLE_DETECTED"
        ]
        self.assertTrue(allowed_cycle)
        self.assertTrue(all(item.severity != "error" for item in allowed_cycle))

    def test_tool_must_exist(self) -> None:
        spec = _base_spec()
        spec["nodes"][0]["config"]["tool_name"] = "tool.custom.missing"

        result = validate_and_compile_workflow_spec(
            workflow_spec=spec,
            tool_registry=self.tool_registry,
        )
        codes = [item.code for item in result.diagnostics]
        self.assertIn("TOOL_NOT_FOUND", codes)
        self.assertFalse(result.valid)

    def test_write_key_must_be_declared(self) -> None:
        spec = _base_spec()
        spec["nodes"][0]["writes"] = ["unknown_key"]

        result = validate_and_compile_workflow_spec(
            workflow_spec=spec,
            tool_registry=self.tool_registry,
        )
        codes = [item.code for item in result.diagnostics]
        self.assertIn("STATE_KEY_WRITE_UNDECLARED", codes)
        self.assertFalse(result.valid)

    def test_conditional_expression_safety(self) -> None:
        spec = _base_spec()
        spec["nodes"] = [
            {
                "id": "conditional_1",
                "type": "conditional",
                "label": "Conditional",
                "reads": ["task"],
                "writes": [],
                "config": {"expression": "__import__('os').system('echo bad')"},
                "position": {"x": 100, "y": 100},
            },
            spec["nodes"][1],
        ]
        spec["edges"] = [
            {
                "id": "edge_1",
                "from": "conditional_1",
                "to": "finalize_1",
                "label": "true",
            },
            {
                "id": "edge_2",
                "from": "conditional_1",
                "to": "finalize_1",
                "label": "false",
            },
        ]

        result = validate_and_compile_workflow_spec(
            workflow_spec=spec,
            tool_registry=self.tool_registry,
        )
        codes = [item.code for item in result.diagnostics]
        self.assertIn("CONDITIONAL_EXPRESSION_INVALID", codes)
        self.assertFalse(result.valid)


class WorkflowMigrationTests(unittest.TestCase):
    def test_legacy_workflow_auto_migrates_to_version_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.sqlite"
            now = _now_iso()
            legacy_graph = {
                "nodes": [
                    {"id": "start", "type": "start", "config": {}},
                    {"id": "end", "type": "end", "config": {"response_template": "{{task}}"}},
                ],
                "edges": [{"from": "start", "to": "end", "condition": "always"}],
            }

            conn = sqlite3.connect(str(db_path))
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA user_version = 1")
                conn.execute(
                    """
                    CREATE TABLE workflows (
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
                    CREATE TABLE runs (
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
                    INSERT INTO workflows (
                        id, name, description, enabled, graph_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "workflow.custom.legacy_case",
                        "Legacy Workflow",
                        "Migrated",
                        1,
                        json.dumps(legacy_graph),
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            db.init_db(str(db_path))
            latest = db.get_latest_workflow_version("workflow.custom.legacy_case")
            self.assertIsNotNone(latest)
            if latest is None:
                return
            self.assertEqual(int(latest.get("version_num") or 0), 1)
            spec = latest.get("spec") if isinstance(latest.get("spec"), dict) else {}
            self.assertEqual(str(spec.get("workflow_id") or ""), "workflow.custom.legacy_case")


if __name__ == "__main__":
    unittest.main()
