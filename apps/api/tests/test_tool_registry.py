from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from app import db, graph

try:
    from fastapi.testclient import TestClient
    from app import main as main_app
    from saturday_agent.tools.langgraph_adapter import build_langgraph_tool_adapter
    from saturday_agent.tools.registry import ToolRegistry

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency/environment gated
    TestClient = None  # type: ignore[assignment]
    main_app = None  # type: ignore[assignment]
    build_langgraph_tool_adapter = None  # type: ignore[assignment]
    ToolRegistry = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_record(tool_id: str) -> dict:
    now = _now_iso()
    return {
        "id": tool_id,
        "name": "Echo Tool",
        "kind": "local",
        "type": "python",
        "description": "Echoes the provided query.",
        "enabled": True,
        "config": {
            "code": (
                "def run(input, context):\n"
                "    return {'echo': str(input.get('query', ''))}\n"
            ),
            "timeout_ms": 1000,
            "allowed_imports": ["json"],
        },
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
        },
        "created_at": now,
        "updated_at": now,
    }


@unittest.skipIf(_IMPORT_ERROR is not None, f"Tool registry deps unavailable: {_IMPORT_ERROR}")
class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tools.sqlite"
        if main_app is not None:
            main_app.DB_PATH = self.db_path
        db.init_db(str(self.db_path))
        self.client = TestClient(main_app.app) if TestClient is not None and main_app is not None else None

    def tearDown(self) -> None:
        if self.client is not None:
            self.client.close()
        self.temp_dir.cleanup()

    def test_upsert_tool_creates_versions_and_enable_toggle_versions(self) -> None:
        db.upsert_tool(_tool_record("tool.custom.echo"), created_by="tests")

        payload = db.get_tool("tool.custom.echo")
        self.assertIsNotNone(payload)
        if payload is None:
            return
        self.assertEqual(int(payload.get("version") or 0), 1)

        latest_version = db.get_latest_tool_version("tool.custom.echo")
        self.assertIsNotNone(latest_version)
        if latest_version is None:
            return
        self.assertEqual(int(latest_version.get("version_num") or 0), 1)

        updated = db.set_tool_enabled("tool.custom.echo", False, created_by="tests")
        self.assertTrue(updated)

        refreshed = db.get_tool("tool.custom.echo")
        self.assertIsNotNone(refreshed)
        if refreshed is None:
            return
        self.assertFalse(bool(refreshed.get("enabled")))
        self.assertEqual(int(refreshed.get("version") or 0), 2)

        versions = db.list_tool_versions("tool.custom.echo")
        self.assertEqual([int(item.get("version_num") or 0) for item in versions], [2, 1])

    def test_legacy_tool_row_is_backfilled_into_tool_versions(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                INSERT INTO tools (
                    id, name, kind, type, description, enabled,
                    config_json, input_schema_json, output_schema_json,
                    created_at, updated_at, deleted_at, current_version_num
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "tool.custom.legacy",
                    "Legacy Tool",
                    "local",
                    "python",
                    "Legacy row without version history",
                    1,
                    "{}",
                    '{"type":"object"}',
                    '{"type":"object"}',
                    _now_iso(),
                    _now_iso(),
                    None,
                    0,
                ),
            )
            conn.commit()

        listed = db.list_tools()
        legacy = next(
            (item for item in listed if str(item.get("id") or "") == "tool.custom.legacy"),
            None,
        )
        self.assertIsNotNone(legacy)
        if legacy is None:
            return
        self.assertEqual(int(legacy.get("version") or 0), 1)

        latest_version = db.get_latest_tool_version("tool.custom.legacy")
        self.assertIsNotNone(latest_version)
        if latest_version is None:
            return
        self.assertEqual(int(latest_version.get("version_num") or 0), 1)

    def test_adapter_validates_sync_and_async_handlers(self) -> None:
        spec = {
            "id": "tool.custom.adapter",
            "tool_id": "tool.custom.adapter",
            "name": "Adapter Tool",
            "description": "Adapter validation test",
            "type": "python",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {"echo": {"type": "string"}},
                "required": ["echo"],
            },
        }

        sync_adapter = build_langgraph_tool_adapter(
            tool_spec=spec,
            handler=lambda tool_input, context: {"echo": str(tool_input.get("query") or "")},
        )
        ok_result = sync_adapter.invoke({"query": "hello"})
        self.assertTrue(bool(ok_result.get("ok")))
        self.assertEqual((ok_result.get("data") or {}).get("echo"), "hello")

        invalid_result = sync_adapter.invoke({"extra": "bad"})
        self.assertFalse(bool(invalid_result.get("ok")))
        self.assertEqual(
            str(((invalid_result.get("error") or {}).get("kind")) or ""),
            "validation",
        )

        async def _async_handler(tool_input: dict, context: dict | None) -> dict:
            return {"echo": str(tool_input.get("query") or "").upper()}

        async_adapter = build_langgraph_tool_adapter(
            tool_spec=spec,
            handler=_async_handler,
        )
        async_result = async_adapter.invoke({"query": "hello"})
        self.assertTrue(bool(async_result.get("ok")))
        self.assertEqual((async_result.get("data") or {}).get("echo"), "HELLO")

    def test_builtin_placeholder_tools_are_disabled(self) -> None:
        registry = ToolRegistry()
        filesystem = registry.get_tool("filesystem.read")
        inspector = registry.get_tool("workflow.inspect")
        self.assertIsNotNone(filesystem)
        self.assertIsNotNone(inspector)
        if filesystem is None or inspector is None:
            return
        self.assertFalse(bool(filesystem.get("enabled")))
        self.assertFalse(bool(inspector.get("enabled")))
        self.assertTrue(str(filesystem.get("deprecated_reason") or "").strip())
        self.assertTrue(str(inspector.get("deprecated_reason") or "").strip())
        self.assertIsNotNone(registry.get_langgraph_tool("search.web"))

    def test_tool_invocation_writes_step_logs(self) -> None:
        db.upsert_tool(_tool_record("tool.custom.echo"), created_by="tests")
        result = graph.invoke_tool(
            tool_id="tool.custom.echo",
            tool_input={"query": "write logs"},
            context={},
        )
        self.assertEqual(str(result.get("status") or ""), "ok")
        run_id = str(result.get("run_id") or "")
        self.assertTrue(run_id)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT step_id, node_id, node_type, tool_name
                FROM step_logs
                WHERE run_id = ?
                ORDER BY step_index ASC, id ASC
                """,
                (run_id,),
            ).fetchall()
        self.assertGreaterEqual(len(rows), 2)
        tool_rows = [row for row in rows if str(row["node_type"] or "") == "tool"]
        self.assertTrue(tool_rows)
        self.assertEqual(str(tool_rows[0]["tool_name"] or ""), "tool.custom.echo")

    def test_tools_invoke_route_returns_error_payload_for_failed_tool(self) -> None:
        if self.client is None:
            self.skipTest("FastAPI TestClient unavailable")

        class _FailingRegistry:
            def invoke_tool(self, *, tool_id: str, tool_input: dict, context: dict | None = None) -> dict:
                _ = (tool_id, tool_input, context)
                return {
                    "ok": False,
                    "deleted": True,
                    "error": {
                        "type": "qdrant",
                        "message": "Qdrant collection missing.",
                    },
                }

        with mock.patch.object(graph, "_runtime_registry", return_value=_FailingRegistry()):
            response = self.client.post(
                "/tools/invoke",
                json={"tool_id": "rag.delete_doc", "input": {"doc_id": "doc-123"}},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(str(payload.get("status") or ""), "error")
        self.assertEqual(
            ((((payload.get("output") or {}).get("error") or {}).get("message"))),
            "Qdrant collection missing.",
        )

    def test_tools_routes_expose_versioned_payloads_and_invoke(self) -> None:
        if self.client is None:
            self.skipTest("FastAPI TestClient unavailable")
        db.upsert_tool(_tool_record("tool.custom.echo"), created_by="tests")

        tools_response = self.client.get("/tools")
        self.assertEqual(tools_response.status_code, 200)
        tools = tools_response.json().get("tools") or []
        custom_tool = next(
            (item for item in tools if str(item.get("id") or "") == "tool.custom.echo"),
            None,
        )
        self.assertIsNotNone(custom_tool)
        if custom_tool is None:
            return
        self.assertEqual(int(custom_tool.get("version") or 0), 1)
        self.assertEqual(str(custom_tool.get("implementation_kind") or ""), "python_module")

        invoke_response = self.client.post(
            "/tools/invoke",
            json={"tool_id": "tool.custom.echo", "input": {"query": "through http"}},
        )
        self.assertEqual(invoke_response.status_code, 200)
        payload = invoke_response.json()
        self.assertEqual(str(payload.get("status") or ""), "ok")
        self.assertEqual(
            (((payload.get("output") or {}).get("data") or {}).get("echo")),
            "through http",
        )


if __name__ == "__main__":
    unittest.main()
