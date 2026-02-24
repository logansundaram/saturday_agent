from __future__ import annotations

import tempfile
import time
import unittest
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from app import db
    from app import main as main_app
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency/environment gated
    TestClient = None  # type: ignore[assignment]
    db = None  # type: ignore[assignment]
    main_app = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finalize_only_spec(workflow_id: str) -> dict:
    return {
        "workflow_id": workflow_id,
        "name": "Finalize Only",
        "description": "",
        "allow_cycles": False,
        "state_schema": [
            {"key": "answer", "type": "string", "description": "", "required": False}
        ],
        "nodes": [
            {
                "id": "finalize_1",
                "type": "finalize",
                "label": "Finalize",
                "reads": ["task"],
                "writes": ["answer"],
                "config": {"response_template": "Final: {{task}}", "output_key": "answer"},
                "position": {"x": 200, "y": 120},
            }
        ],
        "edges": [],
        "metadata": {"enabled": True},
    }


def _sandbox_tool_spec(workflow_id: str, tool_id: str) -> dict:
    return {
        "workflow_id": workflow_id,
        "name": "Sandbox Tool Workflow",
        "description": "",
        "allow_cycles": False,
        "state_schema": [
            {"key": "artifacts", "type": "json", "description": "", "required": False},
            {"key": "answer", "type": "string", "description": "", "required": False},
        ],
        "nodes": [
            {
                "id": "tool_1",
                "type": "tool",
                "label": "Tool",
                "reads": ["task"],
                "writes": ["artifacts"],
                "config": {
                    "tool_name": tool_id,
                    "args_map": {"query": "task"},
                    "output_key": "artifacts",
                },
                "position": {"x": 120, "y": 120},
            },
            {
                "id": "finalize_1",
                "type": "finalize",
                "label": "Finalize",
                "reads": ["task"],
                "writes": ["answer"],
                "config": {"response_template": "done {{task}}", "output_key": "answer"},
                "position": {"x": 420, "y": 120},
            },
        ],
        "edges": [
            {"id": "edge_1", "from": "tool_1", "to": "finalize_1", "label": "always"}
        ],
        "metadata": {"enabled": True},
    }


def _seed_python_tool(tool_id: str) -> None:
    if db is None:
        raise RuntimeError("db module unavailable")
    now = _now_iso()
    db.upsert_tool(
        {
            "id": tool_id,
            "name": "Test Echo Tool",
            "kind": "local",
            "type": "python",
            "description": "Local echo tool for tests",
            "enabled": True,
            "config": {
                "code": (
                    "def run(input, context):\n"
                    "    query = str(input.get('query', ''))\n"
                    "    return {'echo': query}\n"
                ),
                "timeout_ms": 1000,
                "allowed_imports": ["json"],
            },
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "created_at": now,
            "updated_at": now,
        }
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"Endpoint deps unavailable: {_IMPORT_ERROR}")
class WorkflowEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        if db is None or main_app is None:
            self.skipTest(f"Endpoint deps unavailable: {_IMPORT_ERROR}")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite"
        main_app.DB_PATH = self.db_path
        db.init_db(str(self.db_path))
        _seed_python_tool("tool.custom.test_echo")
        self.client = TestClient(main_app.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def test_create_version_and_run_non_sandbox(self) -> None:
        workflow_id = "workflow.custom.endpoint_non_sandbox"
        spec = _finalize_only_spec(workflow_id)

        create_resp = self.client.post(
            f"/workflow/{workflow_id}/versions",
            json={"workflow_spec": spec, "created_by": "tests"},
        )
        self.assertEqual(create_resp.status_code, 200)
        created = create_resp.json()
        self.assertEqual(created.get("workflow_id"), workflow_id)
        self.assertEqual(int(created.get("version_num") or 0), 1)

        run_resp = self.client.post(
            "/workflow/run",
            json={
                "workflow_id": workflow_id,
                "input": {"task": "hello non sandbox"},
                "sandbox_mode": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        run_payload = run_resp.json()
        self.assertEqual(run_payload.get("workflow_id"), workflow_id)
        self.assertIn(str(run_payload.get("status") or ""), {"ok", "completed"})
        run_id = str(run_payload.get("run_id") or "")
        self.assertTrue(run_id)

        logs_resp = self.client.get(f"/runs/{run_id}/logs")
        self.assertEqual(logs_resp.status_code, 200)
        steps = logs_resp.json().get("steps") or []
        self.assertGreaterEqual(len(steps), 1)

    def test_sandbox_pending_approve_and_complete(self) -> None:
        workflow_id = "workflow.custom.endpoint_sandbox_approve"
        spec = _sandbox_tool_spec(workflow_id, "tool.custom.test_echo")

        run_resp = self.client.post(
            "/workflow/run",
            json={
                "draft_spec": spec,
                "input": {"task": "approve this tool call"},
                "sandbox_mode": True,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        run_payload = run_resp.json()
        run_id = str(run_payload.get("run_id") or "")
        self.assertTrue(run_id)

        pending = self._wait_for_pending(run_id)
        self.assertGreaterEqual(len(pending), 1)
        tool_call_id = str(pending[0].get("tool_call_id") or "")
        self.assertTrue(tool_call_id)

        approve_resp = self.client.post(
            f"/runs/{run_id}/tool_calls/{tool_call_id}/approve",
            json={"approved": True},
        )
        self.assertEqual(approve_resp.status_code, 200)

        final_status = self._wait_for_terminal_status(run_id)
        self.assertIn(final_status, {"ok", "completed"})

        logs_resp = self.client.get(f"/runs/{run_id}/logs")
        self.assertEqual(logs_resp.status_code, 200)
        steps = logs_resp.json().get("steps") or []
        self.assertTrue(any((step.get("tool_calls") or []) for step in steps))

    def test_sandbox_pending_reject_and_stop(self) -> None:
        workflow_id = "workflow.custom.endpoint_sandbox_reject"
        spec = _sandbox_tool_spec(workflow_id, "tool.custom.test_echo")

        run_resp = self.client.post(
            "/workflow/run",
            json={
                "draft_spec": spec,
                "input": {"task": "reject this tool call"},
                "sandbox_mode": True,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        run_id = str(run_resp.json().get("run_id") or "")
        self.assertTrue(run_id)

        pending = self._wait_for_pending(run_id)
        self.assertGreaterEqual(len(pending), 1)
        tool_call_id = str(pending[0].get("tool_call_id") or "")
        self.assertTrue(tool_call_id)

        reject_resp = self.client.post(
            f"/runs/{run_id}/tool_calls/{tool_call_id}/approve",
            json={"approved": False},
        )
        self.assertEqual(reject_resp.status_code, 200)

        final_status = self._wait_for_terminal_status(run_id)
        self.assertEqual(final_status, "rejected")

    def test_replay_rejects_non_success_step(self) -> None:
        workflow_id = "workflow.custom.endpoint_replay_non_success"
        spec = _sandbox_tool_spec(workflow_id, "tool.custom.test_echo")
        run_id = self._run_custom_workflow(workflow_id, spec, {"task": "replay non success"})
        from_step = self._find_step(run_id, preferred_node_id="tool_1")
        self.assertIsNotNone(from_step)
        if from_step is None:
            return

        internal_step_id = int(str(from_step.get("step_id", "")).replace("step_", ""))
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                UPDATE run_steps
                SET status = 'error'
                WHERE run_id = ?
                  AND step_id = ?
                """,
                (run_id, internal_step_id),
            )
            conn.commit()
        finally:
            conn.close()

        replay_resp = self.client.post(
            f"/runs/{run_id}/replay",
            json={
                "from_step_id": str(from_step.get("step_id") or ""),
                "state_patch": {},
                "patch_mode": "overlay",
                "base_state": "post",
            },
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay_payload = replay_resp.json()
        self.assertIsNone(replay_payload.get("new_run_id"))
        diagnostics = replay_payload.get("diagnostics") or []
        self.assertTrue(any(item.get("code") == "STEP_NOT_REPLAYABLE" for item in diagnostics))

    def test_replay_blocks_state_validation_errors(self) -> None:
        workflow_id = "workflow.custom.endpoint_replay_schema_validation"
        spec = _finalize_only_spec(workflow_id)
        run_id = self._run_custom_workflow(workflow_id, spec, {"task": "schema guard"})
        from_step = self._find_step(run_id)
        self.assertIsNotNone(from_step)
        if from_step is None:
            return

        replay_resp = self.client.post(
            f"/runs/{run_id}/replay",
            json={
                "from_step_id": str(from_step.get("step_id") or ""),
                "state_patch": {"unknown_key_not_allowed": "x"},
                "patch_mode": "overlay",
                "base_state": "post",
            },
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay_payload = replay_resp.json()
        self.assertIsNone(replay_payload.get("new_run_id"))
        diagnostics = replay_payload.get("diagnostics") or []
        self.assertTrue(any(item.get("code") == "UNKNOWN_KEY" for item in diagnostics))

    def test_replay_creates_linked_fork_metadata(self) -> None:
        workflow_id = "workflow.custom.endpoint_replay_linked_fork"
        spec = _finalize_only_spec(workflow_id)
        run_id = self._run_custom_workflow(workflow_id, spec, {"task": "linked fork"})
        from_step = self._find_step(run_id)
        self.assertIsNotNone(from_step)
        if from_step is None:
            return

        replay_resp = self.client.post(
            f"/runs/{run_id}/replay",
            json={
                "from_step_id": str(from_step.get("step_id") or ""),
                "state_patch": {},
                "patch_mode": "overlay",
                "base_state": "post",
            },
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay_payload = replay_resp.json()
        new_run_id = str(replay_payload.get("new_run_id") or "")
        self.assertTrue(new_run_id)

        run_detail = self.client.get(f"/runs/{new_run_id}")
        self.assertEqual(run_detail.status_code, 200)
        detail_payload = run_detail.json()
        self.assertEqual(str(detail_payload.get("parent_run_id") or ""), run_id)
        self.assertEqual(
            str(detail_payload.get("parent_step_id") or ""),
            str(from_step.get("step_id") or ""),
        )
        self.assertEqual(str(detail_payload.get("mode") or ""), "replay")

    def test_replay_resume_skips_until_resume_and_keeps_state(self) -> None:
        workflow_id = "workflow.custom.endpoint_replay_resume_skip"
        spec = _sandbox_tool_spec(workflow_id, "tool.custom.test_echo")
        run_id = self._run_custom_workflow(workflow_id, spec, {"task": "resume skip"})
        from_step = self._find_step(run_id, preferred_node_id="tool_1")
        self.assertIsNotNone(from_step)
        if from_step is None:
            return

        replay_resp = self.client.post(
            f"/runs/{run_id}/replay",
            json={
                "from_step_id": str(from_step.get("step_id") or ""),
                "state_patch": {},
                "patch_mode": "overlay",
                "base_state": "post",
                "replay_this_step": False,
            },
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay_payload = replay_resp.json()
        new_run_id = str(replay_payload.get("new_run_id") or "")
        self.assertTrue(new_run_id)

        replay_steps_resp = self.client.get(f"/runs/{new_run_id}/steps")
        self.assertEqual(replay_steps_resp.status_code, 200)
        replay_steps = replay_steps_resp.json().get("steps") or []
        skipped = None
        resumed = None
        for step in replay_steps:
            node_id = str(step.get("node_id") or "")
            if node_id == "tool_1":
                skipped = step
            if node_id == "finalize_1":
                resumed = step

        self.assertIsNotNone(skipped)
        self.assertIsNotNone(resumed)
        if skipped is None or resumed is None:
            return
        self.assertEqual(str(skipped.get("status") or ""), "skipped")
        self.assertEqual(str(resumed.get("status") or ""), "success")

        skipped_detail_resp = self.client.get(
            f"/runs/{new_run_id}/steps/{str(skipped.get('step_id') or '')}"
        )
        self.assertEqual(skipped_detail_resp.status_code, 200)
        skipped_detail = (skipped_detail_resp.json().get("step") or {})
        self.assertEqual(skipped_detail.get("pre_state"), skipped_detail.get("post_state"))

    def _wait_for_pending(self, run_id: str, timeout_s: float = 6.0) -> list[dict]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            response = self.client.get(f"/runs/{run_id}/pending_tool_calls")
            if response.status_code == 200:
                pending = response.json().get("pending") or []
                if pending:
                    return pending
            time.sleep(0.1)
        return []

    def _wait_for_terminal_status(self, run_id: str, timeout_s: float = 8.0) -> str:
        deadline = time.time() + timeout_s
        terminal = {"ok", "error", "rejected", "completed", "failed"}
        latest = ""
        while time.time() < deadline:
            response = self.client.get(f"/runs/{run_id}")
            if response.status_code != 200:
                time.sleep(0.1)
                continue
            latest = str(response.json().get("status") or "").lower()
            if latest in terminal:
                return latest
            time.sleep(0.1)
        return latest

    def _run_custom_workflow(self, workflow_id: str, spec: dict, run_input: dict) -> str:
        create_resp = self.client.post(
            f"/workflow/{workflow_id}/versions",
            json={"workflow_spec": spec, "created_by": "tests"},
        )
        self.assertEqual(create_resp.status_code, 200)
        run_resp = self.client.post(
            "/workflow/run",
            json={
                "workflow_id": workflow_id,
                "input": run_input,
                "sandbox_mode": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        payload = run_resp.json()
        run_id = str(payload.get("run_id") or "")
        self.assertTrue(run_id)
        return run_id

    def _find_step(self, run_id: str, preferred_node_id: str | None = None) -> dict | None:
        response = self.client.get(f"/runs/{run_id}/steps")
        self.assertEqual(response.status_code, 200)
        steps = response.json().get("steps") or []
        success_steps = [
            step
            for step in steps
            if str(step.get("status") or "") == "success"
        ]
        if preferred_node_id:
            for step in success_steps:
                if str(step.get("node_id") or "") == preferred_node_id:
                    return step
        return success_steps[0] if success_steps else None


if __name__ == "__main__":
    unittest.main()
