from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

try:
    from fastapi.testclient import TestClient
    from app import db
    from app import main as main_app
    from app.projects import service as project_service

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency/environment gated
    TestClient = None  # type: ignore[assignment]
    db = None  # type: ignore[assignment]
    main_app = None  # type: ignore[assignment]
    project_service = None  # type: ignore[assignment]
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


def _project_workflow_spec() -> dict:
    return {
        "name": "Project Workflow",
        "description": "Project-scoped workflow.",
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
                "config": {
                    "response_template": "Project: {{task}}",
                    "output_key": "answer",
                },
                "position": {"x": 220, "y": 120},
            }
        ],
        "edges": [],
        "metadata": {"enabled": True},
    }


@unittest.skipIf(_IMPORT_ERROR is not None, f"Endpoint deps unavailable: {_IMPORT_ERROR}")
class ProjectEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        if db is None or main_app is None or project_service is None:
            self.skipTest(f"Endpoint deps unavailable: {_IMPORT_ERROR}")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "projects.sqlite"
        self.docs_root = Path(self.temp_dir.name) / "project_docs"
        main_app.DB_PATH = self.db_path
        db.init_db(str(self.db_path))
        self.docs_patch = mock.patch.object(project_service, "PROJECT_DOCS_ROOT", self.docs_root)
        self.docs_patch.start()
        self.client = TestClient(main_app.app)

    def tearDown(self) -> None:
        self.client.close()
        self.docs_patch.stop()
        self.temp_dir.cleanup()

    def _create_project(self, *, name: str = "Alpha", description: str = "Project alpha") -> dict:
        response = self.client.post(
            "/projects",
            json={"name": name, "description": description},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_project_crud_round_trip(self) -> None:
        created = self._create_project()
        project_id = str(created.get("id") or "")
        self.assertTrue(project_id)

        list_response = self.client.get("/projects")
        self.assertEqual(list_response.status_code, 200)
        projects = list_response.json().get("projects") or []
        self.assertTrue(any(str(item.get("id") or "") == project_id for item in projects))

        detail_response = self.client.get(f"/projects/{project_id}")
        self.assertEqual(detail_response.status_code, 200)
        detail = detail_response.json()
        self.assertEqual(detail.get("name"), "Alpha")
        self.assertEqual(detail.get("description"), "Project alpha")
        self.assertEqual(detail.get("chats"), [])
        self.assertEqual(detail.get("documents"), [])
        self.assertEqual((detail.get("ground_truth") or {}).get("content"), "")

        delete_response = self.client.delete(f"/projects/{project_id}")
        self.assertEqual(delete_response.status_code, 204)

        projects_after_delete = (self.client.get("/projects").json().get("projects") or [])
        self.assertFalse(
            any(str(item.get("id") or "") == project_id for item in projects_after_delete)
        )

    def test_project_document_upload_is_scoped_and_deletable(self) -> None:
        project = self._create_project()
        project_id = str(project.get("id") or "")
        captured: dict[str, object] = {}

        def _capture_index(*args, **kwargs) -> None:
            captured.update(kwargs)

        with (
            mock.patch.object(project_service, "_extract_document_text", return_value="alpha beta gamma"),
            mock.patch.object(project_service, "chunk_text", return_value=["alpha", "beta"]),
            mock.patch.object(project_service, "_runtime_qdrant_url", return_value=None),
            mock.patch.object(project_service, "index_chunks_to_qdrant", side_effect=_capture_index),
        ):
            upload_response = self.client.post(
                f"/projects/{project_id}/documents",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
            )
            self.assertEqual(upload_response.status_code, 200)
            document = upload_response.json()
            self.assertEqual(document.get("status"), "indexed")
            self.assertEqual(int(document.get("chunk_count") or 0), 2)
            self.assertEqual(
                captured.get("collection"),
                project_service.project_collection_name(project_id),
            )
            self.assertEqual(captured.get("doc_id"), document.get("id"))

            list_response = self.client.get(f"/projects/{project_id}/documents")
            self.assertEqual(list_response.status_code, 200)
            documents = list_response.json().get("documents") or []
            self.assertEqual(len(documents), 1)
            stored_path = Path(str(documents[0].get("filepath") or ""))
            self.assertTrue(stored_path.exists())

            delete_response = self.client.delete(
                f"/projects/{project_id}/documents/{document['id']}"
            )
            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(delete_response.json().get("id"), document.get("id"))

        documents_after_delete = (
            self.client.get(f"/projects/{project_id}/documents").json().get("documents") or []
        )
        self.assertEqual(documents_after_delete, [])
        self.assertFalse(stored_path.parent.exists())

    def test_project_run_uses_project_workflow_and_logs_injected_context(self) -> None:
        project = self._create_project(name="Launch", description="Critical rollout")
        project_id = str(project.get("id") or "")

        chat_response = self.client.post(
            f"/projects/{project_id}/chat",
            json={"name": "Investigation"},
        )
        self.assertEqual(chat_response.status_code, 200)
        chat_id = str(chat_response.json().get("id") or "")
        self.assertTrue(chat_id)

        db.upsert_tool(_tool_record("tool.custom.echo"), created_by="tests")

        ground_truth_response = self.client.put(
            f"/projects/{project_id}/ground-truth",
            json={"content": "Canonical requirement: use the launch checklist."},
        )
        self.assertEqual(ground_truth_response.status_code, 200)

        tools_response = self.client.put(
            f"/projects/{project_id}/tools",
            json={
                "bindings": [
                    {
                        "tool_name": "tool.custom.echo",
                        "enabled": False,
                        "version": 7,
                    }
                ]
            },
        )
        self.assertEqual(tools_response.status_code, 200)
        tool_binding = next(
            (
                item
                for item in (tools_response.json().get("tools") or [])
                if str(item.get("tool_name") or "") == "tool.custom.echo"
            ),
            None,
        )
        self.assertIsNotNone(tool_binding)
        if tool_binding is None:
            return
        self.assertFalse(bool(tool_binding.get("enabled")))
        self.assertEqual(int(tool_binding.get("version") or 0), 7)

        compile_payload = {
            "valid": True,
            "workflow_spec": {},
            "compiled": {"runtime_graph": {"nodes": [], "edges": []}},
            "diagnostics": [],
        }
        workflow_spec = _project_workflow_spec()
        with mock.patch.object(project_service.workflow_service, "compile_workflow_spec", return_value=compile_payload):
            workflow_response = self.client.post(
                f"/projects/{project_id}/workflow",
                json={"workflow_spec": workflow_spec},
            )
        self.assertEqual(workflow_response.status_code, 200)

        retrieval_calls: list[dict[str, object]] = []
        execute_calls: list[dict[str, object]] = []
        runtime_workflow_id = project_service.project_runtime_workflow_id(project_id)

        def _fake_retrieval(*, tool_defs, tool_input, context):
            retrieval_calls.append(
                {
                    "tool_defs": tool_defs,
                    "tool_input": tool_input,
                    "context": context,
                }
            )
            return (
                {
                    "ok": True,
                    "data": {
                        "results": [
                            {
                                "doc_id": "doc-launch",
                                "chunk_id": "chunk-1",
                                "text": "Launch checklist excerpt",
                                "score": 0.91,
                                "metadata": {
                                    "source": "/tmp/doc-launch.txt",
                                    "doc_id": "doc-launch",
                                    "chunk_id": "chunk-1",
                                },
                            }
                        ]
                    },
                },
                "ok",
            )

        def _fake_execute(**kwargs):
            execute_calls.append(kwargs)
            self.assertEqual(kwargs.get("workflow_id"), runtime_workflow_id)
            workflow_input = kwargs.get("input") or {}
            self.assertEqual(workflow_input.get("task"), "What should we ship?")
            self.assertEqual(workflow_input.get("tool_ids"), [])
            system_message = ((workflow_input.get("messages") or [])[0] or {}).get("content") or ""
            self.assertIn("Canonical requirement: use the launch checklist.", system_message)
            self.assertIn("Launch checklist excerpt", system_message)
            self.assertEqual(
                (((workflow_input.get("context") or {}).get("project_retrieval") or {}).get("results") or [])[0].get("doc_id"),
                "doc-launch",
            )
            self.assertEqual(
                ((kwargs.get("initial_state") or {}).get("project_ground_truth") or ""),
                "Canonical requirement: use the launch checklist.",
            )
            return {"status": "ok", "output": {"answer": "Use the checklist and ship."}}

        with (
            mock.patch.object(project_service.workflow_service, "compile_workflow_spec", return_value=compile_payload),
            mock.patch.object(
                project_service.workflow_service,
                "compile_draft_to_runtime_defs",
                return_value={
                    "workflow_def": {
                        "id": runtime_workflow_id,
                        "name": "Project Workflow",
                        "enabled": True,
                    }
                },
            ),
            mock.patch.object(project_service, "_run_project_retrieval", side_effect=_fake_retrieval),
            mock.patch.object(project_service.graph, "_execute_workflow_run", side_effect=_fake_execute),
        ):
            run_response = self.client.post(
                f"/projects/{project_id}/run",
                json={
                    "message": "What should we ship?",
                    "chat_id": chat_id,
                    "workflow_id": "complex.v1",
                    "model_id": "llama3.2",
                    "tool_ids": ["tool.custom.echo"],
                    "messages": [{"role": "user", "content": "What should we ship?"}],
                },
            )

        self.assertEqual(run_response.status_code, 200)
        payload = run_response.json()
        run_id = str(payload.get("run_id") or "")
        self.assertTrue(run_id)
        self.assertEqual(payload.get("project_id"), project_id)
        self.assertEqual(payload.get("chat_id"), chat_id)
        self.assertEqual(payload.get("workflow_id"), runtime_workflow_id)
        self.assertEqual(payload.get("workflow_source"), "project")
        self.assertEqual(payload.get("tool_ids"), [])
        self.assertEqual(payload.get("output_text"), "Use the checklist and ship.")
        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(len(execute_calls), 1)
        self.assertEqual(
            retrieval_calls[0]["tool_input"].get("collection"),  # type: ignore[index]
            project_service.project_collection_name(project_id),
        )

        run_detail_response = self.client.get(f"/runs/{run_id}")
        self.assertEqual(run_detail_response.status_code, 200)
        run_detail = run_detail_response.json()
        self.assertEqual(run_detail.get("project_id"), project_id)
        self.assertEqual(run_detail.get("chat_id"), chat_id)
        self.assertEqual(run_detail.get("workflow_id"), runtime_workflow_id)

        logs_response = self.client.get(f"/runs/{run_id}/logs")
        self.assertEqual(logs_response.status_code, 200)
        steps = logs_response.json().get("steps") or []
        context_step = next(
            (
                step for step in steps
                if step.get("name") == "node.project_context_injection"
                and step.get("output")
            ),
            None,
        )
        retrieval_step = next(
            (
                step for step in steps
                if step.get("name") == "tool.rag.retrieve"
                and step.get("output")
            ),
            None,
        )
        self.assertIsNotNone(context_step)
        self.assertIsNotNone(retrieval_step)
        if context_step is None or retrieval_step is None:
            return
        self.assertEqual(
            (context_step.get("output") or {}).get("project_ground_truth"),
            "Canonical requirement: use the launch checklist.",
        )
        retrieval_results = (
            (((retrieval_step.get("output") or {}).get("response") or {}).get("data") or {}).get("results")
            or []
        )
        self.assertEqual(retrieval_results[0].get("doc_id"), "doc-launch")

        ground_truth_detail = self.client.get(f"/projects/{project_id}/ground-truth").json()
        self.assertTrue(bool(ground_truth_detail.get("used_in_last_run")))
        self.assertEqual(ground_truth_detail.get("last_run_id"), run_id)
