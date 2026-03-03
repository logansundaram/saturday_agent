from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "apps" / "api"
DESKTOP_DIR = REPO_ROOT / "apps" / "desktop"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pythonpath() -> str:
    paths = [
        str(API_DIR),
        str(REPO_ROOT / "apps" / "agent" / "src"),
    ]
    existing = str(os.environ.get("PYTHONPATH") or "").strip()
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def _http_json(
    method: str,
    url: str,
    payload: Dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw or "{}")


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error = "health check not attempted"
    while time.time() < deadline:
        try:
            payload = _http_json("GET", f"{base_url}/health", timeout=5.0)
            if str(payload.get("api") or "") == "ok":
                return
            last_error = f"unexpected payload: {payload}"
        except Exception as exc:  # pragma: no cover - external process polling
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"Backend did not become healthy: {last_error}")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - windows fallback
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - windows fallback
            process.kill()
        process.wait(timeout=5)


def _launch_backend(
    *,
    python_bin: str,
    port: int,
    db_path: Path,
    trace_path: Path,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["SATURDAY_DB_PATH"] = str(db_path)
    env["SATURDAY_USAGE_TRACE_PATH"] = str(trace_path)
    env.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    return subprocess.Popen(
        [
            python_bin,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(API_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _create_smoke_tool(base_url: str) -> None:
    payload = {
        "id": "tool.custom.smoke_echo",
        "name": "Smoke Echo Tool",
        "kind": "local",
        "description": "Minimal smoke-test echo tool.",
        "type": "python",
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
    }
    _http_json("POST", f"{base_url}/builder/tools", payload)


def _finalize_only_workflow_spec() -> Dict[str, Any]:
    return {
        "workflow_id": "workflow.smoke.finalize_only",
        "name": "Smoke Finalize Workflow",
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
                "config": {
                    "response_template": "Smoke: {{task}}",
                    "output_key": "answer",
                },
                "position": {"x": 200, "y": 120},
            }
        ],
        "edges": [],
        "metadata": {"enabled": True},
    }


def _run_backend_smoke(base_url: str) -> Dict[str, Any]:
    health = _http_json("GET", f"{base_url}/health")
    if str(health.get("api") or "") != "ok":
        raise RuntimeError(f"/health failed: {health}")

    tools = _http_json("GET", f"{base_url}/tools")
    tool_rows = tools.get("tools") if isinstance(tools.get("tools"), list) else []
    if len(tool_rows) == 0:
        raise RuntimeError("/tools returned no tools.")

    workflow = _http_json(
        "POST",
        f"{base_url}/workflow/run",
        {
            "workflow_spec": _finalize_only_workflow_spec(),
            "input": {"task": "debloat smoke"},
            "sandbox_mode": False,
        },
    )
    if str(workflow.get("status") or "").lower() not in {"ok", "completed"}:
        raise RuntimeError(f"/workflow/run failed: {workflow}")

    _create_smoke_tool(base_url)
    tool_run = _http_json(
        "POST",
        f"{base_url}/tools/invoke",
        {
            "tool_id": "tool.custom.smoke_echo",
            "input": {"query": "smoke tool"},
            "context": {"origin": "smoke"},
        },
    )
    output = tool_run.get("output") if isinstance(tool_run.get("output"), dict) else {}
    data = output.get("data") if isinstance(output.get("data"), dict) else {}
    if str(data.get("echo") or "") != "smoke tool":
        raise RuntimeError(f"/tools/invoke failed: {tool_run}")

    return {
        "health": health,
        "tool_count": len(tool_rows),
        "workflow_run_id": str(workflow.get("run_id") or ""),
        "tool_run_id": str(tool_run.get("run_id") or ""),
    }


def _run_desktop_smoke(
    *,
    api_base_url: str,
    trace_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    if report_path.exists():
        report_path.unlink()

    env = os.environ.copy()
    env["SATURDAY_SMOKE"] = "1"
    env["SATURDAY_SKIP_QDRANT"] = "1"
    env["SATURDAY_USAGE_TRACE_PATH"] = str(trace_path)
    env["SATURDAY_SMOKE_REPORT_PATH"] = str(report_path)
    env["SATURDAY_API_URL"] = api_base_url
    env["VITE_API_BASE_URL"] = api_base_url
    env["VITE_USAGE_TRACE"] = "1"
    env["CI"] = "1"

    command = ["npm", "run", "dev"]
    if sys.platform == "darwin":
        command = ["arch", "-arm64", *command]

    process = subprocess.Popen(
        command,
        cwd=str(DESKTOP_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 90.0
        while time.time() < deadline:
            if report_path.exists():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                if not bool(payload.get("ok")):
                    raise RuntimeError(f"Desktop smoke failed: {payload}")
                return payload
            if process.poll() is not None:
                output = ""
                if process.stdout is not None:
                    output = process.stdout.read()
                raise RuntimeError(
                    f"Desktop smoke exited early with code {process.returncode}.\n{output}"
                )
            time.sleep(0.5)
        raise RuntimeError("Desktop smoke timed out without producing a report.")
    finally:
        _terminate_process(process)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Saturday smoke checks.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter with API dependencies installed.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temp smoke artifacts instead of deleting them.",
    )
    args = parser.parse_args()

    temp_dir_ctx = tempfile.TemporaryDirectory(prefix="saturday-smoke-")
    artifacts_dir = Path(temp_dir_ctx.name)
    keep_artifacts = bool(args.keep_artifacts)
    try:
        trace_path = artifacts_dir / "usage-trace.jsonl"
        report_path = artifacts_dir / "desktop-smoke-report.json"
        db_path = artifacts_dir / "smoke.sqlite"
        port = _pick_free_port()
        base_url = f"http://127.0.0.1:{port}"

        backend = _launch_backend(
            python_bin=args.python_bin,
            port=port,
            db_path=db_path,
            trace_path=trace_path,
        )
        try:
            _wait_for_health(base_url)
            backend_summary = _run_backend_smoke(base_url)
            desktop_summary = _run_desktop_smoke(
                api_base_url=base_url,
                trace_path=trace_path,
                report_path=report_path,
            )
        finally:
            _terminate_process(backend)

        summary = {
            "ok": True,
            "backend": backend_summary,
            "desktop": desktop_summary,
            "trace_path": str(trace_path),
            "report_path": str(report_path),
        }
        print(json.dumps(summary, indent=2))
        if keep_artifacts:
            temp_dir_ctx._finalizer.detach()  # type: ignore[attr-defined]
            print(str(artifacts_dir))
        return 0
    finally:
        if not keep_artifacts:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
