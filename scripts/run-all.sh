#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="$ROOT_DIR/apps/api"
AGENT_SRC_DIR="$ROOT_DIR/apps/agent/src"
DESKTOP_DIR="$ROOT_DIR/apps/desktop"

PIDS=()
PID_NAMES=()

print_info() {
  printf '[run-all] %s\n' "$1"
}

print_warn() {
  printf '[run-all][warn] %s\n' "$1"
}

load_local_env() {
  local env_file="$ROOT_DIR/.local.env"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "$env_file"
    set +a
    print_info "Loaded env from $env_file"
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[run-all][error] Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

build_backend_pythonpath() {
  local backend_pythonpath="$API_DIR:$AGENT_SRC_DIR"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    backend_pythonpath="${backend_pythonpath}:$PYTHONPATH"
  fi
  printf '%s\n' "$backend_pythonpath"
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  local venv_python="$API_DIR/.venv/bin/python"
  if [[ -x "$venv_python" ]]; then
    echo "$venv_python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi

  printf '[run-all][error] Missing required command: python3 (or python)\n' >&2
  exit 1
}

is_ollama_up() {
  curl -fsS --max-time 2 "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1
}

fetch_ollama_tags() {
  curl -fsS --max-time 5 "$OLLAMA_BASE_URL/api/tags"
}

check_ollama_embedding_model() {
  if ! is_ollama_up; then
    return 0
  fi

  local status
  status="$(
    OLLAMA_TAGS_JSON="$(fetch_ollama_tags 2>/dev/null || true)" \
    OLLAMA_EMBED_MODEL="$OLLAMA_EMBED_MODEL" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

payload_raw = os.getenv("OLLAMA_TAGS_JSON", "")
requested = os.getenv("OLLAMA_EMBED_MODEL", "").strip()
if not payload_raw or not requested:
    print("skip")
    raise SystemExit(0)

try:
    payload = json.loads(payload_raw)
except json.JSONDecodeError:
    print("unknown")
    raise SystemExit(0)

raw_models = payload.get("models")
if not isinstance(raw_models, list):
    print("unknown")
    raise SystemExit(0)

available = []
for item in raw_models:
    if not isinstance(item, dict):
        continue
    name = str(item.get("name") or item.get("model") or "").strip()
    if name:
        available.append(name)

if not available:
    print("missing")
    raise SystemExit(0)

requested_lower = requested.lower()
for candidate in available:
    if candidate.lower() == requested_lower:
        print(f"exact:{candidate}")
        raise SystemExit(0)

requested_base = requested.split(":", 1)[0].strip().lower()
base_matches = [
    candidate
    for candidate in available
    if candidate.split(":", 1)[0].strip().lower() == requested_base
]
if len(base_matches) == 1:
    print(f"alias:{base_matches[0]}")
elif len(base_matches) > 1:
    print("ambiguous:" + ",".join(base_matches))
else:
    print("missing")
PY
  )"

  case "$status" in
    exact:*)
      print_info "Ollama embedding model ready: ${status#exact:}"
      ;;
    alias:*)
      print_info "Ollama embedding model '${OLLAMA_EMBED_MODEL}' will resolve to installed '${status#alias:}'."
      ;;
    ambiguous:*)
      print_warn "Multiple installed Ollama embedding model variants match '${OLLAMA_EMBED_MODEL}': ${status#ambiguous:}"
      print_warn "Set OLLAMA_EMBED_MODEL explicitly to one installed tag to avoid ambiguous RAG ingest/retrieval."
      ;;
    missing)
      print_warn "Configured Ollama embedding model '${OLLAMA_EMBED_MODEL}' is not installed at $OLLAMA_BASE_URL."
      print_warn "Install it with 'ollama pull ${OLLAMA_EMBED_MODEL}' or set OLLAMA_EMBED_MODEL to an installed model."
      ;;
    *)
      print_warn "Could not verify Ollama embedding model availability for '${OLLAMA_EMBED_MODEL}'."
      ;;
  esac
}

is_backend_up() {
  curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1
}

collect_backend_contract_failures() {
  local failures=()
  local required_path
  for required_path in /models /models/vision /workflows /tools /projects /rag/health; do
    if ! curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}${required_path}" >/dev/null 2>&1; then
      failures+=("GET ${required_path}")
    fi
  done

  # POST-only routes: verify they exist in OpenAPI before reusing backend.
  local openapi_payload
  openapi_payload="$(curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/openapi.json" 2>/dev/null || true)"
  if [[ -z "$openapi_payload" ]]; then
    failures+=("GET /openapi.json")
  else
    local required_openapi_path
    for required_openapi_path in \
      '/artifacts/upload' \
      '/internal/qdrant/config' \
      '/chat/stream' \
      '/tools/invoke' \
      '/builder/tools' \
      '/builder/workflows' \
      '/workflow/run' \
      '/runs/{run_id}' \
      '/runs/{run_id}/logs' \
      '/runs/{run_id}/steps' \
      '/runs/{run_id}/steps/{step_id}' \
      '/runs/{run_id}/state' \
      '/runs/{run_id}/replay/dry_run' \
      '/runs/{run_id}/replay' \
      '/runs/{run_id}/rerun_from_state' \
      '/runs/{run_id}/pending_tool_calls' \
      '/runs/{run_id}/tool_calls/{tool_call_id}/approve' \
      '/projects/{project_id}' \
      '/projects/{project_id}/documents' \
      '/projects/{project_id}/ground-truth' \
      '/projects/{project_id}/tools' \
      '/projects/{project_id}/workflow' \
      '/projects/{project_id}/chat' \
      '/projects/{project_id}/run' \
      '/projects/{project_id}/run/stream'
    do
      if ! grep -q "\"${required_openapi_path}\"" <<<"$openapi_payload"; then
        failures+=("OpenAPI ${required_openapi_path}")
      fi
    done
  fi

  if [[ "${#failures[@]}" -gt 0 ]]; then
    printf '%s\n' "${failures[@]}"
  fi
}

backend_has_required_routes() {
  local failures
  failures="$(collect_backend_contract_failures)"
  [[ -z "$failures" ]]
}

wait_for_backend_contract() {
  local retries="${1:-60}"
  local sleep_s="${2:-0.5}"
  local failures=""

  local i
  for ((i = 1; i <= retries; i++)); do
    failures="$(collect_backend_contract_failures)"
    if [[ -z "$failures" ]]; then
      return 0
    fi
    sleep "$sleep_s"
  done

  if [[ -n "$failures" ]]; then
    print_warn "Backend contract check failed:"
    while IFS= read -r failure; do
      [[ -n "$failure" ]] || continue
      print_warn "  missing ${failure}"
    done <<<"$failures"
  fi
  return 1
}

verify_backend_python() {
  if ! PYTHONPATH="$BACKEND_PYTHONPATH" \
    SATURDAY_DB_PATH="$SATURDAY_DB_PATH" \
    SATURDAY_DATA_DIR="$SATURDAY_DATA_DIR" \
    OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
    "$PYTHON_BIN" -c \
    "import app.main; import saturday_agent.tools.registry; import qdrant_client; import langchain_ollama; import langchain_qdrant; import pypdf; import uvicorn" \
    >/dev/null 2>&1; then
    printf '[run-all][error] The selected Python cannot import the full API stack.\n' >&2
    printf '[run-all][error] PYTHON_BIN=%s\n' "$PYTHON_BIN" >&2
    printf '[run-all][error] Expected imports: app.main, saturday_agent, uvicorn, qdrant-client, langchain-ollama, langchain-qdrant, pypdf.\n' >&2
    exit 1
  fi
}

stop_backend_listener_on_port() {
  if ! command -v lsof >/dev/null 2>&1; then
    print_warn "Cannot stop stale backend automatically because 'lsof' is not available."
    return 1
  fi

  local stale_pids
  stale_pids="$(lsof -tiTCP:${API_PORT} -sTCP:LISTEN || true)"
  if [[ -z "$stale_pids" ]]; then
    return 0
  fi

  local matched=()
  local skipped=()
  local pid
  for pid in $stale_pids; do
    if pid_belongs_to_project "$pid"; then
      matched+=("$pid")
    else
      skipped+=("$pid")
    fi
  done

  if [[ "${#matched[@]}" -gt 0 ]]; then
    print_warn "Stopping stale backend listener(s) on :${API_PORT} (${matched[*]})..."
    kill_pids_with_grace "backend-listener" "${matched[@]}"
  fi

  if [[ "${#skipped[@]}" -gt 0 ]]; then
    print_warn "Refusing to stop non-project listener(s) on :${API_PORT} (${skipped[*]})."
    print_warn "Stop that process manually or choose a different API_PORT."
    return 1
  fi
}

is_pid_alive() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

wait_for_pid_exit() {
  local pid="$1"
  local retries="${2:-20}"
  local sleep_s="${3:-0.2}"

  local i
  for ((i = 1; i <= retries; i++)); do
    if ! is_pid_alive "$pid"; then
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

kill_pid_with_grace() {
  local pid="$1"
  local label="${2:-process}"

  if ! is_pid_alive "$pid"; then
    return 0
  fi

  kill "$pid" >/dev/null 2>&1 || true
  if wait_for_pid_exit "$pid" 20 0.2; then
    return 0
  fi

  print_warn "Force-killing stubborn ${label} process ${pid}."
  kill -9 "$pid" >/dev/null 2>&1 || true
}

kill_pids_with_grace() {
  local label="$1"
  shift || true
  local pid
  for pid in "$@"; do
    if [[ -n "$pid" ]]; then
      kill_pid_with_grace "$pid" "$label"
    fi
  done
}

cleanup_pidfile_process() {
  local pid_file="$1"
  local label="$2"

  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]]; then
    print_info "Stopping previous ${label} process from ${pid_file} (pid ${pid})..."
    kill_pid_with_grace "$pid" "$label"
  fi
  rm -f "$pid_file"
}

write_pidfile() {
  local pid_file="$1"
  local pid="$2"
  printf '%s\n' "$pid" >"$pid_file"
}

pid_belongs_to_project() {
  local pid="$1"
  if ! command -v ps >/dev/null 2>&1; then
    return 0
  fi

  local cmdline
  cmdline="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  if [[ -z "$cmdline" ]]; then
    return 1
  fi

  [[ "$cmdline" == *"$ROOT_DIR"* ]]
}

stop_listener_on_port_for_project() {
  local port="$1"
  local label="$2"

  if ! command -v lsof >/dev/null 2>&1; then
    print_warn "Cannot stop stale ${label} automatically because 'lsof' is not available."
    return 1
  fi

  local stale_pids
  stale_pids="$(lsof -tiTCP:${port} -sTCP:LISTEN || true)"
  if [[ -z "$stale_pids" ]]; then
    return 0
  fi

  local matched=()
  local pid
  for pid in $stale_pids; do
    if pid_belongs_to_project "$pid"; then
      matched+=("$pid")
    else
      print_warn "Skipping ${label} pid ${pid} on :${port} (not in this project tree)."
    fi
  done

  if [[ "${#matched[@]}" -eq 0 ]]; then
    return 0
  fi

  print_warn "Stopping stale ${label} listener(s) on :${port} (${matched[*]})..."
  kill_pids_with_grace "$label-listener" "${matched[@]}"
}

assert_backend_port_available() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local listener_pids
  listener_pids="$(lsof -tiTCP:${API_PORT} -sTCP:LISTEN || true)"
  if [[ -z "$listener_pids" ]]; then
    return 0
  fi

  local project_pids=()
  local external_pids=()
  local pid
  for pid in $listener_pids; do
    if pid_belongs_to_project "$pid"; then
      project_pids+=("$pid")
    else
      external_pids+=("$pid")
    fi
  done

  if [[ "${#external_pids[@]}" -gt 0 ]]; then
    print_warn "API port :${API_PORT} is occupied by non-project listener(s) (${external_pids[*]})."
    print_warn "Stop that process manually or choose a different API_PORT."
    return 1
  fi

  if [[ "${#project_pids[@]}" -gt 0 ]]; then
    print_warn "API port :${API_PORT} is still occupied by project listener(s) (${project_pids[*]})."
    print_warn "Stop those processes or rerun with FORCE_RESTART=1."
    return 1
  fi

  return 0
}

stop_stale_app_processes() {
  print_info "Stopping stale app processes before restart..."
  cleanup_pidfile_process "$BACKEND_PID_FILE" "backend"
  cleanup_pidfile_process "$DESKTOP_PID_FILE" "desktop"
  cleanup_pidfile_process "$OLLAMA_PID_FILE" "ollama"
  stop_listener_on_port_for_project "$API_PORT" "backend" || true
  stop_listener_on_port_for_project "$FRONTEND_PORT" "frontend" || true
}

start_backend_process() {
  (
    cd "$API_DIR"
    if [[ "$API_RELOAD" == "1" ]]; then
      PYTHONPATH="$BACKEND_PYTHONPATH" SATURDAY_DB_PATH="$SATURDAY_DB_PATH" SATURDAY_DATA_DIR="$SATURDAY_DATA_DIR" OLLAMA_BASE_URL="$OLLAMA_BASE_URL" "$PYTHON_BIN" -m uvicorn "$API_APP_MODULE" --host "$API_HOST" --port "$API_PORT" --reload
    else
      PYTHONPATH="$BACKEND_PYTHONPATH" SATURDAY_DB_PATH="$SATURDAY_DB_PATH" SATURDAY_DATA_DIR="$SATURDAY_DATA_DIR" OLLAMA_BASE_URL="$OLLAMA_BASE_URL" "$PYTHON_BIN" -m uvicorn "$API_APP_MODULE" --host "$API_HOST" --port "$API_PORT"
    fi
  ) >"$LOG_DIR/backend.log" 2>&1 &
  local backend_pid="$!"
  PIDS+=("$backend_pid")
  PID_NAMES+=("backend")
  write_pidfile "$BACKEND_PID_FILE" "$backend_pid"
}

wait_for_url() {
  local url="$1"
  local retries="${2:-40}"
  local sleep_s="${3:-0.5}"

  local i
  for ((i = 1; i <= retries; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  print_info "Stopping child processes..."

  local idx
  for ((idx = ${#PIDS[@]} - 1; idx >= 0; idx--)); do
    local pid="${PIDS[$idx]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
  rm -f "$BACKEND_PID_FILE" "$DESKTOP_PID_FILE" "$OLLAMA_PID_FILE"
}

load_local_env

API_APP_MODULE="${API_APP_MODULE:-app.main:app}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://${API_HOST}:${API_PORT}}"
START_OLLAMA="${START_OLLAMA:-auto}" # auto | always | never
API_RELOAD="${API_RELOAD:-0}" # 1 to enable uvicorn --reload
FORCE_RESTART="${FORCE_RESTART:-1}" # 1 to stop stale app processes before start
SATURDAY_DB_PATH="${SATURDAY_DB_PATH:-$API_DIR/saturday.db}"
SATURDAY_DATA_DIR="${SATURDAY_DATA_DIR:-$API_DIR/data}"
LOG_DIR="${LOG_DIR:-/tmp/saturday-agent}"
mkdir -p "$LOG_DIR"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
DESKTOP_PID_FILE="$LOG_DIR/desktop.pid"
OLLAMA_PID_FILE="$LOG_DIR/ollama.pid"

require_cmd npm
require_cmd curl
PYTHON_BIN="$(pick_python)"
BACKEND_PYTHONPATH="$(build_backend_pythonpath)"
print_info "Using Python: $PYTHON_BIN"
print_info "Database: $SATURDAY_DB_PATH"
print_info "Data dir: $SATURDAY_DATA_DIR"
print_info "Ollama base URL: $OLLAMA_BASE_URL"
print_info "Ollama embedding model: $OLLAMA_EMBED_MODEL"
verify_backend_python

trap cleanup EXIT INT TERM

if [[ "$FORCE_RESTART" == "1" ]]; then
  stop_stale_app_processes
elif [[ "$FORCE_RESTART" != "0" ]]; then
  printf '[run-all][error] Invalid FORCE_RESTART value: %s (use 0|1)\n' "$FORCE_RESTART" >&2
  exit 1
fi

if [[ "$START_OLLAMA" == "always" ]]; then
  require_cmd ollama
fi

if [[ "$START_OLLAMA" == "always" ]]; then
  print_info "Starting Ollama server..."
  ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
  ollama_pid="$!"
  PIDS+=("$ollama_pid")
  PID_NAMES+=("ollama")
  write_pidfile "$OLLAMA_PID_FILE" "$ollama_pid"
  if ! wait_for_url "$OLLAMA_BASE_URL/api/tags" 60 0.5; then
    print_warn "Ollama did not become ready. Check $LOG_DIR/ollama.log"
  fi
elif [[ "$START_OLLAMA" == "auto" ]]; then
  if is_ollama_up; then
    print_info "Ollama already running at $OLLAMA_BASE_URL"
  else
    if command -v ollama >/dev/null 2>&1; then
      print_info "Ollama not detected, starting it..."
      ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
      ollama_pid="$!"
      PIDS+=("$ollama_pid")
      PID_NAMES+=("ollama")
      write_pidfile "$OLLAMA_PID_FILE" "$ollama_pid"
      if ! wait_for_url "$OLLAMA_BASE_URL/api/tags" 60 0.5; then
        print_warn "Ollama did not become ready. Check $LOG_DIR/ollama.log"
      fi
    else
      print_warn "Ollama CLI not found and service is not reachable at $OLLAMA_BASE_URL"
    fi
  fi
elif [[ "$START_OLLAMA" == "never" ]]; then
  if ! is_ollama_up; then
    print_warn "Ollama is not reachable at $OLLAMA_BASE_URL"
  fi
else
  printf '[run-all][error] Invalid START_OLLAMA value: %s (use auto|always|never)\n' "$START_OLLAMA" >&2
  exit 1
fi

check_ollama_embedding_model

print_info "Starting FastAPI LangGraph backend on http://${API_HOST}:${API_PORT}"
if is_backend_up; then
  if backend_has_required_routes; then
    print_info "Backend already running at http://${API_HOST}:${API_PORT} (reusing existing process)."
  else
    print_warn "Existing backend is missing required routes."
    if ! stop_backend_listener_on_port; then
      exit 1
    fi
    if ! assert_backend_port_available; then
      exit 1
    fi
    start_backend_process
  fi
else
  if ! assert_backend_port_available; then
    exit 1
  fi
  start_backend_process
fi

if wait_for_backend_contract 60 0.5; then
  print_info "Backend is ready."
elif wait_for_url "http://${API_HOST}:${API_PORT}/health" 5 0.5; then
  print_warn "Backend health check passed, but required routes are still missing. Check $LOG_DIR/backend.log"
else
  print_warn "Backend health check did not pass yet. Check $LOG_DIR/backend.log"
fi

print_info "Starting Electron + React frontend"
(
  cd "$DESKTOP_DIR"
  SATURDAY_API_URL="$VITE_API_BASE_URL" VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run dev
) >"$LOG_DIR/desktop.log" 2>&1 &
desktop_pid="$!"
PIDS+=("$desktop_pid")
PID_NAMES+=("desktop")
write_pidfile "$DESKTOP_PID_FILE" "$desktop_pid"

print_info "All services started."
print_info "Backend log: $LOG_DIR/backend.log"
print_info "Desktop log: $LOG_DIR/desktop.log"
if [[ "$START_OLLAMA" != "never" ]]; then
  print_info "Ollama log: $LOG_DIR/ollama.log"
fi
print_info "Press Ctrl+C to stop."

while true; do
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    pid_name="${PID_NAMES[$idx]:-child}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      wait "$pid" || true
      print_warn "A child process exited unexpectedly: ${pid_name} (pid ${pid})."
      exit 1
    fi
  done
  sleep 1
done
