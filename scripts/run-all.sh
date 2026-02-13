#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="$ROOT_DIR/apps/api"
DESKTOP_DIR="$ROOT_DIR/apps/desktop"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://${API_HOST}:${API_PORT}}"
START_OLLAMA="${START_OLLAMA:-auto}" # auto | always | never
API_RELOAD="${API_RELOAD:-0}" # 1 to enable uvicorn --reload
FORCE_RESTART="${FORCE_RESTART:-1}" # 1 to stop stale app processes before start

LOG_DIR="${LOG_DIR:-/tmp/saturday-agent}"
mkdir -p "$LOG_DIR"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
DESKTOP_PID_FILE="$LOG_DIR/desktop.pid"
OLLAMA_PID_FILE="$LOG_DIR/ollama.pid"

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

is_backend_up() {
  curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1
}

backend_has_required_routes() {
  local required_path
  for required_path in /models /workflows /tools; do
    if ! curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}${required_path}" >/dev/null 2>&1; then
      return 1
    fi
  done

  # /chat/stream is POST-only; verify it exists in OpenAPI before reusing backend.
  if ! curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/openapi.json" | grep -q '"/chat/stream"'; then
    return 1
  fi

  return 0
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

  print_warn "Stopping stale backend listener(s) on :${API_PORT} (${stale_pids})..."
  local pid
  for pid in $stale_pids; do
    kill_pid_with_grace "$pid" "backend-listener"
  done
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
      OLLAMA_BASE_URL="$OLLAMA_BASE_URL" "$PYTHON_BIN" -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --reload
    else
      OLLAMA_BASE_URL="$OLLAMA_BASE_URL" "$PYTHON_BIN" -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT"
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

trap cleanup EXIT INT TERM

require_cmd npm
require_cmd curl
PYTHON_BIN="$(pick_python)"
print_info "Using Python: $PYTHON_BIN"
load_local_env

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

print_info "Starting FastAPI LangGraph backend on http://${API_HOST}:${API_PORT}"
if is_backend_up; then
  if backend_has_required_routes; then
    print_info "Backend already running at http://${API_HOST}:${API_PORT} (reusing existing process)."
  else
    print_warn "Existing backend is missing required routes (/models, /workflows, /tools)."
    stop_backend_listener_on_port || true
    start_backend_process
  fi
else
  start_backend_process
fi

if wait_for_url "http://${API_HOST}:${API_PORT}/health" 60 0.5; then
  if backend_has_required_routes; then
    print_info "Backend is ready."
  else
    print_warn "Backend is up but required routes are missing. Check $LOG_DIR/backend.log"
  fi
else
  print_warn "Backend health check did not pass yet. Check $LOG_DIR/backend.log"
fi

print_info "Starting Electron + React frontend"
(
  cd "$DESKTOP_DIR"
  VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run dev
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
