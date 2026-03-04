# Baseline Runbook

## Backend (FastAPI)

Start the API from [`/Users/Logan/Documents/saturday_agent/apps/api`](/Users/Logan/Documents/saturday_agent/apps/api):

```bash
PYTHONPATH=/Users/Logan/Documents/saturday_agent/apps/api:/Users/Logan/Documents/saturday_agent/apps/agent/src \
SATURDAY_DB_PATH=/Users/Logan/Documents/saturday_agent/apps/api/saturday.db \
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
/Users/Logan/Documents/saturday_agent/apps/api/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Expected API URL: `http://127.0.0.1:8000`

Useful env vars:

- `SATURDAY_DB_PATH`: SQLite file path. Defaults to [`/Users/Logan/Documents/saturday_agent/apps/api/saturday.db`](/Users/Logan/Documents/saturday_agent/apps/api/saturday.db).
- `OLLAMA_BASE_URL`: Ollama base URL. Defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: default text model.
- `VISION_DEFAULT_MODEL` / `VITE_VISION_DEFAULT_MODEL`: default vision model.

Key health URLs:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/rag/health`

## Desktop (Electron + Vite)

Start the desktop app from [`/Users/Logan/Documents/saturday_agent/apps/desktop`](/Users/Logan/Documents/saturday_agent/apps/desktop):

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Expected frontend dev URL: `http://127.0.0.1:5173`

Expected Electron behavior:

- Main window opens from the Vite dev server.
- Default route/page is `chat`.
- The renderer talks to the API at `VITE_API_BASE_URL`.

## Combined Dev Startup

Existing combined script:

```bash
cd /Users/Logan/Documents/saturday_agent/apps/desktop
npm run dev:all
```

This uses [`/Users/Logan/Documents/saturday_agent/scripts/run-all.sh`](/Users/Logan/Documents/saturday_agent/scripts/run-all.sh) and defaults to:

- `API_HOST=127.0.0.1`
- `API_PORT=8000`
- `FRONTEND_PORT=5173`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `VITE_API_BASE_URL=http://127.0.0.1:8000`
- `START_OLLAMA=auto`
- `API_RELOAD=0`

## Smoke Runner

Run the new fast smoke flow from the repo root:

```bash
/Users/Logan/Documents/saturday_agent/apps/api/.venv/bin/python /Users/Logan/Documents/saturday_agent/scripts/smoke.py
```

What it checks:

- Backend `/health` returns `api=ok`
- Backend `/tools` returns at least one tool
- Backend `/workflow/run` executes a trivial inline workflow
- Backend `/tools/invoke` executes a tiny custom echo tool
- Desktop boots the Electron main window
- Chat page renders
- Tools page renders
- Builder page renders
- Inspect page renders
