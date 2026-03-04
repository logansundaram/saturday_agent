# Debloat Audit

Generated from static scan plus runtime smoke trace on 2026-03-03.

Runtime evidence source:

- Smoke runner: [`/Users/Logan/Documents/saturday_agent/scripts/smoke.py`](/Users/Logan/Documents/saturday_agent/scripts/smoke.py)
- Baseline runbook: [`/Users/Logan/Documents/saturday_agent/docs/debloat-baseline-runbook.md`](/Users/Logan/Documents/saturday_agent/docs/debloat-baseline-runbook.md)
- Usage trace captured at `/var/folders/vt/1pky6jvs4tq1rmvf2pwg_m6c0000gn/T/saturday-smoke-z8xt_ep4/usage-trace.jsonl`

## A) Entrypoints + Dependency Map

Desktop:

- Electron main entry: [`/Users/Logan/Documents/saturday_agent/apps/desktop/electron/main.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/electron/main.ts)
- Electron preload: [`/Users/Logan/Documents/saturday_agent/apps/desktop/electron/preload.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/electron/preload.ts)
- React renderer entry: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/main.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/main.tsx)
- App shell: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/App.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/App.tsx)
- Active route switch: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/dashboard.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/dashboard.tsx)

Active renderer pages from the dashboard route switch:

- Chat: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ChatPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ChatPage.tsx)
- Models: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ModelsPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ModelsPage.tsx)
- Tools: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ToolsPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ToolsPage.tsx)
- Builder: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/BuilderPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/BuilderPage.tsx)
- Workflows: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/WorkflowsPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/WorkflowsPage.tsx)
- Local Docs: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/LocalDocsPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/LocalDocsPage.tsx)
- Inspect: [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/InspectPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/InspectPage.tsx)

Backend:

- FastAPI entry: [`/Users/Logan/Documents/saturday_agent/apps/api/app/main.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/main.py)
- Backend orchestration/runtime bridge: [`/Users/Logan/Documents/saturday_agent/apps/api/app/graph.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/graph.py)
- Tool service layer: [`/Users/Logan/Documents/saturday_agent/apps/api/app/tools/service.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/tools/service.py)
- Workflow service layer: [`/Users/Logan/Documents/saturday_agent/apps/api/app/workflows/service.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/workflows/service.py)
- Persistence layer: [`/Users/Logan/Documents/saturday_agent/apps/api/app/db.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/db.py)

LangGraph/runtime:

- Runtime registry entry: [`/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/runtime/registry.py`](/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/runtime/registry.py)
- Tool registry: [`/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py`](/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py)
- LangGraph tool adapter: [`/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/langgraph_adapter.py`](/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/langgraph_adapter.py)

## B) Usage Evidence

Runtime trace summary from the smoke run:

- `49` events total
- Route hits:
  - `/tools`: `8`
  - `/health`: `6`
  - `/workflows`: `6`
  - `/models`: `4`
  - `/models/vision`: `4`
  - `/workflow/run`: `1`
  - `/builder/tools`: `1`
  - `/tools/invoke`: `1`
- Renderer route mounts:
  - `chat`: `2`
  - `tools`: `1`
  - `builder`: `1`
  - `inspect`: `1`
- Tool invocation:
  - `tool.custom.smoke_echo` from `api:/tools/invoke`

Observed runtime path:

1. Electron main window created from [`/Users/Logan/Documents/saturday_agent/apps/desktop/electron/main.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/electron/main.ts)
2. Chat route mounted in [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/dashboard.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/dashboard.tsx)
3. Tools route mounted and `/tools` fetched by [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ToolsPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/ToolsPage.tsx)
4. Builder route mounted and `/tools` plus `/workflows` fetched by [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/BuilderPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/BuilderPage.tsx)
5. Inspect route mounted in [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/InspectPage.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/pages/InspectPage.tsx)
6. Custom echo tool created via `/builder/tools` and invoked through `/tools/invoke`

Static evidence highlights:

- Legacy component cluster has zero production imports and only self-references:
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_page.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_page.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_page.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_page.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chats.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chats.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_tile.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_tile.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tool_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tool_card.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflow_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflow_card.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_card.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/card.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor_pill.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor_pill.tsx)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/header.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/header.tsx)
- Orphaned chat transport stack is self-contained and not imported by the active chat page:
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatClient.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatClient.ts)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatTransport.ts)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/langgraphApiTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/langgraphApiTransport.ts)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/ollamaTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/ollamaTransport.ts)
  - [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/ollama.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/ollama.ts)
- Placeholder tools remain present but intentionally disabled in [`/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py`](/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py): `filesystem.read`, `workflow.inspect`
- Legacy workflow builder endpoints still exist in [`/Users/Logan/Documents/saturday_agent/apps/api/app/main.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/main.py) and client helpers still exist in [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts), but they were not exercised by the smoke flow

## C) Classification

| Candidate | Status | Evidence | Risk | Action |
| --- | --- | --- | --- | --- |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_page.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_page.tsx) | `UNUSED` | Zero imports outside itself; no dashboard route reference; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_page.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_page.tsx) | `UNUSED` | Zero production imports; only references old `model_card.tsx`/`card.tsx`; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chats.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chats.tsx) | `UNUSED` | Zero production imports; only references old `chat_tile.tsx`; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_tile.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/chat_tile.tsx) | `UNUSED` | Imported only by unused `chats.tsx`; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tool_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tool_card.tsx) | `DUPLICATE` | Unused old card; active card is [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tools/ToolCard.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/tools/ToolCard.tsx) | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflow_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflow_card.tsx) | `DUPLICATE` | Unused old card; active card is [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflows/WorkflowCard.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/workflows/WorkflowCard.tsx) | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/model_card.tsx) | `DUPLICATE` | Imported only by unused old `model_page.tsx`; active card is [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/models/ModelCard.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/models/ModelCard.tsx) | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/card.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/card.tsx) | `UNUSED` | Only referenced by unused old `model_page.tsx`; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor_pill.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor_pill.tsx) | `UNUSED` | Zero imports; active monitor uses [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/monitor.tsx) | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/header.tsx`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/components/header.tsx) | `UNUSED` | Imported only by unused old `tool_card.tsx`; no runtime hit | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatClient.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatClient.ts) | `UNUSED` | No imports from active chat path; only references orphaned transport files | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/chatTransport.ts) | `UNUSED` | Only referenced inside orphaned transport stack | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/langgraphApiTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/langgraphApiTransport.ts) | `UNUSED` | Only referenced by orphaned `chatClient.ts`; active chat uses [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts) streaming directly | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/ollamaTransport.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/transports/ollamaTransport.ts) | `UNUSED` | Only referenced by orphaned `chatClient.ts` | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/ollama.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/ollama.ts) | `UNUSED` | No imports found under `apps/desktop/src` | `LOW` | `REMOVE` |
| [`/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py`](/Users/Logan/Documents/saturday_agent/apps/agent/src/saturday_agent/tools/registry.py) placeholders | `PLACEHOLDER` | `filesystem.read` and `workflow.inspect` explicitly disabled with deprecation reason | `MED` | `KEEP` |
| [`/Users/Logan/Documents/saturday_agent/apps/api/app/main.py`](/Users/Logan/Documents/saturday_agent/apps/api/app/main.py) legacy `/builder/workflows` routes | `LIKELY USED` | Client helpers still exist in [`/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts`](/Users/Logan/Documents/saturday_agent/apps/desktop/src/lib/api.ts); not hit in smoke | `MED` | `DEPRECATE` |

## D) Immediate Cleanup Order

1. Remove the isolated legacy component cluster in `apps/desktop/src/components/*.tsx` listed above.
2. Re-run smoke.
3. Remove the orphaned chat transport stack in `apps/desktop/src/lib/`.
4. Re-run smoke.
5. Leave placeholder tools and legacy workflow builder routes in place for now; they need deprecation telemetry, not blind deletion.
