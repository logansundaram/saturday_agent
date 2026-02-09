# saturday-agent

LangGraph workflow runtime for Saturday Agent.

## Structure

- `state/`: shared workflow state models
- `llms/`: model adapters and model registry stubs
- `tools/`: tool registry stubs
- `routing/`: workflow selection policies and router
- `agents/`: `simple`, `moderate`, `complex` workflow graphs
- `runtime/`: graph runtime, tracing, and workflow registry

## Development

Install from `apps/api` with:

```bash
pip install -r requirements.txt
```

This uses `-e ../agent` so `saturday_agent` is importable by the API app.
