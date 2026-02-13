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

## RAG Environment

Set these variables to enable `rag.retrieve` with Qdrant + Ollama embeddings:

```bash
SATURDAY_DATA_DIR=./data
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=saturday_docs
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```
