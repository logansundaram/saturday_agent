from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from saturday_agent.agents.simple.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from saturday_agent.llms.ollama_chat import extract_assistant_text, ollama_chat
from saturday_agent.llms.registry import ModelRegistry
from saturday_agent.runtime.config import RuntimeConfig
from saturday_agent.runtime.tracing import StepEmitter, instrument_node
from saturday_agent.state.models import WorkflowState, append_trace
from saturday_agent.tools.registry import ToolRegistry

WORKFLOW_ID = "simple.v1"
WORKFLOW_TYPE = "simple"
WORKFLOW_TITLE = "Simple Answer"
WORKFLOW_DESCRIPTION = "Minimal workflow for quick responses."

GRAPH_SPEC = {
    "nodes": [
        {"id": "START", "type": "start"},
        {"id": "build_messages", "type": "node"},
        {"id": "llm_answer", "type": "node"},
        {"id": "finalize", "type": "node"},
        {"id": "END", "type": "end"},
    ],
    "edges": [
        {"from": "START", "to": "build_messages"},
        {"from": "build_messages", "to": "llm_answer"},
        {"from": "llm_answer", "to": "finalize"},
        {"from": "finalize", "to": "END"},
    ],
}


def build_graph(
    *,
    config: RuntimeConfig,
    model_registry: ModelRegistry,
    tool_registry: ToolRegistry,
    step_emitter: Optional[StepEmitter] = None,
) -> Any:
    _ = tool_registry

    def build_messages(state: WorkflowState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        task = str(state.get("task") or "")
        context = dict(state.get("context") or {})

        if not any(msg.get("role") == "system" for msg in messages):
            system_content = SYSTEM_PROMPT
            if context:
                system_content = (
                    f"{system_content}\n\nContext:\n"
                    f"{json.dumps(context, default=str, ensure_ascii=True)}"
                )
            messages.insert(0, {"role": "system", "content": system_content})

        if task and not any(msg.get("role") == "user" for msg in messages):
            messages.append({"role": "user", "content": USER_PROMPT_TEMPLATE.format(task=task)})

        return {
            "messages": messages,
            "trace": append_trace(
                state,
                name="build_messages",
                summary="Prepared system and user messages.",
            ),
        }

    def llm_answer(state: WorkflowState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        raw = ollama_chat(
            messages=messages,
            model=model,
            base_url=config.ollama_base_url,
            timeout_seconds=config.ollama_timeout_seconds,
            options=options,
        )
        answer = extract_assistant_text(raw).strip()
        updated_messages = messages + [{"role": "assistant", "content": answer}]

        return {
            "answer": answer,
            "messages": updated_messages,
            "trace": append_trace(
                state,
                name="llm_answer",
                summary="Generated answer from LLM.",
            ),
        }

    def finalize(state: WorkflowState) -> Dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        artifacts.update(
            {
                "workflow_id": WORKFLOW_ID,
                "workflow_type": WORKFLOW_TYPE,
            }
        )
        return {
            "answer": str(state.get("answer") or ""),
            "artifacts": artifacts,
            "trace": append_trace(
                state,
                name="finalize",
                summary="Packed final simple workflow output.",
            ),
        }

    builder = StateGraph(WorkflowState)
    builder.add_node(
        "build_messages",
        instrument_node(
            name="build_messages",
            node_fn=build_messages,
            step_emitter=step_emitter,
        ),
    )
    builder.add_node(
        "llm_answer",
        instrument_node(
            name="llm_answer",
            node_fn=llm_answer,
            step_emitter=step_emitter,
        ),
    )
    builder.add_node(
        "finalize",
        instrument_node(
            name="finalize",
            node_fn=finalize,
            step_emitter=step_emitter,
        ),
    )
    builder.add_edge(START, "build_messages")
    builder.add_edge("build_messages", "llm_answer")
    builder.add_edge("llm_answer", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
