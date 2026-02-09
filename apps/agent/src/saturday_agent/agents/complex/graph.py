from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from saturday_agent.agents.complex.prompts import (
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_PROMPT,
)
from saturday_agent.llms.ollama_chat import extract_assistant_text, ollama_chat
from saturday_agent.llms.registry import ModelRegistry
from saturday_agent.runtime.config import RuntimeConfig
from saturday_agent.runtime.tracing import StepEmitter, StepEvent, instrument_node
from saturday_agent.state.models import WorkflowState, append_trace
from saturday_agent.tools.registry import ToolRegistry

WORKFLOW_ID = "complex.v1"
WORKFLOW_TYPE = "complex"
WORKFLOW_TITLE = "Complex Plan + Tools + Verify"
WORKFLOW_DESCRIPTION = "Adds tool planning, execution, and a stronger verification loop."

GRAPH_SPEC = {
    "nodes": [
        {"id": "START", "type": "start"},
        {"id": "plan", "type": "node"},
        {"id": "decide_tools", "type": "node"},
        {"id": "execute_tools", "type": "node"},
        {"id": "llm_synthesize", "type": "node"},
        {"id": "verify", "type": "node"},
        {"id": "finalize", "type": "node"},
        {"id": "END", "type": "end"},
    ],
    "edges": [
        {"from": "START", "to": "plan"},
        {"from": "plan", "to": "decide_tools"},
        {"from": "decide_tools", "to": "execute_tools", "condition": "has_tools"},
        {"from": "decide_tools", "to": "llm_synthesize", "condition": "no_tools"},
        {"from": "execute_tools", "to": "llm_synthesize"},
        {"from": "llm_synthesize", "to": "verify"},
        {"from": "verify", "to": "plan", "condition": "retry_once"},
        {"from": "verify", "to": "finalize", "condition": "ok_or_limit"},
        {"from": "finalize", "to": "END"},
    ],
}


def _parse_verifier_output(text: str) -> tuple[bool, str]:
    cleaned = text.strip()
    lowered = cleaned.lower()
    ok = lowered.startswith("ok") or lowered.startswith("pass")
    if "fail" in lowered or "not ok" in lowered:
        ok = False
    if not cleaned:
        return False, "Verifier returned empty output."
    return ok, cleaned


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_graph(
    *,
    config: RuntimeConfig,
    model_registry: ModelRegistry,
    tool_registry: ToolRegistry,
    step_emitter: Optional[StepEmitter] = None,
) -> Any:
    def plan(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        context_text = json.dumps(state.get("context") or {}, default=str, ensure_ascii=True)
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        plan_text = ""
        try:
            raw = ollama_chat(
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PLAN_USER_PROMPT.format(task=task, context=context_text),
                    },
                ],
                model=model,
                base_url=config.ollama_base_url,
                timeout_seconds=config.ollama_timeout_seconds,
                options=options,
            )
            plan_text = extract_assistant_text(raw).strip()
        except Exception:
            plan_text = ""

        if not plan_text:
            plan_text = (
                f"1. Analyze task: {task}\n"
                "2. Identify missing info and constraints.\n"
                "3. Synthesize final answer with explicit assumptions."
            )

        return {
            "plan": plan_text,
            "trace": append_trace(
                state,
                name="plan",
                summary="Created complex workflow plan.",
            ),
        }

    def decide_tools(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        context = dict(state.get("context") or {})
        planned_calls = tool_registry.decide_tools(task=task, context=context)

        return {
            "tool_calls": planned_calls,
            "trace": append_trace(
                state,
                name="decide_tools",
                summary=f"Planned {len(planned_calls)} tool calls.",
            ),
        }

    def execute_tools(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        planned_calls = list(state.get("tool_calls") or [])

        executed_calls: list[Dict[str, Any]] = []
        tool_results: list[Dict[str, Any]] = []
        error_count = 0

        for call in planned_calls:
            if not isinstance(call, dict):
                continue

            tool_id = str(call.get("tool_id") or call.get("id") or "").strip()
            if not tool_id:
                continue

            raw_input = call.get("input")
            tool_input = dict(raw_input) if isinstance(raw_input, dict) else {}
            if tool_id == "search.web" and not str(tool_input.get("query", "")).strip() and task:
                tool_input["query"] = task

            started_at = _utc_now_iso()
            status = "ok"
            try:
                tool_output = tool_registry.invoke_tool(tool_id, tool_input)
                if not isinstance(tool_output, dict):
                    tool_output = {"value": tool_output}
                if "error" in tool_output:
                    status = "error"
                    error_count += 1
            except Exception as exc:
                status = "error"
                error_count += 1
                tool_output = {
                    "error": {
                        "type": "tool_exception",
                        "message": str(exc),
                    }
                }
            ended_at = _utc_now_iso()

            executed_calls.append(
                {
                    "tool_id": tool_id,
                    "input": tool_input,
                    "output": tool_output,
                    "status": status,
                }
            )
            tool_results.append(
                {
                    "tool_id": tool_id,
                    "status": status,
                    "output": tool_output,
                }
            )

            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=f"tool.{tool_id}",
                        status=status,
                        started_at=started_at,
                        ended_at=ended_at,
                        input={"tool_id": tool_id, "input": tool_input},
                        output={"tool_id": tool_id, "output": tool_output},
                    )
                )

        trace_status = "ok" if error_count == 0 else "error"
        trace_summary = (
            f"Executed {len(executed_calls)} tool calls."
            if error_count == 0
            else f"Executed {len(executed_calls)} tool calls with {error_count} errors."
        )

        return {
            "tool_calls": executed_calls,
            "tool_results": tool_results,
            "trace": append_trace(
                state,
                name="execute_tools",
                summary=trace_summary,
                status=trace_status,
            ),
        }

    def llm_synthesize(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        plan_text = str(state.get("plan") or "")
        tool_results = list(state.get("tool_results") or [])
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        messages = [
            {
                "role": "system",
                "content": SYNTHESIZE_SYSTEM_PROMPT.format(
                    plan=plan_text,
                    tool_results=json.dumps(tool_results, default=str, ensure_ascii=True),
                ),
            },
            {"role": "user", "content": task},
        ]

        raw = ollama_chat(
            messages=messages,
            model=model,
            base_url=config.ollama_base_url,
            timeout_seconds=config.ollama_timeout_seconds,
            options=options,
        )
        answer = extract_assistant_text(raw).strip()

        updated_messages = list(state.get("messages") or [])
        updated_messages.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "messages": updated_messages,
            "trace": append_trace(
                state,
                name="llm_synthesize",
                summary="Synthesized answer from plan and tool results.",
            ),
        }

    def verify(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        plan_text = str(state.get("plan") or "")
        answer = str(state.get("answer") or "")
        tool_results = json.dumps(state.get("tool_results") or [], default=str, ensure_ascii=True)
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        verifier_text = ""
        try:
            raw = ollama_chat(
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": VERIFY_USER_PROMPT.format(
                            task=task,
                            plan=plan_text,
                            answer=answer,
                            tool_results=tool_results,
                        ),
                    },
                ],
                model=model,
                base_url=config.ollama_base_url,
                timeout_seconds=config.ollama_timeout_seconds,
                options=options,
            )
            verifier_text = extract_assistant_text(raw)
        except Exception:
            verifier_text = ""

        if verifier_text:
            verify_ok, verify_notes = _parse_verifier_output(verifier_text)
        else:
            verify_ok = bool(answer.strip())
            verify_notes = (
                "Heuristic verification used because verifier call failed."
                if verify_ok
                else "Answer is empty."
            )

        retry_count = int(state.get("retry_count") or 0)
        if not verify_ok:
            retry_count += 1

        return {
            "verify_ok": verify_ok,
            "verify_notes": verify_notes,
            "retry_count": retry_count,
            "trace": append_trace(
                state,
                name="verify",
                summary=verify_notes,
                status="ok" if verify_ok else "needs_retry",
            ),
        }

    def finalize(state: WorkflowState) -> Dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        artifacts.update(
            {
                "workflow_id": WORKFLOW_ID,
                "workflow_type": WORKFLOW_TYPE,
                "verify_ok": bool(state.get("verify_ok")),
                "verify_notes": state.get("verify_notes"),
                "tool_call_count": len(state.get("tool_calls") or []),
                "tool_result_count": len(state.get("tool_results") or []),
            }
        )
        return {
            "plan": state.get("plan"),
            "answer": str(state.get("answer") or ""),
            "artifacts": artifacts,
            "trace": append_trace(
                state,
                name="finalize",
                summary="Packed final complex workflow output.",
            ),
        }

    def route_after_decide_tools(state: WorkflowState) -> Literal["execute_tools", "llm_synthesize"]:
        if state.get("tool_calls"):
            return "execute_tools"
        return "llm_synthesize"

    def route_after_verify(state: WorkflowState) -> Literal["plan", "finalize"]:
        if bool(state.get("verify_ok")):
            return "finalize"
        if int(state.get("retry_count") or 0) <= config.max_verify_retries:
            return "plan"
        return "finalize"

    builder = StateGraph(WorkflowState)
    builder.add_node("plan", instrument_node(name="plan", node_fn=plan, step_emitter=step_emitter))
    builder.add_node(
        "decide_tools",
        instrument_node(
            name="decide_tools",
            node_fn=decide_tools,
            step_emitter=step_emitter,
        ),
    )
    builder.add_node(
        "execute_tools",
        instrument_node(
            name="execute_tools",
            node_fn=execute_tools,
            step_emitter=step_emitter,
        ),
    )
    builder.add_node(
        "llm_synthesize",
        instrument_node(
            name="llm_synthesize",
            node_fn=llm_synthesize,
            step_emitter=step_emitter,
        ),
    )
    builder.add_node(
        "verify",
        instrument_node(name="verify", node_fn=verify, step_emitter=step_emitter),
    )
    builder.add_node(
        "finalize",
        instrument_node(name="finalize", node_fn=finalize, step_emitter=step_emitter),
    )

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "decide_tools")
    builder.add_conditional_edges(
        "decide_tools",
        route_after_decide_tools,
        {
            "execute_tools": "execute_tools",
            "llm_synthesize": "llm_synthesize",
        },
    )
    builder.add_edge("execute_tools", "llm_synthesize")
    builder.add_edge("llm_synthesize", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "plan": "plan",
            "finalize": "finalize",
        },
    )
    builder.add_edge("finalize", END)
    return builder.compile()
