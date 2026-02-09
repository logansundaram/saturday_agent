from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from saturday_agent.agents.moderate.prompts import (
    EXECUTE_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_PROMPT,
)
from saturday_agent.llms.ollama_chat import extract_assistant_text, ollama_chat
from saturday_agent.llms.registry import ModelRegistry
from saturday_agent.runtime.config import RuntimeConfig
from saturday_agent.runtime.tracing import StepEmitter, StepEvent, instrument_node
from saturday_agent.state.models import WorkflowState, append_trace
from saturday_agent.tools.registry import ToolRegistry

WORKFLOW_ID = "moderate.v1"
WORKFLOW_TYPE = "moderate"
WORKFLOW_TITLE = "Moderate Plan + Verify"
WORKFLOW_DESCRIPTION = "Adds planning and a single repair loop before finalize."

GRAPH_SPEC = {
    "nodes": [
        {"id": "START", "type": "start"},
        {"id": "plan", "type": "node"},
        {"id": "llm_execute", "type": "node"},
        {"id": "verify", "type": "node"},
        {"id": "repair", "type": "node"},
        {"id": "finalize", "type": "node"},
        {"id": "END", "type": "end"},
    ],
    "edges": [
        {"from": "START", "to": "plan"},
        {"from": "plan", "to": "llm_execute"},
        {"from": "llm_execute", "to": "verify"},
        {"from": "verify", "to": "repair", "condition": "needs_repair"},
        {"from": "verify", "to": "finalize", "condition": "ok_or_limit"},
        {"from": "repair", "to": "llm_execute"},
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
        context = json.dumps(state.get("context") or {}, default=str, ensure_ascii=True)
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        plan_text = ""
        try:
            raw = ollama_chat(
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PLAN_USER_PROMPT.format(task=task, context=context),
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
                f"1. Interpret task.\n2. Use provided context.\n"
                f"3. Produce concise final answer for: {task}"
            )

        return {
            "plan": plan_text,
            "trace": append_trace(
                state,
                name="plan",
                summary="Created execution plan.",
            ),
        }

    def llm_execute(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        plan_text = str(state.get("plan") or "")
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}
        context = dict(state.get("context") or {})

        existing_tool_calls = list(state.get("tool_calls") or [])
        existing_tool_results = list(state.get("tool_results") or [])
        tool_calls = existing_tool_calls
        tool_results = existing_tool_results

        if not tool_results:
            selected_ids_raw = context.get("selected_tool_ids")
            selected_ids = (
                [str(item).strip() for item in selected_ids_raw if str(item).strip()]
                if isinstance(selected_ids_raw, list)
                else []
            )
            tool_inputs_raw = context.get("tool_inputs")
            tool_inputs = tool_inputs_raw if isinstance(tool_inputs_raw, dict) else {}

            executed_calls: list[Dict[str, Any]] = []
            executed_results: list[Dict[str, Any]] = []
            for tool_id in selected_ids:
                per_tool_input = tool_inputs.get(tool_id)
                tool_input = dict(per_tool_input) if isinstance(per_tool_input, dict) else {}
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
                except Exception as exc:
                    status = "error"
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
                executed_results.append(
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

            if executed_results:
                tool_calls = executed_calls
                tool_results = executed_results

        system_content = EXECUTE_SYSTEM_PROMPT.format(plan=plan_text)
        if tool_results:
            system_content = (
                f"{system_content}\n\nTool Results:\n"
                f"{json.dumps(tool_results, default=str, ensure_ascii=True)}"
            )

        messages = [{"role": "system", "content": system_content}]
        user_messages = [
            message for message in list(state.get("messages") or []) if message.get("role") == "user"
        ]
        if user_messages:
            messages.extend(user_messages)
        elif task:
            messages.append({"role": "user", "content": task})

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
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "trace": append_trace(
                state,
                name="llm_execute",
                summary=(
                    "Generated draft answer from plan."
                    if not tool_results
                    else f"Generated draft answer using {len(tool_results)} tool result(s)."
                ),
            ),
        }

    def verify(state: WorkflowState) -> Dict[str, Any]:
        task = str(state.get("task") or "")
        answer = str(state.get("answer") or "")
        plan_text = str(state.get("plan") or "")
        model = str(state.get("model") or model_registry.get_default_model())
        options = state.get("options") if isinstance(state.get("options"), dict) else {}

        verify_text = ""
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
                        ),
                    },
                ],
                model=model,
                base_url=config.ollama_base_url,
                timeout_seconds=config.ollama_timeout_seconds,
                options=options,
            )
            verify_text = extract_assistant_text(raw)
        except Exception:
            verify_text = ""

        if verify_text:
            verify_ok, verify_notes = _parse_verifier_output(verify_text)
        else:
            verify_ok = bool(answer.strip())
            verify_notes = (
                "Heuristic verification used because verifier call failed."
                if verify_ok
                else "Answer is empty."
            )

        return {
            "verify_ok": verify_ok,
            "verify_notes": verify_notes,
            "trace": append_trace(
                state,
                name="verify",
                summary=verify_notes,
                status="ok" if verify_ok else "needs_repair",
            ),
        }

    def repair(state: WorkflowState) -> Dict[str, Any]:
        retry_count = int(state.get("retry_count") or 0) + 1
        plan_text = str(state.get("plan") or "")
        verify_notes = str(state.get("verify_notes") or "Improve correctness and clarity.")
        revised_plan = (
            f"{plan_text}\n\nRepair pass {retry_count}:\n"
            f"- Address verifier notes: {verify_notes}\n"
            "- Keep response grounded in task and context."
        )

        return {
            "plan": revised_plan,
            "retry_count": retry_count,
            "trace": append_trace(
                state,
                name="repair",
                summary="Updated plan based on verification feedback.",
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
            }
        )
        return {
            "plan": state.get("plan"),
            "answer": str(state.get("answer") or ""),
            "artifacts": artifacts,
            "trace": append_trace(
                state,
                name="finalize",
                summary="Packed final moderate workflow output.",
            ),
        }

    def route_after_verify(state: WorkflowState) -> Literal["repair", "finalize"]:
        if bool(state.get("verify_ok")):
            return "finalize"
        if int(state.get("retry_count") or 0) >= 1:
            return "finalize"
        return "repair"

    builder = StateGraph(WorkflowState)
    builder.add_node("plan", instrument_node(name="plan", node_fn=plan, step_emitter=step_emitter))
    builder.add_node(
        "llm_execute",
        instrument_node(name="llm_execute", node_fn=llm_execute, step_emitter=step_emitter),
    )
    builder.add_node(
        "verify",
        instrument_node(name="verify", node_fn=verify, step_emitter=step_emitter),
    )
    builder.add_node(
        "repair",
        instrument_node(name="repair", node_fn=repair, step_emitter=step_emitter),
    )
    builder.add_node(
        "finalize",
        instrument_node(name="finalize", node_fn=finalize, step_emitter=step_emitter),
    )

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "llm_execute")
    builder.add_edge("llm_execute", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "repair": "repair",
            "finalize": "finalize",
        },
    )
    builder.add_edge("repair", "llm_execute")
    builder.add_edge("finalize", END)
    return builder.compile()
