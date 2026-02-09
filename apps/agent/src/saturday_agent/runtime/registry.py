from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from saturday_agent.agents.complex import graph as complex_graph
from saturday_agent.agents.moderate import graph as moderate_graph
from saturday_agent.agents.simple import graph as simple_graph
from saturday_agent.llms.registry import ModelRegistry
from saturday_agent.routing.router import route_workflow_type
from saturday_agent.runtime.config import RuntimeConfig, load_runtime_config
from saturday_agent.runtime.tracing import StepEmitter, StepEvent
from saturday_agent.state.models import build_initial_state
from saturday_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    workflow_type: str
    title: str
    description: str
    graph_spec: Dict[str, Any]
    build_graph_fn: Callable[..., Any]


class WorkflowRegistry:
    def __init__(
        self,
        *,
        config: Optional[RuntimeConfig] = None,
        model_registry: Optional[ModelRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._config = config or load_runtime_config()
        self._model_registry = model_registry or ModelRegistry(
            base_url=self._config.ollama_base_url,
            default_model=self._config.default_model,
            timeout_seconds=self._config.ollama_timeout_seconds,
        )
        self._tool_registry = tool_registry or ToolRegistry()

        self._workflows: Dict[str, WorkflowDefinition] = {
            simple_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=simple_graph.WORKFLOW_ID,
                workflow_type=simple_graph.WORKFLOW_TYPE,
                title=simple_graph.WORKFLOW_TITLE,
                description=simple_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(simple_graph.GRAPH_SPEC),
                build_graph_fn=simple_graph.build_graph,
            ),
            moderate_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=moderate_graph.WORKFLOW_ID,
                workflow_type=moderate_graph.WORKFLOW_TYPE,
                title=moderate_graph.WORKFLOW_TITLE,
                description=moderate_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(moderate_graph.GRAPH_SPEC),
                build_graph_fn=moderate_graph.build_graph,
            ),
            complex_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=complex_graph.WORKFLOW_ID,
                workflow_type=complex_graph.WORKFLOW_TYPE,
                title=complex_graph.WORKFLOW_TITLE,
                description=complex_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(complex_graph.GRAPH_SPEC),
                build_graph_fn=complex_graph.build_graph,
            ),
        }

    def list_workflows(self) -> List[Dict[str, str]]:
        order = {"simple": 0, "moderate": 1, "complex": 2}
        ordered = sorted(
            self._workflows.values(),
            key=lambda item: (order.get(item.workflow_type, 99), item.title),
        )
        return [
            {
                "id": workflow.workflow_id,
                "title": workflow.title,
                "description": workflow.description,
                "type": workflow.workflow_type,
                "version": workflow.workflow_id.split(".")[-1],
                "status": "available",
            }
            for workflow in ordered
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._tool_registry.list_tools()

    def list_models(self) -> Dict[str, Any]:
        return self._model_registry.list_models_payload()

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow_id: {workflow_id}")
        return workflow

    def get_workflow_id_for_type(self, workflow_type: str) -> str:
        for workflow in self._workflows.values():
            if workflow.workflow_type == workflow_type:
                return workflow.workflow_id
        raise ValueError(f"Unknown workflow_type: {workflow_type}")

    def compile_workflow(
        self,
        *,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_type = self.route_workflow_type(
            task=task,
            context=context,
            workflow_type=workflow_type,
        )
        workflow_id = self.get_workflow_id_for_type(selected_type)
        workflow = self.get_workflow(workflow_id)
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "graph": copy.deepcopy(workflow.graph_spec),
        }

    def run_workflow(
        self,
        *,
        workflow_id: str,
        input: Dict[str, Any],
        step_emitter: Optional[StepEmitter] = None,
    ) -> Dict[str, Any]:
        workflow = self.get_workflow(workflow_id)

        raw_messages = input.get("messages") if isinstance(input.get("messages"), list) else []
        task = str(input.get("task") or "")
        if not task:
            for message in reversed(raw_messages):
                if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
                    task = str(message.get("content", ""))
                    break

        context = input.get("context") if isinstance(input.get("context"), dict) else {}
        model = str(input.get("model")) if input.get("model") else None
        options = input.get("options") if isinstance(input.get("options"), dict) else {}
        raw_tool_ids = input.get("tool_ids")
        tool_ids = (
            [str(item).strip() for item in raw_tool_ids if str(item).strip()]
            if isinstance(raw_tool_ids, list)
            else []
        )

        if tool_ids:
            context = dict(context)
            context["selected_tool_ids"] = tool_ids

        state = build_initial_state(
            task=task,
            context=context,
            messages=raw_messages,
            model=model,
            options=options,
            default_model=self._model_registry.get_default_model(),
        )

        if not state.get("task") and not state.get("messages"):
            raise ValueError("Workflow input must include task or messages.")

        compiled_graph = workflow.build_graph_fn(
            config=self._config,
            model_registry=self._model_registry,
            tool_registry=self._tool_registry,
            step_emitter=step_emitter,
        )
        final_state = compiled_graph.invoke(state)

        raw_planned_calls = final_state.get("tool_calls")
        planned_calls = raw_planned_calls if isinstance(raw_planned_calls, list) else []
        existing_results = final_state.get("tool_results")
        has_existing_results = isinstance(existing_results, list) and len(existing_results) > 0
        post_graph_tool_execution = False
        if workflow.workflow_type == "moderate":
            # Moderate may execute tools during llm_execute to make outputs available
            # to the model before synthesis. Keep fallback for compatibility only.
            post_graph_tool_execution = not has_existing_results and bool(planned_calls or tool_ids)
        elif workflow.workflow_type == "complex":
            # Complex graph executes tools inside the execute_tools node so synthesis
            # can consume real outputs. Keep a fallback for compatibility only.
            post_graph_tool_execution = not has_existing_results and bool(planned_calls or tool_ids)

        if post_graph_tool_execution:
            executed_calls, executed_results = self._execute_selected_tools(
                task=task,
                context=context,
                tool_ids=tool_ids,
                planned_calls=planned_calls,
                step_emitter=step_emitter,
            )
            if executed_calls:
                final_state["tool_calls"] = executed_calls
                final_state["tool_results"] = executed_results

        output: Dict[str, Any] = {
            "answer": str(final_state.get("answer") or ""),
            "plan": final_state.get("plan"),
            "artifacts": dict(final_state.get("artifacts") or {}),
        }
        if tool_ids:
            output["selected_tool_ids"] = tool_ids
        if final_state.get("tool_calls"):
            output["tool_calls"] = list(final_state.get("tool_calls") or [])
        if final_state.get("tool_results"):
            output["tool_results"] = list(final_state.get("tool_results") or [])

        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "output": output,
            "final_state": final_state,
        }

    def route_workflow_type(
        self,
        *,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_type: Optional[str] = None,
    ) -> str:
        return route_workflow_type(
            task=task,
            context=context,
            workflow_type=workflow_type,
        )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _resolve_tool_input(
        self,
        *,
        tool_id: str,
        task: str,
        context: Dict[str, Any],
        planned_call: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resolved_input: Dict[str, Any] = {}

        raw_tool_inputs = context.get("tool_inputs")
        tool_inputs = raw_tool_inputs if isinstance(raw_tool_inputs, dict) else {}
        context_input = tool_inputs.get(tool_id)
        if isinstance(context_input, dict):
            resolved_input.update(context_input)

        if isinstance(planned_call, dict):
            planned_input = planned_call.get("input")
            if isinstance(planned_input, dict):
                resolved_input.update(planned_input)

        if tool_id == "search.web" and not str(resolved_input.get("query", "")).strip():
            resolved_input["query"] = task

        return resolved_input

    def _execute_selected_tools(
        self,
        *,
        task: str,
        context: Dict[str, Any],
        tool_ids: List[str],
        planned_calls: List[Dict[str, Any]],
        step_emitter: Optional[StepEmitter],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected_ids = {str(tool_id).strip() for tool_id in tool_ids if str(tool_id).strip()}
        calls_to_run: List[Dict[str, Any]] = []

        if planned_calls:
            for planned_call in planned_calls:
                if not isinstance(planned_call, dict):
                    continue
                tool_id = str(planned_call.get("tool_id") or planned_call.get("id") or "").strip()
                if not tool_id:
                    continue
                if selected_ids and tool_id not in selected_ids:
                    continue
                calls_to_run.append(
                    {
                        "tool_id": tool_id,
                        "input": self._resolve_tool_input(
                            tool_id=tool_id,
                            task=task,
                            context=context,
                            planned_call=planned_call,
                        ),
                    }
                )
        else:
            for tool_id in sorted(selected_ids):
                calls_to_run.append(
                    {
                        "tool_id": tool_id,
                        "input": self._resolve_tool_input(
                            tool_id=tool_id,
                            task=task,
                            context=context,
                            planned_call=None,
                        ),
                    }
                )

        executed_calls: List[Dict[str, Any]] = []
        executed_results: List[Dict[str, Any]] = []

        for call in calls_to_run:
            tool_id = str(call.get("tool_id", "")).strip()
            tool_input = call.get("input") if isinstance(call.get("input"), dict) else {}
            tool_meta = self._tool_registry.get_tool(tool_id) or {}
            tool_name = str(tool_meta.get("name") or tool_id)
            started_at = self._utc_now_iso()
            status = "ok"

            try:
                tool_output = self._tool_registry.invoke_tool(tool_id, tool_input)
                if not isinstance(tool_output, dict):
                    tool_output = {"value": tool_output}
                if "error" in tool_output:
                    status = "error"
            except Exception as exc:
                status = "error"
                tool_output = {
                    "query": str(tool_input.get("query", "")),
                    "results": [],
                    "error": {
                        "type": "tool_exception",
                        "message": str(exc),
                    },
                }

            ended_at = self._utc_now_iso()
            call_record = {
                "tool_id": tool_id,
                "input": tool_input,
                "output": tool_output,
                "status": status,
            }
            executed_calls.append(call_record)
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
                        input={
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "input": tool_input,
                        },
                        output={
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "output": tool_output,
                        },
                    )
                )

        return executed_calls, executed_results
