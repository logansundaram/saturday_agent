from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from saturday_agent.agents.complex import graph as complex_graph
from saturday_agent.agents.moderate import graph as moderate_graph
from saturday_agent.agents.simple import graph as simple_graph
from saturday_agent.llms.ollama_chat import extract_assistant_text, ollama_chat
from saturday_agent.llms.registry import ModelRegistry
from saturday_agent.routing.router import route_workflow_type
from saturday_agent.runtime.config import RuntimeConfig, load_runtime_config
from saturday_agent.runtime.tracing import ReplayControl, StepEmitter, StepEvent, instrument_node
from saturday_agent.state.models import ToolDefinition, WorkflowDefinition as WorkflowDefModel
from saturday_agent.state.models import WorkflowState, build_initial_state
from saturday_agent.tools.registry import ToolRegistry

_TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")
_ALLOWED_EXPR_NODES = {
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Slice,
}


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    workflow_type: str
    title: str
    description: str
    graph_spec: Dict[str, Any]
    source: str = "builtin"
    enabled: bool = True
    build_graph_fn: Optional[Callable[..., Any]] = None


class WorkflowRegistry:
    def __init__(
        self,
        *,
        config: Optional[RuntimeConfig] = None,
        model_registry: Optional[ModelRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        dynamic_tool_definitions: Optional[List[ToolDefinition]] = None,
        dynamic_workflow_definitions: Optional[List[WorkflowDefModel]] = None,
    ) -> None:
        self._config = config or load_runtime_config()
        self._model_registry = model_registry or ModelRegistry(
            base_url=self._config.ollama_base_url,
            default_model=self._config.default_model,
            timeout_seconds=self._config.ollama_timeout_seconds,
        )
        if tool_registry is not None:
            self._tool_registry = tool_registry
            if dynamic_tool_definitions:
                self._tool_registry.register_dynamic_tools(
                    [dict(item) for item in dynamic_tool_definitions if isinstance(item, dict)]
                )
        else:
            self._tool_registry = ToolRegistry(
                dynamic_tools=[
                    dict(item)
                    for item in (dynamic_tool_definitions or [])
                    if isinstance(item, dict)
                ]
            )

        self._workflows: Dict[str, WorkflowDefinition] = {
            simple_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=simple_graph.WORKFLOW_ID,
                workflow_type=simple_graph.WORKFLOW_TYPE,
                title=simple_graph.WORKFLOW_TITLE,
                description=simple_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(simple_graph.GRAPH_SPEC),
                source="builtin",
                enabled=True,
                build_graph_fn=simple_graph.build_graph,
            ),
            moderate_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=moderate_graph.WORKFLOW_ID,
                workflow_type=moderate_graph.WORKFLOW_TYPE,
                title=moderate_graph.WORKFLOW_TITLE,
                description=moderate_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(moderate_graph.GRAPH_SPEC),
                source="builtin",
                enabled=True,
                build_graph_fn=moderate_graph.build_graph,
            ),
            complex_graph.WORKFLOW_ID: WorkflowDefinition(
                workflow_id=complex_graph.WORKFLOW_ID,
                workflow_type=complex_graph.WORKFLOW_TYPE,
                title=complex_graph.WORKFLOW_TITLE,
                description=complex_graph.WORKFLOW_DESCRIPTION,
                graph_spec=copy.deepcopy(complex_graph.GRAPH_SPEC),
                source="builtin",
                enabled=True,
                build_graph_fn=complex_graph.build_graph,
            ),
        }

        self._register_dynamic_workflows(dynamic_workflow_definitions or [])

    def _register_dynamic_workflows(
        self,
        workflow_defs: List[WorkflowDefModel],
    ) -> None:
        for raw in workflow_defs:
            if not isinstance(raw, dict):
                continue

            workflow_id = str(raw.get("id") or "").strip()
            if not workflow_id or workflow_id in self._workflows:
                continue

            graph = raw.get("graph") if isinstance(raw.get("graph"), dict) else {}
            if not graph:
                compiled = raw.get("compiled") if isinstance(raw.get("compiled"), dict) else {}
                runtime_graph = (
                    compiled.get("runtime_graph")
                    if isinstance(compiled.get("runtime_graph"), dict)
                    else None
                )
                if isinstance(runtime_graph, dict):
                    graph = runtime_graph
            self._workflows[workflow_id] = WorkflowDefinition(
                workflow_id=workflow_id,
                workflow_type=str(raw.get("type") or "custom"),
                title=str(raw.get("name") or raw.get("title") or workflow_id),
                description=str(raw.get("description") or ""),
                graph_spec=copy.deepcopy(graph),
                source=str(raw.get("source") or "custom"),
                enabled=bool(raw.get("enabled", True)),
                build_graph_fn=None,
            )

    def list_workflows(self) -> List[Dict[str, Any]]:
        order = {"simple": 0, "moderate": 1, "complex": 2}

        def _sort_key(item: WorkflowDefinition) -> tuple[int, str]:
            source_bias = 0 if item.source == "builtin" else 1
            return (source_bias, order.get(item.workflow_type, 99), item.title.lower())

        ordered = sorted(self._workflows.values(), key=_sort_key)
        return [
            {
                "id": workflow.workflow_id,
                "title": workflow.title,
                "name": workflow.title,
                "description": workflow.description,
                "type": workflow.workflow_type,
                "version": workflow.workflow_id.split(".")[-1],
                "status": "available" if workflow.enabled else "disabled",
                "source": workflow.source,
                "enabled": workflow.enabled,
            }
            for workflow in ordered
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._tool_registry.list_tools()

    def invoke_tool(
        self,
        *,
        tool_id: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._tool_registry.invoke_tool(
            tool_id=tool_id,
            tool_input=tool_input,
            context=context,
        )

    def list_models(self) -> Dict[str, Any]:
        return self._model_registry.list_models_payload()

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow_id: {workflow_id}")
        return workflow

    def get_workflow_id_for_type(self, workflow_type: str) -> str:
        for workflow in self._workflows.values():
            if workflow.workflow_type == workflow_type and workflow.enabled:
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
        tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        replay_mode: bool = False,
        resume_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        workflow = self.get_workflow(workflow_id)
        if not workflow.enabled:
            raise ValueError(f"Workflow '{workflow_id}' is disabled.")

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
        raw_tool_defs = input.get("tool_defs")
        tool_defs = (
            [dict(item) for item in raw_tool_defs if isinstance(item, dict)]
            if isinstance(raw_tool_defs, list)
            else []
        )

        if model:
            context = dict(context)
            context["model_id"] = model
        if tool_ids:
            context = dict(context)
            context["selected_tool_ids"] = tool_ids
        if tool_defs:
            context = dict(context)
            context["tool_defs"] = tool_defs

        state = build_initial_state(
            task=task,
            context=context,
            messages=raw_messages,
            model=model,
            options=options,
            default_model=self._model_registry.get_default_model(),
        )
        if isinstance(initial_state, dict):
            hydrated = copy.deepcopy(state)
            for key, value in initial_state.items():
                hydrated[str(key)] = copy.deepcopy(value)
            state = hydrated

        replay_control = ReplayControl(
            replay_mode=replay_mode,
            resume_node_id=resume_node_id,
        )

        if not state.get("task") and not state.get("messages"):
            raise ValueError("Workflow input must include task or messages.")

        if workflow.source == "custom":
            final_state = self._run_custom_workflow(
                workflow=workflow,
                state=state,
                step_emitter=step_emitter,
                tool_call_gate=tool_call_gate,
                replay_control=replay_control,
            )
        else:
            if workflow.build_graph_fn is None:
                raise ValueError(f"Workflow '{workflow_id}' is not executable.")
            compiled_graph = workflow.build_graph_fn(
                config=self._config,
                model_registry=self._model_registry,
                tool_registry=self._tool_registry,
                step_emitter=step_emitter,
                replay_control=replay_control,
            )
            final_state = compiled_graph.invoke(state)
        if replay_mode and str(resume_node_id or "").strip() and not replay_control.resume_matched:
            raise ValueError(f"Replay resume node '{resume_node_id}' was not reached.")

        raw_planned_calls = final_state.get("tool_calls")
        planned_calls = raw_planned_calls if isinstance(raw_planned_calls, list) else []
        selected_tool_ids = list(tool_ids)
        fallback_tool_ids = list(tool_ids)

        retrieval_state = final_state.get("retrieval")
        retrieval_results = (
            retrieval_state.get("results")
            if isinstance(retrieval_state, dict)
            else None
        )
        retrieval_ran = isinstance(retrieval_results, list)
        if retrieval_ran:
            fallback_tool_ids = [tool_id for tool_id in fallback_tool_ids if tool_id != "rag.retrieve"]
            planned_calls = [
                call
                for call in planned_calls
                if isinstance(call, dict)
                and str(call.get("tool_id") or call.get("id") or "").strip() != "rag.retrieve"
            ]

        existing_results = final_state.get("tool_results")
        has_existing_results = isinstance(existing_results, list) and len(existing_results) > 0
        post_graph_tool_execution = False
        if workflow.workflow_type == "moderate":
            post_graph_tool_execution = not has_existing_results and bool(
                planned_calls or fallback_tool_ids
            )
        elif workflow.workflow_type == "complex":
            post_graph_tool_execution = not has_existing_results and bool(
                planned_calls or fallback_tool_ids
            )

        if post_graph_tool_execution:
            executed_calls, executed_results = self._execute_selected_tools(
                task=task,
                context=context,
                tool_ids=fallback_tool_ids,
                planned_calls=planned_calls,
                step_emitter=step_emitter,
                tool_call_gate=tool_call_gate,
            )
            if executed_calls:
                final_state["tool_calls"] = executed_calls
                final_state["tool_results"] = executed_results

        output: Dict[str, Any] = {
            "answer": str(final_state.get("answer") or ""),
            "plan": final_state.get("plan"),
            "artifacts": dict(final_state.get("artifacts") or {}),
        }
        if selected_tool_ids:
            output["selected_tool_ids"] = selected_tool_ids
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

        if task and not str(resolved_input.get("query", "")).strip():
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
        tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]],
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
            if not bool(tool_meta.get("enabled", False)):
                continue
            started_at = self._utc_now_iso()
            status = "ok"

            if tool_call_gate:
                allowed = bool(
                    tool_call_gate(
                        {
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "input": dict(tool_input or {}),
                            "started_at": started_at,
                        }
                    )
                )
                if not allowed:
                    raise ValueError(f"Tool '{tool_id}' rejected by sandbox policy.")

            try:
                invoke_context = dict(context)
                invoke_context.setdefault("task", task)
                tool_output = self._tool_registry.invoke(
                    tool_id=tool_id,
                    tool_input=tool_input,
                    context=invoke_context,
                )
                if not isinstance(tool_output, dict):
                    tool_output = {"ok": True, "type": "unknown", "data": tool_output, "meta": {}}
                if not bool(tool_output.get("ok", True)) or "error" in tool_output:
                    status = "error"
            except Exception as exc:
                status = "error"
                tool_output = {
                    "ok": False,
                    "type": str(tool_meta.get("type") or "unknown"),
                    "error": {
                        "kind": "tool_exception",
                        "message": str(exc),
                    },
                    "meta": {"tool_id": tool_id},
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
                            "payload": tool_input,
                        },
                        output={
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "response": tool_output,
                        },
                    )
                )

        return executed_calls, executed_results

    def _run_custom_workflow(
        self,
        *,
        workflow: WorkflowDefinition,
        state: WorkflowState,
        step_emitter: Optional[StepEmitter],
        tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        replay_control: Optional[ReplayControl] = None,
    ) -> WorkflowState:
        compiled = self._compile_custom_workflow(
            workflow=workflow,
            step_emitter=step_emitter,
            tool_call_gate=tool_call_gate,
            replay_control=replay_control,
        )
        final_state = compiled.invoke(state)
        if not isinstance(final_state, dict):
            raise ValueError("Custom workflow returned invalid state.")
        return final_state

    def _compile_custom_workflow(
        self,
        *,
        workflow: WorkflowDefinition,
        step_emitter: Optional[StepEmitter],
        tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        replay_control: Optional[ReplayControl] = None,
    ) -> Any:
        graph = workflow.graph_spec if isinstance(workflow.graph_spec, dict) else {}
        raw_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        raw_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("id") or "").strip()
            if not node_id:
                continue
            nodes_by_id[node_id] = {
                "id": node_id,
                "type": str(raw_node.get("type") or "").strip().lower(),
                "config": dict(raw_node.get("config") or {}),
            }

        start_nodes = [node for node in nodes_by_id.values() if node.get("type") == "start"]
        end_nodes = [node for node in nodes_by_id.values() if node.get("type") == "end"]
        if len(start_nodes) != 1 or len(end_nodes) != 1:
            raise ValueError("Custom workflow must include exactly one start and one end node.")

        start_id = str(start_nodes[0]["id"])
        end_id = str(end_nodes[0]["id"])

        edges_by_from: Dict[str, List[Dict[str, str]]] = {}
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            from_id = str(raw_edge.get("from") or "").strip()
            to_id = str(raw_edge.get("to") or "").strip()
            if not from_id or not to_id:
                continue
            if from_id not in nodes_by_id or to_id not in nodes_by_id:
                continue
            condition = str(raw_edge.get("condition") or "always").strip().lower()
            if condition not in {"always", "true", "false"}:
                condition = "always"
            edges_by_from.setdefault(from_id, []).append(
                {
                    "from": from_id,
                    "to": to_id,
                    "condition": condition,
                }
            )

        builder = StateGraph(WorkflowState)

        for node_id, node in nodes_by_id.items():
            node_type = str(node.get("type") or "")
            config = dict(node.get("config") or {})
            if node_type == "start":
                continue

            if node_type == "llm":
                node_fn = self._build_custom_llm_node(
                    node_id=node_id,
                    config=config,
                )
            elif node_type == "tool":
                node_fn = self._build_custom_tool_node(
                    node_id=node_id,
                    config=config,
                    step_emitter=step_emitter,
                    tool_call_gate=tool_call_gate,
                )
            elif node_type in {"condition", "conditional"}:
                node_fn = self._build_custom_condition_node(
                    node_id=node_id,
                    config=config,
                )
            elif node_type in {"end", "finalize"}:
                node_fn = self._build_custom_end_node(
                    workflow=workflow,
                    node_id=node_id,
                    config=config,
                )
            elif node_type == "verify":
                node_fn = self._build_custom_verify_node(
                    node_id=node_id,
                    config=config,
                )
            else:
                node_fn = self._build_noop_node(node_id=node_id)

            builder.add_node(
                node_id,
                instrument_node(
                    name=f"node.{node_id}",
                    node_fn=node_fn,
                    step_emitter=step_emitter,
                    replay_control=replay_control,
                ),
            )

        start_outgoing = edges_by_from.get(start_id, [])
        if not start_outgoing:
            builder.add_edge(START, end_id)
        else:
            added = False
            for edge in start_outgoing:
                target_id = edge.get("to")
                if not target_id:
                    continue
                builder.add_edge(START, target_id)
                added = True
            if not added:
                builder.add_edge(START, end_id)

        for node_id, node in nodes_by_id.items():
            if node_id == start_id:
                continue
            if node_id == end_id:
                builder.add_edge(node_id, END)
                continue

            node_type = str(node.get("type") or "")
            outgoing = edges_by_from.get(node_id, [])
            if node_type in {"condition", "conditional"}:
                true_target = ""
                false_target = ""
                for edge in outgoing:
                    condition = str(edge.get("condition") or "always")
                    if condition == "true" and not true_target:
                        true_target = str(edge.get("to") or "")
                    elif condition == "false" and not false_target:
                        false_target = str(edge.get("to") or "")

                if not true_target and outgoing:
                    true_target = str(outgoing[0].get("to") or "")
                if not false_target:
                    false_target = end_id
                if not true_target:
                    true_target = end_id

                router = self._build_condition_router(config=dict(node.get("config") or {}))
                builder.add_conditional_edges(
                    node_id,
                    router,
                    {
                        "true": true_target,
                        "false": false_target,
                    },
                )
                continue

            targets: List[str] = []
            for edge in outgoing:
                if str(edge.get("condition") or "always") == "always":
                    target = str(edge.get("to") or "")
                    if target and target not in targets:
                        targets.append(target)
            if not targets:
                if outgoing:
                    fallback_target = str(outgoing[0].get("to") or "")
                    if fallback_target:
                        targets.append(fallback_target)
                else:
                    targets.append(end_id)
            for target in targets:
                builder.add_edge(node_id, target)

        return builder.compile()

    def _build_custom_llm_node(
        self,
        *,
        node_id: str,
        config: Dict[str, Any],
    ) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            prompt_template = str(config.get("prompt_template") or "").strip()
            if not prompt_template:
                prompt_template = "{{query}}"

            prompt = self._render_state_template(
                template=prompt_template,
                state=state,
                payload={},
            ).strip()
            if not prompt:
                prompt = str(state.get("task") or "")

            model_id = str(state.get("model") or self._model_registry.get_default_model())
            options = dict(state.get("options") or {})
            if isinstance(config.get("temperature"), (int, float)) and "temperature" not in options:
                options["temperature"] = float(config.get("temperature") or 0.2)

            raw = ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                base_url=self._config.ollama_base_url,
                timeout_seconds=self._config.ollama_timeout_seconds,
                options=options,
            )
            text = extract_assistant_text(raw).strip()

            output_key = str(config.get("output_key") or f"{node_id}_output").strip() or node_id
            artifacts = dict(state.get("artifacts") or {})
            artifacts[output_key] = text
            artifacts["last_output"] = text

            messages = list(state.get("messages") or [])
            messages.append({"role": "assistant", "content": text})

            return {
                "artifacts": artifacts,
                "messages": messages,
            }

        return _node

    def _build_custom_tool_node(
        self,
        *,
        node_id: str,
        config: Dict[str, Any],
        step_emitter: Optional[StepEmitter],
        tool_call_gate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            tool_id = str(config.get("tool_name") or config.get("tool_id") or "").strip()
            if not tool_id:
                raise ValueError(f"Tool node '{node_id}' is missing tool_name.")

            input_map = (
                config.get("args_map")
                if isinstance(config.get("args_map"), dict)
                else config.get("input_map")
            )
            input_map = input_map if isinstance(input_map, dict) else {}
            payload: Dict[str, Any] = {}
            for key, expression in input_map.items():
                output_key = str(key).strip()
                if not output_key:
                    continue
                payload[output_key] = self._resolve_input_expression(
                    expression=expression,
                    state=state,
                )

            if not payload:
                payload = {"query": str(state.get("task") or "")}
            elif "query" not in payload:
                payload["query"] = str(state.get("task") or "")

            invoke_context = dict(state.get("context") or {})
            invoke_context["task"] = str(state.get("task") or "")
            if state.get("model"):
                invoke_context["model_id"] = str(state.get("model"))

            started_at = self._utc_now_iso()
            tool_meta = self._tool_registry.get_tool(tool_id) or {}
            if tool_call_gate:
                allowed = bool(
                    tool_call_gate(
                        {
                            "tool_id": tool_id,
                            "tool_name": str(tool_meta.get("name") or tool_id),
                            "input": dict(payload),
                            "started_at": started_at,
                        }
                    )
                )
                if not allowed:
                    raise ValueError(f"Tool '{tool_id}' rejected by sandbox policy.")

            output = self._tool_registry.invoke(tool_id=tool_id, tool_input=payload, context=invoke_context)
            status = "ok" if bool(output.get("ok", False)) else "error"
            ended_at = self._utc_now_iso()

            if step_emitter:
                step_emitter(
                    StepEvent(
                        name=f"tool.{tool_id}",
                        status=status,
                        started_at=started_at,
                        ended_at=ended_at,
                        input={"tool_id": tool_id, "payload": payload},
                        output={"tool_id": tool_id, "response": output},
                    )
                )

            output_key = str(config.get("output_key") or f"{node_id}_output").strip() or node_id
            artifacts = dict(state.get("artifacts") or {})
            artifacts[output_key] = output
            if bool(output.get("ok", False)):
                data = output.get("data")
                if isinstance(data, dict) and isinstance(data.get("text"), str):
                    artifacts["last_output"] = str(data.get("text"))
                elif isinstance(data, str):
                    artifacts["last_output"] = data

            tool_calls = list(state.get("tool_calls") or [])
            tool_results = list(state.get("tool_results") or [])
            tool_calls.append(
                {
                    "tool_id": tool_id,
                    "input": payload,
                    "output": output,
                    "status": status,
                }
            )
            tool_results.append(
                {
                    "tool_id": tool_id,
                    "status": status,
                    "output": output,
                }
            )

            return {
                "artifacts": artifacts,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "plan": state.get("plan"),
                "answer": state.get("answer"),
                "messages": list(state.get("messages") or []),
                "context": dict(state.get("context") or {}),
                "task": str(state.get("task") or ""),
                "model": state.get("model"),
                "options": dict(state.get("options") or {}),
                "trace": list(state.get("trace") or []),
                "verify_ok": state.get("verify_ok"),
                "verify_notes": state.get("verify_notes"),
                "retry_count": int(state.get("retry_count") or 0),
                "artifact_ids": list(state.get("artifact_ids") or []),
                "vision_model_id": state.get("vision_model_id"),
                "tool_defs": list(state.get("tool_defs") or []),
                "workflow_defs": list(state.get("workflow_defs") or []),
            }

        return _node

    def _build_custom_condition_node(
        self,
        *,
        node_id: str,
        config: Dict[str, Any],
    ) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            result = self._evaluate_condition(state=state, config=config)
            artifacts = dict(state.get("artifacts") or {})
            artifacts[f"{node_id}.result"] = result
            artifacts["last_condition"] = result
            return {
                "artifacts": artifacts,
            }

        return _node

    def _build_custom_verify_node(
        self,
        *,
        node_id: str,
        config: Dict[str, Any],
    ) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            mode = str(config.get("mode") or "rule").strip().lower()
            verify_ok = False
            verify_notes = ""

            if mode == "llm":
                prompt_template = str(config.get("prompt_template") or "").strip()
                if not prompt_template:
                    raise ValueError(f"Verify node '{node_id}' requires prompt_template in llm mode.")
                prompt = self._render_state_template(
                    template=prompt_template,
                    state=state,
                    payload={},
                )
                model_id = str(state.get("model") or self._model_registry.get_default_model())
                raw = ollama_chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    base_url=self._config.ollama_base_url,
                    timeout_seconds=self._config.ollama_timeout_seconds,
                    options=dict(state.get("options") or {}),
                )
                verify_notes = extract_assistant_text(raw).strip()
                verify_ok = verify_notes.lower().startswith("true") or "[pass]" in verify_notes.lower()
            else:
                expression = str(config.get("expression") or "").strip()
                if not expression:
                    raise ValueError(f"Verify node '{node_id}' requires expression in rule mode.")
                verify_ok = self._safe_eval_expression(expression=expression, state=state)
                verify_notes = (
                    str(config.get("fail_message") or "").strip()
                    if not verify_ok
                    else "Rule verification passed."
                )

            output_key = str(config.get("output_key") or f"{node_id}_verified").strip() or f"{node_id}_verified"
            artifacts = dict(state.get("artifacts") or {})
            artifacts[output_key] = verify_ok
            artifacts[f"{node_id}.notes"] = verify_notes
            return {
                "verify_ok": verify_ok,
                "verify_notes": verify_notes,
                "artifacts": artifacts,
            }

        return _node

    def _build_custom_end_node(
        self,
        *,
        workflow: WorkflowDefinition,
        node_id: str,
        config: Dict[str, Any],
    ) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            response_template = str(config.get("response_template") or "")
            artifacts = dict(state.get("artifacts") or {})

            if response_template.strip():
                answer = self._render_state_template(
                    template=response_template,
                    state=state,
                    payload={},
                ).strip()
            else:
                answer = str(artifacts.get("last_output") or state.get("answer") or "").strip()

            if not answer:
                answer = str(state.get("task") or "")

            artifacts.update(
                {
                    "workflow_id": workflow.workflow_id,
                    "workflow_type": workflow.workflow_type,
                    "workflow_source": workflow.source,
                    "terminal_node": node_id,
                }
            )

            return {
                "answer": answer,
                "artifacts": artifacts,
            }

        return _node

    @staticmethod
    def _build_noop_node(*, node_id: str) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
        def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
            artifacts = dict(state.get("artifacts") or {})
            artifacts[f"{node_id}.noop"] = True
            return {"artifacts": artifacts}

        return _node

    def _build_condition_router(
        self,
        *,
        config: Dict[str, Any],
    ) -> Callable[[Mapping[str, Any]], Literal["true", "false"]]:
        def _router(state: Mapping[str, Any]) -> Literal["true", "false"]:
            return "true" if self._evaluate_condition(state=state, config=config) else "false"

        return _router

    def _evaluate_condition(
        self,
        *,
        state: Mapping[str, Any],
        config: Dict[str, Any],
    ) -> bool:
        expression = str(config.get("expression") or "").strip()
        if expression:
            return self._safe_eval_expression(expression=expression, state=state)

        field = str(config.get("field") or "").strip()
        operator = str(config.get("operator") or "exists").strip().lower()
        expected = config.get("value")

        value = self._resolve_state_value(state=state, path=field)

        if operator == "exists":
            return value is not None
        if operator == "not_exists":
            return value is None
        if operator == "equals":
            return value == expected
        if operator == "contains":
            if isinstance(value, (list, tuple, set)):
                return expected in value
            return str(expected) in str(value or "")
        if operator == "gt":
            try:
                return float(value) > float(expected)
            except (TypeError, ValueError):
                return False
        if operator == "lt":
            try:
                return float(value) < float(expected)
            except (TypeError, ValueError):
                return False
        if operator == "in":
            if isinstance(expected, (list, tuple, set)):
                return value in expected
            return False
        return False

    def _safe_eval_expression(
        self,
        *,
        expression: str,
        state: Mapping[str, Any],
    ) -> bool:
        parsed = ast.parse(expression, mode="eval")
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                raise ValueError("Expression calls are not allowed.")
            if type(node) not in _ALLOWED_EXPR_NODES:
                raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        state_dict = dict(state)
        artifacts = dict(state.get("artifacts") or {})
        context = dict(state.get("context") or {})

        locals_map: Dict[str, Any] = {
            "state": self._to_namespace(state_dict),
            "artifacts": self._to_namespace(artifacts),
            "context": self._to_namespace(context),
            "task": str(state.get("task") or ""),
            "answer": state.get("answer"),
            "verify_ok": state.get("verify_ok"),
            "verify_notes": state.get("verify_notes"),
            "retry_count": int(state.get("retry_count") or 0),
        }

        for source in (state_dict, artifacts, context):
            for key, value in source.items():
                key_text = str(key).strip()
                if key_text.isidentifier():
                    locals_map[key_text] = value

        value = eval(
            compile(parsed, "<workflow-expression>", "eval"),
            {"__builtins__": {}},
            locals_map,
        )
        return bool(value)

    def _to_namespace(self, value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(
                **{
                    str(key): self._to_namespace(item)
                    for key, item in value.items()
                }
            )
        if isinstance(value, list):
            return [self._to_namespace(item) for item in value]
        return value

    def _resolve_input_expression(self, *, expression: Any, state: Mapping[str, Any]) -> Any:
        if isinstance(expression, str):
            trimmed = expression.strip()
            if "{{" in trimmed and "}}" in trimmed:
                return self._render_state_template(template=trimmed, state=state, payload={})
            resolved = self._resolve_state_value(state=state, path=trimmed)
            if resolved is not None:
                return resolved
            return trimmed
        return expression

    def _render_state_template(
        self,
        *,
        template: str,
        state: Mapping[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        if not template:
            return ""

        def _replace(match: re.Match[str]) -> str:
            token = str(match.group(1) or "").strip()
            if not token:
                return ""

            if token == "query":
                return str(state.get("task") or "")
            if token == "task":
                return str(state.get("task") or "")

            if token.startswith("input."):
                value = self._resolve_path(payload, token[len("input.") :])
                return "" if value is None else str(value)
            if token.startswith("artifacts."):
                value = self._resolve_path(dict(state.get("artifacts") or {}), token[len("artifacts.") :])
                return "" if value is None else str(value)
            if token.startswith("context."):
                value = self._resolve_path(dict(state.get("context") or {}), token[len("context.") :])
                return "" if value is None else str(value)
            if token.startswith("state."):
                value = self._resolve_path(dict(state), token[len("state.") :])
                return "" if value is None else str(value)

            value = self._resolve_state_value(state=state, path=token)
            return "" if value is None else str(value)

        return _TEMPLATE_PATTERN.sub(_replace, template)

    def _resolve_state_value(self, *, state: Mapping[str, Any], path: str) -> Any:
        normalized = str(path or "").strip()
        if not normalized:
            return None

        if normalized.startswith("artifacts."):
            return self._resolve_path(dict(state.get("artifacts") or {}), normalized[len("artifacts.") :])
        if normalized.startswith("context."):
            return self._resolve_path(dict(state.get("context") or {}), normalized[len("context.") :])
        if normalized.startswith("state."):
            return self._resolve_path(dict(state), normalized[len("state.") :])
        if normalized.startswith("input."):
            # Input aliases to artifacts for workflow node interoperability.
            return self._resolve_path(dict(state.get("artifacts") or {}), normalized[len("input.") :])

        artifacts_value = self._resolve_path(dict(state.get("artifacts") or {}), normalized)
        if artifacts_value is not None:
            return artifacts_value

        context_value = self._resolve_path(dict(state.get("context") or {}), normalized)
        if context_value is not None:
            return context_value

        return self._resolve_path(dict(state), normalized)

    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        current = root
        for part in str(path).split("."):
            segment = part.strip()
            if not segment:
                continue
            if isinstance(current, dict):
                current = current.get(segment)
                continue
            return None
        return current
