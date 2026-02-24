from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from app.workflows.schemas import ValidationDiagnosticModel

BASE_STATE_KEYS: Dict[str, str] = {
    "task": "string",
    "context": "json",
    "messages": "json",
    "plan": "string",
    "answer": "string",
    "artifacts": "json",
    "verify_ok": "bool",
    "verify_notes": "string",
    "retry_count": "number",
}

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


@dataclass
class ValidationResult:
    workflow_spec: Dict[str, Any]
    compiled: Dict[str, Any]
    diagnostics: List[ValidationDiagnosticModel]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)


def validate_and_compile_workflow_spec(
    *,
    workflow_spec: Mapping[str, Any],
    tool_registry: Iterable[Mapping[str, Any]],
) -> ValidationResult:
    diagnostics: List[ValidationDiagnosticModel] = []
    normalized = _normalize_workflow_spec(workflow_spec)

    node_ids = [str(item.get("id") or "") for item in normalized["nodes"]]
    seen: Set[str] = set()
    for node_id in node_ids:
        if not node_id:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="SCHEMA_INVALID",
                    severity="error",
                    message="Node id is required.",
                    path="nodes.id",
                )
            )
            continue
        if node_id in seen:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="DUPLICATE_NODE_ID",
                    severity="error",
                    message=f"Duplicate node id '{node_id}'.",
                    node_id=node_id,
                )
            )
        seen.add(node_id)

    node_map = {str(item.get("id") or ""): item for item in normalized["nodes"]}
    finalize_ids = [
        str(item.get("id") or "")
        for item in normalized["nodes"]
        if str(item.get("type") or "") == "finalize"
    ]
    if not finalize_ids:
        diagnostics.append(
            ValidationDiagnosticModel(
                code="MISSING_FINALIZE_NODE",
                severity="error",
                message="Workflow must include at least one finalize node.",
            )
        )

    edge_ids: Set[str] = set()
    adjacency: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_map.keys()}
    edge_records: List[Dict[str, Any]] = []

    for raw in normalized["edges"]:
        edge_id = str(raw.get("id") or "").strip()
        if not edge_id:
            edge_id = f"edge_{len(edge_records) + 1}"
            raw["id"] = edge_id
        if edge_id in edge_ids:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="SCHEMA_INVALID",
                    severity="warning",
                    message=f"Duplicate edge id '{edge_id}' detected; IDs should be unique.",
                    edge_id=edge_id,
                )
            )
        edge_ids.add(edge_id)

        from_node = str(raw.get("from") or "").strip()
        to_node = str(raw.get("to") or "").strip()
        if from_node not in node_map:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="EDGE_NODE_MISSING",
                    severity="error",
                    message=f"Edge '{edge_id}' references unknown source node '{from_node}'.",
                    edge_id=edge_id,
                    path="edges.from",
                )
            )
            continue
        if to_node not in node_map:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="EDGE_NODE_MISSING",
                    severity="error",
                    message=f"Edge '{edge_id}' references unknown target node '{to_node}'.",
                    edge_id=edge_id,
                    path="edges.to",
                )
            )
            continue

        adjacency[from_node].append(to_node)
        indegree[to_node] = int(indegree.get(to_node, 0)) + 1
        edge_records.append(raw)

    entry_nodes = sorted([node_id for node_id, count in indegree.items() if count == 0])
    if not entry_nodes and node_map:
        diagnostics.append(
            ValidationDiagnosticModel(
                code="SCHEMA_INVALID",
                severity="error",
                message="No entry node found. At least one node must have no incoming edges.",
            )
        )

    has_cycle, cycle_path = _detect_cycle(node_map.keys(), adjacency)
    allow_cycles = bool(normalized.get("allow_cycles", False))
    if has_cycle and not allow_cycles:
        diagnostics.append(
            ValidationDiagnosticModel(
                code="CYCLE_DETECTED",
                severity="error",
                message=f"Cycle detected: {' -> '.join(cycle_path)}.",
            )
        )
    elif has_cycle and allow_cycles:
        diagnostics.append(
            ValidationDiagnosticModel(
                code="CYCLE_DETECTED",
                severity="warning",
                message=f"Cycle detected but allowed: {' -> '.join(cycle_path)}.",
            )
        )

    reachable = _reachable_from_entries(entry_nodes, adjacency)
    for node_id in sorted(node_map.keys()):
        if node_id not in reachable:
            diagnostics.append(
                ValidationDiagnosticModel(
                    code="UNREACHABLE_NODE",
                    severity="error",
                    message=f"Node '{node_id}' is unreachable.",
                    node_id=node_id,
                )
            )

    if finalize_ids and not any(node_id in reachable for node_id in finalize_ids):
        diagnostics.append(
            ValidationDiagnosticModel(
                code="FINALIZE_UNREACHABLE",
                severity="error",
                message="No reachable finalize node from workflow entry points.",
            )
        )

    declared_state_keys: Dict[str, str] = {
        str(item.get("key") or "").strip(): str(item.get("type") or "json")
        for item in normalized.get("state_schema", [])
        if str(item.get("key") or "").strip()
    }

    available_tool_ids = {
        str(item.get("id") or "").strip()
        for item in tool_registry
        if str(item.get("id") or "").strip()
    }

    for node in normalized["nodes"]:
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "")
        writes = [str(item).strip() for item in list(node.get("writes") or []) if str(item).strip()]
        reads = [str(item).strip() for item in list(node.get("reads") or []) if str(item).strip()]

        for key in writes:
            if key not in declared_state_keys:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="STATE_KEY_WRITE_UNDECLARED",
                        severity="error",
                        message=f"Node '{node_id}' writes undeclared state key '{key}'.",
                        node_id=node_id,
                    )
                )

        config = dict(node.get("config") or {})
        if node_type == "tool":
            tool_name = str(config.get("tool_name") or config.get("tool_id") or "").strip()
            if not tool_name:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="NODE_CONFIG_INVALID",
                        severity="error",
                        message=f"Tool node '{node_id}' requires config.tool_name.",
                        node_id=node_id,
                    )
                )
            elif tool_name not in available_tool_ids:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="TOOL_NOT_FOUND",
                        severity="error",
                        message=f"Tool node '{node_id}' references unknown tool '{tool_name}'.",
                        node_id=node_id,
                    )
                )

        if node_type == "conditional":
            expression = _normalize_conditional_expression(config)
            if not expression:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="CONDITIONAL_EXPRESSION_INVALID",
                        severity="error",
                        message=f"Conditional node '{node_id}' requires a valid expression.",
                        node_id=node_id,
                    )
                )
            else:
                refs, error_text = _validate_expression(expression)
                if error_text:
                    diagnostics.append(
                        ValidationDiagnosticModel(
                            code="CONDITIONAL_EXPRESSION_INVALID",
                            severity="error",
                            message=f"Conditional node '{node_id}' expression invalid: {error_text}",
                            node_id=node_id,
                        )
                    )
                allowed_expr_refs = set(declared_state_keys.keys()) | set(BASE_STATE_KEYS.keys())
                for ref in sorted(refs):
                    if ref in {"True", "False", "None"}:
                        continue
                    root = ref.split(".")[0]
                    if ref not in allowed_expr_refs and root not in allowed_expr_refs:
                        diagnostics.append(
                            ValidationDiagnosticModel(
                                code="CONDITIONAL_EXPRESSION_KEY_UNKNOWN",
                                severity="error",
                                message=f"Conditional node '{node_id}' references unknown key '{ref}'.",
                                node_id=node_id,
                            )
                        )
                config["expression"] = expression
                node["config"] = config

        if node_type == "verify":
            mode = str(config.get("mode") or "rule").strip().lower()
            if mode not in {"rule", "llm"}:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="VERIFY_CONFIG_INVALID",
                        severity="error",
                        message=f"Verify node '{node_id}' mode must be rule or llm.",
                        node_id=node_id,
                    )
                )
            if mode == "rule":
                expression = str(config.get("expression") or "").strip()
                if not expression:
                    diagnostics.append(
                        ValidationDiagnosticModel(
                            code="VERIFY_CONFIG_INVALID",
                            severity="error",
                            message=f"Verify node '{node_id}' in rule mode requires expression.",
                            node_id=node_id,
                        )
                    )
            if mode == "llm":
                prompt = str(config.get("prompt_template") or "").strip()
                if not prompt:
                    diagnostics.append(
                        ValidationDiagnosticModel(
                            code="VERIFY_CONFIG_INVALID",
                            severity="error",
                            message=f"Verify node '{node_id}' in llm mode requires prompt_template.",
                            node_id=node_id,
                        )
                    )

        node["reads"] = sorted(set(reads))
        node["writes"] = sorted(set(writes))

    if not has_cycle:
        _validate_reads_availability(
            normalized=normalized,
            adjacency=adjacency,
            indegree=indegree,
            declared_state_keys=declared_state_keys,
            diagnostics=diagnostics,
        )

    normalized["nodes"] = sorted(
        normalized["nodes"],
        key=lambda item: str(item.get("id") or ""),
    )
    normalized["edges"] = sorted(
        edge_records,
        key=lambda item: (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("label") or ""),
            str(item.get("id") or ""),
        ),
    )

    compiled = _compile_from_normalized(normalized, entry_nodes=entry_nodes)
    return ValidationResult(workflow_spec=normalized, compiled=compiled, diagnostics=diagnostics)


def _normalize_workflow_spec(workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "workflow_id": str(workflow_spec.get("workflow_id") or "").strip() or None,
        "name": str(workflow_spec.get("name") or "").strip(),
        "description": str(workflow_spec.get("description") or "").strip(),
        "allow_cycles": bool(workflow_spec.get("allow_cycles", False)),
        "state_schema": [],
        "nodes": [],
        "edges": [],
        "metadata": dict(workflow_spec.get("metadata") or {}),
    }

    raw_state_schema = workflow_spec.get("state_schema")
    if isinstance(raw_state_schema, list):
        seen_keys: Set[str] = set()
        for raw in raw_state_schema:
            if not isinstance(raw, Mapping):
                continue
            key = str(raw.get("key") or "").strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            output["state_schema"].append(
                {
                    "key": key,
                    "type": str(raw.get("type") or "json").strip().lower() or "json",
                    "description": str(raw.get("description") or "").strip(),
                    "required": bool(raw.get("required", False)),
                }
            )

    raw_nodes = workflow_spec.get("nodes")
    if isinstance(raw_nodes, list):
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                continue
            node = {
                "id": str(raw.get("id") or "").strip(),
                "type": str(raw.get("type") or "").strip().lower(),
                "label": str(raw.get("label") or "").strip(),
                "reads": [
                    str(item).strip()
                    for item in list(raw.get("reads") or [])
                    if str(item).strip()
                ],
                "writes": [
                    str(item).strip()
                    for item in list(raw.get("writes") or [])
                    if str(item).strip()
                ],
                "config": dict(raw.get("config") or {}),
                "position": dict(raw.get("position") or {}) if isinstance(raw.get("position"), Mapping) else None,
            }
            output["nodes"].append(node)

    raw_edges = workflow_spec.get("edges")
    if isinstance(raw_edges, list):
        for idx, raw in enumerate(raw_edges):
            if not isinstance(raw, Mapping):
                continue
            edge = {
                "id": str(raw.get("id") or f"edge_{idx + 1}").strip(),
                "from": str(raw.get("from") or raw.get("from_node") or "").strip(),
                "to": str(raw.get("to") or "").strip(),
                "label": str(raw.get("label") or "always").strip() or "always",
            }
            output["edges"].append(edge)

    return output


def _detect_cycle(nodes: Iterable[str], adjacency: Mapping[str, List[str]]) -> Tuple[bool, List[str]]:
    state: Dict[str, int] = {str(node): 0 for node in nodes}
    parent: Dict[str, Optional[str]] = {str(node): None for node in nodes}

    def _dfs(node: str) -> Tuple[bool, List[str]]:
        state[node] = 1
        for nxt in adjacency.get(node, []):
            if state.get(nxt, 0) == 0:
                parent[nxt] = node
                has_cycle, path = _dfs(nxt)
                if has_cycle:
                    return True, path
            elif state.get(nxt) == 1:
                path: List[str] = [nxt]
                current = node
                while current and current != nxt:
                    path.append(current)
                    current = parent.get(current)
                path.append(nxt)
                path.reverse()
                return True, path
        state[node] = 2
        return False, []

    for node in state.keys():
        if state[node] == 0:
            has_cycle, cycle_path = _dfs(node)
            if has_cycle:
                return True, cycle_path
    return False, []


def _reachable_from_entries(entry_nodes: List[str], adjacency: Mapping[str, List[str]]) -> Set[str]:
    seen: Set[str] = set()
    queue: deque[str] = deque(entry_nodes)
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for nxt in adjacency.get(node, []):
            if nxt not in seen:
                queue.append(nxt)
    return seen


def _normalize_conditional_expression(config: Mapping[str, Any]) -> str:
    expression = str(config.get("expression") or "").strip()
    if expression:
        return expression

    field = str(config.get("field") or "").strip()
    operator = str(config.get("operator") or "").strip().lower()
    value = config.get("value")
    if not field or not operator:
        return ""

    if operator == "equals":
        return f"{field} == {repr(value)}"
    if operator == "contains":
        return f"{repr(value)} in {field}"
    if operator == "gt":
        return f"{field} > {repr(value)}"
    if operator == "lt":
        return f"{field} < {repr(value)}"
    if operator == "exists":
        return f"{field} is not None"
    if operator == "not_exists":
        return f"{field} is None"
    if operator == "in":
        return f"{field} in {repr(value)}"
    return ""


def _validate_expression(expression: str) -> Tuple[Set[str], Optional[str]]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return set(), str(exc)

    refs: Set[str] = set()

    for node in ast.walk(parsed):
        if type(node) not in _ALLOWED_EXPR_NODES:
            return set(), f"Unsupported expression node: {type(node).__name__}"
        if isinstance(node, ast.Call):
            return set(), "Function calls are not allowed"
        if isinstance(node, ast.Name):
            refs.add(str(node.id))
        if isinstance(node, ast.Attribute):
            ref = _attribute_to_path(node)
            if ref:
                refs.add(ref)

    return refs, None


def _attribute_to_path(node: ast.Attribute) -> Optional[str]:
    parts: List[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(str(current.attr))
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(str(current.id))
        parts.reverse()
        return ".".join(parts)
    return None


def _validate_reads_availability(
    *,
    normalized: Dict[str, Any],
    adjacency: Mapping[str, List[str]],
    indegree: Mapping[str, int],
    declared_state_keys: Mapping[str, str],
    diagnostics: List[ValidationDiagnosticModel],
) -> None:
    node_map = {str(item.get("id") or ""): item for item in normalized["nodes"]}
    predecessors: Dict[str, Set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            predecessors[target].add(source)

    queue: deque[str] = deque([node_id for node_id, value in indegree.items() if int(value) == 0])
    in_deg = {node_id: int(value) for node_id, value in indegree.items()}
    available_by_node: Dict[str, Set[str]] = {}

    while queue:
        node_id = queue.popleft()
        preds = predecessors.get(node_id, set())

        available = set(BASE_STATE_KEYS.keys()) | set(declared_state_keys.keys())
        for pred in preds:
            available |= available_by_node.get(pred, set())

        node = node_map.get(node_id) or {}
        reads = [str(item).strip() for item in list(node.get("reads") or []) if str(item).strip()]
        writes = [str(item).strip() for item in list(node.get("writes") or []) if str(item).strip()]

        for key in reads:
            if key not in available:
                diagnostics.append(
                    ValidationDiagnosticModel(
                        code="STATE_READ_NOT_AVAILABLE",
                        severity="error",
                        message=(
                            f"Node '{node_id}' reads key '{key}' that is not declared "
                            "or produced by upstream nodes."
                        ),
                        node_id=node_id,
                    )
                )

        available_by_node[node_id] = available | set(writes)

        for nxt in adjacency.get(node_id, []):
            in_deg[nxt] = int(in_deg.get(nxt, 0)) - 1
            if in_deg[nxt] == 0:
                queue.append(nxt)


def _compile_from_normalized(
    normalized: Mapping[str, Any],
    *,
    entry_nodes: List[str],
) -> Dict[str, Any]:
    nodes = sorted(
        [dict(item) for item in list(normalized.get("nodes") or []) if isinstance(item, Mapping)],
        key=lambda item: str(item.get("id") or ""),
    )
    edges = sorted(
        [dict(item) for item in list(normalized.get("edges") or []) if isinstance(item, Mapping)],
        key=lambda item: (
            str(item.get("from") or ""),
            str(item.get("to") or ""),
            str(item.get("label") or ""),
            str(item.get("id") or ""),
        ),
    )

    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source and target:
            adjacency[source].append(target)

    compiled = {
        "nodes": nodes,
        "edges": edges,
        "entry_nodes": sorted(entry_nodes),
        "adjacency": {key: sorted(value) for key, value in sorted(adjacency.items())},
        "allow_cycles": bool(normalized.get("allow_cycles", False)),
        "runtime_graph": _runtime_graph_from_spec(normalized),
    }

    canonical = json.dumps(compiled, sort_keys=True, ensure_ascii=True, default=str)
    compiled["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return compiled


def _runtime_graph_from_spec(normalized: Mapping[str, Any]) -> Dict[str, Any]:
    raw_nodes = [
        dict(item)
        for item in list(normalized.get("nodes") or [])
        if isinstance(item, Mapping)
    ]
    raw_edges = [
        dict(item)
        for item in list(normalized.get("edges") or [])
        if isinstance(item, Mapping)
    ]

    runtime_nodes: List[Dict[str, Any]] = [{"id": "start", "type": "start", "config": {}}]
    node_ids: Set[str] = set()

    for node in raw_nodes:
        node_id = str(node.get("id") or "").strip()
        node_type = str(node.get("type") or "").strip().lower()
        if not node_id or not node_type:
            continue
        node_ids.add(node_id)
        runtime_nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "config": dict(node.get("config") or {}),
                "reads": list(node.get("reads") or []),
                "writes": list(node.get("writes") or []),
            }
        )

    runtime_nodes.append({"id": "end", "type": "end", "config": {}})

    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    runtime_edges: List[Dict[str, Any]] = []

    for edge in raw_edges:
        source = str(edge.get("from") or "").strip()
        target = str(edge.get("to") or "").strip()
        label = str(edge.get("label") or "always").strip() or "always"
        if source not in node_ids or target not in node_ids:
            continue
        runtime_edges.append(
            {
                "from": source,
                "to": target,
                "condition": label,
            }
        )
        indegree[target] = int(indegree.get(target, 0)) + 1

    entry_nodes = sorted([node_id for node_id, count in indegree.items() if count == 0])
    for node_id in entry_nodes:
        runtime_edges.append({"from": "start", "to": node_id, "condition": "always"})

    finalize_nodes = {
        str(item.get("id") or "")
        for item in runtime_nodes
        if str(item.get("type") or "") == "finalize"
    }
    outgoing: Dict[str, int] = defaultdict(int)
    for edge in runtime_edges:
        outgoing[str(edge.get("from") or "")] += 1

    for node_id in sorted(finalize_nodes):
        if outgoing[node_id] == 0:
            runtime_edges.append({"from": node_id, "to": "end", "condition": "always"})

    return {
        "nodes": runtime_nodes,
        "edges": runtime_edges,
    }
