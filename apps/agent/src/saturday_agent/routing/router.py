from __future__ import annotations

from typing import Any, Dict, Optional

from saturday_agent.routing.policies import (
    DEFAULT_ROUTING_POLICY,
    RoutingPolicy,
    score_task_complexity,
    score_to_workflow_type,
)

VALID_WORKFLOW_TYPES = {"simple", "moderate", "complex"}


def route_workflow_type(
    *,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    workflow_type: Optional[str] = None,
    policy: RoutingPolicy = DEFAULT_ROUTING_POLICY,
) -> str:
    if workflow_type:
        normalized = workflow_type.strip().lower()
        if normalized not in VALID_WORKFLOW_TYPES:
            raise ValueError(
                "workflow_type must be one of: simple, moderate, complex"
            )
        return normalized

    score = score_task_complexity(task=task, context=context, policy=policy)
    return score_to_workflow_type(score, policy=policy)
