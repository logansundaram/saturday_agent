from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class RoutingPolicy:
    simple_max_score: int = 3
    moderate_max_score: int = 7
    complex_keywords: set[str] = field(
        default_factory=lambda: {
            "multi-step",
            "architecture",
            "analyze",
            "research",
            "workflow",
            "tool",
            "tradeoff",
            "complex",
        }
    )
    moderate_keywords: set[str] = field(
        default_factory=lambda: {
            "plan",
            "compare",
            "summarize",
            "explain",
            "verify",
            "check",
        }
    )


DEFAULT_ROUTING_POLICY = RoutingPolicy()


def _contains_any(text: str, words: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def score_task_complexity(
    task: str,
    context: Dict[str, Any] | None,
    policy: RoutingPolicy = DEFAULT_ROUTING_POLICY,
) -> int:
    score = 0
    task_words = task.split()

    if len(task_words) > 40:
        score += 3
    elif len(task_words) > 15:
        score += 2
    elif len(task_words) > 6:
        score += 1

    if _contains_any(task, policy.complex_keywords):
        score += 8
    elif _contains_any(task, policy.moderate_keywords):
        score += 4

    context_size = len(context or {})
    if context_size >= 8:
        score += 3
    elif context_size >= 3:
        score += 1

    return score


def score_to_workflow_type(
    score: int,
    policy: RoutingPolicy = DEFAULT_ROUTING_POLICY,
) -> str:
    if score <= policy.simple_max_score:
        return "simple"
    if score <= policy.moderate_max_score:
        return "moderate"
    return "complex"
