"""Adaptive computation with task complexity classification and pipeline selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class AdaptiveDecision:
    complexity_class: int
    expert_count: int
    tool_count: int
    pipeline: str
    latency_budget_ms: float
    confidence_target: float


class AdaptiveRouter:
    def __init__(self, default_latency_budget_ms: float = 10_000.0):
        self.default_budget = default_latency_budget_ms
        self._history: list[dict[str, Any]] = []

    def classify_complexity(self, task: dict[str, Any]) -> int:
        tokens = len(task.get("user_input", "")) + len(str(task.get("task_representation", {})))
        requires_tools = len(task.get("tools", [])) > 0
        requires_verification = task.get("requires_verification", False)
        if requires_verification or requires_tools or tokens > 200:
            return 4
        if tokens > 80 or requires_tools:
            return 3
        if tokens > 30:
            return 2
        return 1

    def select_pipeline(self, complexity_class: int, budget_ms: float) -> str:
        pipelines = {
            1: "fast_direct",
            2: "single_expert",
            3: "multi_expert",
            4: "full_mom",
        }
        if budget_ms < 2000 and complexity_class >= 3:
            return "fast_direct"
        return pipelines.get(complexity_class, "full_mom")

    def decide(self, task: dict[str, Any], budget_ms: float | None = None) -> AdaptiveDecision:
        budget_ms = budget_ms or self.default_budget
        complexity = self.classify_complexity(task)
        pipeline = self.select_pipeline(complexity, budget_ms)
        expert_count = {1: 1, 2: 2, 3: 4, 4: 8}.get(complexity, 4)
        tool_count = 1 if task.get("tools") else 0
        if budget_ms < 1000:
            expert_count = max(1, expert_count // 2)
            tool_count = 0
        confidence_target = 0.9 if complexity >= 3 else 0.8
        decision = AdaptiveDecision(
            complexity_class=complexity,
            expert_count=expert_count,
            tool_count=tool_count,
            pipeline=pipeline,
            latency_budget_ms=budget_ms,
            confidence_target=confidence_target,
        )
        self._history.append({
            "complexity": complexity,
            "pipeline": pipeline,
            "expert_count": expert_count,
            "budget_ms": budget_ms,
        })
        return decision

    def adjust_experts(self, current: int, latency_ms: float, budget_ms: float, quality: float) -> int:
        if latency_ms > budget_ms * 0.8:
            return max(1, current - 1)
        if latency_ms < budget_ms * 0.3 and quality < 0.8:
            return min(16, current + 1)
        return current

    def should_activate_tools(self, task: dict[str, Any], budget_ms: float) -> bool:
        return bool(task.get("tools")) and budget_ms > 2000
