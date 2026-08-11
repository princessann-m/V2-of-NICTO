"""Master Router — coordinates routing decisions across the NICTO stack."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from ...models.model_registry import ModelRegistry


@dataclass
class TaskAnalysis:
    task_type: str
    complexity: str
    required_tools: list[str]
    modality: str
    latency_budget_ms: float
    selected_experts: list[dict] = field(default_factory=list)
    sparsity: float = 0.5


@dataclass
class LatencyBudget:
    total_ms: float
    mme_ms: float
    verification_ms: float
    judging_ms: float
    reserve_ms: float

    @classmethod
    def allocate(cls, total_ms: float) -> LatencyBudget:
        mme_ms = total_ms * 0.45
        verification_ms = total_ms * 0.15
        judging_ms = total_ms * 0.15
        reserve_ms = total_ms - mme_ms - verification_ms - judging_ms
        return cls(total_ms, mme_ms, verification_ms, judging_ms, reserve_ms)


class MasterRouter:
    def __init__(self, registry: ModelRegistry, sparsity: float = 0.5):
        self.registry = registry
        self.sparsity = sparsity

    def analyze_task(self, user_input: str) -> TaskAnalysis:
        text = user_input.lower()
        if any(k in text for k in ["+", "-", "*", "/", "=", "calculate", "solve"]):
            task_type = "math"
        elif any(k in text for k in ["code", "function", "implement", "python", "test"]):
            task_type = "coding"
        elif any(k in text for k in ["image", "picture", "describe", "vision"]):
            task_type = "image"
        elif any(k in text for k in ["simulate", "model", "experiment", "science"]):
            task_type = "science"
        elif any(k in text for k in ["why", "how", "reason", "plan", "logic"]):
            task_type = "reasoning"
        else:
            task_type = "general"

        if len(user_input.split()) < 8:
            complexity = "easy"
        elif len(user_input.split()) > 25:
            complexity = "hard"
        else:
            complexity = "medium"

        tools: list[str] = []
        if task_type == "math":
            tools.append("calculator")
        if task_type == "coding":
            tools.extend(["coding_studio", "linter"])
        if task_type == "science":
            tools.append("simulator")

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            required_tools=tools,
            modality="text",
            latency_budget_ms=500.0,
            sparsity=self.sparsity,
        )

    def select_mme(self, analysis: TaskAnalysis) -> dict[str, Any]:
        candidates = self.registry.find_experts_for(analysis.task_type)
        n = max(1, int(len(candidates) * (1.0 - analysis.sparsity)))
        n = min(n, len(candidates))
        selected = random.sample(candidates, n) if n < len(candidates) else candidates[:]
        return {
            "selected_experts": selected,
            "task_type": analysis.task_type,
            "complexity": analysis.complexity,
            "modality_pipeline": self._modality_pipeline(analysis),
            "latency_budget": LatencyBudget.allocate(analysis.latency_budget_ms),
            "seed": int(time.time() * 1000) % 10000,
        }

    def activate_tools(self, analysis: TaskAnalysis) -> list[str]:
        return list(analysis.required_tools)

    def _modality_pipeline(self, analysis: TaskAnalysis) -> str:
        if analysis.modality == "image":
            return "vision_encoder -> fusion -> decoder"
        if analysis.task_type == "science":
            return "simulator_pipeline -> result_aggregator"
        return "text_encoder -> mme -> text_decoder"
