"""Stopping criteria with agreement detection and budget awareness."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class StopDecision:
    should_stop: bool
    reason: str
    confidence: float
    budget_remaining_ms: float


class StoppingCriterion(ABC):
    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> StopDecision:
        raise NotImplementedError


class EarlyStopping:
    def __init__(self, confidence_threshold: float = 0.85, max_rounds: int = 3, budget_aware: bool = True):
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.budget_aware = budget_aware
        self._round = 0
        self._history: list[dict[str, Any]] = []

    def should_stop(self, context: dict[str, Any]) -> StopDecision:
        self._round += 1
        confidence = context.get("confidence", 0.0)
        budget_remaining = context.get("budget_remaining_ms", float("inf"))
        agreement = context.get("agreement", 1.0)

        if confidence >= self.confidence_threshold:
            return StopDecision(True, "confidence_threshold_met", confidence, budget_remaining)

        if self._round >= self.max_rounds:
            return StopDecision(True, "max_rounds_reached", confidence, budget_remaining)

        if self.budget_aware and budget_remaining < 500:
            return StopDecision(True, "budget_exhausted", confidence, budget_remaining)

        if agreement < 0.3:
            return StopDecision(False, "disagreement_continue", confidence, budget_remaining)

        if context.get("verification_passed") and confidence >= self.confidence_threshold * 0.8:
            return StopDecision(True, "verification_sufficient", confidence, budget_remaining)

        return StopDecision(False, "continue", confidence, budget_remaining)

    def reset(self) -> None:
        self._round = 0
        self._history.clear()

    def record(self, decision: StopDecision, context: dict[str, Any]) -> None:
        self._history.append({
            "round": self._round,
            "should_stop": decision.should_stop,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "budget_remaining_ms": decision.budget_remaining_ms,
        })
