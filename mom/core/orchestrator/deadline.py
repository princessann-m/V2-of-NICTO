"""Deadline management with hard deadline tracking."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class Deadline:
    task_id: str
    absolute_ns: int
    soft_ns: int
    hard_ns: int
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions_used: int = 0
    max_extensions: int = 1
    extension_policy: str = "linear"

    @property
    def remaining_ns(self) -> int:
        return max(0, self.hard_ns - time.perf_counter_ns())

    @property
    def remaining_ms(self) -> float:
        return self.remaining_ns / 1_000_000

    @property
    def is_soft_deadline_passed(self) -> bool:
        return time.perf_counter_ns() >= self.soft_ns

    @property
    def is_hard_deadline_passed(self) -> bool:
        return time.perf_counter_ns() >= self.hard_ns

    @property
    def slack_ns(self) -> int:
        return max(0, self.hard_ns - time.perf_counter_ns())

    def can_extend(self) -> bool:
        return self.extensions_used < self.max_extensions

    def extend(self, budget_ms: float = 500.0) -> None:
        if not self.can_extend():
            raise RuntimeError("Max deadline extensions reached")
        extra = int(budget_ms * 1_000_000)
        if self.extension_policy == "linear":
            self.hard_ns += extra
        elif self.extension_policy == "exponential":
            self.hard_ns += extra * (2 ** self.extensions_used)
        self.extensions_used += 1
        LOG.info("Extended deadline for %s by %.1f ms", self.task_id, budget_ms)


class DeadlineManager:
    def __init__(self, default_hard_ms: float = 10_000.0, default_soft_ms: float = 9_500.0):
        self.default_hard_ms = default_hard_ms
        self.default_soft_ms = default_soft_ms
        self._deadlines: dict[str, Deadline] = {}
        self._history: list[dict[str, Any]] = []

    def create(self, task_id: str, hard_ms: float | None = None, soft_ms: float | None = None,
               priority: int = 0, metadata: dict[str, Any] | None = None) -> Deadline:
        hard = hard_ms if hard_ms is not None else self.default_hard_ms
        soft = soft_ms if soft_ms is not None else hard * 0.95
        now = time.perf_counter_ns()
        deadline = Deadline(
            task_id=task_id,
            absolute_ns=now,
            soft_ns=now + int(soft * 1_000_000),
            hard_ns=now + int(hard * 1_000_000),
            priority=priority,
            metadata=metadata or {},
        )
        self._deadlines[task_id] = deadline
        return deadline

    def get(self, task_id: str) -> Deadline | None:
        return self._deadlines.get(task_id)

    def remaining_budget(self, task_id: str) -> float:
        d = self._deadlines.get(task_id)
        return d.remaining_ms if d else self.default_hard_ms

    def is_expired(self, task_id: str) -> bool:
        d = self._deadlines.get(task_id)
        return d.is_hard_deadline_passed if d else False

    def should_early_stop(self, task_id: str) -> bool:
        d = self._deadlines.get(task_id)
        if not d:
            return False
        if d.is_hard_deadline_passed:
            self._record(task_id, "hard_expired")
            return True
        if d.is_soft_deadline_passed and d.priority < 5:
            self._record(task_id, "soft_expired_low_priority")
            return True
        return False

    def record_completion(self, task_id: str, actual_ms: float, outcome: str = "completed") -> None:
        d = self._deadlines.get(task_id)
        if d:
            self._history.append({
                "task_id": task_id,
                "outcome": outcome,
                "actual_ms": actual_ms,
                "budget_ms": self.default_hard_ms,
                "remaining_ms": d.remaining_ms,
                "extensions_used": d.extensions_used,
            })

    def _record(self, task_id: str, reason: str) -> None:
        LOG.warning("Deadline event for %s: %s", task_id, reason)

    def stats(self) -> dict[str, Any]:
        if not self._history:
            return {"count": 0}
        met = sum(1 for h in self._history if h["outcome"] == "completed")
        return {
            "count": len(self._history),
            "met": met,
            "miss_rate": 1.0 - met / len(self._history),
            "avg_remaining_ms": sum(h["remaining_ms"] for h in self._history) / len(self._history),
        }
