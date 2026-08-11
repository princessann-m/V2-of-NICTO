"""Task scheduler with priority, deadline awareness, and speculative execution."""

from __future__ import annotations

import time
import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

LOG = logging.getLogger(__name__)


@dataclass(order=True)
class ScheduledTask:
    priority: int
    created_ns: int
    task_id: str
    fn: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict[str, Any] = field(default_factory=dict, compare=False)
    deadline_ns: int = 0
    estimated_ms: float = 0.0
    speculative: bool = False


class TaskScheduler:
    def __init__(self, max_workers: int = 4):
        self._queue: list[ScheduledTask] = []
        self._running: dict[str, ScheduledTask] = {}
        self.max_workers = max_workers
        self._completed = 0
        self._dropped = 0

    def submit(self, task_id: str, fn: Callable, priority: int = 0, deadline_ms: float = 0,
               estimated_ms: float = 0.0, args: tuple = (), kwargs: dict[str, Any] | None = None,
               speculative: bool = False) -> ScheduledTask:
        now = time.perf_counter_ns()
        deadline_ns = now + int(deadline_ms * 1_000_000) if deadline_ms > 0 else 0
        task = ScheduledTask(
            priority=-priority,
            created_ns=now,
            task_id=task_id,
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            deadline_ns=deadline_ns,
            estimated_ms=estimated_ms,
            speculative=speculative,
        )
        heapq.heappush(self._queue, task)
        return task

    def run_next(self, now_ns: int | None = None) -> tuple[Any, ScheduledTask] | None:
        now_ns = now_ns or time.perf_counter_ns()
        dropped = []
        while self._queue:
            task = heapq.heappop(self._queue)
            if task.deadline_ns and now_ns > task.deadline_ns:
                self._dropped += 1
                LOG.warning("Dropped task %s: deadline expired", task.task_id)
                continue
            if len(self._running) >= self.max_workers:
                heapq.heappush(self._queue, task)
                break
            self._running[task.task_id] = task
            try:
                result = task.fn(*task.args, **task.kwargs)
            except Exception as exc:
                result = exc
            del self._running[task.task_id]
            self._completed += 1
            return result, task
        return None

    def run_all(self, now_ns: int | None = None) -> list[tuple[Any, ScheduledTask]]:
        now_ns = now_ns or time.perf_counter_ns()
        results = []
        while self._queue and len(self._running) < self.max_workers:
            res = self.run_next(now_ns)
            if res is None:
                break
            results.append(res)
        return results

    def speculative_submit(self, task_id: str, fn: Callable, priority: int = 0, **kwargs: Any) -> ScheduledTask:
        return self.submit(task_id, fn, priority=priority, speculative=True, **kwargs)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": self._completed,
            "dropped": self._dropped,
        }
