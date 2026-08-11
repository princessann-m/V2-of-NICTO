"""Distributed tracing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class Tracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.traces: dict[str, list[Span]] = {}
        self._counter = 0

    def start_span(self, trace_id: str, name: str, parent_id: str | None = None) -> Span:
        with self._lock:
            self._counter += 1
            span_id = f"span-{self._counter}"
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
        )
        with self._lock:
            self.traces.setdefault(trace_id, []).append(span)
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        span.end_time = time.time()
        span.status = status

    def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            spans = self.traces.get(trace_id, [])
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "duration_ms": s.duration_ms(),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in spans
        ]

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                tid: [
                    {
                        "trace_id": s.trace_id,
                        "span_id": s.span_id,
                        "parent_id": s.parent_id,
                        "name": s.name,
                        "duration_ms": s.duration_ms(),
                        "status": s.status,
                    }
                    for s in spans
                ]
                for tid, spans in self.traces.items()
            }
