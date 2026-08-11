"""Prometheus-compatible metrics collection."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.gauges: dict[str, float] = defaultdict(float)

    def increment(self, name: str, amount: int = 1, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.counters[key] += amount

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.histograms[key].append(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.gauges[key] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "histograms": {k: {
                    "count": len(v),
                    "sum": sum(v),
                    "min": min(v),
                    "max": max(v),
                    "avg": sum(v) / len(v) if v else 0,
                } for k, v in self.histograms.items()},
                "gauges": dict(self.gauges),
            }

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        return f"{name}{{{','.join(f'{k}=\"{v}\"' for k, v in labels.items())}}}"
