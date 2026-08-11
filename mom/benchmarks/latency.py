"""Latency benchmark with percentile measurement and quality curves."""

from __future__ import annotations

import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

LOG = logging.getLogger(__name__)


@dataclass
class LatencyBenchmarkResult:
    config_name: str
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    samples: int = 0
    target_met: bool = False
    quality: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LatencyBenchmark:
    def __init__(self, target_ms: float = 10_000.0):
        self.target_ms = target_ms
        self._results: list[LatencyBenchmarkResult] = []

    def measure(self, config_name: str, fn: Callable, dataset: list[dict[str, Any]], quality_fn: Callable | None = None) -> LatencyBenchmarkResult:
        latencies = []
        qualities = []
        for sample in dataset:
            start = time.perf_counter()
            try:
                out = fn(sample.get("input", ""))
                latencies.append((time.perf_counter() - start) * 1000)
                if quality_fn:
                    qualities.append(quality_fn(out, sample))
            except Exception:
                latencies.append(self.target_ms)
        if not latencies:
            latencies = [0.0]
        result = self._compute_result(config_name, latencies, qualities)
        self._results.append(result)
        return result

    def _compute_result(self, config_name: str, latencies: list[float], qualities: list[float]) -> LatencyBenchmarkResult:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_lat[idx]
        result = LatencyBenchmarkResult(
            config_name=config_name,
            p50_ms=percentile(0.50),
            p90_ms=percentile(0.90),
            p95_ms=percentile(0.95),
            p99_ms=percentile(0.99),
            mean_ms=statistics.mean(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            samples=n,
            target_met=percentile(0.95) <= self.target_ms,
            quality=statistics.mean(qualities) if qualities else 0.0,
        )
        LOG.info("Benchmark %s: p95=%.1fms target_met=%s", config_name, result.p95_ms, result.target_met)
        return result

    def compare(self, configs: dict[str, Callable], dataset: list[dict[str, Any]], quality_fn: Callable | None = None) -> list[LatencyBenchmarkResult]:
        return [self.measure(name, fn, dataset, quality_fn) for name, fn in configs.items()]

    def validate_10s_target(self, result: LatencyBenchmarkResult) -> bool:
        return result.p95_ms <= 10_000.0
