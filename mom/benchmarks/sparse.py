"""Sparse activation benchmark comparing k values and measuring accuracy vs compute."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

LOG = logging.getLogger(__name__)


@dataclass
class SparseBenchmarkResult:
    k: int
    accuracy: float
    avg_latency_ms: float
    memory_mb: float
    gpu_utilization: float = 0.0
    samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseActivationBenchmark:
    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 2, 4, 8, 16]
        self._results: list[SparseBenchmarkResult] = []

    def run(self, router_fn: Callable, dataset: list[dict[str, Any]], quality_fn: Callable | None = None) -> list[SparseBenchmarkResult]:
        results = []
        for k in self.k_values:
            latencies = []
            correct = 0
            for sample in dataset:
                start = time.perf_counter()
                try:
                    out = router_fn(sample, k=k)
                    latencies.append((time.perf_counter() - start) * 1000)
                    if quality_fn and quality_fn(out, sample):
                        correct += 1
                except Exception:
                    latencies.append(1000.0)
            accuracy = correct / len(dataset) if dataset else 0.0
            result = SparseBenchmarkResult(
                k=k,
                accuracy=accuracy,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                memory_mb=self._estimate_memory(k),
                gpu_utilization=self._estimate_gpu_util(k),
                samples=len(dataset),
            )
            results.append(result)
            self._results.append(result)
            LOG.info("k=%d accuracy=%.3f avg_latency=%.1fms", k, accuracy, result.avg_latency_ms)
        return results

    def _estimate_memory(self, k: int) -> float:
        return 100 + k * 50

    def _estimate_gpu_util(self, k: int) -> float:
        return min(1.0, k * 0.1)

    def accuracy_vs_compute(self) -> dict[str, list[float]]:
        return {
            "k": [r.k for r in self._results],
            "accuracy": [r.accuracy for r in self._results],
            "latency": [r.avg_latency_ms for r in self._results],
            "memory": [r.memory_mb for r in self._results],
        }
