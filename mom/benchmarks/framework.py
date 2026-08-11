"""Benchmark framework with A/B/C/D/E/F/G/H/I/J comparison modes."""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from ..core.orchestrator import Orchestrator
from ..config import MoMConfig, LLMConfig


@dataclass
class BenchmarkResult:
    mode: str
    accuracy: float = 0.0
    hallucination_rate: float = 1.0
    latency_ms: float = 0.0
    compute_cost: float = 0.0
    samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


COMPARISON_MODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class BenchmarkFramework:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_comparison(
        self,
        mode: str,
        dataset: list[dict[str, Any]],
        handler: Callable | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        if mode not in COMPARISON_MODES:
            raise ValueError(f"Unknown mode {mode}. Choose from {COMPARISON_MODES}")
        handler = handler or self._default_handler(mode)
        correct = 0
        hallucinations = 0
        total_latency = 0.0
        total_cost = 0.0
        for sample in dataset:
            start = time.perf_counter()
            try:
                out = handler(sample.get("input", ""), **kwargs)
                latency = (time.perf_counter() - start) * 1000
                total_latency += latency
                if out.get("answer") and out.get("hallucination_issues", []) == []:
                    correct += 1
                if out.get("hallucination_issues"):
                    hallucinations += 1
                total_cost += out.get("compute_cost", latency * 0.001)
            except Exception:
                total_latency += 0.0

        n = len(dataset) or 1
        result = BenchmarkResult(
            mode=mode,
            accuracy=correct / n,
            hallucination_rate=hallucinations / n,
            latency_ms=total_latency / n,
            compute_cost=total_cost / n,
            samples=len(dataset),
            metadata=kwargs,
        )
        self.results.append(result)
        self._save_result(result)
        return result

    def ablation_study(self, base_mode: str, dataset: list[dict[str, Any]], knobs: dict[str, list[Any]]) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        keys = list(knobs.keys())
        values = list(knobs.values())

        def backtrack(idx: int, current: dict[str, Any]):
            if idx == len(keys):
                mode_label = f"{base_mode}_{'_'.join(str(current[k]) for k in keys)}"
                res = self.run_comparison(mode_label, dataset, **current)
                results.append(res)
                return
            key = keys[idx]
            for v in values[idx]:
                current[key] = v
                backtrack(idx + 1, current)

        backtrack(0, {})
        return results

    def _default_handler(self, mode: str) -> Callable:
        cfg = MoMConfig(llm=LLMConfig(provider="heuristic"), global_deadline=2.0)
        orch = Orchestrator(cfg)

        def handler(user_input: str, **_: Any) -> dict[str, Any]:
            return orch.handle_request(user_input, request_id=f"bench_{mode}")

        return handler

    def _save_result(self, result: BenchmarkResult):
        path = os.path.join(self.output_dir, f"result_{result.mode}.json")
        with open(path, "w") as fh:
            json.dump(asdict(result), fh, indent=2)
