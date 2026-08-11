"""Latency estimation with historical tracking and hardware awareness."""

from __future__ import annotations

import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class LatencyEstimate:
    predicted_ms: float
    confidence: float
    lower_bound_ms: float
    upper_bound_ms: float
    model_id: str = ""
    tool_id: str = ""
    hardware_profile: str = "cpu"
    sample_count: int = 0


class LatencyEstimator:
    def __init__(self, hardware_profile: str = "cpu"):
        self.hardware_profile = hardware_profile
        self._model_history: dict[str, list[float]] = {}
        self._tool_history: dict[str, list[float]] = {}
        self._max_history: int = 200
        self._base_latencies: dict[str, float] = {
            "gpu-small": 120.0,
            "gpu-medium": 350.0,
            "gpu-large": 900.0,
            "cpu-small": 800.0,
            "cpu-medium": 2000.0,
            "cpu-large": 5000.0,
        }

    def estimate_model(self, model_id: str, input_tokens: int = 512, output_tokens: int = 256) -> LatencyEstimate:
        history = self._model_history.get(model_id, [])
        token_factor = (input_tokens + output_tokens) / 1024.0
        if history:
            mean = statistics.mean(history)
            stdev = statistics.pstdev(history) if len(history) > 1 else mean * 0.2
            predicted = mean * token_factor
            ci = 1.96 * stdev * token_factor
        else:
            base = self._base_latencies.get(self.hardware_profile, 2000.0)
            predicted = base * token_factor
            ci = predicted * 0.3

        confidence = max(0.1, min(0.99, 1.0 / (1.0 + ci / max(1.0, predicted))))
        return LatencyEstimate(
            predicted_ms=predicted,
            confidence=confidence,
            lower_bound_ms=max(0.0, predicted - ci),
            upper_bound_ms=predicted + ci,
            model_id=model_id,
            hardware_profile=self.hardware_profile,
            sample_count=len(history),
        )

    def estimate_tool(self, tool_id: str, args_size: int = 0) -> LatencyEstimate:
        history = self._tool_history.get(tool_id, [])
        size_factor = 1.0 + (args_size / 4096.0)
        if history:
            mean = statistics.mean(history)
            stdev = statistics.pstdev(history) if len(history) > 1 else mean * 0.15
            predicted = mean * size_factor
            ci = 1.96 * stdev * size_factor
        else:
            predicted = 500.0 * size_factor
            ci = predicted * 0.25

        confidence = max(0.2, min(0.95, 1.0 / (1.0 + ci / max(1.0, predicted))))
        return LatencyEstimate(
            predicted_ms=predicted,
            confidence=confidence,
            lower_bound_ms=max(0.0, predicted - ci),
            upper_bound_ms=predicted + ci,
            tool_id=tool_id,
            hardware_profile=self.hardware_profile,
            sample_count=len(history),
        )

    def record_model_latency(self, model_id: str, latency_ms: float) -> None:
        self._model_history.setdefault(model_id, []).append(latency_ms)
        if len(self._model_history[model_id]) > self._max_history:
            self._model_history[model_id] = self._model_history[model_id][-self._max_history:]

    def record_tool_latency(self, tool_id: str, latency_ms: float) -> None:
        self._tool_history.setdefault(tool_id, []).append(latency_ms)
        if len(self._tool_history[tool_id]) > self._max_history:
            self._tool_history[tool_id] = self._tool_history[tool_id][-self._max_history:]

    def measure(self, fn: Any, *args, **kwargs) -> tuple[Any, float]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed

    def set_hardware_profile(self, profile: str) -> None:
        self.hardware_profile = profile
