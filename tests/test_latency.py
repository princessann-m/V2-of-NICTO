"""Tests for LatencyEstimator."""

from __future__ import annotations

import time
import pytest

from mom.core.orchestrator.latency import LatencyEstimator, LatencyEstimate


def test_model_estimation():
    est = LatencyEstimator(hardware_profile="gpu-medium")
    estimate = est.estimate_model("model_1", input_tokens=512, output_tokens=256)
    assert estimate.predicted_ms > 0
    assert 0.0 < estimate.confidence <= 1.0
    assert estimate.lower_bound_ms <= estimate.predicted_ms <= estimate.upper_bound_ms


def test_tool_estimation():
    est = LatencyEstimator(hardware_profile="cpu-large")
    estimate = est.estimate_tool("tool_1", args_size=1024)
    assert estimate.tool_id == "tool_1"
    assert estimate.predicted_ms > 0


def test_record_and_reestimate():
    est = LatencyEstimator(hardware_profile="gpu-large")
    est.record_model_latency("model_1", 300.0)
    est.record_model_latency("model_1", 400.0)
    estimate = est.estimate_model("model_1")
    assert estimate.sample_count == 2
    assert estimate.predicted_ms > 0


def test_hardware_profile_switch():
    est = LatencyEstimator(hardware_profile="cpu-small")
    est.set_hardware_profile("gpu-large")
    assert est.hardware_profile == "gpu-large"


def test_measure_wrapper():
    est = LatencyEstimator()
    def slow_fn():
        time.sleep(0.01)
        return 42
    result, elapsed = est.measure(slow_fn)
    assert result == 42
    assert elapsed >= 10.0
