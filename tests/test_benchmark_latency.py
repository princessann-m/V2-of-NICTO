"""Tests for latency benchmark."""

from __future__ import annotations

import pytest

from mom.benchmarks.latency import LatencyBenchmark


def test_benchmark_measure():
    bench = LatencyBenchmark(target_ms=10_000.0)
    dataset = [{"input": "hello"}, {"input": "world"}]
    result = bench.measure("cfg1", lambda x: {"answer": x}, dataset)
    assert result.samples == 2
    assert result.mean_ms >= 0


def test_benchmark_target_met():
    bench = LatencyBenchmark(target_ms=10_000.0)
    result = bench.measure("cfg2", lambda x: {"answer": x}, [{"input": "a"}])
    assert isinstance(result.target_met, bool)


def test_benchmark_percentiles():
    bench = LatencyBenchmark()
    dataset = [{"input": str(i)} for i in range(100)]
    result = bench.measure("cfg3", lambda x: {"answer": x}, dataset)
    assert result.p50_ms <= result.p95_ms <= result.p99_ms
    assert result.min_ms <= result.p50_ms
    assert result.p99_ms <= result.max_ms
