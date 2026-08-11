"""Tests for production monitoring."""

from __future__ import annotations

import pytest

from mom.monitoring.health import HealthCheck
from mom.monitoring.metrics import MetricsCollector
from mom.monitoring.tracing import Tracer


def test_metrics_increment():
    mc = MetricsCollector()
    mc.increment("requests_total")
    snap = mc.snapshot()
    assert snap["counters"]["requests_total"] == 1


def test_metrics_observe():
    mc = MetricsCollector()
    mc.observe("latency_ms", 12.5)
    snap = mc.snapshot()
    assert snap["histograms"]["latency_ms"]["count"] == 1
    assert snap["histograms"]["latency_ms"]["avg"] == 12.5


def test_metrics_gauge():
    mc = MetricsCollector()
    mc.set_gauge("cpu_usage", 0.5)
    snap = mc.snapshot()
    assert snap["gauges"]["cpu_usage"] == 0.5


def test_metrics_labels():
    mc = MetricsCollector()
    mc.increment("requests", labels={"method": "POST"})
    snap = mc.snapshot()
    assert any("method=\"POST\"" in k for k in snap["counters"])


def test_tracer_span_lifecycle():
    tracer = Tracer()
    span = tracer.start_span("trace-1", "step1")
    tracer.end_span(span, "ok")
    trace = tracer.get_trace("trace-1")
    assert len(trace) == 1
    assert trace[0]["name"] == "step1"
    assert trace[0]["status"] == "ok"


def test_tracer_export():
    tracer = Tracer()
    span = tracer.start_span("trace-2", "step2")
    tracer.end_span(span)
    exported = tracer.export()
    assert "trace-2" in exported


def test_health_check_register_and_status():
    hc = HealthCheck()
    hc.register("db", lambda: None)
    result = hc.check("db")
    assert result["status"] == "healthy"
    status = hc.status()
    assert status["total"] == 1
    assert status["healthy"] == 1


def test_health_check_alert_threshold():
    hc = HealthCheck()
    hc.register("svc", lambda: None)
    hc.components["svc"].details["last_latency_ms"] = 100.0
    assert hc.alert_threshold("svc", 50.0) is True
    assert hc.alert_threshold("svc", 200.0) is False
