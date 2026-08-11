"""Tests for adaptive router."""

from __future__ import annotations

import pytest

from mom.core.adaptive.router import AdaptiveRouter, AdaptiveDecision


def test_easy_complexity():
    router = AdaptiveRouter()
    decision = router.decide({"user_input": "hi", "task_representation": {}}, budget_ms=10_000.0)
    assert decision.complexity_class == 1


def test_hard_complexity():
    router = AdaptiveRouter()
    task = {"user_input": "x" * 300, "task_representation": {"requires_verification": True}}
    decision = router.decide(task, budget_ms=10_000.0)
    assert decision.complexity_class == 4


def test_pipeline_selection():
    router = AdaptiveRouter()
    decision = router.decide({"user_input": "test", "task_representation": {}}, budget_ms=500.0)
    assert decision.pipeline in {"fast_direct", "single_expert", "multi_expert", "full_mom"}


def test_adjust_experts_down():
    router = AdaptiveRouter()
    new_k = router.adjust_experts(current=8, latency_ms=9000.0, budget_ms=10_000.0, quality=0.5)
    assert new_k == 7


def test_adjust_experts_up():
    router = AdaptiveRouter()
    new_k = router.adjust_experts(current=2, latency_ms=1000.0, budget_ms=10_000.0, quality=0.6)
    assert new_k == 3


def test_should_activate_tools():
    router = AdaptiveRouter()
    assert router.should_activate_tools({"tools": ["t1"]}, budget_ms=5000.0) is True
    assert router.should_activate_tools({"tools": []}, budget_ms=5000.0) is False
    assert router.should_activate_tools({"tools": ["t1"]}, budget_ms=500.0) is False
