"""Tests for sparse expert router."""

from __future__ import annotations

import pytest

from mom.core.routing.expert_registry import ExpertRegistry
from mom.core.routing.expert_router import ExpertRouter, ConfidenceAgreementMonitor
from mom.core.routing.top_k import TopKSelector, RoutingResult
from mom.core.routing.hardware import HardwareDetector


def test_registry_size():
    registry = ExpertRegistry()
    assert registry.size() == 310


def test_registry_query_by_domain():
    registry = ExpertRegistry()
    experts = registry.query(domain="nlp")
    assert len(experts) > 0
    assert all(e.domain == "nlp" for e in experts)


def test_registry_query_by_subdomain():
    registry = ExpertRegistry()
    experts = registry.query(domain="nlp", subdomain="sentiment")
    assert len(experts) > 0
    assert all(e.subdomain == "sentiment" for e in experts)


def test_expert_router_route():
    registry = ExpertRegistry()
    router = ExpertRouter(registry, k=2)
    result = router.route({"task_type": "qa", "domain": "nlp"})
    assert len(result.selected_ids) <= 2
    assert len(result.selected_ids) > 0


def test_top_k_selector():
    selector = TopKSelector(k=3)
    scores = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
    result = selector.select(scores)
    assert result.k_used == 3
    assert result.selected_ids == ["a", "b", "c"]


def test_top_k_sparse_activate():
    selector = TopKSelector(k=4)
    scores = {"a": 0.9, "b": 0.4, "c": 0.1, "d": 0.05, "e": 0.0}
    result = selector.sparse_activate(scores, threshold=0.2)
    assert "a" in result.selected_ids
    assert "e" not in result.selected_ids


def test_hardware_detector():
    hw = HardwareDetector()
    profile = hw.get_profile()
    assert profile.cpu_cores > 0


def test_router_dynamic_k():
    registry = ExpertRegistry()
    router = ExpertRouter(registry, k=4)
    assert router.dynamic_k({"complexity": "easy"}) == 1
    assert router.dynamic_k({"complexity": "hard"}) == 4


def test_disagreement_trigger():
    registry = ExpertRegistry()
    router = ExpertRouter(registry, k=2)
    triggered = router.disagreement_trigger({"domain": "general"})
    assert isinstance(triggered, bool)
