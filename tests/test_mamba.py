"""Tests for Mamba-related components."""

from __future__ import annotations

import pytest

from mom.models.model_registry import ModelRegistry
from mom.core.routing.master_router import MasterRouter
from mom.training.pipelines import get_pipeline


@pytest.fixture
def registry():
    return ModelRegistry()


def test_mamba_pipeline_exists():
    fn = get_pipeline("mamba")
    assert callable(fn)


def test_master_router_mamba_mode(registry):
    router = MasterRouter(registry)
    analysis = router.analyze_task("Simulate a physics experiment with mamba")
    cfg = router.select_mme(analysis)
    assert "selected_experts" in cfg
    assert "latency_budget" in cfg


def test_registry_science_expert_for_mamba(registry):
    experts = registry.find_experts_for("science")
    domains = {e["domain"] for e in experts}
    assert "science" in domains or "reasoning" in domains
