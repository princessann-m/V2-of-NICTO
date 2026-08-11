"""Tests for routing layer."""

from __future__ import annotations

import pytest

from mom.models.model_registry import ModelRegistry
from mom.core.routing.master_router import MasterRouter, TaskAnalysis, LatencyBudget
from mom.core.routing.expert_router import ExpertRouter
from mom.core.routing.top_k import RoutingResult
from mom.core.routing.learned import LearnedRouter, LearnedRouterConfig


@pytest.fixture
def registry():
    return ModelRegistry()


@pytest.fixture
def master_router(registry):
    return MasterRouter(registry, sparsity=0.5)


@pytest.fixture
def expert_router(registry):
    return ExpertRouter(registry, k=2)


class TestTaskAnalysis:
    def test_math_task(self, master_router):
        analysis = master_router.analyze_task("Calculate 2 + 2")
        assert analysis.task_type == "math"
        assert "calculator" in analysis.required_tools

    def test_coding_task(self, master_router):
        analysis = master_router.analyze_task("Write a python function")
        assert analysis.task_type == "coding"
        assert "coding_studio" in analysis.required_tools

    def test_complexity_easy(self, master_router):
        analysis = master_router.analyze_task("add")
        assert analysis.complexity == "easy"

    def test_complexity_hard(self, master_router):
        long_input = " ".join(["word"] * 30)
        analysis = master_router.analyze_task(long_input)
        assert analysis.complexity == "hard"


class TestLatencyBudget:
    def test_allocates(self):
        budget = LatencyBudget.allocate(1000.0)
        assert budget.total_ms == 1000.0
        assert budget.mme_ms + budget.verification_ms + budget.judging_ms + budget.reserve_ms == 1000.0

    def test_default(self):
        budget = LatencyBudget(total_ms=500.0, mme_ms=250.0, verification_ms=50.0, judging_ms=50.0, reserve_ms=150.0)
        assert budget.total_ms == 500.0


class TestMasterRouter:
    def test_select_mme_returns_config(self, master_router):
        analysis = TaskAnalysis(task_type="math", complexity="medium", required_tools=[], modality="text", latency_budget_ms=500.0)
        cfg = master_router.select_mme(analysis)
        assert "selected_experts" in cfg
        assert "latency_budget" in cfg
        assert "modality_pipeline" in cfg

    def test_activate_tools(self, master_router):
        analysis = TaskAnalysis(task_type="math", complexity="medium", required_tools=["calculator"], modality="text", latency_budget_ms=500.0)
        tools = master_router.activate_tools(analysis)
        assert "calculator" in tools


class TestExpertRouter:
    def test_route_returns_top_k(self, expert_router):
        task = {"task_type": "math", "required_capabilities": ["algebra", "calculus"]}
        result = expert_router.route(task)
        assert isinstance(result, RoutingResult)
        assert len(result.selected_ids) <= expert_router.k
        for score in result.scores:
            assert 0.0 <= score <= 1.0

    def test_route_ordering(self, expert_router):
        task = {"task_type": "coding", "required_capabilities": ["python", "unit_tests"]}
        result = expert_router.route(task)
        scores = result.scores
        assert scores == sorted(scores, reverse=True)


class TestLearnedRouter:
    def test_forward(self):
        router = LearnedRouter(LearnedRouterConfig())
        out = router.forward([0.1] * 768)
        assert len(out) == router.config.num_experts
