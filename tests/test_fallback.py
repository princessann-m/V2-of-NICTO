"""Tests for fallback handlers and strategies."""

from __future__ import annotations

import pytest

from mom.core.fallback.handlers import FallbackHandlers
from mom.core.fallback.strategies import FallbackStrategies


class TestFallbackHandlers:
    def test_handle_untrained_model(self):
        result = FallbackHandlers.handle_untrained_model("test_model")
        assert "untrained" in result["answer"].lower()
        assert result["confidence"] == 0.0
        assert result["metadata"]["fallback"] is True
        assert result["metadata"]["reason"] == "untrained_model"
        assert "recovery_suggestion" in result["metadata"]

    def test_handle_timeout_with_best_available(self):
        best = {"answer": "partial", "confidence": 0.5}
        result = FallbackHandlers.handle_timeout("req_1", best)
        assert result["answer"] == "partial"
        assert result["metadata"]["fallback"] is True
        assert result["metadata"]["reason"] == "deadline"

    def test_handle_timeout_without_best_available(self):
        result = FallbackHandlers.handle_timeout("req_2")
        assert "deadline" in result["answer"].lower()
        assert result["confidence"] == 0.0

    def test_handle_expert_failure_with_remaining(self):
        result = FallbackHandlers.handle_expert_failure(
            "math_expert", "RuntimeError", ["reasoning_expert"]
        )
        assert result["metadata"]["failed_expert"] == "math_expert"
        assert result["metadata"]["remaining_experts"] == ["reasoning_expert"]
        assert result["confidence"] == 0.3

    def test_handle_expert_failure_no_remaining(self):
        result = FallbackHandlers.handle_expert_failure(
            "math_expert", "RuntimeError", []
        )
        assert result["confidence"] == 0.0
        assert result["metadata"]["remaining_experts"] == []

    def test_handle_tool_failure(self):
        result = FallbackHandlers.handle_tool_failure("browser", "timeout")
        assert result["metadata"]["tool"] == "browser"
        assert "timeout" in result["metadata"]["error"]
        assert result["metadata"]["pipeline_continues"] is True

    def test_handle_judge_failure(self):
        result = FallbackHandlers.handle_judge_failure("judge_alpha", "judge_beta")
        assert result["failed_judge"] == "judge_alpha"
        assert result["remaining_judge"] == "judge_beta"
        assert result["reason"] == "judge_failure"


class TestFallbackStrategies:
    def test_cascade_fallback_succeeds_first(self):
        def model_a(task):
            return {"answer": "a", "confidence": 0.9}

        def model_b(task):
            raise RuntimeError("fail")

        models = [("a", model_a), ("b", model_b)]
        result, used = FallbackStrategies.cascade_fallback(models, {"x": 1})
        assert result is not None
        assert result["answer"] == "a"
        assert used == "a"

    def test_cascade_fallback_all_fail(self):
        def bad(task):
            raise RuntimeError("fail")

        models = [("a", bad), ("b", bad)]
        result, used = FallbackStrategies.cascade_fallback(models, {"x": 1})
        assert result is None
        assert used is None

    def test_parallel_fallback_multiple_succeed(self):
        def model_a(task):
            return {"answer": "a", "confidence": 0.9}

        def model_b(task):
            return {"answer": "b", "confidence": 0.7}

        models = [("a", model_a), ("b", model_b)]
        results, used = FallbackStrategies.parallel_fallback(models, {"x": 1})
        assert len(results) == 2
        assert set(used) == {"a", "b"}

    def test_parallel_fallback_some_fail(self):
        def good(task):
            return {"answer": "ok", "confidence": 0.8}

        def bad(task):
            raise RuntimeError("fail")

        models = [("good", good), ("bad", bad)]
        results, used = FallbackStrategies.parallel_fallback(models, {"x": 1})
        assert len(results) == 1
        assert results[0]["answer"] == "ok"
        assert used == ["good"]

    def test_cached_fallback_hit(self):
        cache = {"task_key": {"answer": "cached", "confidence": 0.9}}
        result = FallbackStrategies.cached_fallback(cache, {"original": "task_key"})
        assert result is not None
        assert result["answer"] == "cached"
        assert result["metadata"]["fallback"] is True
        assert result["metadata"]["reason"] == "cache_hit"

    def test_cached_fallback_miss(self):
        cache = {}
        result = FallbackStrategies.cached_fallback(cache, {"original": "missing"})
        assert result is None

    def test_cached_fallback_custom_key_fn(self):
        cache = {"custom_key": {"answer": "custom", "confidence": 0.9}}
        result = FallbackStrategies.cached_fallback(
            cache, {"original": "anything"},
            key_fn=lambda t: "custom_key",
        )
        assert result is not None
        assert result["answer"] == "custom"
