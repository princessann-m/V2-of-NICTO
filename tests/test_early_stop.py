"""Tests for early stopping."""

from __future__ import annotations

import pytest

from mom.core.stopping.early_stop import EarlyStopping, StopDecision


def test_confidence_stop():
    es = EarlyStopping(confidence_threshold=0.85, max_rounds=5)
    decision = es.should_stop({"confidence": 0.9, "budget_remaining_ms": 5000.0})
    assert decision.should_stop is True
    assert decision.reason == "confidence_threshold_met"


def test_max_rounds_stop():
    es = EarlyStopping(confidence_threshold=0.99, max_rounds=2)
    es.should_stop({"confidence": 0.5, "budget_remaining_ms": 5000.0})
    decision = es.should_stop({"confidence": 0.5, "budget_remaining_ms": 5000.0})
    assert decision.should_stop is True
    assert decision.reason == "max_rounds_reached"


def test_budget_aware_stop():
    es = EarlyStopping(confidence_threshold=0.99, max_rounds=5, budget_aware=True)
    decision = es.should_stop({"confidence": 0.5, "budget_remaining_ms": 100.0})
    assert decision.should_stop is True
    assert decision.reason == "budget_exhausted"


def test_continue():
    es = EarlyStopping(confidence_threshold=0.99, max_rounds=5)
    decision = es.should_stop({"confidence": 0.5, "budget_remaining_ms": 5000.0, "agreement": 0.8})
    assert decision.should_stop is False


def test_reset():
    es = EarlyStopping(max_rounds=2)
    es.should_stop({"confidence": 0.5, "budget_remaining_ms": 5000.0})
    es.reset()
    decision = es.should_stop({"confidence": 0.5, "budget_remaining_ms": 5000.0})
    assert decision.reason == "continue"
