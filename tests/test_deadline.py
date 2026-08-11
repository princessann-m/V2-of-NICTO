"""Tests for DeadlineManager."""

from __future__ import annotations

import time
import pytest

from mom.core.orchestrator.deadline import DeadlineManager, Deadline


def test_deadline_creation():
    mgr = DeadlineManager(default_hard_ms=1000.0, default_soft_ms=800.0)
    d = mgr.create("t1")
    assert d.task_id == "t1"
    assert d.remaining_ms <= 1000.0


def test_deadline_expiry():
    mgr = DeadlineManager(default_hard_ms=100.0, default_soft_ms=50.0)
    d = mgr.create("t2", hard_ms=100.0, soft_ms=50.0)
    time.sleep(0.12)
    assert d.is_hard_deadline_passed is True


def test_deadline_extension():
    mgr = DeadlineManager(default_hard_ms=500.0, default_soft_ms=400.0)
    d = mgr.create("t3", hard_ms=500.0, soft_ms=400.0)
    before = d.remaining_ns
    d.extend(budget_ms=200.0)
    assert d.remaining_ns > before
    assert d.extensions_used == 1


def test_deadline_stats():
    mgr = DeadlineManager()
    mgr.create("t1")
    mgr.record_completion("t1", 500.0)
    stats = mgr.stats()
    assert stats["count"] == 1


def test_deadline_early_stop():
    mgr = DeadlineManager(default_hard_ms=50.0, default_soft_ms=20.0)
    mgr.create("t_low", hard_ms=50.0, soft_ms=20.0, priority=1)
    time.sleep(0.06)
    assert mgr.should_early_stop("t_low") is True
