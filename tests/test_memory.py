"""Tests for memory systems."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mom.memory.context import ContextMemory
from mom.memory.experiment_memory import ExperimentMemory, ExperimentRecord


def test_context_memory_add_and_history():
    mem = ContextMemory()
    mem.add("user", "hello")
    mem.add("assistant", "world")
    hist = mem.get_history(1)
    assert len(hist) == 1
    assert hist[0]["role"] == "assistant"


def test_context_memory_store_retrieve():
    mem = ContextMemory()
    mem.store("key", "value")
    assert mem.retrieve("key") == "value"
    assert mem.retrieve("missing", "default") == "default"


def test_context_memory_summarize():
    mem = ContextMemory()
    mem.add("user", "one")
    mem.add("assistant", "two")
    summary = mem.summarize()
    assert "user:" in summary


def test_context_memory_save_load():
    mem = ContextMemory()
    mem.add("user", "persist me")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mem.save(path)
        mem2 = ContextMemory()
        mem2.load(path)
        assert len(mem2.get_history()) == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_experiment_memory_start_complete():
    mem = ExperimentMemory()
    rec = mem.start("exp1", {"lr": 0.01})
    assert rec.experiment_id == "exp1"
    rec = mem.complete("exp1", {"accuracy": 0.9}, ["model.pth"])
    assert rec.metrics["accuracy"] == 0.9


def test_experiment_memory_compare():
    mem = ExperimentMemory()
    mem.start("a", {"lr": 0.01})
    mem.complete("a", {"accuracy": 0.9})
    mem.start("b", {"lr": 0.001})
    mem.complete("b", {"accuracy": 0.85})
    comparison = mem.compare(["a", "b"])
    assert comparison["a"]["metrics"]["accuracy"] == 0.9
