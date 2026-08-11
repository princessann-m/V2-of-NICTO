"""Tests for caching subsystems."""

from __future__ import annotations

import pytest

from mom.cache.model_cache import ModelCache
from mom.cache.tokenizer_cache import TokenizerCache
from mom.cache.tool_cache import ToolResultCache
from mom.cache.simulation_cache import SimulationResultCache


def test_model_cache_put_get():
    cache = ModelCache(max_entries=4)
    cache.put("m1", {"weights": [1, 2]}, size_bytes=1024)
    assert cache.get("m1") == {"weights": [1, 2]}


def test_model_cache_miss():
    cache = ModelCache()
    assert cache.get("missing") is None


def test_model_cache_eviction():
    cache = ModelCache(max_entries=2)
    cache.put("a", 1, size_bytes=512)
    cache.put("b", 2, size_bytes=512)
    cache.put("c", 3, size_bytes=512)
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_model_cache_stats():
    cache = ModelCache()
    cache.put("x", 1, size_bytes=100)
    cache.get("x")
    cache.get("missing")
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_tokenizer_cache():
    cache = TokenizerCache(max_size=2)
    cache.put("tok1", object())
    assert "tok1" in cache
    assert cache.get("tok1") is not None
    cache.clear()
    assert "tok1" not in cache


def test_tool_cache_hash():
    cache = ToolResultCache()
    key = cache.hash_input("fn", (1, 2), {"a": 3})
    assert isinstance(key, str)
    assert len(key) == 64


def test_tool_cache_put_get():
    cache = ToolResultCache(default_ttl_seconds=60.0)
    cache.put("k1", {"result": 42})
    assert cache.get("k1") == {"result": 42}


def test_tool_cache_ttl():
    cache = ToolResultCache(default_ttl_seconds=0.1)
    cache.put("k1", 1, ttl_seconds=0.1)
    import time
    time.sleep(0.15)
    assert cache.get("k1") is None


def test_simulation_cache():
    cache = SimulationResultCache()
    cache.put("cfg1", {"sim": True})
    assert cache.get("cfg1") == {"sim": True}
    assert cache.get("missing") is None
