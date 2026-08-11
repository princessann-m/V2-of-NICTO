"""LRU model cache with priority eviction and memory tracking."""

from __future__ import annotations

import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    key: str
    value: Any
    size_bytes: int
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class ModelCache:
    def __init__(self, max_memory_bytes: int = 2 * 1024 ** 3, max_entries: int = 16):
        self.max_memory = max_memory_bytes
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}
        self._order: list[str] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        entry.last_accessed = time.time()
        entry.hits += 1
        self._hits += 1
        self._order.remove(key)
        self._order.append(key)
        return entry.value

    def put(self, key: str, value: Any, size_bytes: int = 0, priority: int = 0) -> None:
        if key in self._entries:
            self._entries[key].value = value
            self._entries[key].size_bytes = size_bytes
            self._entries[key].priority = priority
            self._entries[key].last_accessed = time.time()
            return
        self._evict_if_needed(size_bytes)
        entry = CacheEntry(key=key, value=value, size_bytes=size_bytes, priority=priority)
        self._entries[key] = entry
        self._order.append(key)

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        while (len(self._entries) >= self.max_entries or
               self._memory_usage() + incoming_bytes > self.max_memory):
            if not self._order:
                break
            evict_key = self._order.pop(0)
            evicted = self._entries.pop(evict_key, None)
            if evicted:
                LOG.debug("Evicted model cache entry %s", evict_key)

    def _memory_usage(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)
        if key in self._order:
            self._order.remove(key)

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "memory_bytes": self._memory_usage(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }
