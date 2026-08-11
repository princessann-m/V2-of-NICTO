"""Tool result cache with input hashing, validation, and TTL."""

from __future__ import annotations

import time
import hashlib
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable

LOG = logging.getLogger(__name__)


@dataclass
class ToolCacheEntry:
    key: str
    result: Any
    created_at: float
    ttl_seconds: float
    valid: bool = True

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class ToolResultCache:
    def __init__(self, default_ttl_seconds: float = 300.0, max_entries: int = 1024):
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, ToolCacheEntry] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_input(fn_name: str, args: tuple, kwargs: dict[str, Any]) -> str:
        payload = pickle.dumps((fn_name, args, kwargs))
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None or not entry.valid or entry.expired:
            self._misses += 1
            if entry:
                entry.valid = False
            return None
        entry.created_at = time.time()
        self._hits += 1
        return entry.result

    def put(self, key: str, result: Any, ttl_seconds: float | None = None, validate: Callable | None = None) -> None:
        if validate is not None and not validate(result):
            return
        if len(self._entries) >= self.max_entries:
            self._evict()
        self._entries[key] = ToolCacheEntry(
            key=key,
            result=result,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
        )

    def _evict(self) -> None:
        if not self._entries:
            return
        oldest = min(self._entries, key=lambda k: self._entries[k].created_at)
        del self._entries[oldest]

    def clear(self) -> None:
        self._entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
