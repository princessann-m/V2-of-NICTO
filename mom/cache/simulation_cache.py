"""Simulation result cache for deterministic reuse."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class SimulationEntry:
    config_hash: str
    result: Any
    created_at: float
    ttl_seconds: float
    valid: bool = True

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class SimulationResultCache:
    def __init__(self, default_ttl_seconds: float = 600.0, max_entries: int = 256):
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, SimulationEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, config_hash: str) -> Any | None:
        entry = self._entries.get(config_hash)
        if entry is None or not entry.valid or entry.expired:
            self._misses += 1
            if entry:
                entry.valid = False
            return None
        entry.created_at = time.time()
        self._hits += 1
        return entry.result

    def put(self, config_hash: str, result: Any, ttl_seconds: float | None = None) -> None:
        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].created_at)
            del self._entries[oldest]
        self._entries[config_hash] = SimulationEntry(
            config_hash=config_hash,
            result=result,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
        )

    def clear(self) -> None:
        self._entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
