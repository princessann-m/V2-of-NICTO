"""Tokenizer cache with shared vocabulary."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class TokenizerEntry:
    name: str
    tokenizer: Any
    vocab_size: int
    eos_token_id: int | None = None


class TokenizerCache:
    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self._cache: dict[str, TokenizerEntry] = {}

    def get(self, name: str) -> Any | None:
        entry = self._cache.get(name)
        return entry.tokenizer if entry else None

    def put(self, name: str, tokenizer: Any, vocab_size: int = 0, eos_token_id: int | None = None) -> None:
        if len(self._cache) >= self.max_size:
            first = next(iter(self._cache))
            del self._cache[first]
        self._cache[name] = TokenizerEntry(
            name=name,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
        )

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._cache
