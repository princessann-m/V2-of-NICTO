"""Conversation context memory."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextMemory:
    history: list[dict[str, Any]] = field(default_factory=list)
    long_term: dict[str, Any] = field(default_factory=dict)
    max_history: int = 200

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        item = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.history.append(item)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_history(self, last_n: int = 20) -> list[dict[str, Any]]:
        return self.history[-last_n:]

    def store(self, key: str, value: Any) -> None:
        self.long_term[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        return self.long_term.get(key, default)

    def summarize(self) -> str:
        return " | ".join(f"{h['role']}: {h['content'][:120]}" for h in self.history[-10:])

    def save(self, path: str | Path) -> None:
        data = {"history": self.history, "long_term": self.long_term}
        Path(path).write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.history = data.get("history", [])
        self.long_term = data.get("long_term", {})
