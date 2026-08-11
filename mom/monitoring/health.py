"""Component health checks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentHealth:
    name: str
    status: str = "unknown"
    last_check: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class HealthCheck:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.components: dict[str, ComponentHealth] = {}

    def register(self, name: str, checker: Any) -> None:
        self.components[name] = ComponentHealth(name=name)

    def check(self, name: str) -> dict[str, Any]:
        comp = self.components.get(name)
        if comp is None:
            return {"name": name, "status": "unknown"}
        comp.last_check = time.time()
        comp.status = "healthy"
        comp.details.setdefault("checks", 0)
        comp.details["checks"] += 1
        return {
            "name": comp.name,
            "status": comp.status,
            "last_check": comp.last_check,
            "details": comp.details,
        }

    def status(self) -> dict[str, Any]:
        healthy = sum(1 for c in self.components.values() if c.status == "healthy")
        total = len(self.components)
        return {
            "healthy": healthy,
            "total": total,
            "components": [
                {
                    "name": c.name,
                    "status": c.status,
                    "last_check": c.last_check,
                }
                for c in self.components.values()
            ],
        }

    def alert_threshold(self, name: str, threshold: float) -> bool:
        comp = self.components.get(name)
        if comp is None:
            return False
        latency = comp.details.get("last_latency_ms", 0)
        return latency > threshold
