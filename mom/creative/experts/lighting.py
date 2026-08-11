from __future__ import annotations

from typing import Any


class LightingExpert:
    def __init__(self, meta: dict | None = None) -> None:
        self.meta = meta or {"name": "lighting_expert"}

    def analyze(self, specifications: dict[str, Any]) -> dict[str, Any]:
        lighting = specifications.get("lighting", {})
        return {
            "expert": self.meta.get("name"),
            "lighting": lighting,
            "score": 0.5,
            "issues": [],
            "suggestions": ["increase_ambient_occlusion"],
        }

    def iterate(self, current: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
        result = dict(current)
        result["lighting"] = feedback.get("lighting_adjustments", current.get("lighting", {}))
        result["iterated"] = True
        return result
