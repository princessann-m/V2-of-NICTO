from __future__ import annotations

from typing import Any


class TextureExpert:
    def __init__(self, meta: dict | None = None) -> None:
        self.meta = meta or {"name": "texture_expert"}

    def analyze(self, specifications: dict[str, Any]) -> dict[str, Any]:
        texture = specifications.get("texture", {})
        return {
            "expert": self.meta.get("name"),
            "texture": texture,
            "score": 0.5,
            "issues": [],
            "suggestions": ["increase_material_detail"],
        }

    def iterate(self, current: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
        result = dict(current)
        result["texture"] = feedback.get("texture_adjustments", current.get("texture", {}))
        result["iterated"] = True
        return result
