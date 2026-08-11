"""Safety policy definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SafetyPolicy:
    level: Literal["low", "medium", "high"] = "medium"
    blocked_categories: list[str] = field(
        default_factory=lambda: ["violence", "hate", "self-harm", "sexual", "pii"]
    )
    pii_patterns: list[str] = field(default_factory=lambda: ["email", "phone", "ssn", "credit_card"])
    audit_log: list[dict] = field(default_factory=list)

    def is_allowed(self, content: str) -> bool:
        lowered = content.lower()
        return not any(cat in lowered for cat in self.blocked_categories)

    def severity(self, content: str) -> str:
        lowered = content.lower()
        hits = sum(1 for cat in self.blocked_categories if cat in lowered)
        if hits == 0:
            return "none"
        if hits == 1:
            return "low"
        if hits <= 3:
            return "medium"
        return "high"

    def log(self, event: dict) -> None:
        self.audit_log.append(event)
