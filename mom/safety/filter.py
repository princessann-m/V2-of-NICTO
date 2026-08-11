"""Input and output safety filtering."""

from __future__ import annotations

import re
from typing import Any

from .policy import SafetyPolicy


class SafetyFilter:
    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()

    def filter_input(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        severity = self.policy.severity(content)
        blocked = severity != "none"
        self.policy.log({"type": "input", "severity": severity, "blocked": blocked})
        return {
            "content": content if not blocked else "[FILTERED]",
            "blocked": blocked,
            "severity": severity,
        }

    def filter_output(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        severity = self.policy.severity(content)
        blocked = severity != "none"
        self.policy.log({"type": "output", "severity": severity, "blocked": blocked})
        return {
            "content": content if not blocked else "[FILTERED]",
            "blocked": blocked,
            "severity": severity,
        }

    def detect_pii(self, content: str) -> list[str]:
        findings: list[str] = []
        if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", content):
            findings.append("email")
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", content):
            findings.append("phone")
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", content):
            findings.append("ssn")
        return findings

    def enforce(self, content: str, context: str = "input") -> dict[str, Any]:
        if context == "input":
            return self.filter_input(content)
        return self.filter_output(content)
