"""Hallucination and validity checker."""

from __future__ import annotations

import re

from ..tools.calculator import Calculator


class HallucinationChecker:
    def __init__(self):
        self.calc = Calculator()

    def check(self, candidate: dict, task: dict) -> tuple[bool, list]:
        issues = []
        ans = candidate.get("answer", "")
        if not ans or not ans.strip():
            issues.append("empty_answer")
        if task.get("task_type") == "math":
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", ans)
            if numbers:
                exprs = re.findall(r"[\d\s\+\-\*\/\.\(\)\^]+", ans)
                for ex in exprs:
                    ex = ex.strip()
                    if len(ex) > 0 and any(ch.isdigit() for ch in ex):
                        try:
                            self.calc.evaluate(ex)
                        except Exception:
                            issues.append(f"bad_expr:{ex}")
        ok = len(issues) == 0
        return ok, issues
