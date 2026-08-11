"""Concrete math expert."""

from __future__ import annotations

from ..tools.calculator import Calculator
from ..models.llm_client import LLMClient


class MathExpert:
    def __init__(self, meta: dict, llm: LLMClient | None = None):
        self.meta = meta
        self.llm = llm
        self.calc = Calculator()

    def compute(self, task: dict) -> dict:
        prompt = task.get("original", "")
        expr = self._extract_expression(prompt)
        answer = ""
        evidence = []
        if expr:
            try:
                answer = str(self.calc.evaluate(expr))
                evidence.append(f"calculator:{expr}={answer}")
            except Exception:
                answer = "(calculation failed)"
        if not answer and self.llm:
            try:
                answer = self.llm.chat([
                    {"role": "system", "content": "You are a math expert. Provide a concise, correct answer."},
                    {"role": "user", "content": prompt},
                ])
                evidence.append("llm:math")
            except Exception:
                answer = "(math expert unavailable)"
        if not answer:
            answer = "(no answer)"
        return {
            "answer": answer,
            "evidence": evidence,
            "metadata": {"expert": self.meta.get("name"), "tool": "calculator"},
        }

    def _extract_expression(self, text: str) -> str | None:
        import re
        m = re.search(r"[\d\.\+\-\*\/\(\)\s\^]+", text)
        if m:
            candidate = m.group(0).strip()
            if any(ch.isdigit() for ch in candidate):
                return candidate
        return None
