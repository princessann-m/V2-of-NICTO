"""Concrete vision expert placeholder (ready for real integration)."""

from __future__ import annotations

from ..models.llm_client import LLMClient


class VisionExpert:
    def __init__(self, meta: dict, llm: LLMClient | None = None):
        self.meta = meta
        self.llm = llm

    def compute(self, task: dict) -> dict:
        prompt = task.get("original", "")
        answer = ""
        evidence = []
        if self.llm:
            try:
                answer = self.llm.chat([
                    {"role": "system", "content": "You are a vision expert. Describe images and visual tasks."},
                    {"role": "user", "content": prompt},
                ])
                evidence.append("llm:vision")
            except Exception:
                answer = "(vision expert unavailable)"
        if not answer:
            answer = "(no visual analysis produced)"
        return {
            "answer": answer,
            "evidence": evidence,
            "metadata": {"expert": self.meta.get("name")},
        }
