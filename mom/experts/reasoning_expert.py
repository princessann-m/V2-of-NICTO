"""Concrete reasoning expert."""

from __future__ import annotations

from ..models.llm_client import LLMClient


class ReasoningExpert:
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
                    {"role": "system", "content": "You are a reasoning expert. Think step by step and give a clear final answer."},
                    {"role": "user", "content": prompt},
                ])
                evidence.append("llm:reasoning")
            except Exception:
                answer = "(reasoning expert unavailable)"
        if not answer:
            answer = "(no reasoning produced)"
        return {
            "answer": answer,
            "evidence": evidence,
            "metadata": {"expert": self.meta.get("name")},
        }
