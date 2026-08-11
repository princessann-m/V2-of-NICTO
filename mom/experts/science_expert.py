"""Concrete science / simulation expert."""

from __future__ import annotations

from ..tools.simulator import Simulator
from ..models.llm_client import LLMClient


class ScienceExpert:
    def __init__(self, meta: dict, llm: LLMClient | None = None):
        self.meta = meta
        self.llm = llm
        self.sim = Simulator()

    def compute(self, task: dict) -> dict:
        prompt = task.get("original", "")
        answer = ""
        evidence = []
        sim_params = self._extract_params(prompt)
        if sim_params:
            try:
                sim_result = self.sim.run("generic", sim_params)
                answer = str(sim_result)
                evidence.append(f"simulator:{sim_result}")
            except Exception:
                answer = "(simulation failed)"
        if not answer and self.llm:
            try:
                answer = self.llm.chat([
                    {"role": "system", "content": "You are a science expert. Provide analysis and results."},
                    {"role": "user", "content": prompt},
                ])
                evidence.append("llm:science")
            except Exception:
                answer = "(science expert unavailable)"
        if not answer:
            answer = "(no analysis produced)"
        return {
            "answer": answer,
            "evidence": evidence,
            "metadata": {"expert": self.meta.get("name")},
        }

    def _extract_params(self, text: str) -> dict:
        params = {}
        import re
        for m in re.finditer(r"(\w+)\s*=\s*([\d\.]+)", text):
            params[m.group(1)] = float(m.group(2))
        return params
