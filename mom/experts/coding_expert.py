"""Concrete coding expert."""

from __future__ import annotations

from ..tools.coding_studio import CodingStudio
from ..models.llm_client import LLMClient


class CodingExpert:
    def __init__(self, meta: dict, llm: LLMClient | None = None):
        self.meta = meta
        self.llm = llm
        self.studio = CodingStudio()

    def compute(self, task: dict) -> dict:
        prompt = task.get("original", "")
        code = ""
        evidence = []
        if self.llm:
            try:
                code = self.llm.chat([
                    {"role": "system", "content": "You are a coding expert. Return only the code block, no extra text."},
                    {"role": "user", "content": prompt},
                ])
                evidence.append("llm:code")
            except Exception:
                code = "# code generation failed"
        test_result = {"tests_run": 0, "tests_passed": 0, "success": False}
        if code:
            try:
                test_result = self.studio.run_tests({"solution.py": code})
                evidence.append(f"studio:{test_result}")
            except Exception:
                pass
        return {
            "answer": code or "(no code generated)",
            "evidence": evidence,
            "metadata": {"expert": self.meta.get("name"), "tests": test_result},
        }
