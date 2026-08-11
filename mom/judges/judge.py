"""Judge implementations with real LLM-based evaluation."""

from __future__ import annotations

import random

from ..models.llm_client import LLMClient


class Judge:
    def __init__(self, name: str = "judge", llm: LLMClient | None = None):
        self.name = name
        self.llm = llm

    def score(self, candidates: list[dict], task: dict) -> dict:
        scores = {}
        for idx, c in enumerate(candidates):
            if self.llm:
                try:
                    score = self._score_with_llm(c, task)
                    scores[str(idx)] = score
                    continue
                except Exception:
                    pass
            scores[str(idx)] = self._score_heuristic(c)
        return scores

    def _score_with_llm(self, candidate: dict, task: dict) -> float:
        answer = candidate.get("answer", "")
        evidence = candidate.get("evidence", [])
        issues = candidate.get("hallucination_issues", [])
        prompt = (
            "Rate the following answer for a task from 0.0 to 1.0 based on correctness, "
            "completeness, and relevance. Return only a number.\n\n"
            f"Task: {task.get('original', '')}\n"
            f"Answer: {answer}\n"
            f"Evidence: {', '.join(evidence) if evidence else 'none'}\n"
            f"Issues: {', '.join(issues) if issues else 'none'}\n"
        )
        text = self.llm.chat([
            {"role": "system", "content": "You are a strict evaluator. Output only a float between 0.0 and 1.0."},
            {"role": "user", "content": prompt},
        ])
        import re
        m = re.search(r"\d*\.?\d+", text)
        if m:
            val = float(m.group(0))
            return max(0.0, min(1.0, val))
        return self._score_heuristic(candidate)

    def _score_heuristic(self, candidate: dict) -> float:
        s = 0.0
        s += float(candidate.get("confidence", 0)) * 2.0
        if candidate.get("evidence"):
            s += 0.5
        if candidate.get("hallucination_issues"):
            s -= len(candidate.get("hallucination_issues")) * 0.5
        s += (random.random() - 0.5) * 0.1
        return s
