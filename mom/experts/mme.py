"""Mixed Mamba Expert System — orchestrates selected experts."""

from __future__ import annotations

import random
import time

from ..experts import build_expert
from ..models.llm_client import LLMClient


class MixedMambaExpertSystem:
    def __init__(self, config: dict, llm: LLMClient | None = None):
        self.config = config or {"selected_experts": []}
        self.llm = llm

    def process(self, task: dict, deadline: float) -> dict:
        selected = self.config.get("selected_experts", [])
        results = []
        for m in selected:
            expert = build_expert(m, llm=self.llm)
            start = time.time()
            try:
                out = expert.compute(task)
                out.setdefault("latency_ms", int((time.time() - start) * 1000))
                results.append(out)
            except Exception as e:
                results.append({
                    "answer": f"(expert {m.get('name')} failed: {e})",
                    "metadata": {"expert": m.get("name"), "error": str(e)},
                    "latency_ms": int((time.time() - start) * 1000),
                })

        answer, confidence, reasoning = self._aggregate(results)
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": reasoning,
            "evidence": [e for r in results for e in r.get("evidence", [])],
            "metadata": {"experts_used": [m.get("name") for m in selected]},
            "mme_config": self.config,
            "candidate_results": results,
        }

    def _aggregate(self, results: list[dict]) -> tuple[str, float, str]:
        answers = [r.get("answer", "") for r in results if r.get("answer")]
        if not answers:
            return "(no answer)", 0.0, "No experts produced output"
        if len(answers) == 1:
            return answers[0], 0.7, "Single expert result"
        avg_len = sum(len(a) for a in answers) / len(answers)
        if avg_len < 20:
            combined = " | ".join(answers)
            confidence = min(0.95, 0.5 + 0.1 * len(answers))
            return combined, confidence, f"Combined {len(answers)} short answers"
        best = max(results, key=lambda r: len(r.get("evidence", [])))
        confidence = min(0.95, 0.6 + 0.1 * len(best.get("evidence", [])))
        return best.get("answer", answers[0]), confidence, "Selected best evidence-rich candidate"
