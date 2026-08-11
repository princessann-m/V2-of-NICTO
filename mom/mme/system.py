"""Mixed Mamba Expert system."""

from __future__ import annotations

import time
import random

from ..models.mamba.expert import MambaExpert
from ..models.mamba.config import MambaConfig
from ..core.routing.expert_router import ExpertRouter
from ..config import MoMConfig, load_config


class MMESystem:
    def __init__(self, config: dict, registry=None, llm_client=None, seed: int = 0):
        self.config = config or {"selected_experts": []}
        self.registry = registry
        self.llm = llm_client
        self.seed = seed
        self.expert_router = ExpertRouter(registry, k=config.get("top_k", 2))

    def process(self, task: dict, deadline: float) -> dict:
        selected = self.config.get("selected_experts", [])
        if not selected and self.registry:
            routed = self.expert_router.route(task)
            selected = []
            for eid in routed.selected_ids:
                expert_meta = self.registry.query(domain=task.get("task_type", "general"))
                for e in expert_meta:
                    if e.get("id") == eid or e.get("name") == eid:
                        selected.append(e)
                        break
            selected = selected[: self.config.get("top_k", 2)]
        results = []
        remaining = max(0.1, deadline - time.time())
        max_new = max(1, min(20, int(remaining * 10)))
        for m in selected:
            cfg = MambaConfig(
                vocab_size=100,
                d_model=32,
                n_layers=2,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            expert = MambaExpert(
                config=cfg,
                role=m.get("role", m.get("domain", "general")),
                capabilities=m.get("capabilities", []),
            )
            start = time.time()
            try:
                out = expert.compute(task, max_new_tokens=max_new)
                out.setdefault("latency_ms", int((time.time() - start) * 1000))
                results.append(out)
            except Exception as e:
                results.append(
                    {
                        "answer": f"(expert {m.get('name')} failed: {e})",
                        "metadata": {"expert": m.get("name"), "error": str(e)},
                        "latency_ms": int((time.time() - start) * 1000),
                    }
                )

        answer, confidence, reasoning = self._aggregate(results)
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": reasoning,
            "evidence": [e for r in results for e in r.get("evidence", [])],
            "metadata": {"experts_used": [m.get("name") for m in selected], "mme_id": self.seed},
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
            confidence = min(0.95, 0.5 + 0.1 * len(answers))
            return "\n".join(answers), confidence, f"Combined {len(answers)} short answers"
        best = max(results, key=lambda r: len(r.get("evidence", [])))
        confidence = min(0.95, 0.6 + 0.1 * len(best.get("evidence", [])))
        return best.get("answer", answers[0]), confidence, "Selected best evidence-rich candidate"
