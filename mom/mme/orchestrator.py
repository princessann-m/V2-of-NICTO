"""MME orchestrator running three independent systems."""

from __future__ import annotations

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from .system import MMESystem
from ..models.model_registry import ModelRegistry
from ..config import MoMConfig, load_config


class MMEOrchestrator:
    def __init__(self, registry: ModelRegistry, config: MoMConfig | None = None, llm_client=None):
        self.config = config or load_config()
        self.registry = registry
        self.llm = llm_client
        self.systems = [
            MMESystem({"selected_experts": [], "top_k": 2}, registry=registry, llm_client=llm_client, seed=i)
            for i in range(3)
        ]

    def run(self, task: dict, deadline: float) -> list[dict]:
        candidates = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(mme_sys.process, task, deadline) for mme_sys in self.systems]
            for f in as_completed(futures):
                try:
                    cand = f.result()
                    candidates.append(cand)
                except Exception as e:
                    candidates.append(
                        {
                            "answer": f"(MME failed: {e})",
                            "confidence": 0.0,
                            "metadata": {"error": str(e)},
                        }
                    )
        return candidates
