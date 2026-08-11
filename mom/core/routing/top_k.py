"""Top-K selector with sparse activation and learned routing support."""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    selected_ids: list[str]
    scores: list[float]
    all_scores: dict[str, float]
    k_used: int
    method: str = "topk"


class TopKSelector:
    def __init__(self, k: int = 4, learned_router: Any = None):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.learned_router = learned_router

    def select(self, scores: dict[str, float], k: int | None = None) -> RoutingResult:
        k = k or self.k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [item[0] for item in sorted_items[:k]]
        selected_scores = [item[1] for item in sorted_items[:k]]
        return RoutingResult(
            selected_ids=selected,
            scores=selected_scores,
            all_scores=dict(sorted_items),
            k_used=min(k, len(sorted_items)),
        )

    def route_dataset(self, dataset: list[dict[str, Any]], score_fn: Any) -> list[RoutingResult]:
        results = []
        for sample in dataset:
            scores = score_fn(sample)
            results.append(self.select(scores))
        return results

    def sparse_activate(self, scores: dict[str, float], threshold: float = 0.0) -> RoutingResult:
        filtered = {k: v for k, v in scores.items() if v > threshold}
        return self.select(filtered)

    def learned_select(self, task: dict[str, Any], candidates: list[str]) -> RoutingResult:
        if self.learned_router is None:
            scores = {c: random.random() for c in candidates}
            return self.select(scores)
        scores = self.learned_router.score(task, candidates)
        return self.select(scores)
