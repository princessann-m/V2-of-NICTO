"""Sparse expert router with 310 experts, hierarchical routing, and performance awareness."""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any

from .expert_registry import ExpertRegistry, ExpertMetadata
from .top_k import TopKSelector, RoutingResult
from .hardware import HardwareDetector

LOG = logging.getLogger(__name__)


@dataclass
class ExpertScore:
    expert: ExpertMetadata
    score: float
    latency_ms: float
    quality: float
    compute_cost: float
    confidence: float
    agreement: float = 1.0


class ConfidenceAgreementMonitor:
    def __init__(self, disagreement_threshold: float = 0.2):
        self.disagreement_threshold = disagreement_threshold
        self._history: list[dict[str, Any]] = []

    def check(self, scores: list[ExpertScore]) -> bool:
        if len(scores) < 2:
            return False
        values = [s.score for s in scores]
        spread = max(values) - min(values)
        triggered = spread > self.disagreement_threshold
        self._history.append({"spread": spread, "triggered": triggered, "count": len(scores)})
        return triggered

    def record_routing(self, task: dict[str, Any], scores: list[ExpertScore]) -> None:
        self._history.append({
            "task_type": task.get("task_type", "unknown"),
            "expert_count": len(scores),
            "spread": max((s.score for s in scores), default=0.0) - min((s.score for s in scores), default=0.0),
        })


class ExpertCache:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        if len(self._cache) >= self.max_size:
            first = next(iter(self._cache))
            del self._cache[first]
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


class LoadBalancer:
    def __init__(self):
        self._load: dict[str, int] = {}

    def record_invocation(self, expert_id: str) -> None:
        self._load[expert_id] = self._load.get(expert_id, 0) + 1

    def load_factor(self, expert_id: str) -> float:
        return self._load.get(expert_id, 0)

    def least_loaded(self, candidates: list[ExpertMetadata]) -> ExpertMetadata | None:
        if not candidates:
            return None
        return min(candidates, key=lambda e: self._load.get(e.id, 0))


class ExpertRouter:
    def __init__(self, registry: ExpertRegistry, k: int = 4, hardware: HardwareDetector | None = None):
        self.registry = registry
        self.k = k
        self.hardware = hardware or HardwareDetector()
        self.selector = TopKSelector(k=k)
        self.cache = ExpertCache()
        self.load_balancer = LoadBalancer()
        self.monitor = ConfidenceAgreementMonitor()
        self._default_k = k

    def route(self, task: dict[str, Any], k: int | None = None) -> RoutingResult:
        k = k or self._default_k
        cache_key = self._cache_key(task, k)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = self._hierarchical_filter(task)
        if not candidates:
            candidates = self.registry.all()

        scores = [self._score_expert(e, task) for e in candidates]
        score_map = {s.expert.id: s.score for s in scores}
        result = self.selector.select(score_map, k=k)

        expert_scores = {s.expert.id: s for s in scores}
        selected_scores = [expert_scores[eid] for eid in result.selected_ids if eid in expert_scores]
        self.monitor.record_routing(task, selected_scores)

        self.cache.put(cache_key, result)
        return result

    def route_with_metadata(self, task: dict[str, Any], k: int | None = None) -> tuple[RoutingResult, list[ExpertScore]]:
        k = k or self._default_k
        candidates = self._hierarchical_filter(task)
        if not candidates:
            candidates = self.registry.all()
        scores = [self._score_expert(e, task) for e in candidates]
        score_map = {s.expert.id: s.score for s in scores}
        result = self.selector.select(score_map, k=k)
        expert_scores = {s.expert.id: s for s in scores}
        selected = [expert_scores[eid] for eid in result.selected_ids if eid in expert_scores]
        return result, selected

    def _hierarchical_filter(self, task: dict[str, Any]) -> list[ExpertMetadata]:
        domain = task.get("domain") or task.get("task_domain") or "general"
        subdomain = task.get("subdomain") or ""
        modality = task.get("modality")
        candidates = self.registry.query(domain=domain, subdomain=subdomain or None, modality=modality)
        if candidates:
            return candidates
        candidates = self.registry.query(domain=domain)
        if candidates:
            return candidates
        return self.registry.query(domain="general")

    def _score_expert(self, expert: ExpertMetadata, task: dict[str, Any]) -> ExpertScore:
        if isinstance(expert, dict):
            capabilities = expert.get("capabilities", [])
            latency_baseline_ms = expert.get("latency_baseline_ms", 500.0)
            quality_score = expert.get("quality_score", 0.7)
            compute_cost = expert.get("compute_cost", 0.1)
            expert_id = expert.get("id", expert.get("name", "unknown"))
        else:
            capabilities = expert.capabilities
            latency_baseline_ms = expert.latency_baseline_ms
            quality_score = expert.quality_score
            compute_cost = expert.compute_cost
            expert_id = expert.id
        task_caps = set(task.get("required_capabilities", []))
        expert_caps = set(capabilities)
        overlap = len(task_caps & expert_caps) / max(1, len(task_caps | expert_caps))

        latency = self.hardware.profile.placement_strategy
        latency_ms = latency_baseline_ms * (0.8 if latency == "cuda" else 1.2)
        quality = quality_score
        cost = compute_cost
        load = self.load_balancer.load_factor(expert_id)

        score = (
            0.4 * overlap +
            0.3 * quality +
            0.2 * (1.0 / max(1.0, latency_ms / 1000.0)) -
            0.1 * load
        )
        confidence = min(1.0, 0.5 + 0.5 * overlap + 0.1 * len(getattr(expert, "performance_history", [])) / 10.0)

        if isinstance(expert, dict):
            metadata = ExpertMetadata(
                id=expert_id,
                name=expert.get("name", expert_id),
                domain=expert.get("domain", "general"),
                subdomain=expert.get("subdomain", ""),
                capabilities=capabilities,
                modality=expert.get("modality", "text"),
                latency_baseline_ms=latency_baseline_ms,
                quality_score=quality_score,
                compute_cost=compute_cost,
            )
        else:
            metadata = expert
        return ExpertScore(
            expert=metadata,
            score=max(0.0, score),
            latency_ms=latency_ms,
            quality=quality,
            compute_cost=cost,
            confidence=confidence,
        )

    def dynamic_k(self, task: dict[str, Any]) -> int:
        complexity = task.get("complexity", "medium")
        if complexity == "easy":
            return 1
        if complexity == "medium":
            return 2
        if complexity == "hard":
            return 4
        return self._default_k

    def disagreement_trigger(self, task: dict[str, Any]) -> bool:
        _, scores = self.route_with_metadata(task, k=2)
        return self.monitor.check(scores)

    def _cache_key(self, task: dict[str, Any], k: int) -> str:
        return f"{task.get('task_type', '')}:{task.get('domain', '')}:{k}"
