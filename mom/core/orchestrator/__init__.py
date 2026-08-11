"""Orchestrator subpackage with latency-aware architecture."""

from __future__ import annotations

import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .deadline import DeadlineManager, Deadline
from .latency import LatencyEstimator, LatencyEstimate
from .scheduler import TaskScheduler, ScheduledTask
from mom.core.router import Router
from mom.core.routing.expert_router import ExpertRouter, ExpertScore, ExpertCache
from mom.core.stopping.early_stop import EarlyStopping, StopDecision
from mom.core.adaptive.router import AdaptiveRouter, AdaptiveDecision
from mom.models.top_model import TopModel
from mom.models.model_registry import ModelRegistry
from mom.models.llm_client import LLMClient
from mom.config import MoMConfig, load_config
from mom.mme.orchestrator import MMEOrchestrator
from mom.models.verifier import VerifierModel, VerifierConfig
from mom.models.judge import JudgeModel, JudgeConfig
from mom.verification.hallucination import HallucinationChecker

LOG_PATH = "logs/mom_requests.log"
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


@dataclass
class OrchestratorMetrics:
    total_requests: int = 0
    p95_latency_ms: float = 0.0
    target_met: bool = False
    expert_routing_ms: float = 0.0
    tool_execution_ms: float = 0.0
    judge_ms: float = 0.0


class Orchestrator:
    def __init__(self, config: MoMConfig | None = None):
        self.config = config or load_config()
        self.top_model = TopModel(self.config)
        self.registry = ModelRegistry()
        self.router = Router(self.registry)
        self.expert_router = ExpertRouter(self.registry)
        self.deadline_mgr = DeadlineManager(default_hard_ms=10_000.0, default_soft_ms=8_000.0)
        self.latency_estimator = LatencyEstimator()
        self.scheduler = TaskScheduler(max_workers=4)
        self.early_stopping = EarlyStopping()
        self.adaptive_router = AdaptiveRouter(default_latency_budget_ms=10_000.0)
        self.max_retries = self.config.max_retries
        self.global_deadline = self.config.global_deadline
        self.h_checker = HallucinationChecker()
        self.llm = LLMClient(self.config.llm) if self.config.llm.provider != "heuristic" else None
        self.mme_orchestrator = MMEOrchestrator(self.registry, self.config, llm_client=self.llm)
        self.verifier = VerifierModel(VerifierConfig())
        self.judge1 = JudgeModel(JudgeConfig(), name="judge_alpha")
        self.judge2 = JudgeModel(JudgeConfig(), name="judge_beta")
        self._video_pipeline = None
        self._image_pipeline = None
        self.metrics = OrchestratorMetrics()
        self._request_times: list[float] = []

    def handle_request(self, user_input: str, request_id: str | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        request_id = request_id or f"req_{int(start * 1000)}"
        deadline = self.deadline_mgr.create(request_id, hard_ms=self.global_deadline * 1000.0)
        log: dict[str, Any] = {
            "request_id": request_id,
            "user_input": user_input,
            "start_time": start,
        }

        task = self.top_model.parse(user_input)
        log["task_representation"] = task
        adaptive = self.adaptive_router.decide(task, budget_ms=deadline.remaining_ms)
        k = self.expert_router.dynamic_k(task)

        modality = task.get("modality", "text")
        if modality == "video":
            pipeline = self._get_video_pipeline()
            if pipeline:
                result = pipeline.generate(task)
                log["end_time"] = time.time()
                self._write_log(log)
                return result
        if modality == "image":
            pipeline = self._get_image_pipeline()
            if pipeline:
                result = pipeline.generate(task.get("visual_specifications", task))
                log["end_time"] = time.time()
                self._write_log(log)
                return result

        routing_start = time.perf_counter()
        _, expert_scores = self.expert_router.route_with_metadata(task, k=k)
        routing_ms = (time.perf_counter() - routing_start) * 1000
        log["expert_routing_ms"] = routing_ms
        self.metrics.expert_routing_ms = routing_ms

        candidates = self._run_parallel_candidates(task, deadline, adaptive)
        log["raw_candidates"] = candidates

        verified = []
        tool_start = time.perf_counter()
        for cand in candidates:
            attempts = 0
            while attempts <= self.max_retries:
                stop = self.deadline_mgr.should_early_stop(request_id)
                if stop:
                    LOG.warning("Early stopping for %s", request_id)
                    break
                ok, issues = self.h_checker.check(cand, task)
                cand.setdefault("hallucination_issues", []).extend(issues)
                if ok:
                    verified.append(cand)
                    break
                attempts += 1
                if not stop:
                    new_candidates = self._run_parallel_candidates(task, deadline, adaptive, speculative=True)
                    if new_candidates:
                        cand = new_candidates[0]
        tool_ms = (time.perf_counter() - tool_start) * 1000
        log["tool_execution_ms"] = tool_ms
        self.metrics.tool_execution_ms = tool_ms

        log["verified_candidates"] = verified

        judge_start = time.perf_counter()
        scores1 = self.judge1.score(verified, task)
        scores2 = self.judge2.score(verified, task)
        judge_ms = (time.perf_counter() - judge_start) * 1000
        log["judge_ms"] = judge_ms
        self.metrics.judge_ms = judge_ms
        log["judge1_scores"] = scores1
        log["judge2_scores"] = scores2

        final = self._arbitrate(verified, scores1, scores2)
        log["final_choice"] = final
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        log["latency_ms"] = latency_ms
        log["end_time"] = end
        self._request_times.append(latency_ms)
        if len(self._request_times) >= 20:
            sorted_times = sorted(self._request_times)
            idx = int(0.95 * (len(sorted_times) - 1))
            self.metrics.p95_latency_ms = sorted_times[idx]
            self.metrics.target_met = self.metrics.p95_latency_ms <= 10_000.0
        self.deadline_mgr.record_completion(request_id, latency_ms)
        self._write_log(log)
        return final

    def _run_parallel_candidates(self, task: dict[str, Any], deadline: Any, adaptive: Any, speculative: bool = False) -> list[dict[str, Any]]:
        try:
            absolute_deadline = time.time() + max(0.1, deadline.remaining_ms / 1000.0)
            return self.mme_orchestrator.run(task, absolute_deadline)
        except Exception as exc:
            LOG.debug("Parallel execution failed: %s", exc)
            return []

    def _arbitrate(self, candidates: list[dict[str, Any]], scores1: dict[str, Any], scores2: dict[str, Any]) -> dict[str, Any]:
        if not candidates:
            return {"answer": "", "metadata": {"reason": "no_candidates"}}
        if not scores1 or not scores2:
            best = max(candidates, key=lambda c: c.get("confidence", 0))
            return best
        top1 = max(scores1, key=lambda k: scores1[k]) if scores1 else "0"
        top2 = max(scores2, key=lambda k: scores2[k]) if scores2 else "0"
        if top1 == top2:
            return candidates[int(top1)] if top1.isdigit() and int(top1) < len(candidates) else candidates[0]
        combined = {}
        for idx, cand in enumerate(candidates):
            s1 = scores1.get(str(idx), 0)
            s2 = scores2.get(str(idx), 0)
            combined[idx] = s1 + s2
        best_idx = max(combined, key=lambda k: combined[k])
        return candidates[int(best_idx)]

    def _get_video_pipeline(self):
        if self._video_pipeline is None:
            try:
                from ..generation.video.pipeline import VideoPipeline
                from ..generation.video.config import VideoPipelineConfig
                self._video_pipeline = VideoPipeline(config=VideoPipelineConfig())
            except Exception:
                self._video_pipeline = False
        return self._video_pipeline if self._video_pipeline is not False else None

    def _get_image_pipeline(self):
        if self._image_pipeline is None:
            try:
                from ..generation.image.pipeline import ImagePipeline
                self._image_pipeline = ImagePipeline()
            except Exception:
                self._image_pipeline = False
        return self._image_pipeline if self._image_pipeline is not False else None

    def _write_log(self, record: dict[str, Any]) -> None:
        try:
            import os
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            logging.exception("Failed to write log")
