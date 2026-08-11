"""FastAPI server for NICTO."""

from __future__ import annotations

import time
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..core.orchestrator import Orchestrator
from ..benchmarks.framework import BenchmarkFramework
from ..models.model_registry import ModelRegistry
from ..config import MoMConfig, LLMConfig

logger = logging.getLogger(__name__)
app = FastAPI(title="NICTO API", version="0.1.0")

_orchestrator: Orchestrator | None = None
_benchmarks = BenchmarkFramework()
_registry = ModelRegistry()


class HandleRequest(BaseModel):
    input: str
    request_id: str | None = None


class HandleResponse(BaseModel):
    request_id: str
    answer: str
    metadata: dict[str, Any]
    latency_ms: float | None = None


class BenchmarkRequest(BaseModel):
    mode: str = "A"
    dataset: list[dict[str, Any]] | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return {"experts": _registry.experts}


@app.post("/handle", response_model=HandleResponse)
async def handle(req: HandleRequest):
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(MoMConfig(llm=LLMConfig(provider="heuristic"), global_deadline=5.0))
    start = time.perf_counter()
    try:
        result = _orchestrator.handle_request(req.input, request_id=req.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency = (time.perf_counter() - start) * 1000
    return HandleResponse(
        request_id=req.request_id or f"api_{int(start)}",
        answer=result.get("answer", ""),
        metadata=result.get("metadata", {}),
        latency_ms=latency,
    )


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    dataset = req.dataset or [{"input": "Calculate 2+2"}]
    result = _benchmarks.run_comparison(req.mode, dataset)
    return result


def create_app() -> FastAPI:
    return app
