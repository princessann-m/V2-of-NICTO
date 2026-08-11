"""Demo script showcasing MoM capabilities with graceful degradation."""

from __future__ import annotations

import sys

from mom.core.orchestrator import Orchestrator
from mom.config import MoMConfig, LLMConfig
from mom.tools.calculator import Calculator
from mom.core.errors import MoMError, ModelError, ToolError, DeadlineError, VerificationError, JudgeError, ResourceError
from mom.core.fallback.handlers import FallbackHandlers
from mom.core.fallback.strategies import FallbackStrategies


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_basic_calculation(orch: Orchestrator) -> None:
    print_section("Demo 1: Basic Calculation with Calculator Tool")
    prompt = "Calculate 12 * (3 + 4) and explain briefly."
    try:
        res = orch.handle_request(prompt, request_id="demo1")
        answer = res.get("answer", "(no answer)")
        metadata = res.get("metadata", {})
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Metadata: {metadata}")
        if metadata.get("fallback"):
            print("[INFO] System operated in fallback mode — model may be untrained.")
    except Exception as exc:
        print(f"[ERROR] Request failed: {exc}")


def demo_calculator_direct() -> None:
    print_section("Demo 2: Direct Calculator Tool (No Model Needed)")
    calc = Calculator()
    expressions = [
        "12 * (3 + 4)",
        "2 ** 10",
        "(100 - 37) / 3",
    ]
    for expr in expressions:
        try:
            result = calc.evaluate(expr)
            print(f"  {expr} = {result}")
        except Exception as exc:
            print(f"  [ERROR] {expr}: {exc}")


def demo_untrained_model_message() -> None:
    print_section("Demo 3: Untrained Model Fallback Message")
    fallback = FallbackHandlers.handle_untrained_model("mme_expert")
    print(f"Fallback response: {fallback['answer']}")
    print(f"Recovery suggestion: {fallback['metadata'].get('recovery_suggestion')}")


def demo_error_hierarchy() -> None:
    print_section("Demo 4: MoM Error Hierarchy")
    errors = [
        ModelError("Weights not loaded", model_name="mamba_expert"),
        ToolError("Division by zero", tool_name="calculator"),
        DeadlineError("Budget exceeded", task_id="req_001"),
        VerificationError("Hallucination detected", issues=["unverified_claim"]),
        JudgeError("Scoring failed", judge_name="judge_alpha"),
        ResourceError("Out of memory", resource="gpu_0"),
    ]
    for err in errors:
        print(f"  [{type(err).__name__}] {err}")


def demo_fallback_strategies() -> None:
    print_section("Demo 5: Fallback Strategies")

    def fake_model_a(task):
        return {"answer": "result_a", "confidence": 0.9}

    def fake_model_b(task):
        return {"answer": "result_b", "confidence": 0.6}

    def fake_model_c(task):
        return {"answer": "result_c", "confidence": 0.4}

    models = [
        ("model_a", fake_model_a),
        ("model_b", fake_model_b),
        ("model_c", fake_model_c),
    ]
    task = {"original": "demo task"}

    result, used = FallbackStrategies.cascade_fallback(models, task)
    print(f"  Cascade fallback result: {result['answer']} (via {used})")

    results, used_list = FallbackStrategies.parallel_fallback(models, task)
    print(f"  Parallel fallback results: {[r['answer'] for r in results]} (models: {used_list})")

    cache = {str(task.get("original", "")): {"answer": "cached_result", "confidence": 0.8}}
    cached = FallbackStrategies.cached_fallback(cache, task)
    print(f"  Cached fallback result: {cached['answer']}")


def demo_expert_failure_handling() -> None:
    print_section("Demo 6: Expert and Tool Failure Handling")
    expert_fallback = FallbackHandlers.handle_expert_failure(
        "math_expert",
        "RuntimeError: untrained weights",
        ["reasoning_expert"],
    )
    print(f"Expert failure: {expert_fallback['answer']}")

    tool_fallback = FallbackHandlers.handle_tool_failure(
        "browser",
        "ConnectionError: timeout",
    )
    print(f"Tool failure: {tool_fallback['answer']}")

    judge_fallback = FallbackHandlers.handle_judge_failure("judge_alpha", "judge_beta")
    print(f"Judge failure fallback: {judge_fallback}")


def demo_orchestrator_metrics(orch: Orchestrator) -> None:
    print_section("Demo 7: Orchestrator Metrics")
    m = orch.metrics
    print(f"  Total requests: {m.total_requests}")
    print(f"  Fallbacks triggered: {m.fallbacks_triggered}")
    print(f"  P95 latency: {m.p95_latency_ms:.1f} ms")
    print(f"  Target met: {m.target_met}")


def run() -> None:
    print("=" * 60)
    print("  MoM V2 — Fallback & Error Handling Demo")
    print("=" * 60)

    cfg = MoMConfig(
        llm=LLMConfig(provider="heuristic"),
        global_deadline=120.0,
        max_retries=1,
    )
    orch = Orchestrator(cfg)

    demo_basic_calculation(orch)
    demo_calculator_direct()
    demo_untrained_model_message()
    demo_error_hierarchy()
    demo_fallback_strategies()
    demo_expert_failure_handling()
    demo_orchestrator_metrics(orch)

    print_section("Demo Complete")
    print("System demonstrated graceful degradation across:")
    print("  - Untrained models -> informative fallback messages")
    print("  - Timeouts -> best available results")
    print("  - Expert/tool/judge failures -> continued pipeline execution")
    print("  - Error hierarchy -> structured context and recovery suggestions")
    print()


if __name__ == "__main__":
    run()
