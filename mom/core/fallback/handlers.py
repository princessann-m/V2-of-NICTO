"""Fallback handlers for MoM component failures."""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger(__name__)


class FallbackHandlers:
    @staticmethod
    def handle_untrained_model(model_name: str = "model") -> dict[str, Any]:
        LOG.warning("Model %s is untrained, returning informative fallback", model_name)
        return {
            "answer": f"[{model_name} is untrained — no trained weights loaded]",
            "confidence": 0.0,
            "metadata": {
                "fallback": True,
                "reason": "untrained_model",
                "model": model_name,
                "recovery_suggestion": "Run training pipeline to initialize model weights.",
            },
        }

    @staticmethod
    def handle_timeout(request_id: str, best_available: dict[str, Any] | None = None) -> dict[str, Any]:
        LOG.warning("Timeout reached for request %s, returning best available result", request_id)
        if best_available is not None:
            result = dict(best_available)
            result.setdefault("metadata", {})
            result["metadata"].update({
                "fallback": True,
                "reason": "deadline",
                "recovery_suggestion": "Increase global_deadline or reduce task complexity.",
            })
            return result
        return {
            "answer": "[deadline reached — partial result unavailable]",
            "confidence": 0.0,
            "metadata": {
                "fallback": True,
                "reason": "deadline",
                "recovery_suggestion": "Increase global_deadline or reduce task complexity.",
            },
        }

    @staticmethod
    def handle_expert_failure(expert_name: str, error: str, remaining_experts: list[str]) -> dict[str, Any]:
        LOG.warning("Expert %s failed (%s), continuing with remaining: %s", expert_name, error, remaining_experts)
        if remaining_experts:
            return {
                "answer": f"[expert {expert_name} failed: {error}. Proceeding with {', '.join(remaining_experts)}]",
                "confidence": 0.3,
                "metadata": {
                    "fallback": True,
                    "reason": "expert_failure",
                    "failed_expert": expert_name,
                    "error": error,
                    "remaining_experts": remaining_experts,
                    "recovery_suggestion": "Check expert health status and re-run.",
                },
            }
        return {
            "answer": f"[all experts failed — {expert_name}: {error}]",
            "confidence": 0.0,
            "metadata": {
                "fallback": True,
                "reason": "expert_failure",
                "failed_expert": expert_name,
                "error": error,
                "remaining_experts": [],
                "recovery_suggestion": "Verify all expert modules and retry.",
            },
        }

    @staticmethod
    def handle_tool_failure(tool_name: str, error: str, pipeline_continues: bool = True) -> dict[str, Any]:
        LOG.warning("Tool %s failed (%s), skipping", tool_name, error)
        return {
            "answer": f"[tool {tool_name} skipped due to error: {error}]",
            "confidence": 0.2,
            "metadata": {
                "fallback": True,
                "reason": "tool_failure",
                "tool": tool_name,
                "error": error,
                "pipeline_continues": pipeline_continues,
                "recovery_suggestion": f"Verify {tool_name} installation and retry.",
            },
        }

    @staticmethod
    def handle_judge_failure(failed_judge: str, remaining_judge: str | None = None) -> dict[str, Any]:
        LOG.warning("Judge %s failed, falling back to %s", failed_judge, remaining_judge)
        return {
            "fallback": True,
            "reason": "judge_failure",
            "failed_judge": failed_judge,
            "remaining_judge": remaining_judge,
            "recovery_suggestion": "Check judge model health and retry with both judges.",
        }
