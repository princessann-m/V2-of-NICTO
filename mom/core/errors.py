"""MoM error hierarchy with context and recovery suggestions."""

from __future__ import annotations

from typing import Any


class MoMError(Exception):
    """Base exception for MoM framework errors."""

    def __init__(self, message: str = "", context: dict[str, Any] | None = None, recovery: str | None = None):
        super().__init__(message)
        self.context = context or {}
        self.recovery = recovery

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.context:
            parts.append(f"context={self.context}")
        if self.recovery:
            parts.append(f"recovery={self.recovery}")
        return " | ".join(parts)


class ModelError(MoMError):
    """Raised when a model component fails."""

    def __init__(self, message: str = "Model error", model_name: str = "", **kwargs):
        super().__init__(
            message,
            context={"model_name": model_name, **kwargs},
            recovery="Check model weights and configuration.",
        )


class ToolError(MoMError):
    """Raised when a tool invocation fails."""

    def __init__(self, message: str = "Tool error", tool_name: str = "", **kwargs):
        super().__init__(
            message,
            context={"tool_name": tool_name, **kwargs},
            recovery="Verify tool installation and inputs.",
        )


class VerificationError(MoMError):
    """Raised when verification fails."""

    def __init__(self, message: str = "Verification failed", issues: list[str] | None = None, **kwargs):
        super().__init__(
            message,
            context={"issues": issues or [], **kwargs},
            recovery="Review candidate answers for accuracy.",
        )


class JudgeError(MoMError):
    """Raised when the judge component fails."""

    def __init__(self, message: str = "Judge error", judge_name: str = "", **kwargs):
        super().__init__(
            message,
            context={"judge_name": judge_name, **kwargs},
            recovery="Check judge model health and retry.",
        )


class DeadlineError(MoMError):
    """Raised when a deadline is missed."""

    def __init__(self, message: str = "Deadline missed", task_id: str = "", **kwargs):
        super().__init__(
            message,
            context={"task_id": task_id, **kwargs},
            recovery="Increase deadline or reduce task complexity.",
        )


class ResourceError(MoMError):
    """Raised when system resources are insufficient."""

    def __init__(self, message: str = "Resource error", resource: str = "", **kwargs):
        super().__init__(
            message,
            context={"resource": resource, **kwargs},
            recovery="Free resources or adjust configuration.",
        )
