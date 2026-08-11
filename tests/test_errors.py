"""Tests for MoM error hierarchy."""

from __future__ import annotations

import pytest

from mom.core.errors import (
    MoMError,
    ModelError,
    ToolError,
    VerificationError,
    JudgeError,
    DeadlineError,
    ResourceError,
)


class TestMoMError:
    def test_base_error_message(self):
        err = MoMError("something failed")
        assert "something failed" in str(err)

    def test_base_error_with_context(self):
        err = MoMError("failed", context={"key": "value"})
        assert "context=" in str(err)

    def test_base_error_with_recovery(self):
        err = MoMError("failed", recovery="try again")
        assert "recovery=" in str(err)

    def test_is_exception(self):
        with pytest.raises(MoMError):
            raise MoMError("boom")


class TestModelError:
    def test_default_message(self):
        err = ModelError()
        assert "Model error" in str(err)

    def test_model_name_in_context(self):
        err = ModelError(model_name="mamba_expert")
        assert err.context["model_name"] == "mamba_expert"

    def test_recovery_suggestion(self):
        err = ModelError()
        assert "weights" in err.recovery.lower()

    def test_custom_message(self):
        err = ModelError("weights missing")
        assert "weights missing" in str(err)


class TestToolError:
    def test_tool_name_in_context(self):
        err = ToolError(tool_name="calculator")
        assert err.context["tool_name"] == "calculator"

    def test_recovery_suggestion(self):
        err = ToolError()
        assert "tool" in err.recovery.lower()


class TestVerificationError:
    def test_issues_in_context(self):
        issues = ["empty_answer", "bad_expr"]
        err = VerificationError(issues=issues)
        assert err.context["issues"] == issues

    def test_recovery_suggestion(self):
        err = VerificationError()
        assert "candidate" in err.recovery.lower()


class TestJudgeError:
    def test_judge_name_in_context(self):
        err = JudgeError(judge_name="judge_alpha")
        assert err.context["judge_name"] == "judge_alpha"

    def test_recovery_suggestion(self):
        err = JudgeError()
        assert "judge" in err.recovery.lower()


class TestDeadlineError:
    def test_task_id_in_context(self):
        err = DeadlineError(task_id="req_123")
        assert err.context["task_id"] == "req_123"

    def test_recovery_suggestion(self):
        err = DeadlineError()
        assert "deadline" in err.recovery.lower()


class TestResourceError:
    def test_resource_in_context(self):
        err = ResourceError(resource="gpu_0")
        assert err.context["resource"] == "gpu_0"

    def test_recovery_suggestion(self):
        err = ResourceError()
        assert "resource" in err.recovery.lower()


class TestErrorHierarchy:
    def test_all_are_mom_error_subclasses(self):
        assert issubclass(ModelError, MoMError)
        assert issubclass(ToolError, MoMError)
        assert issubclass(VerificationError, MoMError)
        assert issubclass(JudgeError, MoMError)
        assert issubclass(DeadlineError, MoMError)
        assert issubclass(ResourceError, MoMError)

    def test_catch_as_base(self):
        with pytest.raises(MoMError):
            raise ModelError("boom")

        with pytest.raises(MoMError):
            raise ToolError("boom")

        with pytest.raises(MoMError):
            raise DeadlineError("boom")
