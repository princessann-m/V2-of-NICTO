"""Tests for safety layers."""

from __future__ import annotations

import pytest

from mom.safety.filter import SafetyFilter
from mom.safety.policy import SafetyPolicy


def test_policy_is_allowed_clean():
    policy = SafetyPolicy()
    assert policy.is_allowed("hello world") is True


def test_policy_is_allowed_blocked():
    policy = SafetyPolicy(blocked_categories=["violence"])
    assert policy.is_allowed("this contains violence") is False


def test_policy_severity_levels():
    policy = SafetyPolicy()
    assert policy.severity("clean text") == "none"
    assert policy.severity("violence") == "low"


def test_filter_input_safe():
    filt = SafetyFilter()
    result = filt.filter_input("safe content")
    assert result["blocked"] is False
    assert result["content"] == "safe content"


def test_filter_input_blocked():
    filt = SafetyFilter(policy=SafetyPolicy(blocked_categories=["violence"]))
    result = filt.filter_input("violence here")
    assert result["blocked"] is True
    assert result["content"] == "[FILTERED]"


def test_detect_pii_email():
    filt = SafetyFilter()
    findings = filt.detect_pii("contact me at test@example.com")
    assert "email" in findings


def test_detect_pii_phone():
    filt = SafetyFilter()
    findings = filt.detect_pii("call 555-123-4567")
    assert "phone" in findings


def test_enforce_input():
    filt = SafetyFilter()
    result = filt.enforce("hello", context="input")
    assert result["blocked"] is False


def test_enforce_output():
    filt = SafetyFilter()
    result = filt.enforce("hello", context="output")
    assert result["blocked"] is False


def test_audit_log():
    filt = SafetyFilter()
    filt.filter_input("test")
    assert len(filt.policy.audit_log) == 1
    assert filt.policy.audit_log[0]["type"] == "input"
