"""Tests for browser tools."""

from __future__ import annotations

import numpy as np

import pytest

from mom.tools.browser.browser import BrowserTool
from mom.tools.browser.search import SearchEngine


def test_search_returns_list():
    engine = SearchEngine()
    results = engine.query("test")
    assert isinstance(results, list)


def test_browser_open_returns_status():
    browser = BrowserTool(headless=True, timeout=5.0)
    result = browser.open("https://example.com")
    assert "status" in result
    assert result["status"] in ("success", "error")
    assert "latency_ms" in result


def test_browser_screenshot_placeholder():
    browser = BrowserTool()
    result = browser.screenshot("https://example.com")
    assert result["status"] == "placeholder"


def test_browser_close():
    browser = BrowserTool()
    browser.close()
    assert browser._page is None
