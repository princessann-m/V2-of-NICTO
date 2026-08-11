"""Web browser automation and page extraction."""

from __future__ import annotations

import time
from typing import Any

from .search import SearchEngine


class BrowserTool:
    def __init__(self, headless: bool = True, timeout: float = 30.0) -> None:
        self.headless = headless
        self.timeout = timeout
        self.search = SearchEngine()
        self._page = None

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return self.search.query(query, max_results=max_results)

    def open(self, url: str) -> dict[str, Any]:
        start = time.time()
        try:
            content = self._fetch(url)
            return {
                "url": url,
                "content": content,
                "status": "success",
                "latency_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            return {
                "url": url,
                "content": "",
                "status": "error",
                "error": str(e),
                "latency_ms": int((time.time() - start) * 1000),
            }

    def screenshot(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "screenshot": None,
            "status": "placeholder",
            "message": "Screenshot requires Playwright installation",
        }

    def close(self) -> None:
        self._page = None

    def _fetch(self, url: str) -> str:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MoM-Browser/0.1"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return data[:200_000]
