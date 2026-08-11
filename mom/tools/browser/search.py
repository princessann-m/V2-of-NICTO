"""Search engine with result ranking and summarization."""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any


class SearchEngine:
    def query(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MoM-Search/0.1"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        snippets: list[str] = []
        parts = html.split("<a class=\"result__a\"")
        for part in parts[1:]:
            href = part.split("href=\"", 1)[-1].split("\"", 1)[0]
            title = part.split(">", 1)[-1].split("<", 1)[0]
            snippet_block = part.split("<a class=\"result__snippet\"", 1)[-1]
            snippet = snippet_block.split(">", 1)[-1].split("<", 1)[0] if "<a class=\"result__snippet\"" in part else ""
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
                snippets.append(snippet)
            if len(results) >= max_results:
                break

        if not results:
            return [{"title": query, "url": url, "snippet": "No results parsed"}]

        summary = " ".join(snippets)[:500]
        for r in results:
            r["summary"] = summary
        return results
