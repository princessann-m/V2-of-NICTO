"""Fallback strategies for MoM."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

LOG = logging.getLogger(__name__)


class FallbackStrategies:
    @staticmethod
    def cascade_fallback(
        models: list[tuple[str, Callable]],
        task: dict[str, Any],
        timeout_per_model: float = 2.0,
    ) -> tuple[dict[str, Any] | None, str | None]:
        LOG.info("Cascade fallback: trying %d models", len(models))
        for idx, (name, fn) in enumerate(models):
            try:
                LOG.debug("Cascade: attempting model %d/%d (%s)", idx + 1, len(models), name)
                result = fn(task)
                if result is not None and result.get("answer"):
                    LOG.info("Cascade fallback succeeded with %s", name)
                    return result, name
            except Exception as exc:
                LOG.warning("Cascade: model %s failed: %s", name, exc)
                continue
        LOG.warning("Cascade fallback exhausted all models")
        return None, None

    @staticmethod
    def parallel_fallback(
        models: list[tuple[str, Callable]],
        task: dict[str, Any],
        max_workers: int = 4,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        LOG.info("Parallel fallback: launching %d models", len(models))
        results: list[dict[str, Any]] = []
        used: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fn, task): name for name, fn in models}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    result = fut.result()
                    if result is not None:
                        results.append(result)
                        used.append(name)
                except Exception as exc:
                    LOG.warning("Parallel fallback: model %s failed: %s", name, exc)
        LOG.info("Parallel fallback: %d succeeded out of %d", len(results), len(models))
        return results, used

    @staticmethod
    def cached_fallback(
        cache: dict[str, Any],
        task: dict[str, Any],
        key_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> dict[str, Any] | None:
        if key_fn is None:
            key = str(task.get("original", ""))
        else:
            key = key_fn(task)
        LOG.debug("Looking up cached fallback for key: %s", key)
        entry = cache.get(key)
        if entry is not None:
            LOG.info("Cached fallback hit for key: %s", key)
            result = dict(entry)
            result.setdefault("metadata", {})
            result["metadata"].update({
                "fallback": True,
                "reason": "cache_hit",
                "cache_key": key,
            })
            return result
        LOG.debug("No cached fallback for key: %s", key)
        return None
