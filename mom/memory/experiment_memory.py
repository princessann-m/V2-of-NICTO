"""Experiment memory for reproducibility and cross-experiment analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    experiment_id: str
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    tags: list[str] = field(default_factory=list)


class ExperimentMemory:
    def __init__(self, storage_path: str | Path = "logs/experiments") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, ExperimentRecord] = {}

    def start(self, experiment_id: str, config: dict[str, Any]) -> ExperimentRecord:
        rec = ExperimentRecord(experiment_id=experiment_id, config=config)
        self.records[experiment_id] = rec
        self._save(rec)
        return rec

    def complete(self, experiment_id: str, metrics: dict[str, Any], artifacts: list[str] | None = None) -> ExperimentRecord:
        rec = self.records.get(experiment_id)
        if rec is None:
            raise KeyError(f"Unknown experiment {experiment_id}")
        rec.metrics = metrics
        rec.artifacts = artifacts or []
        rec.completed_at = time.time()
        self._save(rec)
        return rec

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self.records.get(experiment_id)

    def list_by_tag(self, tag: str) -> list[ExperimentRecord]:
        return [r for r in self.records.values() if tag in r.tags]

    def compare(self, experiment_ids: list[str]) -> dict[str, Any]:
        return {
            eid: {
                "config": self.records[eid].config,
                "metrics": self.records[eid].metrics,
            }
            for eid in experiment_ids
            if eid in self.records
        }

    def _save(self, rec: ExperimentRecord) -> None:
        path = self.storage_path / f"{rec.experiment_id}.json"
        data = {
            "experiment_id": rec.experiment_id,
            "config": rec.config,
            "metrics": rec.metrics,
            "artifacts": rec.artifacts,
            "started_at": rec.started_at,
            "completed_at": rec.completed_at,
            "tags": rec.tags,
        }
        path.write_text(json.dumps(data))
