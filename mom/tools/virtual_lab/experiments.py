"""Experiment framework with full scientific workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ExperimentStage:
    """Represents a single stage in the experiment workflow."""

    name: str
    status: str = "pending"
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def complete(self, data: Dict[str, Any]) -> None:
        self.status = "complete"
        self.data.update(data)
        self.completed_at = datetime.now().isoformat()


@dataclass
class Experiment:
    """Scientific experiment with full workflow tracking."""

    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    question: str = ""
    hypothesis: str = ""
    design: str = ""
    stages: List[ExperimentStage] = field(default_factory=list)
    readings: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def update_stage(self, name: str, status: str = "complete", data: Optional[Dict[str, Any]] = None) -> None:
        for stage in self.stages:
            if stage.name == name:
                stage.status = status
                if data:
                    stage.data.update(data)
                if status == "complete":
                    stage.completed_at = datetime.now().isoformat()
                return
        raise ValueError(f"Stage not found: {name}")

    def add_reading(self, reading: Dict[str, Any]) -> None:
        self.readings.append(reading)

    def workflow(self) -> List[str]:
        expected = ["question", "hypothesis", "design", "setup", "run", "measure", "analyze", "conclusion"]
        return expected

    def reproducibility_payload(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "design": self.design,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "created_at": s.created_at,
                    "completed_at": s.completed_at,
                }
                for s in self.stages
            ],
            "readings_count": len(self.readings),
            "created_at": self.created_at,
        }


class ExperimentRegistry:
    """Registry for experiments with lookup and reproducibility support."""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}

    def register(self, experiment: Experiment) -> None:
        self._experiments[experiment.experiment_id] = experiment

    def get(self, experiment_id: str) -> Experiment:
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]

    def list_experiments(self) -> List[Dict[str, Any]]:
        return [{"id": e.experiment_id, "question": e.question} for e in self._experiments.values()]
