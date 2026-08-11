"""Virtual lab with experiment tracking, hypothesis management, and reproducibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .experiments import Experiment, ExperimentStage, ExperimentRegistry
from .instruments import VirtualInstrument


@dataclass
class LabResult:
    """Container for virtual experiment results."""

    label: str = "VIRTUAL EXPERIMENT RESULT"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instrument_readings: List[Dict[str, Any]] = field(default_factory=list)
    experiment_id: str = ""

    def summary(self) -> str:
        return f"[{self.label}] experiment={self.experiment_id}, readings={len(self.instrument_readings)}"


class VirtualLab:
    """Virtual scientific laboratory with full experiment workflow."""

    def __init__(self):
        self._instruments: Dict[str, VirtualInstrument] = {}
        self._experiments = ExperimentRegistry()
        self._results: List[LabResult] = []
        self._register_default_instruments()

    def _register_default_instruments(self):
        self.register_instrument("thermometer", lambda: __import__("mom.tools.virtual_lab.instruments", fromlist=["Thermometer"]).Thermometer())
        self.register_instrument("voltmeter", lambda: __import__("mom.tools.virtual_lab.instruments", fromlist=["Voltmeter"]).Voltmeter())
        self.register_instrument("stopwatch", lambda: __import__("mom.tools.virtual_lab.instruments", fromlist=["Stopwatch"]).Stopwatch())
        self.register_instrument("spectrometer", lambda: __import__("mom.tools.virtual_lab.instruments", fromlist=["Spectrometer"]).Spectrometer())

    def register_instrument(self, name: str, factory: Callable[[], VirtualInstrument]) -> None:
        self._instruments[name] = factory()

    def get_instrument(self, name: str) -> VirtualInstrument:
        if name not in self._instruments:
            raise KeyError(f"Unknown instrument: {name}")
        return self._instruments[name]

    def create_experiment(self, question: str, hypothesis: str, design: str) -> Experiment:
        experiment = Experiment(
            question=question,
            hypothesis=hypothesis,
            design=design,
            stages=[
                ExperimentStage(name="question", status="complete", data={"question": question}),
                ExperimentStage(name="hypothesis", status="complete", data={"hypothesis": hypothesis}),
                ExperimentStage(name="design", status="complete", data={"design": design}),
                ExperimentStage(name="setup", status="pending"),
                ExperimentStage(name="run", status="pending"),
                ExperimentStage(name="measure", status="pending"),
                ExperimentStage(name="analyze", status="pending"),
                ExperimentStage(name="conclusion", status="pending"),
            ],
        )
        self._experiments.register(experiment)
        return experiment

    def setup(self, experiment_id: str, setup_data: Dict[str, Any]) -> None:
        experiment = self._experiments.get(experiment_id)
        experiment.update_stage("setup", status="complete", data=setup_data)

    def run(self, experiment_id: str, run_fn: Callable[[], Dict[str, Any]]) -> LabResult:
        experiment = self._experiments.get(experiment_id)
        run_data = run_fn()
        instrument_readings = run_data.get("readings", [])
        experiment.update_stage("run", status="complete", data=run_data)
        result = LabResult(
            label="VIRTUAL EXPERIMENT RESULT",
            data=run_data,
            instrument_readings=instrument_readings,
            experiment_id=experiment_id,
            metadata={
                "experiment_question": experiment.question,
                "experiment_hypothesis": experiment.hypothesis,
                "created": datetime.now().isoformat(),
            },
        )
        self._results.append(result)
        return result

    def measure(self, experiment_id: str, instrument_name: str, measurement_fn: Callable[[VirtualInstrument], Dict[str, Any]]) -> Dict[str, Any]:
        instrument = self.get_instrument(instrument_name)
        reading = measurement_fn(instrument)
        reading["instrument"] = instrument_name
        reading["timestamp"] = datetime.now().isoformat()
        experiment = self._experiments.get(experiment_id)
        experiment.add_reading(reading)
        experiment.update_stage("measure", status="complete")
        return reading

    def analyze(self, experiment_id: str, analysis_fn: Callable[[Experiment], Dict[str, Any]]) -> Dict[str, Any]:
        experiment = self._experiments.get(experiment_id)
        analysis = analysis_fn(experiment)
        experiment.update_stage("analyze", status="complete", data=analysis)
        return analysis

    def conclude(self, experiment_id: str, conclusion: str) -> None:
        experiment = self._experiments.get(experiment_id)
        experiment.update_stage("conclusion", status="complete", data={"conclusion": conclusion})

    def reproducibility_report(self, experiment_id: str) -> Dict[str, Any]:
        experiment = self._experiments.get(experiment_id)
        return experiment.reproducibility_payload()

    @property
    def history(self) -> List[LabResult]:
        return list(self._results)
