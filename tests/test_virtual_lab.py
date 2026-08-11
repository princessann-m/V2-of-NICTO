"""Tests for virtual lab components."""

from __future__ import annotations

import pytest

from mom.tools.virtual_lab.lab import VirtualLab
from mom.tools.virtual_lab.instruments import (
    VirtualInstrument,
    Thermometer,
    Voltmeter,
    Stopwatch,
    Spectrometer,
)
from mom.tools.virtual_lab.experiments import Experiment, ExperimentStage, ExperimentRegistry


class TestInstruments:
    def test_thermometer(self):
        t = Thermometer()
        reading = t.measure(initial_temp=100.0, time=1.0, ambient=25.0)
        assert reading["type"] == "TEMPERATURE"
        assert "value" in reading

    def test_voltmeter(self):
        v = Voltmeter()
        reading = v.measure(resistance=100.0, current=0.05)
        assert reading["type"] == "VOLTAGE"
        assert reading["value"] == 5.0

    def test_stopwatch(self):
        s = Stopwatch()
        reading = s.measure(distance=100.0, speed=10.0)
        assert reading["type"] == "TIME"
        assert reading["value"] == 10.0

    def test_spectrometer(self):
        s = Spectrometer()
        reading = s.measure(wavelength_range=(400.0, 700.0), peaks=2)
        assert reading["type"] == "SPECTRUM"
        assert "wavelengths" in reading
        assert "intensity" in reading


class TestVirtualLab:
    def test_lab_setup(self):
        lab = VirtualLab()
        assert lab is not None

    def test_create_experiment(self):
        lab = VirtualLab()
        exp = lab.create_experiment(
            question="Does temperature affect reaction rate?",
            hypothesis="Higher temp increases rate",
            design="Measure at three temperatures",
        )
        assert exp.question == "Does temperature affect reaction rate?"

    def test_measure_instrument(self):
        lab = VirtualLab()
        exp = lab.create_experiment("q", "h", "d")
        lab.setup(exp.experiment_id, {})
        reading = lab.measure(
            exp.experiment_id,
            "thermometer",
            lambda inst: inst.measure(initial_temp=50.0, time=2.0, ambient=20.0),
        )
        assert reading["instrument"] == "thermometer"

    def test_reproducibility_report(self):
        lab = VirtualLab()
        exp = lab.create_experiment("q", "h", "d")
        report = lab.reproducibility_report(exp.experiment_id)
        assert report["experiment_id"] == exp.experiment_id


class TestExperimentRegistry:
    def test_register_and_get(self):
        registry = ExperimentRegistry()
        exp = Experiment(question="test", hypothesis="h", design="d")
        registry.register(exp)
        found = registry.get(exp.experiment_id)
        assert found is exp

    def test_unknown_experiment(self):
        registry = ExperimentRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")
