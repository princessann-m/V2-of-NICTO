"""Tests for simulation tools."""

from __future__ import annotations

import pytest
import numpy as np

from mom.tools.simulator import Simulator
from mom.tools.simulation.engine import SimulationEngine
from mom.tools.simulation.physics import (
    PendulumSimulator,
    ProjectileSimulator,
    SpringSimulator,
    CircuitSimulator,
)
from mom.tools.simulation.domains import PhysicsSolver, MathSolver, EngineeringSolver


class TestSimulator:
    def test_run_basic(self):
        sim = Simulator()
        result = sim.run("model", {"x": 2.0})
        assert result["success"] is True
        assert "measurements" in result
        assert "x" in result["measurements"]

    def test_run_multiple_params(self):
        sim = Simulator()
        result = sim.run("model", {"a": 1.0, "b": 2.0})
        assert len(result["measurements"]) == 2


class TestPhysicsSimulators:
    def test_pendulum(self):
        engine = SimulationEngine()
        result = PendulumSimulator.run(
            engine,
            {
                "length": 1.0,
                "gravity": 9.81,
                "damping": 0.0,
                "theta0": 0.1,
                "omega0": 0.0,
                "t_span": (0.0, 10.0),
            },
        )
        assert result.label == "SIMULATION RESULT"
        assert len(result.time) > 0

    def test_projectile(self):
        engine = SimulationEngine()
        result = ProjectileSimulator.run(
            engine,
            {
                "v0": 10.0,
                "angle_deg": 45.0,
                "g": 9.81,
                "drag_coeff": 0.0,
                "mass": 1.0,
                "t_span": (0.0, 5.0),
            },
        )
        assert result.label == "SIMULATION RESULT"
        assert len(result.time) > 0

    def test_spring(self):
        engine = SimulationEngine()
        result = SpringSimulator.run(
            engine,
            {
                "k": 10.0,
                "m": 1.0,
                "damping": 0.0,
                "x0": 0.5,
                "v0": 0.0,
                "t_span": (0.0, 10.0),
            },
        )
        assert result.label == "SIMULATION RESULT"
        assert len(result.time) > 0

    def test_circuit(self):
        engine = SimulationEngine()
        result = CircuitSimulator.run(
            engine,
            {
                "R": 100.0,
                "L": 0.1,
                "C": 1e-6,
                "V_source": 5.0,
                "i0": 0.0,
                "v0": 0.0,
                "t_span": (0.0, 0.1),
            },
        )
        assert result.label == "SIMULATION RESULT"
        assert len(result.time) > 0


class TestDomains:
    def test_physics_solver_ode(self):
        engine = SimulationEngine()
        def fun(t, y):
            return np.array([y[1], -y[0]])
        result = PhysicsSolver.solve_ode(engine, fun, (0.0, 1.0), np.array([1.0, 0.0]))
        assert result.label == "SIMULATION RESULT"

    def test_math_solver_integral(self):
        result = MathSolver.definite_integral(np.sin, 0.0, np.pi)
        assert "value" in result
        assert abs(result["value"] - 2.0) < 1e-3

    def test_math_solver_root(self):
        result = MathSolver.root_find(lambda x: x**2 - 2, 1.0)
        assert "root" in result
        assert abs(result["root"] - np.sqrt(2)) < 1e-3

    def test_engineering_solver(self):
        x = np.linspace(0, 1, 10)
        result = EngineeringSolver.beam_deflection(L=1.0, I=1e-6, E=200e9, w=1000.0, x=x)
        assert "deflection" in result
