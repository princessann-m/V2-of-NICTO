"""Physics simulators with real differential equation solving."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .engine import SimulationEngine, SimulationResult


def _pendulum_simulator(engine: SimulationEngine, params: Dict[str, Any], solver: str) -> SimulationResult:
    schema = {
        "length": float,
        "gravity": float,
        "damping": float,
        "theta0": float,
        "omega0": float,
        "t_span": tuple,
    }
    engine.validate_parameters(params, schema)
    L = params["length"]
    g = params["gravity"]
    b = params["damping"]
    theta0 = params["theta0"]
    omega0 = params["omega0"]
    t_span = params["t_span"]

    def deriv(t, y):
        theta, omega = y
        return np.array([omega, -(g / L) * np.sin(theta) - (b / L) * omega])

    y0 = np.array([theta0, omega0])
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = engine.run(
        name="pendulum",
        state_function=deriv,
        t_span=t_span,
        y0=y0,
        solver=solver,
        t_eval=t_eval,
        parameters=params,
        label="SIMULATION RESULT",
    )
    return result


def _projectile_simulator(engine: SimulationEngine, params: Dict[str, Any], solver: str) -> SimulationResult:
    schema = {
        "v0": float,
        "angle_deg": float,
        "g": float,
        "drag_coeff": float,
        "mass": float,
        "t_span": tuple,
    }
    engine.validate_parameters(params, schema)
    v0 = params["v0"]
    angle = np.deg2rad(params["angle_deg"])
    g = params["g"]
    c = params["drag_coeff"]
    m = params["mass"]
    t_span = params["t_span"]

    def deriv(t, y):
        x, y_pos, vx, vy = y
        v = np.sqrt(vx**2 + vy**2)
        ax = -(c / m) * v * vx
        ay = -g - (c / m) * v * vy
        return np.array([vx, vy, ax, ay])

    y0 = np.array([0.0, 0.0, v0 * np.cos(angle), v0 * np.sin(angle)])
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = engine.run(
        name="projectile",
        state_function=deriv,
        t_span=t_span,
        y0=y0,
        solver=solver,
        t_eval=t_eval,
        parameters=params,
        label="SIMULATION RESULT",
    )
    return result


def _spring_simulator(engine: SimulationEngine, params: Dict[str, Any], solver: str) -> SimulationResult:
    schema = {
        "k": float,
        "m": float,
        "damping": float,
        "x0": float,
        "v0": float,
        "t_span": tuple,
    }
    engine.validate_parameters(params, schema)
    k = params["k"]
    m = params["m"]
    c = params["damping"]
    x0 = params["x0"]
    v0 = params["v0"]
    t_span = params["t_span"]

    def deriv(t, y):
        x, v = y
        return np.array([v, -(k / m) * x - (c / m) * v])

    y0 = np.array([x0, v0])
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = engine.run(
        name="spring",
        state_function=deriv,
        t_span=t_span,
        y0=y0,
        solver=solver,
        t_eval=t_eval,
        parameters=params,
        label="SIMULATION RESULT",
    )
    return result


def _circuit_simulator(engine: SimulationEngine, params: Dict[str, Any], solver: str) -> SimulationResult:
    schema = {
        "R": float,
        "L": float,
        "C": float,
        "V_source": float,
        "i0": float,
        "v0": float,
        "t_span": tuple,
    }
    engine.validate_parameters(params, schema)
    R = params["R"]
    L = params["L"]
    C = params["C"]
    V = params["V_source"]
    i0 = params["i0"]
    v0 = params["v0"]
    t_span = params["t_span"]

    def deriv(t, y):
        i, v_c = y
        di = (V - R * i - v_c) / L
        dv = (i - v_c / (1e12 if C == 0 else C)) / (1e-12 if C == 0 else C)
        dv = i / C
        return np.array([di, dv])

    y0 = np.array([i0, v0])
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = engine.run(
        name="circuit",
        state_function=deriv,
        t_span=t_span,
        y0=y0,
        solver=solver,
        t_eval=t_eval,
        parameters=params,
        label="SIMULATION RESULT",
    )
    return result


class PendulumSimulator:
    """Simple pendulum with damping."""

    @staticmethod
    def run(engine: SimulationEngine, params: Dict[str, Any], solver: str = "rk4") -> SimulationResult:
        return _pendulum_simulator(engine, params, solver)


class ProjectileSimulator:
    """Projectile motion with quadratic drag."""

    @staticmethod
    def run(engine: SimulationEngine, params: Dict[str, Any], solver: str = "rk4") -> SimulationResult:
        return _projectile_simulator(engine, params, solver)


class SpringSimulator:
    """Damped harmonic oscillator."""

    @staticmethod
    def run(engine: SimulationEngine, params: Dict[str, Any], solver: str = "rk4") -> SimulationResult:
        return _spring_simulator(engine, params, solver)


class CircuitSimulator:
    """RLC circuit dynamics."""

    @staticmethod
    def run(engine: SimulationEngine, params: Dict[str, Any], solver: str = "rk4") -> SimulationResult:
        return _circuit_simulator(engine, params, solver)
