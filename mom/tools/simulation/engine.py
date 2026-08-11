"""Core simulation engine with modular backends and numerical solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SimulationResult:
    """Container for simulation results with metadata."""

    label: str = "SIMULATION RESULT"
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    state: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    solver: str = ""
    duration_seconds: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.label}] solver={self.solver}, "
            f"steps={len(self.time)}, "
            f"t=[{self.time.min():.4f}, {self.time.max():.4f}], "
            f"duration={self.duration_seconds:.4f}s, "
            f"params={self.parameters}"
        )


class SolverInterface:
    """Base numerical solver interface."""

    name: str = "base"

    def solve(self, fun, t_span, y0, t_eval=None, **kwargs):
        raise NotImplementedError


class EulerSolver(SolverInterface):
    """Explicit Euler method."""

    name = "euler"

    def solve(self, fun, t_span, y0, t_eval=None, **kwargs):
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.linspace(t0, tf, 1000)
        dt = t_eval[1] - t_eval[0]
        n_steps = len(t_eval)
        y = np.zeros((n_steps, len(y0)))
        y[0] = y0
        t = t_eval[0]
        y_curr = y0.copy()
        for i in range(1, n_steps):
            dy = fun(t, y_curr)
            y_curr = y_curr + dt * np.asarray(dy)
            y[i] = y_curr
            t = t_eval[i]
        return t_eval, y


class RK4Solver(SolverInterface):
    """Classical Runge-Kutta (RK4) method."""

    name = "rk4"

    def solve(self, fun, t_span, y0, t_eval=None, **kwargs):
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.linspace(t0, tf, 1000)
        dt = t_eval[1] - t_eval[0]
        n_steps = len(t_eval)
        y = np.zeros((n_steps, len(y0)))
        y[0] = y0
        t = t_eval[0]
        y_curr = y0.copy()
        for i in range(1, n_steps):
            k1 = fun(t, y_curr)
            k2 = fun(t + dt / 2, y_curr + dt / 2 * np.asarray(k1))
            k3 = fun(t + dt / 2, y_curr + dt / 2 * np.asarray(k2))
            k4 = fun(t + dt, y_curr + dt * np.asarray(k3))
            y_curr = y_curr + (dt / 6) * (
                np.asarray(k1) + 2 * np.asarray(k2) + 2 * np.asarray(k3) + np.asarray(k4)
            )
            y[i] = y_curr
            t = t_eval[i]
        return t_eval, y


class ScipyIVPSolver(SolverInterface):
    """Wrapper around scipy.integrate.solve_ivp for adaptive methods."""

    name = "scipy_ivp"

    def solve(self, fun, t_span, y0, t_eval=None, method="RK45", **kwargs):
        sol = solve_ivp(
            fun,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            dense_output=False,
            **kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.t, sol.y.T


class SimulationEngine:
    """Modular simulation engine supporting multiple backends and domains."""

    def __init__(self):
        self._backends: Dict[str, SolverInterface] = {
            "euler": EulerSolver(),
            "rk4": RK4Solver(),
            "scipy_ivp": ScipyIVPSolver(),
        }
        self._results: List[SimulationResult] = []
        self._registry: Dict[str, Callable] = {}

    def register_backend(self, name: str, backend: SolverInterface) -> None:
        self._backends[name] = backend

    def register_simulator(self, name: str, simulator: Callable) -> None:
        self._registry[name] = simulator

    def validate_parameters(self, params: Dict[str, Any], schema: Dict[str, type]) -> None:
        for key, expected_type in schema.items():
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
            if not isinstance(params[key], expected_type):
                raise TypeError(
                    f"Parameter {key} must be {expected_type.__name__}, got {type(params[key]).__name__}"
                )

    def run(
        self,
        name: str,
        state_function: Callable[[float, np.ndarray], np.ndarray],
        t_span: tuple[float, float],
        y0: np.ndarray,
        solver: str = "rk4",
        t_eval: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
        label: str = "SIMULATION RESULT",
    ) -> SimulationResult:
        if solver not in self._backends:
            raise ValueError(f"Unknown solver: {solver}")
        if parameters is None:
            parameters = {}
        start = datetime.now()
        backend = self._backends[solver]
        t, y = backend.solve(state_function, t_span, y0, t_eval=t_eval)
        duration = (datetime.now() - start).total_seconds()
        result = SimulationResult(
            label=label,
            time=t,
            state=y,
            solver=solver,
            duration_seconds=duration,
            parameters=parameters,
            metadata={"name": name, "created": datetime.now().isoformat()},
        )
        self._results.append(result)
        return result

    def run_registered(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        solver: str = "rk4",
    ) -> SimulationResult:
        if name not in self._registry:
            raise ValueError(f"Unknown simulator: {name}")
        return self._registry[name](self, parameters or {}, solver)

    @property
    def history(self) -> List[SimulationResult]:
        return list(self._results)
