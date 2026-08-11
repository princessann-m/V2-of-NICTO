"""Domain-specific solvers for physics, math, and engineering."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve, minimize_scalar
from scipy.linalg import expm

from .engine import SimulationEngine, SimulationResult


class PhysicsSolver:
    """Physics-domain helper solvers."""

    @staticmethod
    def solve_ode(engine: SimulationEngine, fun, t_span, y0, solver: str = "rk4", t_eval=None) -> SimulationResult:
        return engine.run("physics_ode", fun, t_span, y0, solver=solver, t_eval=t_eval, label="SIMULATION RESULT")

    @staticmethod
    def orbital_decay(r0: float, t_span: tuple[float, float], mu: float = 398600.4418, Cd: float = 2.2, A: float = 1.0, m: float = 100.0) -> Dict[str, Any]:
        def drag(t, y):
            r, v = y
            rho = 1.0 * np.exp(-(r - 6371.0) / 10.0)
            a_drag = -0.5 * Cd * A / m * rho * v * abs(v)
            return np.array([v, -mu / r**2 + a_drag])

        sol = solve_ivp(drag, t_span, [r0, 0.0], method="RK45", dense_output=True, max_step=10.0)
        return {
            "time": sol.t,
            "radius": sol.y[0],
            "velocity": sol.y[1],
            "label": "SIMULATION RESULT",
        }


class MathSolver:
    """Mathematics-domain solvers."""

    @staticmethod
    def definite_integral(fun, a: float, b: float) -> Dict[str, Any]:
        val, err = quad(fun, a, b)
        return {"value": float(val), "error": float(err), "label": "SIMULATION RESULT"}

    @staticmethod
    def root_find(fun, x0: float) -> Dict[str, Any]:
        root = fsolve(fun, x0)[0]
        return {"root": float(root), "label": "SIMULATION RESULT"}

    @staticmethod
    def minimize(fun, bounds) -> Dict[str, Any]:
        res = minimize_scalar(fun, bounds=bounds, method="bounded")
        return {"x": float(res.x), "fun": float(res.fun), "label": "SIMULATION RESULT"}

    @staticmethod
    def matrix_exponential(A: np.ndarray, t: float) -> Dict[str, Any]:
        expA = expm(A * t)
        return {"expA": expA, "label": "SIMULATION RESULT"}

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        x = np.linalg.solve(A, b)
        return {"x": x, "label": "SIMULATION RESULT"}


class EngineeringSolver:
    """Engineering-domain solvers."""

    @staticmethod
    def beam_deflection(L: float, I: float, E: float, w: float, x: np.ndarray) -> Dict[str, Any]:
        deflection = (w * x * (L**3 - 2 * L * x**2 + x**3)) / (24 * E * I)
        return {"x": x, "deflection": deflection, "label": "SIMULATION RESULT"}

    @staticmethod
    def heat_transfer(k: float, A: float, L: float, T_hot: float, T_cold: float, t: np.ndarray) -> Dict[str, Any]:
        h = k * A / L
        T = T_cold + (T_hot - T_cold) * np.exp(-h * t / (1000 * 385.0))
        return {"time": t, "temperature": T, "label": "SIMULATION RESULT"}

    @staticmethod
    def stress_analysis(sigma: np.ndarray, E: float, nu: float) -> Dict[str, Any]:
        eps = np.array([sigma[0] / E - nu * sigma[1] / E, sigma[1] / E - nu * sigma[0] / E])
        return {"strain": eps, "label": "SIMULATION RESULT"}
