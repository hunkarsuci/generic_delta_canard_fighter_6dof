"""Fixed-step numerical integration methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


DerivativeFunction = Callable[[float, np.ndarray], np.ndarray]


def _validate_step_size(dt: float) -> None:
    """Raise an error unless the integration step is finite and positive."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Integration step dt must be finite and positive.")


def euler_step(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance ``x`` by one explicit Euler step."""
    _validate_step_size(dt)
    state = np.asarray(x, dtype=float)
    derivative = np.asarray(derivative_function(t, state), dtype=float)
    return state + dt * derivative


def rk4_step(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance ``x`` by one classical fourth-order Runge--Kutta step."""
    _validate_step_size(dt)
    state = np.asarray(x, dtype=float)

    k1 = np.asarray(derivative_function(t, state), dtype=float)
    k2 = np.asarray(
        derivative_function(t + 0.5 * dt, state + 0.5 * dt * k1),
        dtype=float,
    )
    k3 = np.asarray(
        derivative_function(t + 0.5 * dt, state + 0.5 * dt * k2),
        dtype=float,
    )
    k4 = np.asarray(derivative_function(t + dt, state + dt * k3), dtype=float)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
