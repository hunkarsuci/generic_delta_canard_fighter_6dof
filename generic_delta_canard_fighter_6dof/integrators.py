"""Fixed-step numerical integration methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DerivativeFunction = Callable[[float, np.ndarray], np.ndarray]

# Quaternion slice for post-step normalization: indices [Q0, Q1, Q2, Q3]
_QUAT_SLICE = slice(6, 10)


def _validate_step_size(dt: float) -> None:
    """Raise an error unless the integration step is finite and positive."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Integration step dt must be finite and positive.")


def _normalize_quaternion_in_state(x: np.ndarray) -> np.ndarray:
    """Normalize the quaternion portion of a 13-state vector in place.

    The quaternion is assumed to occupy indices 6–9 (Q0, Q1, Q2, Q3).
    If the state length is not 13, this is a no-op (Euler-state vector).
    """
    if len(x) == 13:
        q = x[_QUAT_SLICE]
        norm = np.linalg.norm(q)
        if norm > 0.0:
            x[_QUAT_SLICE] = q / norm
    return x


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
    result = state + dt * derivative
    return _normalize_quaternion_in_state(result)


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

    result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _normalize_quaternion_in_state(result)
