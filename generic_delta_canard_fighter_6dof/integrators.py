"""Fixed-step numerical integration methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DerivativeFunction = Callable[[float, np.ndarray], np.ndarray]
PostStepHook = Callable[[np.ndarray], np.ndarray]


def _validate_step_size(dt: float) -> None:
    """Raise an error unless the integration step is finite and positive."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Integration step dt must be finite and positive.")


def euler_step(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
    *,
    post_step: PostStepHook | None = None,
) -> np.ndarray:
    """Advance ``x`` by one explicit Euler step.

    Parameters
    ----------
    derivative_function:
        Function returning dx/dt given t and x.
    t:
        Current time [s].
    x:
        Current state vector.
    dt:
        Integration timestep [s]. Must be finite and positive.
    post_step:
        Optional callback applied to the state after the step (e.g. normalization).

    Returns
    -------
    np.ndarray
        Next state vector x(t + dt).
    """
    _validate_step_size(dt)
    state = np.asarray(x, dtype=float)
    derivative = np.asarray(derivative_function(t, state), dtype=float)
    result = state + dt * derivative
    if post_step is not None:
        result = post_step(result)
    return result


def rk4_step(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
    *,
    post_step: PostStepHook | None = None,
) -> np.ndarray:
    """Advance ``x`` by one classical fourth-order Runge--Kutta step.

    Parameters
    ----------
    derivative_function:
        Function returning dx/dt given t and x.
    t:
        Current time [s].
    x:
        Current state vector.
    dt:
        Integration timestep [s]. Must be finite and positive.
    post_step:
        Optional callback applied to the state after the step (e.g. normalization).

    Returns
    -------
    np.ndarray
        Next state vector x(t + dt).
    """
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
    if post_step is not None:
        result = post_step(result)
    return result


def euler_step_quat(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance a 13-state aircraft quaternion vector with Euler and quaternion normalization."""
    from generic_delta_canard_fighter_6dof.quat_state import normalize_quat_state

    return euler_step(derivative_function, t, x, dt, post_step=normalize_quat_state)


def rk4_step_quat(
    derivative_function: DerivativeFunction,
    t: float,
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance a 13-state aircraft quaternion vector with RK4 and quaternion normalization."""
    from generic_delta_canard_fighter_6dof.quat_state import normalize_quat_state

    return rk4_step(derivative_function, t, x, dt, post_step=normalize_quat_state)
