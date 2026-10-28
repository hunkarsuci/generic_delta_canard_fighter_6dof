"""
Deterministic simulation runner for the 6DOF aircraft model.

This module provides a minimal simulation loop with fixed-step integration.
It does NOT perform file I/O, plotting, or configuration parsing — those
belong in the evaluation entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.integrators import euler_step, rk4_step

IntegratorFn = Callable[
    [Callable[[float, np.ndarray], np.ndarray], float, np.ndarray, float],
    np.ndarray,
]


def _validate_state_vector(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError("State must be a numpy array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("State contains non-finite values.")


@dataclass(frozen=True)
class SimulationResult:
    """Result of a completed simulation run."""

    t: np.ndarray
    """Time history [s], shape (n_steps + 1,)."""

    x: np.ndarray
    """State history, shape (n_steps + 1, n_states)."""

    dt: float
    """Integration step size [s]."""

    t_final: float
    """Final simulation time [s]."""

    integrator_name: str
    """Name of the integrator used."""


def simulate(
    derivative_fn: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    dt: float,
    t_final: float,
    *,
    integrator: str = "rk4",
    post_step: Callable[[np.ndarray], np.ndarray] | None = None,
) -> SimulationResult:
    """Run a fixed-step simulation from t=0 to t_final.

    Parameters
    ----------
    derivative_fn:
        Function f(t, x) returning the state derivative x_dot.
    x0:
        Initial state vector.
    dt:
        Fixed integration step [s]. Must be finite and positive.
    t_final:
        Final simulation time [s]. Must be >= 0.
    integrator:
        Integration method: ``"rk4"``, ``"euler"``, ``"rk4_quat"``, or ``"euler_quat"``.
    post_step:
        Optional callback applied to the state after each step (e.g. normalization).

    Returns
    -------
    SimulationResult
        Time and state history.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(t_final) or t_final < 0.0:
        raise ValueError("t_final must be finite and non-negative.")

    x0 = np.asarray(x0, dtype=float)
    _validate_state_vector(x0)

    if integrator == "rk4":
        step_fn = (
            rk4_step
            if post_step is None
            else (lambda fn, t, x, s_dt: rk4_step(fn, t, x, s_dt, post_step=post_step))
        )
    elif integrator == "euler":
        step_fn = (
            euler_step
            if post_step is None
            else (
                lambda fn, t, x, s_dt: euler_step(fn, t, x, s_dt, post_step=post_step)
            )
        )
    elif integrator == "rk4_quat":
        from generic_delta_canard_fighter_6dof.quat_state import normalize_quat_state

        actual_post = post_step if post_step is not None else normalize_quat_state
        step_fn = lambda fn, t, x, s_dt: rk4_step(fn, t, x, s_dt, post_step=actual_post)
    elif integrator == "euler_quat":
        from generic_delta_canard_fighter_6dof.quat_state import normalize_quat_state

        actual_post = post_step if post_step is not None else normalize_quat_state
        step_fn = lambda fn, t, x, s_dt: euler_step(
            fn, t, x, s_dt, post_step=actual_post
        )
    else:
        raise ValueError(
            f"Unknown integrator: '{integrator}'. Use 'rk4', 'euler', 'rk4_quat', or 'euler_quat'."
        )

    n_steps = 0 if t_final == 0.0 else int(np.ceil(t_final / dt))

    n_states = len(x0)
    t = np.zeros(n_steps + 1)
    x = np.zeros((n_steps + 1, n_states))

    t[0] = 0.0
    x[0] = x0.copy()

    current_x = x0.copy()
    for i in range(n_steps):
        current_t = t[i]
        step_dt = min(dt, t_final - current_t)
        current_x = step_fn(derivative_fn, current_t, current_x, step_dt)

        if not np.all(np.isfinite(current_x)):
            raise RuntimeError(
                f"Non-finite state detected at t ≈ {current_t + step_dt:.3e}. "
                "Simulation aborted."
            )

        t[i + 1] = current_t + step_dt
        x[i + 1] = current_x.copy()

    return SimulationResult(
        t=t,
        x=x,
        dt=dt,
        t_final=t_final,
        integrator_name=integrator,
    )
