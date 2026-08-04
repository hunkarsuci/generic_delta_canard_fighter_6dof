"""
Tests for numerical integrators.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.integrators import euler_step, rk4_step


def test_euler_step_constant_derivative() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([2.0])

    x_next = euler_step(
        derivative_function=derivative,
        t=0.0,
        x=np.array([1.0]),
        dt=0.5,
    )

    assert x_next[0] == pytest.approx(2.0)


def test_rk4_step_constant_derivative() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([2.0])

    x_next = rk4_step(
        derivative_function=derivative,
        t=0.0,
        x=np.array([1.0]),
        dt=0.5,
    )

    assert x_next[0] == pytest.approx(2.0)


def test_rk4_exponential_growth() -> None:
    """
    Test x_dot = x.

    Exact solution after one step:

        x(dt) = exp(dt)
    """

    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1

    x_next = rk4_step(
        derivative_function=derivative,
        t=0.0,
        x=np.array([1.0]),
        dt=dt,
    )

    assert x_next[0] == pytest.approx(np.exp(dt), rel=1.0e-6)


def test_euler_rejects_invalid_dt() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError):
        euler_step(
            derivative_function=derivative,
            t=0.0,
            x=np.array([1.0]),
            dt=0.0,
        )


def test_rk4_rejects_invalid_dt() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError):
        rk4_step(
            derivative_function=derivative,
            t=0.0,
            x=np.array([1.0]),
            dt=-0.1,
        )


def test_rk4_rejects_nonfinite_dt() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError):
        rk4_step(derivative, 0.0, np.array([1.0]), dt=np.nan)

    with pytest.raises(ValueError):
        rk4_step(derivative, 0.0, np.array([1.0]), dt=np.inf)


# ── RK4 order-of-accuracy (step-halving convergence) ─────────────────


def test_rk4_order_of_accuracy() -> None:
    """Verify RK4 exhibits approximately 4th-order convergence.

    For x_dot = x with x(0) = 1, the exact solution is x(t) = exp(t).
    Halving dt should reduce the error by a factor of ~16 (2^4).
    """

    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    t_final = 1.0
    x0 = np.array([1.0])

    errors = []
    dts = []

    base_dt = 0.1
    for k in range(4):
        dt = base_dt / (2**k)
        n_steps = int(t_final / dt)
        x = x0.copy()
        t = 0.0
        for _ in range(n_steps):
            x = rk4_step(derivative, t, x, dt)
            t += dt

        error = abs(x[0] - np.exp(t_final))
        errors.append(error)
        dts.append(dt)

    # Compute empirical order: error ∝ dt^p
    # log(error_ratio) = p * log(dt_ratio)
    ratios = []
    for i in range(len(errors) - 1):
        if errors[i] > 1e-15 and errors[i + 1] > 1e-15:
            p_empirical = np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            ratios.append(p_empirical)

    # Empirical order should be near 4 for RK4
    avg_order = float(np.mean(ratios))
    assert 3.0 < avg_order < 5.0, f"Expected ~4th order, got {avg_order:.2f}"


def test_rk4_errors_decrease_with_dt() -> None:
    """Smaller timesteps should produce monotonically smaller errors."""

    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    t_final = 0.5
    x0 = np.array([1.0])
    prev_error = float("inf")

    for dt in [0.1, 0.05, 0.025]:
        n_steps = int(t_final / dt)
        x = x0.copy()
        t = 0.0
        for _ in range(n_steps):
            x = rk4_step(derivative, t, x, dt)
            t += dt
        error = abs(x[0] - np.exp(t_final))
        assert error < prev_error
        prev_error = error


# ── Full aircraft model convergence check ────────────────────────────


def _make_derivative_func():
    """Create a closure over the aircraft dynamics for integrator testing."""
    from generic_delta_canard_fighter_6dof.equations import (
        aircraft_dynamics,
        zero_forces_moments,
    )
    from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
    from generic_delta_canard_fighter_6dof.state import make_control, make_state

    geo = create_default_geometry()
    control = make_control()

    x0 = make_state(
        VT=200.0,
        alpha=np.deg2rad(3.0),
        beta=np.deg2rad(1.0),
        p=0.05,
        q=-0.02,
        r=0.01,
        phi=np.deg2rad(5.0),
        theta=np.deg2rad(2.0),
        psi=np.deg2rad(10.0),
        h=1000.0,
    )

    def deriv(t: float, x: np.ndarray) -> np.ndarray:
        return aircraft_dynamics(t, x, control, geo, zero_forces_moments)

    return deriv, x0


def test_full_model_convergence_with_dt_halving() -> None:
    """Integrate aircraft EOM at dt, dt/2, dt/4 and verify solutions converge."""
    deriv, x0 = _make_derivative_func()

    t_final = 1.0

    results = {}
    base_dt = 0.05
    for k in range(3):
        dt = base_dt / (2**k)
        n = int(t_final / dt)
        x = x0.copy()
        t = 0.0
        for _ in range(n):
            x = rk4_step(deriv, t, x, dt)
            t += dt
        results[dt] = x.copy()

    # Compare dt vs dt/2 and dt/2 vs dt/4
    diff_coarse = np.linalg.norm(results[base_dt] - results[base_dt / 2])
    diff_fine = np.linalg.norm(results[base_dt / 2] - results[base_dt / 4])

    # The finer pair should differ less
    # Use a generous ratio: the finer pair difference should be < coarse pair
    assert diff_fine < diff_coarse, (
        f"Convergence check failed: coarse diff={diff_coarse:.2e}, "
        f"fine diff={diff_fine:.2e}"
    )

    # All state values should be finite
    for dt_key, x_val in results.items():
        assert np.all(np.isfinite(x_val)), f"Non-finite state at dt={dt_key}"


def test_euler_rejects_nonfinite_dt() -> None:
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError):
        euler_step(derivative, 0.0, np.array([1.0]), dt=-0.01)
