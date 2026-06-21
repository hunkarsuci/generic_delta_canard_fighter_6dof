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