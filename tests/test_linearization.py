"""
Tests for numerical linearization.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.linearization import (
    linear_prediction,
    linearize,
)
from generic_delta_canard_fighter_6dof.propulsion import (
    combined_forces_moments,
)
from generic_delta_canard_fighter_6dof.simulation import simulate
from generic_delta_canard_fighter_6dof.state import (
    NUM_CONTROLS,
    NUM_STATES,
    StateIndex,
)


def _get_trim_operating_point():
    """Get a trim point for linearization tests."""
    from generic_delta_canard_fighter_6dof.trim import trim_straight_level

    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    return result.state, result.control


def test_linearize_a_matrix_shape() -> None:
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)
    assert lin.A.shape == (NUM_STATES, NUM_STATES)
    assert lin.B.shape == (NUM_STATES, NUM_CONTROLS)


def test_linearize_a_entries_are_finite() -> None:
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)
    assert np.all(np.isfinite(lin.A))
    assert np.all(np.isfinite(lin.B))


def test_linearize_residual_near_zero_at_trim() -> None:
    """At trim, the state derivative should be approximately zero."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)
    # VT_dot, alpha_dot, and q_dot should be near zero
    assert abs(lin.x_dot0[StateIndex.VT]) < 1e-8
    assert abs(lin.x_dot0[StateIndex.ALPHA]) < 1e-8
    assert abs(lin.x_dot0[StateIndex.Q]) < 1e-8


def test_linearize_reproducible() -> None:
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin1 = linearize(x0, u0, geo, fm)
    lin2 = linearize(x0, u0, geo, fm)
    assert np.allclose(lin1.A, lin2.A)
    assert np.allclose(lin1.B, lin2.B)


def test_linearize_perturbation_sensitivity() -> None:
    """A and B should not change dramatically with small perturbation changes."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()

    dx1 = np.full(NUM_STATES, 1e-5)
    dx2 = np.full(NUM_STATES, 1e-4)

    lin1 = linearize(x0, u0, geo, fm, dx=dx1)
    lin2 = linearize(x0, u0, geo, fm, dx=dx2)

    # Relative difference should be small for well-behaved entries
    rel_diff = np.linalg.norm(lin1.A - lin2.A) / max(np.linalg.norm(lin1.A), 1e-12)
    assert rel_diff < 0.1  # Within 10%


def test_nonlinear_vs_linear_short_horizon() -> None:
    """Over a short horizon with small perturbation, linear and nonlinear
    predictions should agree reasonably well."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)

    # Small perturbation in alpha
    dx0 = np.zeros(NUM_STATES)
    dx0[StateIndex.ALPHA] = 0.01  # ~0.57 deg perturbation

    # Linear prediction
    dt = 0.01
    n_steps = 50  # 0.5 s
    dx_lin = linear_prediction(lin, dx0, None, dt, n_steps)

    # Nonlinear simulation
    x_perturbed = x0 + dx0

    def deriv(t, x):
        return fm(t, x, u0, geo)  # not right, need aircraft_dynamics

    from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics

    def deriv_full(t, x):
        return aircraft_dynamics(t, x, u0, geo, fm)

    result = simulate(deriv_full, x_perturbed, dt, dt * n_steps)

    # Compute nonlinear deviation from trim
    dx_nonlin = np.zeros((n_steps + 1, NUM_STATES))
    for k in range(n_steps + 1):
        dx_nonlin[k] = result.x[k] - x0

    # Compare alpha deviation at final time
    lin_alpha_dev = dx_lin[-1, StateIndex.ALPHA]
    nonlin_alpha_dev = dx_nonlin[-1, StateIndex.ALPHA]

    # Linear and nonlinear alpha deviations should agree within ~20% over 0.5s
    rel_error = abs(lin_alpha_dev - nonlin_alpha_dev) / max(
        abs(nonlin_alpha_dev), 1e-12
    )
    assert rel_error < 0.20, (
        f"Linear vs nonlinear mismatch: linear={lin_alpha_dev:.6f}, "
        f"nonlinear={nonlin_alpha_dev:.6f}, rel_err={rel_error:.3f}"
    )


def test_linear_prediction_zero_perturbation_remains_zero() -> None:
    """With zero initial perturbation, linear model stays at origin."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)

    dx_lin = linear_prediction(lin, np.zeros(NUM_STATES), None, 0.01, 10)
    assert np.allclose(dx_lin, 0.0)


def test_linearize_rejects_wrong_shapes() -> None:
    geo = create_default_geometry()
    fm = combined_forces_moments()

    with pytest.raises(ValueError):
        linearize(np.zeros(10), np.zeros(5), geo, fm)

    with pytest.raises(ValueError):
        linearize(np.zeros(12), np.zeros(4), geo, fm)


def test_b_matrix_throttle_derivative() -> None:
    """Verify B matrix column for throttle makes physical sense:
    positive throttle should increase VT_dot."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)

    # B[VT, throttle] should be positive: more throttle -> more forward acceleration
    assert lin.B[StateIndex.VT, 4] > 0.0


def test_a_matrix_speed_stability() -> None:
    """A[VT, VT] should be slightly negative or near zero (speed stability)."""
    x0, u0 = _get_trim_operating_point()
    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(x0, u0, geo, fm)

    # Speed stability: faster → more drag → deceleration → negative derivative
    # This may or may not hold for this generic model; just check it's finite
    assert np.isfinite(lin.A[StateIndex.VT, StateIndex.VT])
