"""
Tests for the straight-and-level trim solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.state import StateIndex
from generic_delta_canard_fighter_6dof.trim import trim_straight_level


def test_trim_converges_at_nominal_condition() -> None:
    """Trim should converge for a reasonable flight condition."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert result.nfev > 0
    assert result.nfev < 500


def test_trim_residuals_below_threshold() -> None:
    """Dimensional residuals should be small at the trim point."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged

    # Force residuals should be small relative to weight
    weight = 9500.0 * 9.80665  # ~93,163 N
    assert np.linalg.norm(result.force_residual_N) < 1e-6 * weight

    # Moment residuals should be small
    assert np.linalg.norm(result.moment_residual_Nm) < 1.0


def test_trim_vt_dot_near_zero() -> None:
    """VT_dot should be negligible at trim."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert abs(result.state_derivative[StateIndex.VT]) < 1e-8


def test_trim_alpha_dot_near_zero() -> None:
    """alpha_dot should be negligible at trim."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert abs(result.state_derivative[StateIndex.ALPHA]) < 1e-8


def test_trim_q_dot_near_zero() -> None:
    """q_dot should be negligible at trim."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert abs(result.state_derivative[StateIndex.Q]) < 1e-8


def test_trim_controls_within_bounds() -> None:
    """Trimmed control deflections should be within physical limits."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert 0.0 <= result.throttle <= 1.0
    assert -25.0 <= result.symmetric_elevon_deg <= 25.0
    # alpha should be within bounds
    assert -5.0 <= result.alpha_deg <= 20.0


def test_trim_state_is_valid() -> None:
    """Trim state should pass state validation."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    # make_state does validation internally, just check it doesn't raise
    from generic_delta_canard_fighter_6dof.state import validate_state, validate_control

    validate_state(result.state)
    validate_control(result.control)


def test_trim_beta_zero() -> None:
    """Trim condition should have zero sideslip."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert result.state[StateIndex.BETA] == 0.0


def test_trim_phi_zero_level_wings() -> None:
    """Trim condition should have level wings."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert result.state[StateIndex.PHI] == 0.0


def test_trim_theta_equals_alpha() -> None:
    """For level flight (gamma=0), theta should equal alpha."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert result.state[StateIndex.THETA] == pytest.approx(
        result.state[StateIndex.ALPHA]
    )


def test_trim_reproducible() -> None:
    """Same inputs should produce the same trim result."""
    r1 = trim_straight_level(5000.0, 200.0)
    r2 = trim_straight_level(5000.0, 200.0)
    assert r1.alpha_deg == pytest.approx(r2.alpha_deg)
    assert r1.throttle == pytest.approx(r2.throttle)
    assert r1.symmetric_elevon_deg == pytest.approx(r2.symmetric_elevon_deg)


def test_trim_rejects_negative_airspeed() -> None:
    """Negative airspeed should fail cleanly."""
    result = trim_straight_level(5000.0, -10.0)
    assert not result.converged


def test_trim_different_altitude() -> None:
    """Trim should converge at a different altitude point."""
    result = trim_straight_level(2000.0, 250.0)
    assert result.converged
    assert abs(result.state_derivative[StateIndex.VT]) < 1e-8
    assert 0.0 <= result.throttle <= 1.0


def test_trim_sea_level() -> None:
    """Trim should converge at sea level."""
    result = trim_straight_level(0.0, 150.0)
    assert result.converged


def test_trim_high_altitude() -> None:
    """Trim should converge near tropopause."""
    result = trim_straight_level(10000.0, 250.0)
    assert result.converged


def test_trim_beta_dot_near_zero() -> None:
    """beta_dot should also be negligible (symmetric condition)."""
    result = trim_straight_level(5000.0, 200.0)
    assert result.converged
    assert abs(result.state_derivative[StateIndex.BETA]) < 1e-12
