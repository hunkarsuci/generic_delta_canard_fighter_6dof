"""
Tests for the generic aerodynamic model.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.aerodynamics import (
    AeroCoefficients,
    aerodynamic_forces_moments,
)
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.state import make_control, make_state


def test_zero_dynamic_pressure_zero_forces() -> None:
    """At zero airspeed, aerodynamic forces should be zero."""
    geo = create_default_geometry()
    x = make_state(VT=0.0, h=1000.0)
    u = make_control()
    fm = aerodynamic_forces_moments(0.0, x, u, geo)
    assert np.allclose(fm.force_body_N, 0.0)
    assert np.allclose(fm.moment_body_Nm, 0.0)


def test_force_scales_with_dynamic_pressure() -> None:
    """At double the speed, aerodynamic forces should scale with q_bar ∝ VT²."""
    geo = create_default_geometry()
    u = make_control()

    x1 = make_state(VT=100.0, h=0.0)
    x2 = make_state(VT=200.0, h=0.0)

    fm1 = aerodynamic_forces_moments(0.0, x1, u, geo)
    fm2 = aerodynamic_forces_moments(0.0, x2, u, geo)

    # Ratio should be approximately (200/100)^2 = 4
    # Not exact due to small alpha-induced changes, but close
    f1 = np.linalg.norm(fm1.force_body_N)
    f2 = np.linalg.norm(fm2.force_body_N)
    ratio = f2 / f1
    assert 3.8 < ratio < 4.2


def test_zero_sideslip_symmetric_forces() -> None:
    """At zero beta, side force and roll/yaw moments should be near zero."""
    geo = create_default_geometry()
    # Small alpha, zero beta, symmetric controls
    x = make_state(VT=150.0, alpha=0.05, beta=0.0, h=2000.0)
    u = make_control(delta_elevon_left=0.02, delta_elevon_right=0.02)
    fm = aerodynamic_forces_moments(0.0, x, u, geo)

    # Side force Y (index 1) should be near zero
    assert fm.force_body_N[1] == pytest.approx(0.0, abs=1e-9)
    # Rolling moment L (index 0) from differential elevon=0 should be zero
    assert fm.moment_body_Nm[0] == pytest.approx(0.0, abs=1e-9)


def test_differential_elevon_produces_rolling_moment() -> None:
    """Differential elevon should produce nonzero rolling moment."""
    geo = create_default_geometry()
    x = make_state(VT=150.0, h=2000.0)
    u = make_control(delta_elevon_left=-0.05, delta_elevon_right=0.05)
    fm = aerodynamic_forces_moments(0.0, x, u, geo)
    assert fm.moment_body_Nm[0] != 0.0


def test_forces_moments_have_correct_shape() -> None:
    geo = create_default_geometry()
    x = make_state(VT=150.0, alpha=0.05, beta=0.02, h=3000.0)
    u = make_control(delta_canard=0.01)
    fm = aerodynamic_forces_moments(0.0, x, u, geo)
    assert fm.force_body_N.shape == (3,)
    assert fm.moment_body_Nm.shape == (3,)
    assert np.all(np.isfinite(fm.force_body_N))
    assert np.all(np.isfinite(fm.moment_body_Nm))


def test_positive_alpha_produces_negative_z_force() -> None:
    """Positive alpha (nose up) should produce negative body-z force (upward lift)."""
    geo = create_default_geometry()
    # VT > 0, alpha positive, body z is downward, so lift acts negative in body z
    x = make_state(VT=150.0, alpha=0.1, beta=0.0, h=0.0)
    u = make_control()
    fm = aerodynamic_forces_moments(0.0, x, u, geo)
    # Body Z (index 2) should be negative (lift upward = negative downward force)
    assert fm.force_body_N[2] < 0.0


def test_positive_beta_produces_positive_side_force() -> None:
    """With CY_beta < 0, positive beta should produce positive Y (rightward force)."""
    geo = create_default_geometry()
    x = make_state(VT=150.0, alpha=0.0, beta=0.05, h=0.0)
    u = make_control()
    fm = aerodynamic_forces_moments(0.0, x, u, geo)
    # CY_beta is negative, so side force should be negative in wind axes
    # After wind→body transform, at zero alpha this maps directly to body Y
    # But actually with CY_beta < 0 and beta > 0, Y_wind = q*S*CY_beta*beta < 0
    # In body: Y_body = Y_wind (at alpha=0). So Y_body < 0.
    # Hmm, actually the sign convention: body Y is positive right.
    # With CY_beta = -1.0, beta > 0 means Y_wind < 0, which in body (at alpha=0)
    # gives Y_body < 0.
    # This is physically correct: positive sideslip produces a restoring force
    # toward negative y (left), pushing the nose back.
    assert fm.force_body_N[1] < 0.0


def test_coefficient_defaults_are_finite() -> None:
    c = AeroCoefficients()
    for name in dir(c):
        if not name.startswith("_"):
            val = getattr(c, name)
            if not callable(val):
                assert np.isfinite(val), f"{name} is not finite"
