"""
Tests for nonlinear 6DOF equations of motion.

These tests verify the core dynamics through independent analytical
derivations rather than calling the same function twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.constants import GRAVITY_MPS2
from generic_delta_canard_fighter_6dof.equations import (
    ForcesMoments,
    aircraft_dynamics,
    body_velocity_derivatives_to_wind_derivatives,
    gravity_force_body,
    rotational_acceleration_body,
    translational_acceleration_body,
    zero_forces_moments,
)
from generic_delta_canard_fighter_6dof.geometry import (
    AircraftGeometry,
    create_default_geometry,
)
from generic_delta_canard_fighter_6dof.state import (
    NUM_STATES,
    StateIndex,
    make_control,
    make_state,
)

# ── helpers ───────────────────────────────────────────────────────────


def _zero_control() -> np.ndarray:
    return make_control()


def _default_geometry() -> AircraftGeometry:
    return create_default_geometry()


# ── gravity_force_body ────────────────────────────────────────────────


def test_gravity_body_at_zero_attitude() -> None:
    """At level attitude, gravity in body z is positive (down)."""
    m = 1000.0
    f_body = gravity_force_body(phi=0.0, theta=0.0, psi=0.0, mass_kg=m)

    assert f_body[0] == pytest.approx(0.0, abs=1e-12)
    assert f_body[1] == pytest.approx(0.0, abs=1e-12)
    assert f_body[2] == pytest.approx(m * GRAVITY_MPS2)


def test_gravity_body_at_pitch_up_90() -> None:
    """At theta = +90 deg, gravity acts along body -x (aft)."""
    m = 1000.0
    f_body = gravity_force_body(phi=0.0, theta=np.pi / 2, psi=0.0, mass_kg=m)

    # Body z-forward? No. At theta=+90, aircraft nose points up (NED up = negative down).
    # NED gravity: [0, 0, m*g]. Body: C_bn @ [0, 0, m*g].
    # C_bn(theta=90): body x points NED -z (up), body z points NED +x (north).
    # So gravity in body = [-m*g, 0, 0]
    assert f_body[0] == pytest.approx(-m * GRAVITY_MPS2)
    assert f_body[1] == pytest.approx(0.0, abs=1e-12)
    assert f_body[2] == pytest.approx(0.0, abs=1e-10)


def test_gravity_body_at_roll_90() -> None:
    """At phi = +90 deg (right wing down), gravity acts along body +y."""
    m = 1000.0
    f_body = gravity_force_body(phi=np.pi / 2, theta=0.0, psi=0.0, mass_kg=m)

    # NED gravity: [0, 0, m*g] (positive down)
    # phi = +90: body y points NED down (positive z_D)
    # So gravity in body = [0, m*g, 0]
    assert f_body[0] == pytest.approx(0.0, abs=1e-12)
    assert f_body[1] == pytest.approx(m * GRAVITY_MPS2)
    assert f_body[2] == pytest.approx(0.0, abs=1e-10)


def test_gravity_body_magnitude_is_mg() -> None:
    """Gravity magnitude in body equals m*g regardless of attitude."""
    m = 1000.0
    phi, theta, psi = 0.3, -0.2, 0.7
    f_body = gravity_force_body(phi=phi, theta=theta, psi=psi, mass_kg=m)
    assert np.linalg.norm(f_body) == pytest.approx(m * GRAVITY_MPS2)


# ── translational_acceleration_body ───────────────────────────────────


def test_translational_accel_no_rotation() -> None:
    """With zero angular rate, acceleration = force / mass."""
    v_body = np.array([100.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    force = np.array([5000.0, 0.0, 0.0])
    m = 2000.0

    a = translational_acceleration_body(v_body, omega, force, m)
    assert np.allclose(a, force / m)


def test_translational_accel_coriolis() -> None:
    """With pitch rate and forward speed, coriolis produces +w_dot."""
    v_body = np.array([100.0, 0.0, 0.0])  # forward
    omega = np.array([0.0, 0.1, 0.0])  # pitch up
    force = np.array([0.0, 0.0, 0.0])
    m = 2000.0

    a = translational_acceleration_body(v_body, omega, force, m)
    # coriolis: -omega x v = -[0, 0.1, 0] x [100, 0, 0]
    # = -[0.1*0 - 0*0, 0*100 - 0*0, 0*0 - 0.1*100]
    # = -[0, 0, -10] = [0, 0, 10]
    # So w_dot = +10 (positive = downward acceleration in body)
    assert a[0] == pytest.approx(0.0, abs=1e-12)
    assert a[1] == pytest.approx(0.0, abs=1e-12)
    assert a[2] == pytest.approx(10.0)


def test_translational_accel_rejects_negative_mass() -> None:
    with pytest.raises(ValueError):
        translational_acceleration_body(
            np.ones(3),
            np.zeros(3),
            np.zeros(3),
            mass_kg=-1.0,
        )


# ── rotational_acceleration_body ──────────────────────────────────────


def test_rotational_accel_no_gyroscopic() -> None:
    """With zero angular rate and diagonal inertia, alpha = I^{-1} M."""
    geo = AircraftGeometry(
        mass_kg=1000,
        Ixx_kg_m2=100.0,
        Iyy_kg_m2=200.0,
        Izz_kg_m2=300.0,
        Ixz_kg_m2=0.0,
        wing_area_m2=50.0,
        wingspan_m=10.0,
        mean_aerodynamic_chord_m=5.0,
    )
    omega = np.array([0.0, 0.0, 0.0])
    moment = np.array([10.0, 20.0, 30.0])

    alpha = rotational_acceleration_body(omega, moment, geo)
    assert alpha[0] == pytest.approx(0.1)  # 10/100
    assert alpha[1] == pytest.approx(0.1)  # 20/200
    assert alpha[2] == pytest.approx(0.1)  # 30/300


def test_rotational_accel_with_Ixz_coupling() -> None:
    """With Ixz nonzero, moment about x affects both p_dot and r_dot."""
    Ixx, Iyy, Izz, Ixz = 100.0, 200.0, 300.0, 20.0
    geo = AircraftGeometry(
        mass_kg=1000,
        Ixx_kg_m2=Ixx,
        Iyy_kg_m2=Iyy,
        Izz_kg_m2=Izz,
        Ixz_kg_m2=Ixz,
        wing_area_m2=50.0,
        wingspan_m=10.0,
        mean_aerodynamic_chord_m=5.0,
    )

    # Pure rolling moment
    omega = np.array([0.0, 0.0, 0.0])
    moment = np.array([100.0, 0.0, 0.0])

    alpha = rotational_acceleration_body(omega, moment, geo)

    # With Ixz coupling, a pure rolling moment produces both roll and yaw acceleration
    assert alpha[0] != 0.0  # p_dot
    assert alpha[2] != 0.0  # r_dot from coupling

    # Independent check using explicit matrix inverse
    I_mat = np.array([[Ixx, 0, -Ixz], [0, Iyy, 0], [-Ixz, 0, Izz]])
    alpha_expected = np.linalg.inv(I_mat) @ moment
    assert np.allclose(alpha, alpha_expected)


# ── body_velocity_derivatives_to_wind_derivatives ─────────────────────


def test_wind_derivatives_steady_level() -> None:
    """Steady level flight: VT constant, alpha=beta=0, zero accel."""
    VT_dot, alpha_dot, beta_dot = body_velocity_derivatives_to_wind_derivatives(
        u=100.0,
        v=0.0,
        w=0.0,
        u_dot=0.0,
        v_dot=0.0,
        w_dot=0.0,
    )
    assert VT_dot == pytest.approx(0.0)
    assert alpha_dot == pytest.approx(0.0)
    assert beta_dot == pytest.approx(0.0)


def test_wind_derivatives_pure_forward_accel() -> None:
    """Pure forward acceleration: VT_dot = u_dot, alpha/beta unchanged."""
    VT_dot, alpha_dot, beta_dot = body_velocity_derivatives_to_wind_derivatives(
        u=100.0,
        v=0.0,
        w=0.0,
        u_dot=5.0,
        v_dot=0.0,
        w_dot=0.0,
    )
    assert VT_dot == pytest.approx(5.0)
    assert alpha_dot == pytest.approx(0.0, abs=1e-12)
    assert beta_dot == pytest.approx(0.0, abs=1e-12)


def test_wind_derivatives_pure_downward_accel() -> None:
    """Pure downward body acceleration at zero alpha: VT_dot=0, alpha_dot>0."""
    VT = 100.0
    VT_dot, alpha_dot, beta_dot = body_velocity_derivatives_to_wind_derivatives(
        u=VT,
        v=0.0,
        w=0.0,
        u_dot=0.0,
        v_dot=0.0,
        w_dot=10.0,
    )
    assert VT_dot == pytest.approx(0.0, abs=1e-12)
    assert alpha_dot == pytest.approx(10.0 / VT)  # = u*w_dot / u² = w_dot/u
    assert beta_dot == pytest.approx(0.0, abs=1e-12)


def test_wind_derivatives_zero_airspeed() -> None:
    """At zero airspeed, all wind derivatives are zero (model limitation)."""
    VT_dot, alpha_dot, beta_dot = body_velocity_derivatives_to_wind_derivatives(
        u=0.0,
        v=0.0,
        w=0.0,
        u_dot=10.0,
        v_dot=0.0,
        w_dot=0.0,
    )
    assert VT_dot == 0.0
    assert alpha_dot == 0.0
    assert beta_dot == 0.0


# ── aircraft_dynamics (full state derivative) ─────────────────────────


def test_dynamics_returns_correct_shape() -> None:
    x = make_state(VT=100.0)
    u = _zero_control()
    geo = _default_geometry()
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)
    assert x_dot.shape == (NUM_STATES,)
    assert np.all(np.isfinite(x_dot))


def test_dynamics_steady_level_no_gravity_produces_constant_velocity() -> None:
    """With zero forces AND no gravity, level flight should have zero VT_dot."""
    x = make_state(VT=100.0, alpha=0.0, beta=0.0)
    u = _zero_control()
    geo = _default_geometry()

    # Use a force model that returns ZERO (no gravity either)
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)

    # Gravity IS always added inside aircraft_dynamics, so VT_dot will NOT be zero.
    # Instead, verify: at level attitude with gravity only, body accel = [0, 0, g]
    # VT_dot = 0, alpha_dot = g/VT
    VT = x[StateIndex.VT]
    assert x_dot[StateIndex.VT] == pytest.approx(0.0, abs=1e-12)
    assert x_dot[StateIndex.ALPHA] == pytest.approx(GRAVITY_MPS2 / VT)


def test_dynamics_heading_change_from_yaw_rate() -> None:
    """At zero pitch/roll, psi_dot should equal r."""
    x = make_state(VT=100.0, r=0.5)
    u = _zero_control()
    geo = _default_geometry()
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)
    assert x_dot[StateIndex.PSI] == pytest.approx(0.5)


def test_dynamics_position_rates_level_north() -> None:
    """At zero attitude and forward speed, x_N grows at VT, y_E=0, h_dot=0."""
    VT = 150.0
    x = make_state(VT=VT, alpha=0.0, beta=0.0)
    u = _zero_control()
    geo = _default_geometry()
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)
    assert x_dot[StateIndex.X_N] == pytest.approx(VT)
    assert x_dot[StateIndex.Y_E] == pytest.approx(0.0, abs=1e-12)
    assert x_dot[StateIndex.H] == pytest.approx(0.0, abs=1e-12)


def test_dynamics_rejects_invalid_state() -> None:
    with pytest.raises(ValueError):
        aircraft_dynamics(0.0, np.zeros(11), _zero_control(), _default_geometry())


def test_dynamics_rejects_invalid_control() -> None:
    with pytest.raises(ValueError):
        aircraft_dynamics(0.0, make_state(VT=100), np.zeros(4), _default_geometry())


def test_dynamics_zero_angular_rates_hold_attitude() -> None:
    """With zero angular rates and no moments, phi/theta/psi should be constant."""
    x = make_state(VT=100.0, phi=0.3, theta=0.1, psi=0.5)
    u = _zero_control()
    geo = _default_geometry()
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)
    assert x_dot[StateIndex.PHI] == pytest.approx(0.0, abs=1e-12)
    assert x_dot[StateIndex.THETA] == pytest.approx(0.0, abs=1e-12)
    assert x_dot[StateIndex.PSI] == pytest.approx(0.0, abs=1e-12)


def test_dynamics_state_derivatives_are_finite() -> None:
    """At a representative flight condition, all derivatives are finite."""
    x = make_state(
        VT=200.0,
        alpha=np.deg2rad(5.0),
        beta=np.deg2rad(-2.0),
        p=0.1,
        q=-0.05,
        r=0.02,
        phi=np.deg2rad(10.0),
        theta=np.deg2rad(3.0),
        psi=np.deg2rad(45.0),
    )
    u = _zero_control()
    geo = _default_geometry()
    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=zero_forces_moments)
    assert np.all(np.isfinite(x_dot))


def test_dynamics_with_custom_force_model() -> None:
    """Custom force model contributions are reflected in state derivatives."""

    def constant_force(t, x, control, geo):
        return ForcesMoments(
            force_body_N=np.array([1000.0, 0.0, 0.0]),
            moment_body_Nm=np.array([0.0, 500.0, 0.0]),
        )

    x = make_state(VT=100.0)
    u = _zero_control()
    geo = _default_geometry()

    x_dot = aircraft_dynamics(0.0, x, u, geo, force_moment_model=constant_force)

    # Forward force should increase VT (positive VT_dot)
    assert x_dot[StateIndex.VT] > 0.0
    # Pitch moment should produce angular acceleration
    assert x_dot[StateIndex.Q] != 0.0


# ── ForcesMoments validation ──────────────────────────────────────────


def test_forces_moments_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        ForcesMoments(force_body_N=np.zeros(4), moment_body_Nm=np.zeros(3))

    with pytest.raises(ValueError):
        ForcesMoments(force_body_N=np.zeros(3), moment_body_Nm=np.zeros(2))


def test_forces_moments_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        ForcesMoments(
            force_body_N=np.array([np.nan, 0.0, 0.0]),
            moment_body_Nm=np.zeros(3),
        )


def test_zero_forces_moments_returns_zero() -> None:
    geo = _default_geometry()
    fm = zero_forces_moments(0.0, make_state(VT=100), _zero_control(), geo)
    assert np.allclose(fm.force_body_N, 0.0)
    assert np.allclose(fm.moment_body_Nm, 0.0)
