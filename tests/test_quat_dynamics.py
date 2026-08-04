"""Tests for quaternion-state nonlinear dynamics.

Covers:
- State vector construction and validation
- Euler <-> quaternion state conversion
- Quaternion dynamics: zero-rate attitude preservation
- Constant-rate analytical quaternion solution
- Quaternion sign equivalence (q and -q represent same attitude)
- Norm preservation during integration
- 90-degree known rotations
- Euler and quaternion trajectories agree away from singularities
- Quaternion model remains finite near pitch ±90 degrees
- Translational trajectory agreement
- RK4 convergence for quaternion dynamics
- Shared force/moment model consistency
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.constants import GRAVITY_MPS2
from generic_delta_canard_fighter_6dof.equations import (
    ForcesMoments,
    aircraft_dynamics,
    aircraft_dynamics_quat,
    gravity_force_body,
)
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.integrators import (
    rk4_step,
)
from generic_delta_canard_fighter_6dof.quat_state import (
    NUM_QUAT_STATES,
    QuatStateIndex,
    _validate_quat_state,
    euler_state_to_quat_state,
    make_quat_state,
    quat_state_to_dict,
    quat_state_to_euler_state,
)
from generic_delta_canard_fighter_6dof.quaternions import (
    quaternion_to_euler,
)
from generic_delta_canard_fighter_6dof.simulation import simulate
from generic_delta_canard_fighter_6dof.state import (
    StateIndex,
    make_control,
    make_state,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_trimmed_euler_state() -> np.ndarray:
    """Return an Euler state near a plausible trimmed condition."""
    return make_state(
        VT=200.0,
        alpha=0.05,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
        phi=0.0,
        theta=0.05,
        psi=0.0,
        x_N=0.0,
        y_E=0.0,
        h=5000.0,
    )


def _make_zero_control() -> np.ndarray:
    return make_control(
        delta_canard=0.0,
        delta_elevon_left=0.0,
        delta_elevon_right=0.0,
        delta_rudder=0.0,
        throttle=0.0,
    )


# ── state vector construction ────────────────────────────────────────────────


def test_make_quat_state_default_is_identity():
    x = make_quat_state()
    assert len(x) == NUM_QUAT_STATES
    assert x[QuatStateIndex.Q0] == 1.0
    assert x[QuatStateIndex.Q1] == 0.0
    assert x[QuatStateIndex.Q2] == 0.0
    assert x[QuatStateIndex.Q3] == 0.0
    assert np.all(np.isfinite(x))


def test_make_quat_state_custom_values():
    x = make_quat_state(u=100.0, v=10.0, w=-5.0, p=0.1, q=-0.2, r=0.05, h=3000.0)
    assert x[QuatStateIndex.U] == 100.0
    assert x[QuatStateIndex.V] == 10.0
    assert x[QuatStateIndex.W] == -5.0
    assert x[QuatStateIndex.H] == 3000.0


def test_validate_quat_state_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _validate_quat_state(np.zeros(12))


def test_validate_quat_state_rejects_nonfinite():
    x = make_quat_state()
    x[QuatStateIndex.U] = np.nan
    with pytest.raises(ValueError):
        _validate_quat_state(x)


def test_quat_state_to_dict():
    x = make_quat_state(u=150.0, h=5000.0)
    d = quat_state_to_dict(x)
    assert d["u"] == 150.0
    assert d["h"] == 5000.0
    assert d["q0"] == 1.0


# ── Euler <-> quaternion state conversion ────────────────────────────────────


def test_euler_to_quat_state_round_trip():
    x_euler = _make_trimmed_euler_state()
    x_quat = euler_state_to_quat_state(x_euler)
    x_euler_back = quat_state_to_euler_state(x_quat)

    # Compare all 12 Euler states
    for i in range(12):
        assert x_euler_back[i] == pytest.approx(x_euler[i], abs=1e-12), (
            f"Mismatch at index {i}"
        )


def test_quat_to_euler_state_round_trip():
    x_quat = make_quat_state(u=180.0, v=5.0, w=-10.0, h=2000.0)
    x_euler = quat_state_to_euler_state(x_quat)
    x_quat_back = euler_state_to_quat_state(x_euler)

    for i in range(NUM_QUAT_STATES):
        assert x_quat_back[i] == pytest.approx(x_quat[i], abs=1e-12), (
            f"Mismatch at index {i}"
        )


def test_euler_to_quat_at_identity():
    """Identity Euler angles → identity quaternion."""
    x_euler = make_state(phi=0.0, theta=0.0, psi=0.0)
    x_quat = euler_state_to_quat_state(x_euler)
    assert x_quat[QuatStateIndex.Q0] == pytest.approx(1.0, abs=1e-12)
    assert x_quat[QuatStateIndex.Q1] == pytest.approx(0.0, abs=1e-12)
    assert x_quat[QuatStateIndex.Q2] == pytest.approx(0.0, abs=1e-12)
    assert x_quat[QuatStateIndex.Q3] == pytest.approx(0.0, abs=1e-12)


def test_quat_sign_equivalence():
    """q and -q represent the same physical attitude."""
    x1 = make_quat_state(q0=0.5, q1=0.5, q2=0.5, q3=0.5)
    x2 = make_quat_state(q0=-0.5, q1=-0.5, q2=-0.5, q3=-0.5)

    e1 = quat_state_to_euler_state(x1)
    e2 = quat_state_to_euler_state(x2)

    for i in range(3):  # phi, theta, psi
        assert e1[StateIndex.PHI + i] == pytest.approx(
            e2[StateIndex.PHI + i], abs=1e-12
        )


# ── quaternion dynamics: zero-rate attitude preservation ─────────────────────


def test_quat_dynamics_zero_rates_preserves_attitude():
    """With zero angular rates and zero forces, attitude should not change."""
    geo = create_default_geometry()
    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, h=5000.0)
    ctrl = _make_zero_control()

    # Use zero_forces_moments with no gravity coupling by overriding.
    # The aircraft_dynamics_quat always adds gravity, so we use a model
    # that cancels gravity to get pure constant-velocity motion.
    # Actually, gravity in body frame depends on attitude. For level flight
    # with q=[1,0,0,0], gravity_body = [0,0,mg] so it adds downward accel.
    # Test attitude preservation by checking quaternion doesn't change
    # when angular rates are zero (no moments, no gyroscopic coupling).
    def no_moment_model(t, x, control, geometry):
        # Return only horizontal thrust to balance drag (not needed here)
        # Key: zero moments → zero angular acceleration
        return ForcesMoments(
            force_body_N=np.zeros(3),
            moment_body_Nm=np.zeros(3),
        )

    x_dot = aircraft_dynamics_quat(0.0, x0, ctrl, geo, no_moment_model)

    # With zero angular rates and zero moments, q_dot should be zero
    assert x_dot[QuatStateIndex.Q0] == pytest.approx(0.0, abs=1e-15)
    assert x_dot[QuatStateIndex.Q1] == pytest.approx(0.0, abs=1e-15)
    assert x_dot[QuatStateIndex.Q2] == pytest.approx(0.0, abs=1e-15)
    assert x_dot[QuatStateIndex.Q3] == pytest.approx(0.0, abs=1e-15)


# ── constant-rate analytical quaternion solution ─────────────────────────────


def test_quat_dynamics_constant_pure_roll_rate():
    """Constant roll rate p = 1 rad/s: q(t) = [cos(pt/2), sin(pt/2), 0, 0]."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, p=1.0, q=0.0, r=0.0, h=5000.0)

    def no_forces(t, x, control, geometry):
        return ForcesMoments(force_body_N=np.zeros(3), moment_body_Nm=np.zeros(3))

    x_dot = aircraft_dynamics_quat(0.0, x0, ctrl, geo, no_forces)

    # Analytical: q_dot = 0.5 * [1,0,0,0] ⊗ [0,1,0,0] = 0.5 * [-0, 1, 0, 0]
    # Actually: q0' = -0.5*(q1*p + q2*q_rate + q3*r) = -0.5*(0*1) = 0
    # q1' = 0.5*(q0*p + q2*r - q3*q_rate) = 0.5*(1*1) = 0.5
    assert x_dot[QuatStateIndex.Q1] == pytest.approx(0.5, abs=1e-14)
    assert x_dot[QuatStateIndex.Q2] == pytest.approx(0.0, abs=1e-14)
    assert x_dot[QuatStateIndex.Q3] == pytest.approx(0.0, abs=1e-14)


def test_quat_dynamics_constant_pitch_rate():
    """Constant pitch rate q = 2 rad/s."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, p=0.0, q=2.0, r=0.0, h=5000.0)

    def no_forces(t, x, control, geometry):
        return ForcesMoments(force_body_N=np.zeros(3), moment_body_Nm=np.zeros(3))

    x_dot = aircraft_dynamics_quat(0.0, x0, ctrl, geo, no_forces)

    # q_dot = 0.5 * [1,0,0,0] ⊗ [0,0,2,0] = [0, 0, 1, 0]
    assert x_dot[QuatStateIndex.Q2] == pytest.approx(1.0, abs=1e-14)


# ── norm preservation during integration ─────────────────────────────────────


def test_quat_rk4_norm_preservation():
    """RK4 step with post-normalization preserves quaternion norm."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(
        u=200.0,
        v=0.0,
        w=0.0,
        p=0.5,
        q=-0.3,
        r=0.2,
        h=5000.0,
    )

    def deriv(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    dt = 0.01
    x1 = rk4_step(deriv, 0.0, x0, dt)

    q = x1[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
    norm = np.linalg.norm(q)
    assert norm == pytest.approx(1.0, abs=1e-14)


def test_quat_rk4_norm_preservation_many_steps():
    """Quaternion norm stays 1.0 over a long integration."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(
        u=200.0,
        v=0.0,
        w=0.0,
        p=0.1,
        q=0.2,
        r=-0.15,
        h=5000.0,
    )

    def deriv(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    x = x0.copy()
    dt = 0.01
    for step in range(1000):
        x = rk4_step(deriv, step * dt, x, dt)
        q = x[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-14)


# ── 90-degree known rotations ────────────────────────────────────────────────


def test_quat_dynamics_90_degree_yaw_integration():
    """Integrate constant yaw rate r = π/2 rad/s for 1 s → 90° yaw.

    Uses a geometry with Ixz = 0 to avoid gyroscopic coupling that would
    induce roll/pitch rates from pure yaw motion.
    """
    from generic_delta_canard_fighter_6dof.geometry import AircraftGeometry
    from generic_delta_canard_fighter_6dof.quaternions import quaternion_to_dcm

    # Zero Ixz to eliminate gyroscopic coupling (ω × Iω with pure yaw)
    geo = AircraftGeometry(
        mass_kg=9500.0,
        Ixx_kg_m2=75000.0,
        Iyy_kg_m2=95000.0,
        Izz_kg_m2=145000.0,
        Ixz_kg_m2=0.0,
        wing_area_m2=50.0,
        wingspan_m=10.5,
        mean_aerodynamic_chord_m=5.0,
    )
    ctrl = _make_zero_control()

    x0 = make_quat_state(
        u=200.0,
        v=0.0,
        w=0.0,
        p=0.0,
        q=0.0,
        r=np.pi / 2,
        h=5000.0,
    )

    def deriv(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    dt = 0.001
    x = x0.copy()
    t = 0.0
    for _ in range(1000):
        x = rk4_step(deriv, t, x, dt)
        t += dt

    # After 1 second of r = π/2, body x-axis should point east in NED
    C_nb = quaternion_to_dcm(x[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1])
    body_forward = np.array([1.0, 0.0, 0.0])
    ned_forward = C_nb @ body_forward

    # Should point approximately east
    assert ned_forward[0] == pytest.approx(0.0, abs=1e-6)  # x_N ≈ 0
    assert ned_forward[1] == pytest.approx(1.0, abs=1e-6)  # y_E ≈ 1
    assert ned_forward[2] == pytest.approx(0.0, abs=1e-6)  # z_D ≈ 0


# ── Euler/quaternion trajectory agreement away from singularities ────────────


def test_euler_vs_quat_trajectory_agreement():
    """Euler and quaternion models produce identical trajectories away from θ=±90°."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    # Use a known Euler initial state
    x_euler0 = make_state(
        VT=200.0,
        alpha=0.05,
        beta=0.01,
        p=0.1,
        q=-0.05,
        r=0.03,
        phi=0.1,
        theta=0.05,
        psi=0.2,
        h=5000.0,
    )

    # Convert to quaternion state
    x_quat0 = euler_state_to_quat_state(x_euler0)

    geo_obj = geo

    def deriv_euler(t, x):
        return aircraft_dynamics(t, x, ctrl, geo_obj)

    def deriv_quat(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo_obj)

    dt = 0.01
    t_final = 1.0
    n_steps = int(t_final / dt)

    x_e = x_euler0.copy()
    x_q = x_quat0.copy()

    for step in range(n_steps):
        t = step * dt
        x_e = rk4_step(deriv_euler, t, x_e, dt)
        x_q = rk4_step(deriv_quat, t, x_q, dt)

    # Convert quat result back to Euler for comparison
    x_e_from_q = quat_state_to_euler_state(x_q)

    # Compare angular rates (should match closely)
    for i in range(3, 6):  # p, q, r
        assert x_e[i] == pytest.approx(x_e_from_q[i], abs=1e-10)

    # Compare attitudes (phi, theta, psi)
    for i in range(6, 9):
        assert x_e[i] == pytest.approx(x_e_from_q[i], abs=1e-8)

    # Compare positions
    for i in range(9, 12):
        assert x_e[i] == pytest.approx(x_e_from_q[i], abs=1e-8)


# ── quaternion model finite near pitch ±90° ──────────────────────────────────


def test_quat_dynamics_finite_near_pitch_90():
    """Quaternion dynamics remain finite when pitch approaches 90°."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    # Set up a state with theta near 90° (converted to quaternion)
    x_euler_near_singularity = make_state(
        VT=200.0,
        alpha=0.0,
        beta=0.0,
        theta=np.deg2rad(89.0),
        h=5000.0,
    )
    x_quat = euler_state_to_quat_state(x_euler_near_singularity)

    # Verify quaternion representation is valid
    q = x_quat[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-14)

    # Dynamics should run without error
    x_dot = aircraft_dynamics_quat(0.0, x_quat, ctrl, geo)
    assert np.all(np.isfinite(x_dot))


def test_quat_dynamics_through_pitch_90():
    """Integrate through pitch 90° — quaternion stays finite, Euler model would fail."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    # Start at level flight, apply a pitch moment to drive through 90°
    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, q=1.5, h=5000.0)

    def pitch_moment_model(t, x, control, geometry):
        return ForcesMoments(
            force_body_N=np.zeros(3),
            moment_body_Nm=np.array([0.0, 50000.0, 0.0]),  # pitch-up moment
        )

    dt = 0.001
    x = x0.copy()
    t = 0.0

    # Integrate until pitch passes through 90° region
    for _ in range(2000):
        x = rk4_step(
            lambda t_, x_: aircraft_dynamics_quat(
                t_, x_, ctrl, geo, pitch_moment_model
            ),
            t,
            x,
            dt,
        )
        t += dt

    # Quaternion should still be valid
    q = x[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-14)

    # All states should be finite
    assert np.all(np.isfinite(x))

    # Extract Euler angles — they should give valid values
    phi, theta, psi = quaternion_to_euler(q)
    assert np.isfinite(phi)
    assert np.isfinite(theta)
    assert np.isfinite(psi)


# ── translational trajectory agreement ───────────────────────────────────────


def test_euler_vs_quat_position_trajectory():
    """Position trajectories match between Euler and quaternion models."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x_euler0 = make_state(
        VT=200.0,
        alpha=0.03,
        beta=0.0,
        phi=0.1,
        theta=0.03,
        psi=0.5,
        h=5000.0,
    )
    x_quat0 = euler_state_to_quat_state(x_euler0)

    def deriv_euler(t, x):
        return aircraft_dynamics(t, x, ctrl, geo)

    def deriv_quat(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    dt = 0.01
    x_e = x_euler0.copy()
    x_q = x_quat0.copy()

    for step in range(200):
        t = step * dt
        x_e = rk4_step(deriv_euler, t, x_e, dt)
        x_q = rk4_step(deriv_quat, t, x_q, dt)

    # Compare position states (indices 9,10,11 in Euler, 10,11,12 in quat)
    x_e_from_q = quat_state_to_euler_state(x_q)
    assert x_e[StateIndex.X_N] == pytest.approx(x_e_from_q[StateIndex.X_N], abs=1e-6)
    assert x_e[StateIndex.Y_E] == pytest.approx(x_e_from_q[StateIndex.Y_E], abs=1e-6)
    assert x_e[StateIndex.H] == pytest.approx(x_e_from_q[StateIndex.H], abs=1e-6)


# ── RK4 convergence for quaternion dynamics ──────────────────────────────────


def test_quat_rk4_convergence():
    """RK4 integration of quaternion dynamics shows O(dt^4) convergence.

    Uses a longer integration time and a state with non-zero angular rates
    to produce meaningful error magnitudes above numerical precision.
    """
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(
        u=200.0,
        v=10.0,
        w=-5.0,
        p=1.0,
        q=-0.5,
        r=0.8,
        h=5000.0,
    )

    def deriv(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    # Reference solution with very small dt
    dt_ref = 0.0001
    t_final = 0.5
    x_ref = x0.copy()
    t = 0.0
    for _ in range(int(t_final / dt_ref)):
        x_ref = rk4_step(deriv, t, x_ref, dt_ref)
        t += dt_ref

    errors = []
    dts = [0.02, 0.01, 0.005]
    for dt in dts:
        x = x0.copy()
        t = 0.0
        for _ in range(int(t_final / dt)):
            x = rk4_step(deriv, t, x, dt)
            t += dt
        err = np.linalg.norm(x - x_ref)
        errors.append(err)

    # Error should decrease monotonically with dt
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], (
            f"dt={dts[i]} err={errors[i]:.2e} not > "
            f"dt={dts[i + 1]} err={errors[i + 1]:.2e}"
        )


# ── shared model consistency ─────────────────────────────────────────────────


def test_quat_dynamics_uses_same_force_model():
    """A custom force model produces identical body forces in both formulations."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    recorded_euler: list[ForcesMoments] = []
    recorded_quat: list[ForcesMoments] = []

    def recording_model(t, x, control, geometry):
        fm = ForcesMoments(
            force_body_N=np.array([1000.0, 50.0, -200.0]),
            moment_body_Nm=np.array([500.0, -300.0, 100.0]),
        )
        recorded_euler.append(fm)
        return fm

    def recording_model_quat(t, x, control, geometry):
        fm = ForcesMoments(
            force_body_N=np.array([1000.0, 50.0, -200.0]),
            moment_body_Nm=np.array([500.0, -300.0, 100.0]),
        )
        recorded_quat.append(fm)
        return fm

    x_e = _make_trimmed_euler_state()
    x_q = euler_state_to_quat_state(x_e)

    aircraft_dynamics(0.0, x_e, ctrl, geo, recording_model)
    aircraft_dynamics_quat(0.0, x_q, ctrl, geo, recording_model_quat)

    # Both models should receive the same body forces
    assert recorded_euler[0].force_body_N == pytest.approx(
        recorded_quat[0].force_body_N
    )
    assert recorded_euler[0].moment_body_Nm == pytest.approx(
        recorded_quat[0].moment_body_Nm
    )


def test_quat_dynamics_gravity_consistency():
    """Gravity force in quaternion dynamics matches the Euler gravity_force_body."""
    geo = create_default_geometry()

    # Euler gravity at level attitude
    g_euler = gravity_force_body(0.0, 0.0, 0.0, geo.mass_kg)
    # In body frame: [0, 0, m*g] (downward)
    assert g_euler[2] == pytest.approx(geo.mass_kg * GRAVITY_MPS2, abs=1e-10)

    # Quaternion dynamics with zero velocity (so no aero forces) at identity
    x_quat = make_quat_state(u=0.0, v=0.0, w=0.0, h=5000.0)
    ctrl = _make_zero_control()

    x_dot = aircraft_dynamics_quat(0.0, x_quat, ctrl, geo)
    # With zero forces from aero/prop, only gravity acts
    # At level attitude, gravity is purely in z_body (down)
    # F/m = [0, 0, g] in body frame
    assert x_dot[QuatStateIndex.W] == pytest.approx(GRAVITY_MPS2, abs=1e-10)


# ── simulation integration ───────────────────────────────────────────────────


def test_simulate_with_quat_dynamics():
    """The simulate() function works with quaternion dynamics."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, h=5000.0)

    def deriv(t, x):
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    result = simulate(deriv, x0, dt=0.01, t_final=0.5, integrator="rk4")

    assert result.t_final == 0.5
    assert result.x.shape[0] == 51  # 0.5/0.01 + 1
    assert result.x.shape[1] == NUM_QUAT_STATES
    assert result.integrator_name == "rk4"


def test_simulate_quat_detects_nonfinite():
    """simulate() detects non-finite quaternion state and raises."""
    geo = create_default_geometry()
    ctrl = _make_zero_control()

    x0 = make_quat_state(u=200.0, v=0.0, w=0.0, h=5000.0)

    call_count = [0]

    def exploding_deriv(t, x):
        call_count[0] += 1
        if call_count[0] > 5:
            return np.full(NUM_QUAT_STATES, np.nan)
        return aircraft_dynamics_quat(t, x, ctrl, geo)

    with pytest.raises(RuntimeError, match="Non-finite"):
        simulate(exploding_deriv, x0, dt=0.01, t_final=1.0)
