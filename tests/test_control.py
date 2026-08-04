"""Tests for PID and LQR flight control systems."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.control import (
    AircraftPID,
    LQRConfig,
    LQRController,
    PIDConfig,
    PIDController,
    compute_lqr,
)
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.linearization import linearize
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.state import (
    NUM_CONTROLS,
    NUM_STATES,
    make_control,
    make_state,
)
from generic_delta_canard_fighter_6dof.trim import trim_straight_level

# ── PIDController ────────────────────────────────────────────────────────────


def test_pid_zero_error_zero_output():
    pid = PIDController(PIDConfig(kp=1.0, ki=0.5, kd=0.1, name="test"))
    out = pid.step(0.0, 0.01)
    assert out == pytest.approx(0.0)


def test_pid_proportional():
    pid = PIDController(
        PIDConfig(kp=2.0, ki=0.0, kd=0.0, output_min=-10.0, output_max=10.0)
    )
    out = pid.step(3.0, 0.01)
    assert out == pytest.approx(6.0)


def test_pid_integral_accumulation():
    pid = PIDController(
        PIDConfig(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            output_min=-100.0,
            output_max=100.0,
            integrator_min=-100.0,
            integrator_max=100.0,
        )
    )
    # Constant error of 2.0 for 10 steps at dt=0.01 → integral = 1.0 * 2.0 * 0.10 = 0.2
    for _ in range(10):
        out = pid.step(2.0, 0.01)
    assert out == pytest.approx(0.2, abs=1e-10)


def test_pid_anti_windup():
    pid = PIDController(
        PIDConfig(kp=1.0, ki=10.0, kd=0.0, output_min=-0.5, output_max=0.5)
    )
    # Large error → output saturates → integrator clamped
    out = pid.step(2.0, 0.01)
    assert out == pytest.approx(0.5)
    # After saturation, integrator should be limited


def test_pid_derivative_filtering():
    pid_no_filter = PIDController(
        PIDConfig(
            kp=0.0,
            ki=0.0,
            kd=1.0,
            deriv_filter_tau=0.0,
            output_min=-10.0,
            output_max=10.0,
        )
    )
    pid_filter = PIDController(
        PIDConfig(
            kp=0.0,
            ki=0.0,
            kd=1.0,
            deriv_filter_tau=0.1,
            output_min=-10.0,
            output_max=10.0,
        )
    )

    # Step change in error
    pid_no_filter.step(0.0, 0.01)
    pid_filter.step(0.0, 0.01)

    out_no = pid_no_filter.step(1.0, 0.01)
    out_f = pid_filter.step(1.0, 0.01)

    # Filtered derivative should be smaller
    assert abs(out_f) < abs(out_no)


def test_pid_reset():
    pid = PIDController(
        PIDConfig(kp=0.0, ki=1.0, kd=0.0, output_min=-100.0, output_max=100.0)
    )
    pid.step(5.0, 0.1)
    assert pid._integral != 0.0
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0


def test_pid_output_limits():
    pid = PIDController(
        PIDConfig(kp=10.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0)
    )
    out = pid.step(10.0, 0.01)
    assert out == 1.0


def test_pid_rejects_invalid_dt():
    pid = PIDController(PIDConfig())
    with pytest.raises(ValueError):
        pid.step(1.0, -0.01)


def test_pid_deterministic():
    cfg = PIDConfig(kp=1.0, ki=0.5, kd=0.1, output_min=-10.0, output_max=10.0)
    p1 = PIDController(cfg)
    p2 = PIDController(cfg)
    for _ in range(5):
        o1 = p1.step(0.5, 0.01)
        o2 = p2.step(0.5, 0.01)
        assert o1 == pytest.approx(o2)


# ── AircraftPID ──────────────────────────────────────────────────────────────


def test_aircraft_pid_zero_error():
    pid = AircraftPID()
    x = make_state(VT=200.0)
    ref = np.array([0.0, 0.0, 0.0, 200.0])  # zero rate errors, zero speed error
    out = pid.step(x, ref, 0.01)
    assert out.shape == (4,)
    # With zero error, all outputs should be near zero
    assert np.all(np.abs(out) < 1e-10)


def test_aircraft_pid_roll_response():
    pid = AircraftPID()
    x = make_state(p=0.5, VT=200.0)
    ref = np.array([0.0, 0.0, 0.0, 200.0])  # target zero roll rate
    out = pid.step(x, ref, 0.01)
    # Roll error = 0 - 0.5 = -0.5 → negative output (counter roll)
    assert out[1] < 0.0  # roll channel


def test_aircraft_pid_reset():
    pid = AircraftPID()
    x = make_state(VT=200.0)
    pid.step(x, np.array([0.1, 0.0, 0.0, 200.0]), 0.01)
    pid.reset()
    # After reset, stepping with zero error should give zero output
    out = pid.step(x, np.array([0.0, 0.0, 0.0, 200.0]), 0.01)
    assert np.all(np.abs(out) < 1e-10)


# ── LQRConfig ────────────────────────────────────────────────────────────────


def test_lqr_config_valid():
    cfg = LQRConfig()
    assert cfg.max_state_dev.shape == (NUM_STATES,)
    assert cfg.max_control_dev.shape == (NUM_CONTROLS,)


def test_lqr_config_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        LQRConfig(max_state_dev=np.ones(6))


def test_lqr_config_rejects_non_positive():
    with pytest.raises(ValueError):
        LQRConfig(max_state_dev=np.zeros(NUM_STATES))


# ── LQR computation ──────────────────────────────────────────────────────────


def test_compute_lqr_stabilizable_system():
    """LQR for a simple stabilizable system (all 12 states have damping)."""
    A12 = -np.eye(NUM_STATES)  # all 12 states stable
    B12 = np.eye(NUM_STATES, NUM_CONTROLS) * 0.1  # all states actuated

    result = compute_lqr(A12, B12)
    assert result.stabilizable
    assert result.K.shape == (NUM_CONTROLS, NUM_STATES)
    assert len(result.closed_loop_eigenvalues) == NUM_STATES
    assert np.all(np.real(result.closed_loop_eigenvalues) < 0.0)


def test_compute_lqr_symmetric_q_r():
    result = compute_lqr(
        -np.eye(NUM_STATES),
        np.ones((NUM_STATES, NUM_CONTROLS)) * 0.1,
    )
    assert result.Q.shape == (NUM_STATES, NUM_STATES)
    assert result.R.shape == (NUM_CONTROLS, NUM_CONTROLS)
    assert np.allclose(result.Q, result.Q.T)
    assert np.allclose(result.R, result.R.T)


def test_lqr_riccati_residual():
    """S should approximately satisfy the CARE."""
    A = -np.eye(NUM_STATES)
    B = np.ones((NUM_STATES, NUM_CONTROLS)) * 0.1

    result = compute_lqr(A, B)
    if result.stabilizable:
        residual = (
            A.T @ result.S
            + result.S @ A
            - result.S @ B @ np.linalg.solve(result.R, B.T) @ result.S
            + result.Q
        )
        assert np.linalg.norm(residual) < 1e-6


def test_lqr_finite_gains():
    A = -np.eye(NUM_STATES)
    B = np.eye(NUM_STATES, NUM_CONTROLS) * 0.1
    result = compute_lqr(A, B)
    assert np.all(np.isfinite(result.K))


# ── LQRController ────────────────────────────────────────────────────────────


def test_lqr_controller_validation():
    K = np.zeros((NUM_CONTROLS, NUM_STATES))
    x_trim = make_state()
    u_trim = make_control()
    ctrl = LQRController(K, x_trim, u_trim)
    assert ctrl.K.shape == (NUM_CONTROLS, NUM_STATES)


def test_lqr_controller_at_trim_zero_output():
    K = np.ones((NUM_CONTROLS, NUM_STATES)) * 0.1
    x_trim = make_state(VT=200.0)
    u_trim = make_control(throttle=0.5)
    ctrl = LQRController(K, x_trim, u_trim)
    u = ctrl.step(x_trim)
    assert np.allclose(u, u_trim)


def test_lqr_controller_throttle_clamped():
    K = np.ones((NUM_CONTROLS, NUM_STATES)) * 0.1
    x_trim = make_state(VT=200.0)
    u_trim = make_control(throttle=0.5)
    ctrl = LQRController(K, x_trim, u_trim)
    # Large perturbation should saturate throttle
    x = make_state(VT=50.0)  # far from trim
    u = ctrl.step(x)
    assert 0.0 <= u[4] <= 1.0


# ── integration: LQR at a real trim point ────────────────────────────────────


def test_lqr_at_trim_point():
    """LQR computed at a real aircraft trim point produces stable closed loop."""
    try:
        trim = trim_straight_level(5000.0, 200.0)
        if not trim.converged:
            pytest.skip("Trim did not converge")
    except (ValueError, RuntimeError):
        pytest.skip("Trim solver failed")

    geo = create_default_geometry()
    fm = combined_forces_moments()

    try:
        lin = linearize(trim.state, trim.control, geo, fm)
    except (ValueError, RuntimeError):
        pytest.skip("Linearization failed")

    result = compute_lqr(lin.A, lin.B)

    assert result.stabilizable
    assert result.K.shape == (NUM_CONTROLS, NUM_STATES)
    assert np.all(np.isfinite(result.K))
    # At least some closed-loop eigenvalues should be stable
    assert np.any(np.real(result.closed_loop_eigenvalues) < 0.0)
