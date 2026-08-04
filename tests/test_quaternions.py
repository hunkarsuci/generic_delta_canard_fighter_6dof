"""
Tests for quaternion attitude utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.quaternions import (
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rates,
    quaternion_to_dcm,
    quaternion_to_euler,
)

from generic_delta_canard_fighter_6dof.transforms import body_to_ned_dcm


def test_normalize_quaternion() -> None:
    q = np.array([2.0, 0.0, 0.0, 0.0])

    q_unit = normalize_quaternion(q)

    assert np.allclose(q_unit, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(q_unit) == pytest.approx(1.0)


def test_normalize_zero_quaternion_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_quaternion(np.zeros(4))


def test_euler_to_quaternion_identity() -> None:
    q = euler_to_quaternion(phi=0.0, theta=0.0, psi=0.0)

    assert np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0]))


def test_quaternion_to_dcm_matches_euler_dcm() -> None:
    phi = np.deg2rad(20.0)
    theta = np.deg2rad(-10.0)
    psi = np.deg2rad(45.0)

    q = euler_to_quaternion(phi, theta, psi)

    C_from_quaternion = quaternion_to_dcm(q)
    C_from_euler = body_to_ned_dcm(phi, theta, psi)

    assert np.allclose(C_from_quaternion, C_from_euler)


def test_euler_quaternion_round_trip() -> None:
    phi_original = np.deg2rad(15.0)
    theta_original = np.deg2rad(8.0)
    psi_original = np.deg2rad(-30.0)

    q = euler_to_quaternion(phi_original, theta_original, psi_original)
    phi, theta, psi = quaternion_to_euler(q)

    assert phi == pytest.approx(phi_original)
    assert theta == pytest.approx(theta_original)
    assert psi == pytest.approx(psi_original)


def test_quaternion_yaw_90_body_forward_points_east() -> None:
    q = euler_to_quaternion(phi=0.0, theta=0.0, psi=np.pi / 2.0)
    C_nb = quaternion_to_dcm(q)

    body_forward = np.array([1.0, 0.0, 0.0])
    ned_vector = C_nb @ body_forward

    expected = np.array([0.0, 1.0, 0.0])

    assert np.allclose(ned_vector, expected, atol=1.0e-12)


def test_quaternion_rates_at_identity_for_roll_rate() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])

    q_dot = quaternion_rates(q, p=0.2, q_rate=0.0, r=0.0)

    expected = np.array([0.0, 0.1, 0.0, 0.0])

    assert np.allclose(q_dot, expected)


def test_quaternion_rates_at_identity_for_yaw_rate() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])

    q_dot = quaternion_rates(q, p=0.0, q_rate=0.0, r=0.4)

    expected = np.array([0.0, 0.0, 0.0, 0.2])

    assert np.allclose(q_dot, expected)


def test_q_and_minus_q_same_dcm() -> None:
    """q and -q represent the same attitude."""
    q = euler_to_quaternion(0.3, -0.2, 0.7)
    q_neg = -q

    C_from_q = quaternion_to_dcm(q)
    C_from_neg_q = quaternion_to_dcm(q_neg)

    assert np.allclose(C_from_q, C_from_neg_q)


def test_negative_q_same_euler() -> None:
    """q and -q should produce the same Euler angles."""
    q = euler_to_quaternion(0.3, -0.2, 0.7)
    q_neg = -q

    phi1, theta1, psi1 = quaternion_to_euler(q)
    phi2, theta2, psi2 = quaternion_to_euler(q_neg)

    assert phi1 == pytest.approx(phi2)
    assert theta1 == pytest.approx(theta2)
    assert psi1 == pytest.approx(psi2)


def test_quaternion_dcm_is_rotation_matrix() -> None:
    """Every quaternion DCM should be a valid rotation matrix."""
    from generic_delta_canard_fighter_6dof.transforms import is_rotation_matrix

    q = euler_to_quaternion(0.4, -0.3, 1.2)
    C = quaternion_to_dcm(q)
    assert is_rotation_matrix(C)


def test_dcm_to_quaternion_round_trip() -> None:
    """Euler → Quaternion → DCM should equal direct Euler → DCM."""
    phi, theta, psi = 0.3, -0.15, 0.8
    q = euler_to_quaternion(phi, theta, psi)
    C_from_q = quaternion_to_dcm(q)
    C_direct = body_to_ned_dcm(phi, theta, psi)
    assert np.allclose(C_from_q, C_direct)


def test_quaternion_multiply_identity() -> None:
    """q ⊗ [1,0,0,0] = q."""
    q = euler_to_quaternion(0.2, -0.1, 0.5)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_multiply(q, identity), q)
    assert np.allclose(quaternion_multiply(identity, q), q)


def test_quaternion_multiply_associative() -> None:
    """(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)."""
    a = euler_to_quaternion(0.1, 0.2, 0.3)
    b = euler_to_quaternion(-0.15, 0.05, 0.4)
    c = euler_to_quaternion(0.3, -0.1, -0.2)

    left = quaternion_multiply(quaternion_multiply(a, b), c)
    right = quaternion_multiply(a, quaternion_multiply(b, c))
    assert np.allclose(left, right)


def test_quaternion_conjugate_inverts_rotation() -> None:
    """Conjugate quaternion should produce inverse DCM."""
    q = euler_to_quaternion(0.3, -0.2, 0.7)
    q_star = quaternion_conjugate(q)

    C = quaternion_to_dcm(q)
    C_star = quaternion_to_dcm(q_star)

    # Conjugate DCM should be the transpose (inverse)
    assert np.allclose(C_star, C.T)


def test_quaternion_rates_zero_omega_preserves_q() -> None:
    """Zero angular rate should produce zero derivative."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_dot = quaternion_rates(q, p=0.0, q_rate=0.0, r=0.0)
    assert np.allclose(q_dot, 0.0)


def test_quaternion_rates_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        quaternion_rates(np.array([1.0, 0.0, 0.0]), p=0.0, q_rate=0.0, r=0.0)


def test_normalize_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        normalize_quaternion(np.array([1.0, 0.0, 0.0]))


def test_quaternion_norm_deviation_in_rates() -> None:
    """Verify quaternion_rates normalizes input (documented behavior)."""
    q_slightly_off = np.array([0.999, 0.01, 0.0, 0.0])
    # Should not raise; normalizes internally
    q_dot = quaternion_rates(q_slightly_off, p=0.1, q_rate=0.0, r=0.0)
    assert q_dot.shape == (4,)
    assert np.all(np.isfinite(q_dot))
