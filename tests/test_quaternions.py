"""
Tests for quaternion attitude utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.quaternions import (
    euler_to_quaternion,
    normalize_quaternion,
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