"""
Tests for coordinate transformation functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.transforms import (
    body_to_ned_dcm,
    is_rotation_matrix,
    ned_to_body_dcm,
)


def test_body_to_ned_identity_at_zero_angles() -> None:
    C_nb = body_to_ned_dcm(phi=0.0, theta=0.0, psi=0.0)

    assert np.allclose(C_nb, np.eye(3))


def test_body_to_ned_is_rotation_matrix() -> None:
    C_nb = body_to_ned_dcm(phi=0.2, theta=-0.1, psi=0.7)

    assert is_rotation_matrix(C_nb)


def test_ned_to_body_is_transpose_inverse() -> None:
    phi = 0.3
    theta = 0.2
    psi = -0.5

    C_nb = body_to_ned_dcm(phi, theta, psi)
    C_bn = ned_to_body_dcm(phi, theta, psi)

    assert np.allclose(C_bn, C_nb.T)
    assert np.allclose(C_nb @ C_bn, np.eye(3))
    assert np.allclose(C_bn @ C_nb, np.eye(3))


def test_yaw_90_degrees_body_forward_points_east() -> None:
    C_nb = body_to_ned_dcm(phi=0.0, theta=0.0, psi=np.pi / 2.0)

    body_forward = np.array([1.0, 0.0, 0.0])
    ned_vector = C_nb @ body_forward

    expected = np.array([0.0, 1.0, 0.0])

    assert np.allclose(ned_vector, expected, atol=1.0e-12)


def test_positive_pitch_body_forward_has_negative_down_component() -> None:
    theta = np.deg2rad(10.0)
    C_nb = body_to_ned_dcm(phi=0.0, theta=theta, psi=0.0)

    body_forward = np.array([1.0, 0.0, 0.0])
    ned_vector = C_nb @ body_forward

    expected = np.array([np.cos(theta), 0.0, -np.sin(theta)])

    assert np.allclose(ned_vector, expected)


def test_invalid_rotation_matrix_shape() -> None:
    bad_matrix = np.eye(2)

    assert not is_rotation_matrix(bad_matrix)


def test_non_rotation_matrix_rejected() -> None:
    bad_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert not is_rotation_matrix(bad_matrix)


def test_dcm_determinant_is_positive_one() -> None:
    """Every valid DCM must have determinant +1."""
    for phi, theta, psi in [
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 0.7),
        (np.pi / 4, np.pi / 6, -np.pi / 3),
        (1.5, -0.8, 2.1),
    ]:
        C = body_to_ned_dcm(phi, theta, psi)
        assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-12)


def test_dcm_columns_are_orthonormal() -> None:
    """Each column of the DCM should be a unit vector, mutually orthogonal."""
    C = body_to_ned_dcm(phi=0.4, theta=-0.3, psi=1.2)
    for i in range(3):
        assert np.linalg.norm(C[:, i]) == pytest.approx(1.0)
        for j in range(i + 1, 3):
            assert np.dot(C[:, i], C[:, j]) == pytest.approx(0.0, abs=1e-12)


def test_yaw_180_degrees() -> None:
    """180-degree yaw: body forward points NED south, body right points west."""
    C = body_to_ned_dcm(phi=0.0, theta=0.0, psi=np.pi)
    forward_ned = C @ np.array([1.0, 0.0, 0.0])
    right_ned = C @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(forward_ned, [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(right_ned, [0.0, -1.0, 0.0], atol=1e-12)


def test_roll_90_body_down_to_west() -> None:
    """90-degree right roll: body z (down) points NED west (negative y_E)."""
    C = body_to_ned_dcm(phi=np.pi / 2, theta=0.0, psi=0.0)
    body_down = np.array([0.0, 0.0, 1.0])
    ned = C @ body_down
    # Right roll: right wing down, left wing up, body z tilts toward left (west)
    assert np.allclose(ned, [0.0, -1.0, 0.0], atol=1e-12)


def test_body_to_ned_and_ned_to_body_round_trip_vector() -> None:
    """v_body → v_ned → v_body should recover the original vector."""
    phi, theta, psi = 0.2, -0.15, 0.8
    v_body = np.array([120.0, 15.0, -5.0])
    C_nb = body_to_ned_dcm(phi, theta, psi)
    C_bn = ned_to_body_dcm(phi, theta, psi)
    v_roundtrip = C_bn @ (C_nb @ v_body)
    assert np.allclose(v_roundtrip, v_body)


def test_composition_two_yaws() -> None:
    """Two successive yaws compose as a single yaw (no pitch/roll coupling)."""
    psi1, psi2 = 0.3, 0.5
    C1 = body_to_ned_dcm(0, 0, psi1)
    # Simpler independent check: apply yaw psi2 to body vector, then psi1
    v_body = np.array([1.0, 0.0, 0.0])
    v_intermediate = np.array(
        [
            np.cos(psi2),
            np.sin(psi2),
            0.0,
        ]
    )
    v_final = C1 @ v_intermediate
    v_direct = body_to_ned_dcm(0, 0, psi1 + psi2) @ v_body
    assert np.allclose(v_final, v_direct)
