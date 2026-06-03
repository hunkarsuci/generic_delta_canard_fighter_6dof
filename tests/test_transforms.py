"""
Tests for coordinate transformation functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.transforms import (
    body_to_nav_dcm,
    body_to_ned_dcm,
    is_rotation_matrix,
    nav_to_body_dcm,
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


def test_alias_functions_match_primary_functions() -> None:
    phi = 0.1
    theta = 0.2
    psi = 0.3

    assert np.allclose(body_to_nav_dcm(phi, theta, psi), body_to_ned_dcm(phi, theta, psi))
    assert np.allclose(nav_to_body_dcm(phi, theta, psi), ned_to_body_dcm(phi, theta, psi))


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