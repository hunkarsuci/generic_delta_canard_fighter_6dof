"""
Coordinate transformation utilities.

This module handles rotations between the aircraft body frame and the navigation frame

Frame convention used in this project: 

Body frame:
    x_b = forward
    y_b = right wing
    z_b = down

Navigation frame:
    x_N = north
    y_E = east
    z_D = down

Euler angle convention:
    phi   = roll angle  [rad]
    theta = pitch angle [rad]
    psi   = yaw angle   [rad]

"""

from __future__ import annotations

import numpy as np


def body_to_ned_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Return the direction cosine matrix from body frame to NED frame.

    Parameters
    phi:
        Roll angle [rad].
    theta:
        Pitch angle [rad].
    psi:
        Yaw / heading angle [rad].

    Returns
    np.ndarray
        3x3 direction cosine matrix C_nb.

    Notes
    The returned matrix transforms body-frame vectors into NED-frame vectors:

        v_ned = C_nb @ v_body

    This uses the standard aerospace 3-2-1 Euler sequence:
        yaw, then pitch, then roll.
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [
                -s_theta,
                s_phi * c_theta,
                c_phi * c_theta,
            ],
        ],
        dtype=float,
    )

def ned_to_body_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Return the direction cosine matrix from NED frame to body frame.

    Since rotation matrices are orthonormal, the inverse is the transpose:

        C_bn = C_nb.T
    """
    return body_to_ned_dcm(phi, theta, psi).T


def body_to_nav_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Alias for body_to_ned_dcm.

    The project uses NED as the navigation frame internally.
    This alias keeps the name intuitive for higher-level code.
    """
    return body_to_ned_dcm(phi, theta, psi)


def nav_to_body_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Alias for ned_to_body_dcm.
    """
    return ned_to_body_dcm(phi, theta, psi)


def is_rotation_matrix(matrix: np.ndarray, tolerance: float = 1.0e-9) -> bool:
    """
    Check whether a matrix is a valid rotation matrix.

    A valid rotation matrix R satisfies:

        R @ R.T = I
        det(R) = +1

    Parameters
    ----------
    matrix:
        Matrix to check.
    tolerance:
        Numerical tolerance for the checks.

    Returns
    -------
    bool
        True if the matrix is approximately a valid rotation matrix.
    """
    matrix = np.asarray(matrix)

    if matrix.shape != (3, 3):
        return False

    identity = np.eye(3)

    orthonormal_error = np.max(np.abs(matrix @ matrix.T - identity))
    determinant_error = abs(np.linalg.det(matrix) - 1.0)

    return orthonormal_error < tolerance and determinant_error < tolerance