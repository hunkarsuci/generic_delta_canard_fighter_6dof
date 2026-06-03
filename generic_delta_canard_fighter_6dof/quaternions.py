"""
Quaternions attitude utilities 

This module provides quaternion tools for fighter-aircraft attitude modeling 

The baseline simulator uses a 12-state Euler-angle model:

    [VT, alpha, beta, p, q, r, phi, theta, psi, x_N, y_E, h]

However, Euler angles become singular at theta = +/- 90 degrees. Fighter aircraft may perform aggressive maneuvers where quaternion attitude
propagation is more robust.

Quaternion convention used here: 

    q = [q0,q1,q2,q3]

where q0 is the scalar part 

The quaternion represents attitude from body fixed frame to NED frame, thereby the corresponding direction cosine matrix transforms vectors as: 

    v_ned = C_nb @ v_body

"""

from __future__ import annotations

import numpy as np 

def normalize_quaternion(q: np.ndarray) -> np.ndarray: 
    """
    Normalize a quaternion to unit length 

    Parameters 

    q:
        Quaternion [q0, q1, q2, q3]
    Returns 
    
    np.ndarray
        Unit quaternion.
    
    """
    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}.")
    
    norm = np.linalg.norm(q)

    if norm <= 0.0: 
        raise ValueError("Cannot normalize a zero quaternion.")
    
    return q / norm 

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of a quaternion.

    For q = [q0, q1, q2, q3], the conjugate is:

        q* = [q0, -q1, -q2, -q3]
    """
    q = normalize_quaternion(q)

    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q_left:
        Left quaternion [q0, q1, q2, q3].
    q_right:
        Right quaternion [q0, q1, q2, q3].

    Returns
    -------
    np.ndarray
        Product q_left ⊗ q_right.
    """
    q_left = np.asarray(q_left, dtype=float)
    q_right = np.asarray(q_right, dtype=float)

    if q_left.shape != (4,):
        raise ValueError(f"q_left must have shape (4,), got {q_left.shape}.")

    if q_right.shape != (4,):
        raise ValueError(f"q_right must have shape (4,), got {q_right.shape}.")

    a0, a1, a2, a3 = q_left
    b0, b1, b2, b3 = q_right

    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        dtype=float,
    )


def euler_to_quaternion(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert 3-2-1 Euler angles to a body-to-NED quaternion.

    Parameters
    ----------
    phi:
        Roll angle [rad].
    theta:
        Pitch angle [rad].
    psi:
        Yaw angle [rad].

    Returns
    -------
    np.ndarray
        Unit quaternion [q0, q1, q2, q3].
    """
    half_phi = 0.5 * phi
    half_theta = 0.5 * theta
    half_psi = 0.5 * psi

    c_phi = np.cos(half_phi)
    s_phi = np.sin(half_phi)

    c_theta = np.cos(half_theta)
    s_theta = np.sin(half_theta)

    c_psi = np.cos(half_psi)
    s_psi = np.sin(half_psi)

    q0 = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    q1 = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    q2 = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    q3 = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return normalize_quaternion(np.array([q0, q1, q2, q3], dtype=float))


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert a body-to-NED quaternion to a direction cosine matrix.

    Returns
    -------
    np.ndarray
        3x3 matrix C_nb such that:

            v_ned = C_nb @ v_body
    """
    q0, q1, q2, q3 = normalize_quaternion(q)

    return np.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=float,
    )


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a body-to-NED quaternion to 3-2-1 Euler angles.

    Returns
    -------
    tuple[float, float, float]
        Tuple (phi, theta, psi) in radians.
    """
    q0, q1, q2, q3 = normalize_quaternion(q)

    phi = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3),
        1.0 - 2.0 * (q1 * q1 + q2 * q2),
    )

    theta_argument = 2.0 * (q0 * q2 - q3 * q1)
    theta_argument = np.clip(theta_argument, -1.0, 1.0)
    theta = np.arcsin(theta_argument)

    psi = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        1.0 - 2.0 * (q2 * q2 + q3 * q3),
    )

    return float(phi), float(theta), float(psi)


def quaternion_rates(
    q: np.ndarray,
    p: float,
    q_rate: float,
    r: float,
) -> np.ndarray:
    """
    Compute quaternion derivative from body angular rates.

    Parameters
    ----------
    q:
        Body-to-NED quaternion [q0, q1, q2, q3].
    p:
        Body roll rate [rad/s].
    q_rate:
        Body pitch rate [rad/s].
    r:
        Body yaw rate [rad/s].

    Returns
    -------
    np.ndarray
        Quaternion derivative q_dot.

    Notes
    -----
    The equation is:

        q_dot = 0.5 * q ⊗ [0, p, q_rate, r]
    """
    q = normalize_quaternion(q)

    omega_quat = np.array([0.0, p, q_rate, r], dtype=float)

    return 0.5 * quaternion_multiply(q, omega_quat)