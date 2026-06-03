"""
Aircraft kinematics utilities.

This module handles:

1. Euler angle kinematics
2. Conversion between wind variables and body-axis velocity
3. Position kinematics helper functions

The state vector uses:

    VT, alpha, beta, p, q, r, phi, theta, psi, x_N, y_E, h

where VT, alpha, and beta are often more useful for aircraft aerodynamics
than direct body velocities u, v, w.
"""

from __future__ import annotations

import numpy as np

from generic_delta_canard_fighter_6dof.constants import EPSILON
from generic_delta_canard_fighter_6dof.transforms import body_to_ned_dcm


def euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """
    Compute Euler angle rates from body angular rates.

    Parameters
    ----------
    phi:
        Roll angle [rad].
    theta:
        Pitch angle [rad].
    p:
        Body roll rate [rad/s].
    q:
        Body pitch rate [rad/s].
    r:
        Body yaw rate [rad/s].

    Returns
    -------
    np.ndarray
        Array [phi_dot, theta_dot, psi_dot] in [rad/s].

    Notes
    -----
    The standard 3-2-1 Euler kinematic equations are:

        phi_dot   = p + q sin(phi) tan(theta) + r cos(phi) tan(theta)
        theta_dot = q cos(phi) - r sin(phi)
        psi_dot   = q sin(phi)/cos(theta) + r cos(phi)/cos(theta)

    These equations become singular when cos(theta) = 0, which happens at
    theta = +/- 90 degrees.
    """
    cos_theta = np.cos(theta)

    if abs(cos_theta) < EPSILON:
        raise ValueError(
            "Euler angle singularity: theta is too close to +/- 90 degrees."
        )

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)

    phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = (q * sin_phi + r * cos_phi) / cos_theta

    return np.array([phi_dot, theta_dot, psi_dot], dtype=float)


def wind_to_body_velocity(VT: float, alpha: float, beta: float) -> np.ndarray:
    """
    Convert wind variables to body-axis velocity components.

    Parameters
   
    VT:
        Total airspeed [m/s].
    alpha:
        Angle of attack [rad].
    beta:
        Sideslip angle [rad].

    Returns

    np.ndarray
        Body-axis velocity vector [u, v, w] in [m/s].

    Notes
   
    Body velocity convention:

        u = forward velocity
        v = right velocity
        w = downward velocity

    Conversion:

        u = VT cos(alpha) cos(beta)
        v = VT sin(beta)
        w = VT sin(alpha) cos(beta)
    """
    if VT < 0.0:
        raise ValueError("Total airspeed VT must be non-negative.")

    u = VT * np.cos(alpha) * np.cos(beta)
    v = VT * np.sin(beta)
    w = VT * np.sin(alpha) * np.cos(beta)

    return np.array([u, v, w], dtype=float)


def body_to_wind_angles(u: float, v: float, w: float) -> tuple[float, float, float]:
    """
    Convert body-axis velocity components to wind variables.

    Parameters

    u:
        Forward body-axis velocity [m/s].
    v:
        Right body-axis velocity [m/s].
    w:
        Downward body-axis velocity [m/s].

    Returns
 
    tuple[float, float, float]
        Tuple (VT, alpha, beta), where:

            VT    = total airspeed [m/s]
            alpha = angle of attack [rad]
            beta  = sideslip angle [rad]

    Notes

    The inverse conversion is:

        VT    = sqrt(u^2 + v^2 + w^2)
        alpha = atan2(w, u)
        beta  = asin(v / VT)

    At zero airspeed, alpha and beta are undefined. For numerical safety,
    this function returns alpha = 0 and beta = 0 when VT is near zero.
    """
    VT = float(np.sqrt(u * u + v * v + w * w))

    if VT < EPSILON:
        return 0.0, 0.0, 0.0

    alpha = float(np.arctan2(w, u))

    beta_argument = np.clip(v / VT, -1.0, 1.0)
    beta = float(np.arcsin(beta_argument))

    return VT, alpha, beta


def body_velocity_to_ned_velocity(
    u: float,
    v: float,
    w: float,
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Convert body-axis velocity components to NED velocity components.

    Returns

    np.ndarray
        Velocity vector [V_N, V_E, V_D] in [m/s].
    """
    velocity_body = np.array([u, v, w], dtype=float)
    C_nb = body_to_ned_dcm(phi, theta, psi)

    return C_nb @ velocity_body


def altitude_rate_from_down_velocity(V_D: float) -> float:
    """
    Convert NED down velocity to altitude rate.

    In the NED frame, positive V_D means moving downward.
    In our state vector, altitude h is positive upward.

    Therefore:

        h_dot = -V_D
    """
    return -V_D