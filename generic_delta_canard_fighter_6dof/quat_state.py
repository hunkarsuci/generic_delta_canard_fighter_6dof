"""
Quaternion-based 13-state vector for the generic delta-canard fighter.

State vector:

    x_quat = [
        u, v, w,           # body-axis velocity [m/s]
        p, q, r,           # body angular rates [rad/s]
        q0, q1, q2, q3,    # body-to-NED quaternion (scalar-first, Hamilton)
        x_N, y_E, h        # NED position [m], h positive up
    ]

This formulation has no Euler-angle singularity and is suitable for
aggressive-maneuver simulation.

The 12-state Euler model (state.py) is preserved for analysis and local
linearization.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

NUM_QUAT_STATES: int = 13


class QuatStateIndex(IntEnum):
    """Indices of the 13-state quaternion aircraft vector."""

    U = 0
    V = 1
    W = 2

    P = 3
    Q = 4
    R = 5

    Q0 = 6
    Q1 = 7
    Q2 = 8
    Q3 = 9

    X_N = 10
    Y_E = 11
    H = 12


def make_quat_state(
    u: float = 0.0,
    v: float = 0.0,
    w: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    q0: float = 1.0,
    q1: float = 0.0,
    q2: float = 0.0,
    q3: float = 0.0,
    x_N: float = 0.0,
    y_E: float = 0.0,
    h: float = 0.0,
) -> np.ndarray:
    """Create a 13-element quaternion-based aircraft state vector."""
    x = np.zeros(NUM_QUAT_STATES, dtype=float)

    x[QuatStateIndex.U] = u
    x[QuatStateIndex.V] = v
    x[QuatStateIndex.W] = w

    x[QuatStateIndex.P] = p
    x[QuatStateIndex.Q] = q
    x[QuatStateIndex.R] = r

    x[QuatStateIndex.Q0] = q0
    x[QuatStateIndex.Q1] = q1
    x[QuatStateIndex.Q2] = q2
    x[QuatStateIndex.Q3] = q3

    x[QuatStateIndex.X_N] = x_N
    x[QuatStateIndex.Y_E] = y_E
    x[QuatStateIndex.H] = h

    _validate_quat_state(x)

    return x


def _validate_quat_state(x: np.ndarray) -> None:
    """Validate the quaternion state vector."""
    x = np.asarray(x)

    if x.shape != (NUM_QUAT_STATES,):
        raise ValueError(
            f"Quaternion state must have shape ({NUM_QUAT_STATES},), got {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Quaternion state contains non-finite values.")


def quat_state_to_dict(x: np.ndarray) -> dict[str, float]:
    """Convert a quaternion state vector to a readable dictionary."""
    _validate_quat_state(x)

    return {
        "u": float(x[QuatStateIndex.U]),
        "v": float(x[QuatStateIndex.V]),
        "w": float(x[QuatStateIndex.W]),
        "p": float(x[QuatStateIndex.P]),
        "q": float(x[QuatStateIndex.Q]),
        "r": float(x[QuatStateIndex.R]),
        "q0": float(x[QuatStateIndex.Q0]),
        "q1": float(x[QuatStateIndex.Q1]),
        "q2": float(x[QuatStateIndex.Q2]),
        "q3": float(x[QuatStateIndex.Q3]),
        "x_N": float(x[QuatStateIndex.X_N]),
        "y_E": float(x[QuatStateIndex.Y_E]),
        "h": float(x[QuatStateIndex.H]),
    }


def quat_state_to_euler_state(x_quat: np.ndarray) -> np.ndarray:
    """Convert a 13-state quaternion vector to the 12-state Euler vector.

    The Euler state stores [VT, alpha, beta, p, q, r, phi, theta, psi, x_N, y_E, h].
    VT, alpha, beta are computed from body u, v, w.
    phi, theta, psi are extracted from the quaternion.
    """
    from generic_delta_canard_fighter_6dof.kinematics import body_to_wind_angles
    from generic_delta_canard_fighter_6dof.quaternions import quaternion_to_euler
    from generic_delta_canard_fighter_6dof.state import (
        NUM_STATES,
        StateIndex,
    )

    _validate_quat_state(x_quat)

    u = float(x_quat[QuatStateIndex.U])
    v = float(x_quat[QuatStateIndex.V])
    w = float(x_quat[QuatStateIndex.W])

    VT, alpha, beta = body_to_wind_angles(u, v, w)

    q = x_quat[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
    phi, theta, psi = quaternion_to_euler(q)

    x_euler = np.zeros(NUM_STATES, dtype=float)

    x_euler[StateIndex.VT] = VT
    x_euler[StateIndex.ALPHA] = alpha
    x_euler[StateIndex.BETA] = beta

    x_euler[StateIndex.P] = float(x_quat[QuatStateIndex.P])
    x_euler[StateIndex.Q] = float(x_quat[QuatStateIndex.Q])
    x_euler[StateIndex.R] = float(x_quat[QuatStateIndex.R])

    x_euler[StateIndex.PHI] = phi
    x_euler[StateIndex.THETA] = theta
    x_euler[StateIndex.PSI] = psi

    x_euler[StateIndex.X_N] = float(x_quat[QuatStateIndex.X_N])
    x_euler[StateIndex.Y_E] = float(x_quat[QuatStateIndex.Y_E])
    x_euler[StateIndex.H] = float(x_quat[QuatStateIndex.H])

    return x_euler


def euler_state_to_quat_state(x_euler: np.ndarray) -> np.ndarray:
    """Convert a 12-state Euler vector to the 13-state quaternion vector."""
    from generic_delta_canard_fighter_6dof.kinematics import wind_to_body_velocity
    from generic_delta_canard_fighter_6dof.quaternions import euler_to_quaternion
    from generic_delta_canard_fighter_6dof.state import StateIndex, validate_state

    validate_state(x_euler)

    VT = float(x_euler[StateIndex.VT])
    alpha = float(x_euler[StateIndex.ALPHA])
    beta = float(x_euler[StateIndex.BETA])

    u, v, w = wind_to_body_velocity(VT, alpha, beta)

    phi = float(x_euler[StateIndex.PHI])
    theta = float(x_euler[StateIndex.THETA])
    psi = float(x_euler[StateIndex.PSI])

    q = euler_to_quaternion(phi, theta, psi)

    return make_quat_state(
        u=float(u),
        v=float(v),
        w=float(w),
        p=float(x_euler[StateIndex.P]),
        q=float(x_euler[StateIndex.Q]),
        r=float(x_euler[StateIndex.R]),
        q0=float(q[0]),
        q1=float(q[1]),
        q2=float(q[2]),
        q3=float(q[3]),
        x_N=float(x_euler[StateIndex.X_N]),
        y_E=float(x_euler[StateIndex.Y_E]),
        h=float(x_euler[StateIndex.H]),
    )
