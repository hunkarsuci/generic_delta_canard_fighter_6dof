"""
State and control vector definitions for the generic delta-canard fighter.
State vector:

    x = [
        VT, alpha, beta,
        p, q, r,
        phi, theta, psi,
        x_N, y_E, h
    ]

Control vector:

    u = [
        delta_canard,
        delta_elevon_left,
        delta_elevon_right,
        delta_rudder,
        throttle
    ]

All angles are in radians.
All angular rates are in radians per second.
"""

from __future__ import annotations
from enum import IntEnum
import numpy as np


class StateIndex(IntEnum):
    """Indices of the 12-state aircraft vector."""

    VT = 0
    ALPHA = 1
    BETA = 2

    P = 3
    Q = 4
    R = 5

    PHI = 6
    THETA = 7
    PSI = 8

    X_N = 9
    Y_E = 10
    H = 11


class ControlIndex(IntEnum):
    """Indices of the aircraft control vector."""

    DELTA_CANARD = 0
    DELTA_ELEVON_LEFT = 1
    DELTA_ELEVON_RIGHT = 2
    DELTA_RUDDER = 3
    THROTTLE = 4


NUM_STATES: int = 12
NUM_CONTROLS: int = 5


def make_state(
    VT: float = 0.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    phi: float = 0.0,
    theta: float = 0.0,
    psi: float = 0.0,
    x_N: float = 0.0,
    y_E: float = 0.0,
    h: float = 0.0,
) -> np.ndarray:
    """
    Create a 12-element aircraft state vector.
    """
    x = np.zeros(NUM_STATES, dtype=float)

    x[StateIndex.VT] = VT
    x[StateIndex.ALPHA] = alpha
    x[StateIndex.BETA] = beta

    x[StateIndex.P] = p
    x[StateIndex.Q] = q
    x[StateIndex.R] = r

    x[StateIndex.PHI] = phi
    x[StateIndex.THETA] = theta
    x[StateIndex.PSI] = psi

    x[StateIndex.X_N] = x_N
    x[StateIndex.Y_E] = y_E
    x[StateIndex.H] = h

    validate_state(x)

    return x


def make_control(
    delta_canard: float = 0.0,
    delta_elevon_left: float = 0.0,
    delta_elevon_right: float = 0.0,
    delta_rudder: float = 0.0,
    throttle: float = 0.0,
) -> np.ndarray:
    """
    Create a 5-element aircraft control vector.
    """
    u = np.zeros(NUM_CONTROLS, dtype=float)

    u[ControlIndex.DELTA_CANARD] = delta_canard
    u[ControlIndex.DELTA_ELEVON_LEFT] = delta_elevon_left
    u[ControlIndex.DELTA_ELEVON_RIGHT] = delta_elevon_right
    u[ControlIndex.DELTA_RUDDER] = delta_rudder
    u[ControlIndex.THROTTLE] = throttle

    validate_control(u)

    return u

# we add here validation to check state numbers and airspeed has to be non-negative.

def validate_state(x: np.ndarray) -> None:
    """
    Validate that the state vector has correct size and valid numerical values.
    """
    x = np.asarray(x)

    if x.shape != (NUM_STATES,):
        raise ValueError(f"State vector must have shape ({NUM_STATES},), got {x.shape}.")

    if not np.all(np.isfinite(x)):
        raise ValueError("State vector contains non-finite values.")

    if x[StateIndex.VT] < 0.0:
        raise ValueError("Total airspeed VT must be non-negative.")


def validate_control(u: np.ndarray) -> None:
    """
    Validate that the control vector has correct size and valid numerical values.
    """
    u = np.asarray(u)

    if u.shape != (NUM_CONTROLS,):
        raise ValueError(f"Control vector must have shape ({NUM_CONTROLS},), got {u.shape}.")

    if not np.all(np.isfinite(u)):
        raise ValueError("Control vector contains non-finite values.")

    throttle = u[ControlIndex.THROTTLE]

    if throttle < 0.0 or throttle > 1.0:
        raise ValueError("Throttle must be between 0 and 1.")


def state_to_dict(x: np.ndarray) -> dict[str, float]:
    """
    Convert a state vector into a readable dictionary.
    """
    validate_state(x)

    return {
        "VT": float(x[StateIndex.VT]),
        "alpha": float(x[StateIndex.ALPHA]),
        "beta": float(x[StateIndex.BETA]),
        "p": float(x[StateIndex.P]),
        "q": float(x[StateIndex.Q]),
        "r": float(x[StateIndex.R]),
        "phi": float(x[StateIndex.PHI]),
        "theta": float(x[StateIndex.THETA]),
        "psi": float(x[StateIndex.PSI]),
        "x_N": float(x[StateIndex.X_N]),
        "y_E": float(x[StateIndex.Y_E]),
        "h": float(x[StateIndex.H]),
    }


def control_to_dict(u: np.ndarray) -> dict[str, float]:
    """
    Convert a control vector into a readable dictionary.
    """
    validate_control(u)

    return {
        "delta_canard": float(u[ControlIndex.DELTA_CANARD]),
        "delta_elevon_left": float(u[ControlIndex.DELTA_ELEVON_LEFT]),
        "delta_elevon_right": float(u[ControlIndex.DELTA_ELEVON_RIGHT]),
        "delta_rudder": float(u[ControlIndex.DELTA_RUDDER]),
        "throttle": float(u[ControlIndex.THROTTLE]),
    }