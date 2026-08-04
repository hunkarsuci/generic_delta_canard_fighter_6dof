"""
Numerical linearization of the 6DOF aircraft dynamics about a trim point.

Computes A = df/dx and B = df/du using central finite differences.

State ordering (12):
    [VT, alpha, beta, p, q, r, phi, theta, psi, x_N, y_E, h]

Input ordering (5):
    [delta_canard, delta_elevon_left, delta_elevon_right, delta_rudder, throttle]

Units per row/column match the state and control definitions:
    states: m/s, rad, rad, rad/s, rad/s, rad/s, rad, rad, rad, m, m, m
    controls: rad, rad, rad, rad, [-]

The linearization uses Euler-angle perturbations directly since the trim
point is away from the Euler singularity (theta = ±90°).

Scale-aware perturbation sizes are used for numerical robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import AircraftGeometry
from generic_delta_canard_fighter_6dof.state import NUM_CONTROLS, NUM_STATES

# Scale-aware perturbation sizes
# Chosen to be ~sqrt(eps) for the most sensitive entries
_DEFAULT_DX = np.array(
    [
        1e-4,  # VT [m/s]
        1e-5,  # alpha [rad]
        1e-5,  # beta [rad]
        1e-5,  # p [rad/s]
        1e-5,  # q [rad/s]
        1e-5,  # r [rad/s]
        1e-5,  # phi [rad]
        1e-5,  # theta [rad]
        1e-5,  # psi [rad]
        1e-3,  # x_N [m]
        1e-3,  # y_E [m]
        1e-3,  # h [m]
    ]
)

_DEFAULT_DU = np.array(
    [
        1e-5,  # delta_canard [rad]
        1e-5,  # delta_elevon_left [rad]
        1e-5,  # delta_elevon_right [rad]
        1e-5,  # delta_rudder [rad]
        1e-5,  # throttle [-]
    ]
)


@dataclass(frozen=True)
class LinearizationResult:
    """Result of numerical linearization about an operating point."""

    A: np.ndarray
    """State matrix (12×12), A[i,j] = d(x_dot_i)/d(x_j)."""

    B: np.ndarray
    """Input matrix (12×5), B[i,j] = d(x_dot_i)/d(u_j)."""

    x0: np.ndarray
    """Operating point state vector (12,)."""

    u0: np.ndarray
    """Operating point control vector (5,)."""

    x_dot0: np.ndarray
    """State derivative at the operating point (12,)."""

    dx: np.ndarray
    """Perturbation sizes used for state [12]."""

    du: np.ndarray
    """Perturbation sizes used for control [5]."""

    method: str = "central"


def linearize(
    x0: np.ndarray,
    u0: np.ndarray,
    geometry: AircraftGeometry,
    force_moment_model,
    *,
    dx: np.ndarray | None = None,
    du: np.ndarray | None = None,
) -> LinearizationResult:
    """Compute A = df/dx and B = df/du via central finite differences.

    Parameters
    ----------
    x0:
        Operating point state [12].
    u0:
        Operating point control [5].
    geometry:
        Aircraft geometry.
    force_moment_model:
        Force/moment model callable.
    dx:
        State perturbation sizes [12]. Uses defaults if None.
    du:
        Control perturbation sizes [5]. Uses defaults if None.

    Returns
    -------
    LinearizationResult
        A, B matrices with metadata.
    """
    if dx is None:
        dx = _DEFAULT_DX.copy()
    if du is None:
        du = _DEFAULT_DU.copy()

    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    dx = np.asarray(dx, dtype=float)
    du = np.asarray(du, dtype=float)

    if x0.shape != (NUM_STATES,):
        raise ValueError(f"x0 must have shape ({NUM_STATES},), got {x0.shape}.")
    if u0.shape != (NUM_CONTROLS,):
        raise ValueError(f"u0 must have shape ({NUM_CONTROLS},), got {u0.shape}.")

    # Reference derivative
    def f(x, u):
        return aircraft_dynamics(0.0, x, u, geometry, force_moment_model)

    f0 = f(x0, u0)

    # ── A matrix: finite differences w.r.t. state ──
    A = np.zeros((NUM_STATES, NUM_STATES))

    for j in range(NUM_STATES):
        e = np.zeros(NUM_STATES)
        e[j] = dx[j]

        x_plus = x0 + e
        x_minus = x0 - e

        f_plus = f(x_plus, u0)
        f_minus = f(x_minus, u0)

        A[:, j] = (f_plus - f_minus) / (2.0 * dx[j])

        # If perturbation goes out of valid range, fall back to forward diff
        if not np.all(np.isfinite(A[:, j])):
            # Try forward difference on the valid side
            try:
                f_forward = f(x_plus, u0)
                A[:, j] = (f_forward - f0) / dx[j]
            except (ValueError, RuntimeError):
                try:
                    f_backward = f(x_minus, u0)
                    A[:, j] = (f0 - f_backward) / dx[j]
                except (ValueError, RuntimeError):
                    A[:, j] = np.nan

    # ── B matrix: finite differences w.r.t. control ──
    B = np.zeros((NUM_STATES, NUM_CONTROLS))

    for j in range(NUM_CONTROLS):
        e = np.zeros(NUM_CONTROLS)
        e[j] = du[j]

        u_plus = u0 + e
        u_minus = u0 - e

        # Clamp throttle to [0, 1]
        u_plus_clamped = u_plus.copy()
        u_minus_clamped = u_minus.copy()
        if j == 4:  # throttle
            u_plus_clamped[4] = np.clip(u_plus[4], 0.0, 1.0)
            u_minus_clamped[4] = np.clip(u_minus[4], 0.0, 1.0)

        f_plus = f(x0, u_plus_clamped)
        f_minus = f(x0, u_minus_clamped)

        # If both perturbations are effectively the same (clamping), use
        # forward difference
        if np.allclose(u_plus_clamped, u_minus_clamped):
            B[:, j] = (f_plus - f0) / du[j]
        else:
            B[:, j] = (f_plus - f_minus) / (2.0 * du[j])

    return LinearizationResult(
        A=A,
        B=B,
        x0=x0.copy(),
        u0=u0.copy(),
        x_dot0=f0,
        dx=dx.copy(),
        du=du.copy(),
    )


def linear_prediction(
    lin: LinearizationResult,
    dx0: np.ndarray,
    du_seq: np.ndarray | None,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Propagate the linear model forward from a perturbation.

    Parameters
    ----------
    lin:
        Linearization result.
    dx0:
        Initial state perturbation [12].
    du_seq:
        Control perturbation sequence, shape (n_steps, 5) or None for zero.
    dt:
        Time step [s].
    n_steps:
        Number of steps.

    Returns
    -------
    np.ndarray
        State perturbation history, shape (n_steps + 1, 12).
    """
    x_lin = np.zeros((n_steps + 1, NUM_STATES))
    x_lin[0] = np.asarray(dx0, dtype=float)

    if du_seq is None:
        du_seq_np = np.zeros((n_steps, NUM_CONTROLS))
    else:
        du_seq_np = np.asarray(du_seq, dtype=float)

    for k in range(n_steps):
        dx_dot = lin.A @ x_lin[k] + lin.B @ du_seq_np[k]
        x_lin[k + 1] = x_lin[k] + dt * dx_dot

    return x_lin
