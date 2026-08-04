"""
Minimal generic aerodynamic model for the delta-canard fighter.

**IMPORTANT**: The coefficient values in this module are educational
placeholder data. They do NOT represent any real aircraft. They are
chosen to produce a plausible straight-and-level trim solution for
validating the trim, linearization, and simulation infrastructure.

The model scope:
- Linear lift, drag, and pitching moment in alpha
- Linear side-force and roll/yaw moments in beta
- Linear control derivatives for symmetric elevon and canard
- No Mach or Reynolds number dependence
- No stall modeling
- No unsteady/hysteresis effects

Coefficient provenance: GENERIC PLACEHOLDER — not from any published
source or aircraft database. Suitable only for infrastructure validation.

Owner should replace all coefficients with actual aircraft data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.atmosphere import (
    dynamic_pressure,
    isa_atmosphere,
)
from generic_delta_canard_fighter_6dof.equations import ForcesMoments
from generic_delta_canard_fighter_6dof.geometry import AircraftGeometry
from generic_delta_canard_fighter_6dof.state import StateIndex


@dataclass(frozen=True)
class AeroCoefficients:
    """Generic placeholder aerodynamic coefficients.

    All derivatives are per radian unless noted otherwise.
    """

    # ── longitudinal ──
    CL0: float = 0.12
    """Zero-alpha lift coefficient [-]."""

    CL_alpha: float = 5.0
    """Lift-curve slope per radian [-]."""

    CL_de: float = 0.4
    """Lift increment per symmetric elevon deflection per radian [-]."""

    CD0: float = 0.022
    """Zero-lift drag coefficient [-]."""

    K: float = 0.08
    """Induced-drag factor K such that CD_i = K * CL^2 [-]."""

    Cm0: float = 0.01
    """Zero-alpha pitching moment coefficient [-]."""

    Cm_alpha: float = -0.55
    """Pitch stiffness derivative per radian [-]."""

    Cm_de: float = -1.2
    """Pitch control derivative per radian [-]."""

    # ── lateral-directional ──
    CY_beta: float = -1.0
    """Side-force derivative per radian [-]."""

    Cl_beta: float = -0.08
    """Dihedral effect per radian [-]."""

    Cl_da: float = 0.06
    """Roll control power per radian differential elevon [-]."""

    Cl_dr: float = 0.012
    """Roll due to rudder per radian [-]."""

    Cn_beta: float = 0.12
    """Weathercock stability per radian [-]."""

    Cn_da: float = -0.01
    """Adverse yaw from differential elevon per radian [-]."""

    Cn_dr: float = -0.06
    """Yaw control power per radian [-]."""


def _symmetric_elevon(control: np.ndarray, state_idx_module) -> float:
    """Average left/right elevon deflection [rad]."""
    return 0.5 * (
        control[state_idx_module.DELTA_ELEVON_LEFT]
        + control[state_idx_module.DELTA_ELEVON_RIGHT]
    )


def _differential_elevon(control: np.ndarray, state_idx_module) -> float:
    """Differential elevon: (right - left) / 2 [rad]."""
    return 0.5 * (
        control[state_idx_module.DELTA_ELEVON_RIGHT]
        - control[state_idx_module.DELTA_ELEVON_LEFT]
    )


def aerodynamic_forces_moments(
    t: float,
    x: np.ndarray,
    control: np.ndarray,
    geometry: AircraftGeometry,
    *,
    coeffs: AeroCoefficients | None = None,
) -> ForcesMoments:
    """Compute body-axis aerodynamic forces and moments.

    This is a ForceMomentModel-compatible callable.

    Parameters
    ----------
    t:
        Current time [s] (unused; kept for interface compatibility).
    x:
        Aircraft state vector.
    control:
        Aircraft control vector.
    geometry:
        Aircraft geometry.
    coeffs:
        Aerodynamic coefficients. Uses defaults if None.

    Returns
    -------
    ForcesMoments
        Body-axis forces [N] and moments [N*m] from aerodynamics only.
    """
    if coeffs is None:
        coeffs = AeroCoefficients()

    from generic_delta_canard_fighter_6dof.state import ControlIndex

    VT = x[StateIndex.VT]
    alpha = x[StateIndex.ALPHA]
    beta = x[StateIndex.BETA]
    h = x[StateIndex.H]

    # Atmosphere
    atm = isa_atmosphere(h)
    q_bar = dynamic_pressure(atm.density_kg_m3, VT)

    S = geometry.wing_area_m2
    b = geometry.wingspan_m
    c = geometry.mean_aerodynamic_chord_m

    # Control mixing
    de = _symmetric_elevon(control, ControlIndex)
    da = _differential_elevon(control, ControlIndex)
    dr = control[ControlIndex.DELTA_RUDDER]

    # ── longitudinal coefficients ──
    CL = coeffs.CL0 + coeffs.CL_alpha * alpha + coeffs.CL_de * de
    CD = coeffs.CD0 + coeffs.K * CL * CL
    Cm = coeffs.Cm0 + coeffs.Cm_alpha * alpha + coeffs.Cm_de * de

    # ── lateral-directional coefficients ──
    CY = coeffs.CY_beta * beta
    Cl = coeffs.Cl_beta * beta + coeffs.Cl_da * da + coeffs.Cl_dr * dr
    Cn = coeffs.Cn_beta * beta + coeffs.Cn_da * da + coeffs.Cn_dr * dr

    # ── wind-axis forces ──
    L_wind = q_bar * S * CL  # lift (positive up, perpendicular to velocity)
    D_wind = q_bar * S * CD  # drag (positive aft, parallel to velocity)
    Y_wind = q_bar * S * CY  # side force (positive to right)

    # ── body-axis forces ──
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)

    # Wind to body: standard rotation by (-alpha, -beta)
    force_body = np.array(
        [
            -D_wind * cos_a * cos_b - Y_wind * cos_a * sin_b + L_wind * sin_a,
            -D_wind * sin_b + Y_wind * cos_b,
            -D_wind * sin_a * cos_b - Y_wind * sin_a * sin_b - L_wind * cos_a,
        ]
    )

    # ── body-axis moments ──
    moment_body = np.array(
        [
            q_bar * S * b * Cl,
            q_bar * S * c * Cm,
            q_bar * S * b * Cn,
        ]
    )

    return ForcesMoments(force_body_N=force_body, moment_body_Nm=moment_body)
