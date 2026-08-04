"""
Straight-and-level trim solver for the 6DOF aircraft model.

Finds steady-state flight conditions where:
  - flight path angle = 0 (level flight)
  - no sideslip
  - no angular rates
  - no acceleration

Uses scipy.optimize with bounds on unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from generic_delta_canard_fighter_6dof.aerodynamics import AeroCoefficients
from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import (
    AircraftGeometry,
    create_default_geometry,
)
from generic_delta_canard_fighter_6dof.propulsion import (
    PropulsionConfig,
    combined_forces_moments,
)
from generic_delta_canard_fighter_6dof.state import (
    StateIndex,
    make_control,
    make_state,
)
from generic_delta_canard_fighter_6dof.units import deg_to_rad, rad_to_deg


@dataclass
class TrimResult:
    """Result of a trim computation."""

    converged: bool
    """Whether the solver converged."""

    message: str
    """Solver termination message."""

    nfev: int
    """Number of function evaluations."""

    alpha_deg: float
    """Trim angle of attack [deg]."""

    theta_deg: float
    """Trim pitch angle [deg] (= alpha for level flight)."""

    throttle: float
    """Trim throttle setting [0, 1]."""

    symmetric_elevon_deg: float
    """Trim symmetric elevon deflection [deg]."""

    state: np.ndarray
    """Full 12-element state at the trim point."""

    control: np.ndarray
    """Full 5-element control at the trim point."""

    force_residual_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Residual body force [X, Y, Z] in N at trim."""

    moment_residual_Nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Residual body moment [L, M, N] in N*m at trim."""

    state_derivative: np.ndarray = field(default_factory=lambda: np.zeros(12))
    """Full state derivative at the converged point."""

    scaled_residual_norm: float = 0.0
    """Final scaled residual norm from the solver."""

    requested_altitude_m: float = 0.0
    """Altitude specified in the trim request [m]."""

    requested_airspeed_mps: float = 0.0
    """Airspeed specified in the trim request [m/s]."""


def trim_straight_level(
    altitude_m: float,
    airspeed_mps: float,
    *,
    geometry: AircraftGeometry | None = None,
    aero_coeffs: AeroCoefficients | None = None,
    prop_config: PropulsionConfig | None = None,
    alpha0_deg: float = 5.0,
) -> TrimResult:
    """Compute a straight-and-level trim point.

    Parameters
    ----------
    altitude_m:
        Trim altitude [m]. Must be within the atmosphere model range [0, 20000].
    airspeed_mps:
        Trim true airspeed [m/s]. Must be positive.
    geometry:
        Aircraft geometry. Uses defaults if None.
    aero_coeffs:
        Aerodynamic coefficients. Uses defaults if None.
    prop_config:
        Propulsion configuration. Uses defaults if None.
    alpha0_deg:
        Initial guess for angle of attack [deg].

    Returns
    -------
    TrimResult
        Structured trim result with convergence status, solution, and residuals.
    """
    if airspeed_mps <= 0.0:
        return TrimResult(
            converged=False,
            message="Airspeed must be positive.",
            nfev=0,
            alpha_deg=0.0,
            theta_deg=0.0,
            throttle=0.0,
            symmetric_elevon_deg=0.0,
            state=make_state(),
            control=make_control(),
            requested_altitude_m=altitude_m,
            requested_airspeed_mps=airspeed_mps,
        )

    if geometry is None:
        geometry = create_default_geometry()
    if aero_coeffs is None:
        aero_coeffs = AeroCoefficients()
    if prop_config is None:
        prop_config = PropulsionConfig()

    force_model = combined_forces_moments(aero_coeffs, prop_config)

    # ── bounds on unknowns ──
    # unknowns: [alpha_rad, symmetric_elevon_rad, throttle]
    alpha_min = deg_to_rad(-5.0)
    alpha_max = deg_to_rad(20.0)
    de_min = deg_to_rad(-25.0)
    de_max = deg_to_rad(25.0)
    throttle_min = 0.0
    throttle_max = 1.0

    lower_bounds = np.array([alpha_min, de_min, throttle_min])
    upper_bounds = np.array([alpha_max, de_max, throttle_max])

    x0 = np.array(
        [
            deg_to_rad(alpha0_deg),
            0.0,
            0.5,
        ]
    )

    nfev = 0

    def residual(u: np.ndarray) -> np.ndarray:
        nonlocal nfev
        nfev += 1

        alpha_r, de_r, thr = u

        # Build state and control at candidate trim point
        state = make_state(
            VT=airspeed_mps,
            alpha=alpha_r,
            beta=0.0,
            p=0.0,
            q=0.0,
            r=0.0,
            phi=0.0,
            theta=alpha_r,  # level flight: theta = alpha
            psi=0.0,
            h=altitude_m,
        )
        control = make_control(
            delta_canard=0.0,
            delta_elevon_left=de_r,
            delta_elevon_right=de_r,
            delta_rudder=0.0,
            throttle=thr,
        )

        x_dot = aircraft_dynamics(0.0, state, control, geometry, force_model)

        # Residuals: VT_dot, alpha_dot, q_dot must all be zero.
        # Also check beta_dot and p_dot, r_dot (should be near zero for
        # a symmetric configuration).
        r = np.array(
            [
                x_dot[StateIndex.VT],
                x_dot[StateIndex.ALPHA],
                x_dot[StateIndex.Q],
            ],
            dtype=float,
        )

        return r

    # Scale residuals for better conditioning
    # VT_dot ~ m/s^2 (order 0.1–10)
    # alpha_dot ~ rad/s (order 0.01–1)
    # q_dot ~ rad/s^2 (order 0.1–10)
    scaling = np.array([1.0 / airspeed_mps, 1.0, 1.0])

    def scaled_residual(u: np.ndarray) -> np.ndarray:
        return residual(u) * scaling

    try:
        result = least_squares(
            scaled_residual,
            x0,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=500,
        )
    except (ValueError, RuntimeError) as exc:
        return TrimResult(
            converged=False,
            message=f"Solver exception: {exc}",
            nfev=nfev,
            alpha_deg=float(rad_to_deg(x0[0])),
            theta_deg=float(rad_to_deg(x0[0])),
            throttle=float(x0[2]),
            symmetric_elevon_deg=float(rad_to_deg(x0[1])),
            state=make_state(),
            control=make_control(),
            requested_altitude_m=altitude_m,
            requested_airspeed_mps=airspeed_mps,
        )

    alpha_r, de_r, thr = result.x
    converged = bool(result.success)

    # Build final state and control
    state = make_state(
        VT=airspeed_mps,
        alpha=alpha_r,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
        phi=0.0,
        theta=alpha_r,
        psi=0.0,
        h=altitude_m,
    )
    control = make_control(
        delta_canard=0.0,
        delta_elevon_left=de_r,
        delta_elevon_right=de_r,
        delta_rudder=0.0,
        throttle=thr,
    )

    # Compute full state derivative for residual reporting
    x_dot = aircraft_dynamics(0.0, state, control, geometry, force_model)

    # Compute dimensional force and moment residuals
    from generic_delta_canard_fighter_6dof.equations import gravity_force_body

    fm_total = force_model(0.0, state, control, geometry)
    grav_body = gravity_force_body(0.0, alpha_r, 0.0, geometry.mass_kg)
    total_force = fm_total.force_body_N + grav_body
    total_moment = fm_total.moment_body_Nm

    # Force residual: F_total - m * a_body
    # At trim, a_body should be zero, so residual = F_total
    force_residual = total_force.copy()
    moment_residual = total_moment.copy()

    return TrimResult(
        converged=converged,
        message=str(result.message),
        nfev=nfev,
        alpha_deg=float(rad_to_deg(alpha_r)),
        theta_deg=float(rad_to_deg(alpha_r)),
        throttle=float(thr),
        symmetric_elevon_deg=float(rad_to_deg(de_r)),
        state=state,
        control=control,
        force_residual_N=force_residual,
        moment_residual_Nm=moment_residual,
        state_derivative=x_dot,
        scaled_residual_norm=float(np.linalg.norm(scaled_residual(result.x)))
        if converged
        else float("inf"),
        requested_altitude_m=altitude_m,
        requested_airspeed_mps=airspeed_mps,
    )
