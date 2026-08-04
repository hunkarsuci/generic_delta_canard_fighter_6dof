"""
Minimal propulsion model.

**IMPORTANT**: The thrust values are educational placeholder data.
They do NOT represent any real engine. Replace with actual data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from generic_delta_canard_fighter_6dof.equations import ForcesMoments
from generic_delta_canard_fighter_6dof.geometry import AircraftGeometry


def _default_thrust_axis() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class PropulsionConfig:
    """Generic placeholder propulsion parameters."""

    max_thrust_N: float = 120_000.0
    """Maximum sea-level static thrust [N]."""

    thrust_axis_body: np.ndarray = field(default_factory=_default_thrust_axis)
    """Unit vector of thrust direction in body axes."""

    def __post_init__(self) -> None:
        axis = np.asarray(self.thrust_axis_body, dtype=float)
        if axis.shape != (3,):
            raise ValueError("thrust_axis_body must have shape (3,).")
        norm = np.linalg.norm(axis)
        if norm <= 0:
            raise ValueError("thrust_axis_body must be nonzero.")
        object.__setattr__(self, "thrust_axis_body", axis / norm)


def propulsion_forces_moments(
    t: float,
    x: np.ndarray,
    control: np.ndarray,
    geometry: AircraftGeometry,
    *,
    config: PropulsionConfig | None = None,
) -> ForcesMoments:
    """Compute propulsion forces and moments.

    ForceMomentModel-compatible.

    The thrust magnitude is proportional to throttle. No Mach/altitude
    dependence in this minimal model.
    """
    if config is None:
        config = PropulsionConfig()

    from generic_delta_canard_fighter_6dof.state import ControlIndex

    throttle = control[ControlIndex.THROTTLE]
    thrust_N = config.max_thrust_N * throttle

    force_body = thrust_N * config.thrust_axis_body
    # Thrust is applied at CG; no thrust-induced moment in this model
    moment_body = np.zeros(3)

    return ForcesMoments(force_body_N=force_body, moment_body_Nm=moment_body)


def combined_forces_moments(
    aero_coeffs=None,
    prop_config=None,
):
    """Return a ForceMomentModel that combines aerodynamics and propulsion."""
    from generic_delta_canard_fighter_6dof.aerodynamics import (
        aerodynamic_forces_moments,
    )

    def model(t, x, control, geometry):
        aero = aerodynamic_forces_moments(
            t,
            x,
            control,
            geometry,
            coeffs=aero_coeffs,
        )
        prop = propulsion_forces_moments(
            t,
            x,
            control,
            geometry,
            config=prop_config,
        )
        return ForcesMoments(
            force_body_N=aero.force_body_N + prop.force_body_N,
            moment_body_Nm=aero.moment_body_Nm + prop.moment_body_Nm,
        )

    return model
