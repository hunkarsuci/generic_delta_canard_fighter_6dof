"""Control allocation and surface mixing for the delta-canard fighter.

Maps virtual control channels to physical surfaces:

    Virtual channels:
        pitch_cmd       — symmetric pitch (positive = nose-up)
        roll_cmd        — differential roll (positive = right-wing-down)
        yaw_cmd         — rudder yaw (positive = nose-right)
        throttle_cmd    — engine power [0, 1]

    Physical surfaces:
        delta_canard            — canard deflection [rad]
        delta_elevon_left       — left elevon [rad]
        delta_elevon_right      — right elevon [rad]
        delta_rudder            — rudder [rad]
        throttle                — engine power [0, 1]

Mapping (sign conventions consistent with aerodynamics.py):
    delta_canard        =  pitch_gain * pitch_cmd
    delta_elevon_left   =  pitch_gain * pitch_cmd  -  roll_gain * roll_cmd
    delta_elevon_right  =  pitch_gain * pitch_cmd  +  roll_gain * roll_cmd
    delta_rudder        =  yaw_gain * yaw_cmd
    throttle            =  throttle_cmd

This module does NOT implement optimal control allocation. It is a fixed
linear mapping with hard saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationConfig:
    """Gains and limits for surface mixing.

    All gains map virtual command to physical surface deflection.
    pitch_gain and roll_gain are dimensionless (rad deflection per rad command).
    yaw_gain is dimensionless (rad rudder per rad command).
    """

    pitch_gain: float = 1.0
    roll_gain: float = 1.0
    yaw_gain: float = 1.0

    def __post_init__(self) -> None:
        for name in ("pitch_gain", "roll_gain", "yaw_gain"):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative, got {val}")


@dataclass
class AllocationResult:
    """Result of control allocation.

    Attributes
    ----------
    surface_commands:
        5-element vector [canard, elev_L, elev_R, rudder, throttle].
    feasible:
        True if all surface commands are within actuator limits.
    infeasible_channels:
        Set of channel indices where limits were violated.
    """

    surface_commands: np.ndarray
    feasible: bool = True
    infeasible_channels: set[int] = ()

    def __post_init__(self) -> None:
        self.infeasible_channels = set(self.infeasible_channels)


def allocate(
    pitch_cmd: float,
    roll_cmd: float,
    yaw_cmd: float,
    throttle_cmd: float,
    config: AllocationConfig | None = None,
    surface_min: np.ndarray | None = None,
    surface_max: np.ndarray | None = None,
) -> AllocationResult:
    """Map virtual commands to physical surface deflections.

    Parameters
    ----------
    pitch_cmd:
        Symmetric pitch command (positive = nose-up moment).
    roll_cmd:
        Differential roll command (positive = right-wing-down moment).
    yaw_cmd:
        Rudder yaw command (positive = nose-right moment).
    throttle_cmd:
        Engine command [0, 1].
    config:
        Allocation gains. Uses defaults if None.
    surface_min:
        5-vector of lower position limits [rad, rad, rad, rad, dim].
        Defaults to symmetric [-0.5, -0.5, -0.5, -0.4, 0.0].
    surface_max:
        5-vector of upper position limits [rad, rad, rad, rad, dim].
        Defaults to symmetric [0.5, 0.5, 0.5, 0.4, 1.0].

    Returns
    -------
    AllocationResult
        Surface commands and feasibility information.
    """
    if config is None:
        config = AllocationConfig()

    if surface_min is None:
        surface_min = np.array([-0.5, -0.5, -0.5, -0.4, 0.0], dtype=float)
    if surface_max is None:
        surface_max = np.array([0.5, 0.5, 0.5, 0.4, 1.0], dtype=float)

    # Linear mixing
    surfaces = np.zeros(5, dtype=float)

    surfaces[0] = config.pitch_gain * pitch_cmd  # canard (pitch only)
    surfaces[1] = (
        config.pitch_gain * pitch_cmd - config.roll_gain * roll_cmd
    )  # left elevon
    surfaces[2] = (
        config.pitch_gain * pitch_cmd + config.roll_gain * roll_cmd
    )  # right elevon
    surfaces[3] = config.yaw_gain * yaw_cmd  # rudder
    surfaces[4] = throttle_cmd  # throttle

    # Check feasibility
    infeasible = set()
    for i in range(5):
        if surfaces[i] < surface_min[i] or surfaces[i] > surface_max[i]:
            infeasible.add(i)

    return AllocationResult(
        surface_commands=surfaces,
        feasible=len(infeasible) == 0,
        infeasible_channels=infeasible,
    )
