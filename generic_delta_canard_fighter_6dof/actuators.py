"""Configurable actuator dynamics for the delta-canard fighter.

Provides a reusable first-order actuator model with:
- First-order lag (tau * delta_dot + delta = delta_cmd)
- Position limits [delta_min, delta_max]
- Rate limits [rate_min, rate_max]
- Saturation reporting
- Deterministic initialization

All actuator parameters are synthetic defaults — not representative of any
real aircraft hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActuatorConfig:
    """Configuration for a single actuator channel.

    Attributes
    ----------
    tau_s:
        First-order time constant [s]. Must be > 0.
    position_min:
        Minimum position [rad] or [0-1] for throttle.
    position_max:
        Maximum position [rad] or [0-1] for throttle.
    rate_min_radps:
        Minimum (negative) rate limit [rad/s] or [1/s] for throttle.
    rate_max_radps:
        Maximum (positive) rate limit [rad/s] or [1/s] for throttle.
    initial_position:
        Initial actuator position. Defaults to midpoint.
    name:
        Human-readable label for diagnostics.
    """

    tau_s: float = 0.05
    position_min: float = -0.5
    position_max: float = 0.5
    rate_min_radps: float = -10.0
    rate_max_radps: float = 10.0
    initial_position: float | None = None
    name: str = "actuator"

    def __post_init__(self) -> None:
        if self.tau_s <= 0.0:
            raise ValueError(
                f"{self.name}: time constant tau_s must be positive, got {self.tau_s}"
            )
        if self.position_min >= self.position_max:
            raise ValueError(
                f"{self.name}: position_min ({self.position_min}) must be < "
                f"position_max ({self.position_max})"
            )
        if self.rate_min_radps >= self.rate_max_radps:
            raise ValueError(
                f"{self.name}: rate_min ({self.rate_min_radps}) must be < "
                f"rate_max ({self.rate_max_radps})"
            )


@dataclass
class ActuatorState:
    """Runtime state of a single actuator channel.

    Attributes
    ----------
    position:
        Current actuator position.
    saturated_upper:
        True if position hit the upper limit this step.
    saturated_lower:
        True if position hit the lower limit this step.
    rate_saturated:
        True if rate limiting was active this step.
    """

    position: float = 0.0
    saturated_upper: bool = False
    saturated_lower: bool = False
    rate_saturated: bool = False


class Actuator:
    """First-order actuator model with position and rate limits.

    Dynamics (forward Euler discretization):
        position_dot = (cmd - position) / tau_s
        position_dot = clamp(position_dot, rate_min, rate_max)
        position_next = clamp(position + dt * position_dot, pos_min, pos_max)

    Parameters
    ----------
    config:
        Actuator configuration.
    """

    def __init__(self, config: ActuatorConfig) -> None:
        self.config = config
        init = (
            config.initial_position
            if config.initial_position is not None
            else 0.5 * (config.position_min + config.position_max)
        )
        self._position = float(np.clip(init, config.position_min, config.position_max))
        self.saturated_upper = False
        self.saturated_lower = False
        self.rate_saturated = False

    @property
    def position(self) -> float:
        """Current actuator position."""
        return self._position

    def reset(self, position: float | None = None) -> None:
        """Reset actuator to a given position or its configured initial position."""
        if position is None:
            position = (
                self.config.initial_position
                if self.config.initial_position is not None
                else 0.5 * (self.config.position_min + self.config.position_max)
            )
        self._position = float(
            np.clip(position, self.config.position_min, self.config.position_max)
        )
        self.saturated_upper = False
        self.saturated_lower = False
        self.rate_saturated = False

    def step(self, command: float, dt: float) -> float:
        """Advance the actuator by one time step.

        Parameters
        ----------
        command:
            Desired position.
        dt:
            Time step [s]. Must be > 0.

        Returns
        -------
        float
            New actuator position.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if not np.isfinite(command):
            raise ValueError("Command must be finite.")

        # Rate-limited first-order response
        desired_rate = (command - self._position) / self.config.tau_s
        clamped_rate = float(
            np.clip(
                desired_rate, self.config.rate_min_radps, self.config.rate_max_radps
            )
        )
        self.rate_saturated = abs(clamped_rate - desired_rate) > 1e-15

        # Euler step
        new_position = self._position + dt * clamped_rate

        # Position limits
        if new_position > self.config.position_max:
            self._position = self.config.position_max
            self.saturated_upper = True
            self.saturated_lower = False
        elif new_position < self.config.position_min:
            self._position = self.config.position_min
            self.saturated_lower = True
            self.saturated_upper = False
        else:
            self._position = new_position
            self.saturated_upper = False
            self.saturated_lower = False

        return self._position


# ── synthetic default configurations ─────────────────────────────────────────
# These are clearly labeled simulation parameters, not real-aircraft data.


def default_canard_config() -> ActuatorConfig:
    """Synthetic default: canard actuator (pitch control surface)."""
    return ActuatorConfig(
        tau_s=0.05,
        position_min=-0.5,  # rad, ~28.6° trailing-edge down
        position_max=0.5,  # rad
        rate_min_radps=-8.0,  # rad/s, ~458°/s
        rate_max_radps=8.0,
        name="canard",
    )


def default_elevon_config() -> ActuatorConfig:
    """Synthetic default: elevon actuator (pitch + roll surface)."""
    return ActuatorConfig(
        tau_s=0.04,
        position_min=-0.5,
        position_max=0.5,
        rate_min_radps=-10.0,
        rate_max_radps=10.0,
        name="elevon",
    )


def default_rudder_config() -> ActuatorConfig:
    """Synthetic default: rudder actuator."""
    return ActuatorConfig(
        tau_s=0.06,
        position_min=-0.4,
        position_max=0.4,
        rate_min_radps=-6.0,
        rate_max_radps=6.0,
        name="rudder",
    )


def default_throttle_config() -> ActuatorConfig:
    """Synthetic default: throttle actuator (engine spool)."""
    return ActuatorConfig(
        tau_s=0.3,  # slower engine response
        position_min=0.0,
        position_max=1.0,
        rate_min_radps=-2.0,  # 1/s (not rad/s — throttle is dimensionless)
        rate_max_radps=2.0,
        initial_position=0.0,
        name="throttle",
    )


class ActuatorBank:
    """Collection of all five aircraft actuators.

    Channels (index order):
        0: canard
        1: left elevon
        2: right elevon
        3: rudder
        4: throttle

    Provides batch step() and reset() for integration into the simulation loop.
    """

    def __init__(
        self,
        canard: Actuator | None = None,
        elevon_left: Actuator | None = None,
        elevon_right: Actuator | None = None,
        rudder: Actuator | None = None,
        throttle: Actuator | None = None,
    ) -> None:
        self.canard = canard or Actuator(default_canard_config())
        self.elevon_left = elevon_left or Actuator(default_elevon_config())
        self.elevon_right = elevon_right or Actuator(default_elevon_config())
        self.rudder = rudder or Actuator(default_rudder_config())
        self.throttle = throttle or Actuator(default_throttle_config())

        self._actuators: list[Actuator] = [
            self.canard,
            self.elevon_left,
            self.elevon_right,
            self.rudder,
            self.throttle,
        ]

    def step(self, command: np.ndarray, dt: float) -> np.ndarray:
        """Advance all actuators and return the achieved positions.

        Parameters
        ----------
        command:
            5-element control command vector (rad/rad/rad/rad/dim).
        dt:
            Time step [s].

        Returns
        -------
        np.ndarray
            5-element vector of achieved actuator positions.
        """
        cmd = np.asarray(command, dtype=float)
        if cmd.shape != (5,):
            raise ValueError(f"Command must have shape (5,), got {cmd.shape}")

        achieved = np.zeros(5, dtype=float)
        for i, act in enumerate(self._actuators):
            achieved[i] = act.step(float(cmd[i]), dt)
        return achieved

    def reset(self, positions: np.ndarray | None = None) -> None:
        """Reset all actuators."""
        if positions is not None:
            pos = np.asarray(positions, dtype=float)
            for i, act in enumerate(self._actuators):
                act.reset(float(pos[i]))
        else:
            for act in self._actuators:
                act.reset()

    @property
    def positions(self) -> np.ndarray:
        """Current positions of all actuators as a 5-vector."""
        return np.array([a.position for a in self._actuators], dtype=float)

    @property
    def any_saturated(self) -> bool:
        """True if any actuator is currently saturated (position or rate)."""
        return any(
            a.saturated_upper or a.saturated_lower or a.rate_saturated
            for a in self._actuators
        )
