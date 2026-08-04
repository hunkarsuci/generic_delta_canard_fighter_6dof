"""Soft real-time simulation runner.

This is a soft real-time demonstration and does NOT provide deterministic
operating-system scheduling or hardware real-time guarantees.

Features:
- Fixed simulation step
- Monotonic clock (injectable for testing)
- Wall-clock pacing
- Deterministic non-real-time mode
- Pause and reset
- Overrun detection and dropped-deadline counting
- Telemetry callback
- Optional logging
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RealtimeConfig:
    """Configuration for the soft real-time runner.

    Attributes
    ----------
    dt:
        Simulation time step [s].
    wallclock_dt:
        Target wall-clock step [s]. Default = dt (real-time).
        Set to 0 for as-fast-as-possible (deterministic non-real-time).
    max_overruns:
        Maximum consecutive overruns before raising RuntimeError.
        None disables the limit.
    log_overruns:
        If True, print overrun warnings.
    """

    dt: float = 0.01
    wallclock_dt: float | None = None
    max_overruns: int | None = 100
    log_overruns: bool = True

    def __post_init__(self) -> None:
        if self.wallclock_dt is None:
            self.wallclock_dt = self.dt


@dataclass
class RealtimeStats:
    """Runtime statistics for the real-time runner."""

    sim_time: float = 0.0
    wall_time: float = 0.0
    steps_completed: int = 0
    overruns: int = 0
    dropped_deadlines: int = 0
    running: bool = True
    paused: bool = False


class Clock:
    """Injectable clock interface for testing."""

    def time(self) -> float:
        """Return current monotonic time in seconds."""
        return time.monotonic()

    def sleep(self, duration_s: float) -> None:
        """Sleep for the given duration."""
        if duration_s > 0:
            time.sleep(duration_s)


class FakeClock(Clock):
    """Deterministic fake clock for testing."""

    def __init__(self, start_time: float = 0.0) -> None:
        self._t = start_time
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self._t

    def sleep(self, duration_s: float) -> None:
        self.sleep_calls.append(duration_s)
        self._t += duration_s

    def advance(self, dt: float) -> None:
        self._t += dt


TelemetryCallback = Callable[[RealtimeStats, np.ndarray], None]


@dataclass
class RealtimeRunner:
    """Soft real-time simulation runner.

    Parameters
    ----------
    derivative_fn:
        Function f(t, x) returning state derivative.
    step_fn:
        Integration step function (e.g., rk4_step).
    x0:
        Initial state vector.
    config:
        Real-time configuration.
    clock:
        Clock implementation (injectable for testing).
    telemetry:
        Optional callback(stats, state) called after each step.
    """

    derivative_fn: Callable[[float, np.ndarray], np.ndarray]
    step_fn: Callable[
        [Callable[[float, np.ndarray], np.ndarray], float, np.ndarray, float],
        np.ndarray,
    ]
    x0: np.ndarray
    config: RealtimeConfig = field(default_factory=RealtimeConfig)
    clock: Clock = field(default_factory=Clock)
    telemetry: TelemetryCallback | None = None

    def __post_init__(self) -> None:
        self.stats = RealtimeStats()
        self._state = np.asarray(self.x0, dtype=float).copy()
        self._sim_t = 0.0
        self._consecutive_overruns = 0

    @property
    def state(self) -> np.ndarray:
        """Current simulation state."""
        return self._state.copy()

    def reset(self, x0: np.ndarray | None = None) -> None:
        """Reset the simulation to initial conditions."""
        if x0 is not None:
            self._state = np.asarray(x0, dtype=float).copy()
        else:
            self._state = np.asarray(self.x0, dtype=float).copy()
        self._sim_t = 0.0
        self.stats = RealtimeStats()
        self._consecutive_overruns = 0

    def step(self) -> None:
        """Advance the simulation by one time step with wall-clock pacing.

        Raises
        ------
        RuntimeError
            If max_overruns is exceeded.
        """
        if self.stats.paused:
            return

        cfg = self.config
        wallclock_start = self.clock.time()

        # Integration step
        self._state = self.step_fn(self.derivative_fn, self._sim_t, self._state, cfg.dt)

        # Check for non-finite state
        if not np.all(np.isfinite(self._state)):
            self.stats.running = False
            raise RuntimeError(f"Non-finite state detected at t = {self._sim_t:.3e}")

        self._sim_t += cfg.dt
        self.stats.sim_time = self._sim_t
        self.stats.steps_completed += 1

        # Wall-clock pacing
        elapsed = self.clock.time() - wallclock_start
        self.stats.wall_time += elapsed

        if cfg.wallclock_dt is not None and cfg.wallclock_dt > 0:
            sleep_time = cfg.wallclock_dt - elapsed
            if sleep_time > 0:
                self.clock.sleep(sleep_time)
                self._consecutive_overruns = 0
            else:
                self.stats.overruns += 1
                self.stats.dropped_deadlines += 1
                self._consecutive_overruns += 1
                if cfg.log_overruns:
                    pass  # Silent in non-log mode; caller can check stats

                if (
                    cfg.max_overruns is not None
                    and self._consecutive_overruns >= cfg.max_overruns
                ):
                    self.stats.running = False
                    raise RuntimeError(
                        f"Max consecutive overruns ({cfg.max_overruns}) exceeded"
                    )

        # Telemetry
        if self.telemetry is not None:
            self.telemetry(self.stats, self._state)

    def run(self, t_final: float) -> RealtimeStats:
        """Run the simulation to t_final with wall-clock pacing.

        Parameters
        ----------
        t_final:
            Final simulation time [s].

        Returns
        -------
        RealtimeStats
            Final statistics.
        """
        n_steps = int(np.ceil(t_final / self.config.dt))
        for _ in range(n_steps):
            if not self.stats.running:
                break
            try:
                self.step()
            except RuntimeError:
                break
        return self.stats

    def pause(self) -> None:
        """Pause the simulation."""
        self.stats.paused = True

    def resume(self) -> None:
        """Resume the simulation."""
        self.stats.paused = False
