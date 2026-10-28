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

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral

import numpy as np

logger = logging.getLogger(__name__)


class StopReason(str, Enum):
    """Reason why the real-time runner stopped or paused."""

    NONE = "none"
    COMPLETED = "completed"
    PAUSED = "paused"
    MAX_OVERRUNS = "max_overruns"
    NON_FINITE_STATE = "non_finite_state"
    RUNTIME_ERROR = "runtime_error"
    STOPPED = "stopped"


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
        Maximum consecutive overruns before stopping or raising.
        None disables the limit.
    log_overruns:
        If True, log overrun warnings via the standard logging module.
    propagate_errors:
        If True, runtime errors (e.g. non-finite states, max overruns) are raised.
        If False (default), errors are captured in RealtimeStats and returned gracefully.
    """

    dt: float = 0.01
    wallclock_dt: float | None = None
    max_overruns: int | None = 100
    log_overruns: bool = True
    propagate_errors: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if self.wallclock_dt is None:
            self.wallclock_dt = self.dt
        if not np.isfinite(self.wallclock_dt) or self.wallclock_dt < 0.0:
            raise ValueError("wallclock_dt must be finite and non-negative.")
        if self.max_overruns is not None and (
            isinstance(self.max_overruns, bool)
            or not isinstance(self.max_overruns, Integral)
            or self.max_overruns <= 0
        ):
            raise ValueError("max_overruns must be a positive integer or None.")
        if not isinstance(self.propagate_errors, bool):
            raise TypeError("propagate_errors must be a boolean.")


@dataclass
class RealtimeStats:
    """Runtime statistics and observability metrics for the real-time runner.

    Attributes
    ----------
    sim_time:
        Elapsed simulation time [s].
    wall_time:
        Total wall-clock time elapsed [s] (sum of compute_time and sleep_time).
    compute_time:
        Total wall-clock time spent computing integration steps [s].
    sleep_time:
        Total wall-clock time spent sleeping for real-time pacing [s].
    steps_completed:
        Number of successfully completed integration steps.
    overruns:
        Total number of steps where compute time exceeded target wallclock step.
    dropped_deadlines:
        Total count of missed timing deadlines.
    running:
        Whether the simulation is currently active (not stopped or errored).
    paused:
        Whether the simulation is currently paused.
    stop_reason:
        Explicit reason why the runner stopped or paused.
    stop_message:
        Detailed explanation if stopped due to an error or manual stop.
    last_compute_time:
        Wall-clock time spent computing the most recent step [s].
    last_sleep_time:
        Wall-clock time spent sleeping in the most recent step [s].
    """

    sim_time: float = 0.0
    wall_time: float = 0.0
    compute_time: float = 0.0
    sleep_time: float = 0.0
    steps_completed: int = 0
    overruns: int = 0
    dropped_deadlines: int = 0
    running: bool = True
    paused: bool = False
    stop_reason: StopReason = StopReason.NONE
    stop_message: str | None = None
    last_compute_time: float = 0.0
    last_sleep_time: float = 0.0


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

    def step(self, step_dt: float | None = None) -> None:
        """Advance the simulation by one time step with wall-clock pacing.

        Raises
        ------
        RuntimeError
            If max_overruns is exceeded or state becomes non-finite when
            propagate_errors is True.
        """
        if self.stats.paused or not self.stats.running:
            return

        cfg = self.config
        integration_dt = cfg.dt if step_dt is None else step_dt
        if not np.isfinite(integration_dt) or integration_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive.")
        if integration_dt > cfg.dt:
            raise ValueError("step_dt must not exceed the configured dt.")

        calc_start = self.clock.time()

        # Integration step
        try:
            self._state = self.step_fn(
                self.derivative_fn,
                self._sim_t,
                self._state,
                integration_dt,
            )
        except Exception as e:
            calc_duration = self.clock.time() - calc_start
            self.stats.compute_time += calc_duration
            self.stats.last_compute_time = calc_duration
            self.stats.wall_time = self.stats.compute_time + self.stats.sleep_time
            self.stats.running = False
            self.stats.stop_reason = StopReason.RUNTIME_ERROR
            self.stats.stop_message = str(e)
            if cfg.propagate_errors:
                raise
            return

        # Check for non-finite state
        if not np.all(np.isfinite(self._state)):
            calc_duration = self.clock.time() - calc_start
            self.stats.compute_time += calc_duration
            self.stats.last_compute_time = calc_duration
            self.stats.wall_time = self.stats.compute_time + self.stats.sleep_time
            self.stats.running = False
            self.stats.stop_reason = StopReason.NON_FINITE_STATE
            self.stats.stop_message = (
                f"Non-finite state detected at t = {self._sim_t:.3e}"
            )
            if cfg.propagate_errors:
                raise RuntimeError(self.stats.stop_message)
            return

        calc_duration = self.clock.time() - calc_start
        self.stats.compute_time += calc_duration
        self.stats.last_compute_time = calc_duration

        self._sim_t += integration_dt
        self.stats.sim_time = self._sim_t
        self.stats.steps_completed += 1

        # Wall-clock pacing
        actual_sleep = 0.0
        if cfg.wallclock_dt is not None and cfg.wallclock_dt > 0:
            target_wallclock_dt = cfg.wallclock_dt * integration_dt / cfg.dt
            sleep_time = target_wallclock_dt - calc_duration
            if sleep_time > 0:
                sleep_start = self.clock.time()
                self.clock.sleep(sleep_time)
                actual_sleep = self.clock.time() - sleep_start
                self._consecutive_overruns = 0
            else:
                self.stats.overruns += 1
                self.stats.dropped_deadlines += 1
                self._consecutive_overruns += 1
                if cfg.log_overruns:
                    logger.warning(
                        "Real-time overrun at t=%.3fs (compute=%.4fs, target=%.4fs, consecutive=%d, total=%d)",
                        self._sim_t,
                        calc_duration,
                        target_wallclock_dt,
                        self._consecutive_overruns,
                        self.stats.overruns,
                    )

                if (
                    cfg.max_overruns is not None
                    and self._consecutive_overruns >= cfg.max_overruns
                ):
                    self.stats.sleep_time += actual_sleep
                    self.stats.last_sleep_time = actual_sleep
                    self.stats.wall_time = (
                        self.stats.compute_time + self.stats.sleep_time
                    )
                    self.stats.running = False
                    self.stats.stop_reason = StopReason.MAX_OVERRUNS
                    self.stats.stop_message = (
                        f"Max consecutive overruns ({cfg.max_overruns}) exceeded"
                    )
                    if cfg.propagate_errors:
                        raise RuntimeError(self.stats.stop_message)
                    return

        self.stats.sleep_time += actual_sleep
        self.stats.last_sleep_time = actual_sleep
        self.stats.wall_time = self.stats.compute_time + self.stats.sleep_time

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
        if not np.isfinite(t_final) or t_final < 0.0:
            raise ValueError("t_final must be finite and non-negative.")

        if self.stats.paused:
            self.stats.stop_reason = StopReason.PAUSED
            return self.stats

        tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(t_final))
        if t_final < self._sim_t - tolerance:
            raise ValueError(
                f"t_final ({t_final}) must not be less than current simulation time ({self._sim_t:.3f})."
            )

        while t_final - self._sim_t > tolerance:
            if not self.stats.running:
                break
            if self.stats.paused:
                self.stats.stop_reason = StopReason.PAUSED
                break
            remaining = t_final - self._sim_t
            try:
                self.step(min(self.config.dt, remaining))
            except RuntimeError:
                if self.config.propagate_errors:
                    raise
                break

        if (
            self.stats.running
            and not self.stats.paused
            and abs(t_final - self._sim_t) <= tolerance
        ):
            self._sim_t = t_final
            self.stats.sim_time = t_final
            self.stats.stop_reason = StopReason.COMPLETED
            self.stats.stop_message = None

        return self.stats

    def stop(self, reason: str = "Stopped by user") -> None:
        """Explicitly stop the simulation runner."""
        self.stats.running = False
        self.stats.stop_reason = StopReason.STOPPED
        self.stats.stop_message = reason

    def pause(self) -> None:
        """Pause the simulation."""
        self.stats.paused = True
        if self.stats.stop_reason in (StopReason.NONE, StopReason.COMPLETED):
            self.stats.stop_reason = StopReason.PAUSED

    def resume(self) -> None:
        """Resume the simulation."""
        self.stats.paused = False
        if self.stats.stop_reason == StopReason.PAUSED:
            self.stats.stop_reason = StopReason.NONE
