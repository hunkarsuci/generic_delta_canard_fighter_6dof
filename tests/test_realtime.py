"""Tests for soft real-time simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.integrators import rk4_step
from generic_delta_canard_fighter_6dof.realtime import (
    FakeClock,
    RealtimeConfig,
    RealtimeRunner,
)
from generic_delta_canard_fighter_6dof.state import make_control, make_state


def _make_runner(duration=0.5, dt=0.01):
    geo = create_default_geometry()
    ctrl = make_control()
    x0 = make_state(VT=200.0, h=5000.0)

    def deriv(t, x):
        return aircraft_dynamics(t, x, ctrl, geo)

    cfg = RealtimeConfig(dt=dt, wallclock_dt=0.0)  # as-fast-as-possible
    clock = FakeClock()
    return RealtimeRunner(deriv, rk4_step, x0, cfg, clock=clock)


def test_deterministic_fake_clock_execution():
    runner = _make_runner(duration=0.5)
    stats = runner.run(0.5)
    assert stats.steps_completed == 50  # 0.5 / 0.01
    assert stats.sim_time == pytest.approx(0.5)
    assert np.all(np.isfinite(runner.state))


def test_correct_number_of_steps():
    runner = _make_runner(duration=0.2, dt=0.01)
    stats = runner.run(0.2)
    assert stats.steps_completed == 20


def test_overrun_detection():
    """With zero wallclock_dt, overruns should be 0 since we aren't sleeping."""
    runner = _make_runner(duration=0.1)
    stats = runner.run(0.1)
    assert stats.overruns == 0  # as-fast-as-possible mode


def test_reset_behavior():
    runner = _make_runner(duration=0.5)
    runner.run(0.2)
    _x_before = runner.state.copy()
    runner.reset()
    assert runner.stats.steps_completed == 0
    assert runner.stats.sim_time == 0.0


def test_pause_and_resume():
    runner = _make_runner(duration=0.5)
    runner.pause()
    assert runner.stats.paused
    runner.resume()
    assert not runner.stats.paused


def test_fake_clock_advance():
    clock = FakeClock(10.0)
    assert clock.time() == 10.0
    clock.sleep(1.0)
    assert clock.time() == 11.0


def test_nonfinite_state_detection():
    geo = create_default_geometry()
    x0 = make_state(VT=200.0)
    ctrl = make_control()

    call_count = [0]

    def deriv(t, x):
        call_count[0] += 1
        if call_count[0] > 5:
            return np.full(12, np.nan)
        return aircraft_dynamics(t, x, ctrl, geo)

    cfg = RealtimeConfig(dt=0.01, wallclock_dt=0.0, max_overruns=100)
    runner = RealtimeRunner(deriv, rk4_step, x0, cfg, clock=FakeClock())

    stats = runner.run(1.0)
    # Runner should have stopped early due to non-finite state
    assert not stats.running
    assert stats.steps_completed < 100
