"""Tests for soft real-time simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.integrators import rk4_step
from generic_delta_canard_fighter_6dof.realtime import (
    Clock,
    FakeClock,
    RealtimeConfig,
    RealtimeRunner,
    StopReason,
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


def test_non_divisible_duration_uses_partial_final_step():
    runner = _make_runner(dt=0.3)
    stats = runner.run(1.0)

    assert stats.steps_completed == 4
    assert stats.sim_time == pytest.approx(1.0)


def test_divisible_decimal_duration_does_not_add_roundoff_step():
    runner = _make_runner(dt=0.1)
    stats = runner.run(1.0)

    assert stats.steps_completed == 10
    assert stats.sim_time == pytest.approx(1.0)


def test_duration_shorter_than_step():
    runner = _make_runner(dt=1.0)
    stats = runner.run(0.2)

    assert stats.steps_completed == 1
    assert stats.sim_time == pytest.approx(0.2)


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


def test_run_returns_without_advancing_while_paused():
    runner = _make_runner()
    runner.pause()

    stats = runner.run(0.5)

    assert stats.steps_completed == 0
    assert stats.sim_time == 0.0


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
    assert stats.stop_reason == StopReason.NON_FINITE_STATE
    assert stats.stop_message is not None and "Non-finite state" in stats.stop_message
    assert stats.steps_completed < 100


def test_stop_reason_completed():
    runner = _make_runner(duration=0.1, dt=0.01)
    stats = runner.run(0.1)
    assert stats.stop_reason == StopReason.COMPLETED
    assert stats.stop_message is None
    assert stats.steps_completed == 10


def test_stop_reason_paused_and_resumed():
    runner = _make_runner(duration=0.5, dt=0.01)
    runner.pause()
    assert runner.stats.paused
    assert runner.stats.stop_reason == StopReason.PAUSED

    stats = runner.run(0.5)
    assert stats.steps_completed == 0
    assert stats.stop_reason == StopReason.PAUSED

    runner.resume()
    assert not runner.stats.paused
    assert runner.stats.stop_reason == StopReason.NONE

    stats = runner.run(0.1)
    assert stats.steps_completed == 10
    assert stats.stop_reason == StopReason.COMPLETED


def test_manual_stop_method():
    runner = _make_runner(duration=0.5, dt=0.01)
    runner.stop("Cancelled by flight operator")
    assert not runner.stats.running
    assert runner.stats.stop_reason == StopReason.STOPPED
    assert runner.stats.stop_message == "Cancelled by flight operator"

    # Further run calls do not advance
    stats = runner.run(0.5)
    assert stats.steps_completed == 0
    assert stats.stop_reason == StopReason.STOPPED


def test_error_propagation_true_on_nonfinite_state():
    x0 = make_state(VT=200.0)

    def deriv(t, x):
        return np.full_like(x, np.nan)

    cfg = RealtimeConfig(dt=0.01, wallclock_dt=0.0, propagate_errors=True)
    runner = RealtimeRunner(deriv, rk4_step, x0, cfg, clock=FakeClock())

    with pytest.raises(RuntimeError, match="Non-finite state detected"):
        runner.run(0.1)

    assert not runner.stats.running
    assert runner.stats.stop_reason == StopReason.NON_FINITE_STATE


def test_error_propagation_true_on_step_exception():
    x0 = make_state(VT=200.0)

    def broken_deriv(t, x):
        raise ArithmeticError("Custom calculation failure")

    cfg = RealtimeConfig(dt=0.01, wallclock_dt=0.0, propagate_errors=True)
    runner = RealtimeRunner(broken_deriv, rk4_step, x0, cfg, clock=FakeClock())

    with pytest.raises(ArithmeticError, match="Custom calculation failure"):
        runner.step(0.01)

    assert not runner.stats.running
    assert runner.stats.stop_reason == StopReason.RUNTIME_ERROR


def test_error_propagation_false_on_step_exception():
    x0 = make_state(VT=200.0)

    def broken_deriv(t, x):
        raise ArithmeticError("Graceful failure")

    cfg = RealtimeConfig(dt=0.01, wallclock_dt=0.0, propagate_errors=False)
    runner = RealtimeRunner(broken_deriv, rk4_step, x0, cfg, clock=FakeClock())

    # Does not raise
    stats = runner.run(0.1)
    assert not stats.running
    assert stats.stop_reason == StopReason.RUNTIME_ERROR
    assert stats.stop_message == "Graceful failure"
    assert stats.steps_completed == 0


def test_timing_observability_accounting():
    """Verify compute_time, sleep_time, and wall_time are tracked accurately."""
    clock = FakeClock()
    x0 = np.array([1.0])

    def slow_deriv(t, x):
        # Simulate compute time by advancing the clock during the step
        clock.advance(0.004)
        return np.zeros_like(x)

    def dummy_step(fn, t, x, dt):
        fn(t, x)
        return x

    # wallclock_dt = 0.01 s. With compute = 0.004 s, sleep should be 0.006 s per step.
    cfg = RealtimeConfig(dt=0.01, wallclock_dt=0.01, max_overruns=10)
    runner = RealtimeRunner(slow_deriv, dummy_step, x0, cfg, clock=clock)

    stats = runner.run(0.05)  # 5 steps
    assert stats.steps_completed == 5
    assert stats.overruns == 0
    assert stats.compute_time == pytest.approx(5 * 0.004, abs=1e-9)
    assert stats.sleep_time == pytest.approx(5 * 0.006, abs=1e-9)
    assert stats.wall_time == pytest.approx(
        stats.compute_time + stats.sleep_time, abs=1e-9
    )
    assert stats.last_compute_time == pytest.approx(0.004, abs=1e-9)
    assert stats.last_sleep_time == pytest.approx(0.006, abs=1e-9)


def test_overrun_logging_and_max_overruns_stop_reason(caplog):
    """Verify overruns are logged when log_overruns=True and max_overruns stops runner."""
    clock = FakeClock()
    x0 = np.array([1.0])

    def very_slow_deriv(t, x):
        # Compute time = 0.015s, which exceeds wallclock_dt = 0.01s (overrun of 0.005s)
        clock.advance(0.015)
        return np.zeros_like(x)

    def dummy_step(fn, t, x, dt):
        fn(t, x)
        return x

    cfg = RealtimeConfig(
        dt=0.01,
        wallclock_dt=0.01,
        max_overruns=3,
        log_overruns=True,
        propagate_errors=False,
    )
    runner = RealtimeRunner(very_slow_deriv, dummy_step, x0, cfg, clock=clock)

    with caplog.at_level("WARNING"):
        stats = runner.run(0.1)

    assert not stats.running
    assert stats.stop_reason == StopReason.MAX_OVERRUNS
    assert "Max consecutive overruns (3) exceeded" in (stats.stop_message or "")
    assert stats.overruns == 3
    assert stats.dropped_deadlines == 3
    assert stats.sleep_time == 0.0

    # Verify overrun warning was logged
    overrun_warnings = [
        rec for rec in caplog.records if "Real-time overrun" in rec.message
    ]
    assert len(overrun_warnings) == 3


def test_overrun_logging_disabled_does_not_log(caplog):
    clock = FakeClock()
    x0 = np.array([1.0])

    def slow_deriv(t, x):
        clock.advance(0.015)
        return np.zeros_like(x)

    cfg = RealtimeConfig(
        dt=0.01,
        wallclock_dt=0.01,
        max_overruns=2,
        log_overruns=False,
        propagate_errors=False,
    )
    runner = RealtimeRunner(
        slow_deriv, lambda fn, t, x, dt: (fn(t, x), x)[1], x0, cfg, clock=clock
    )

    with caplog.at_level("WARNING"):
        runner.run(0.1)

    overrun_warnings = [
        rec for rec in caplog.records if "Real-time overrun" in rec.message
    ]
    assert len(overrun_warnings) == 0


def test_repeated_runs_continuity_and_reset():
    runner = _make_runner(duration=0.5, dt=0.01)

    stats1 = runner.run(0.2)
    assert stats1.steps_completed == 20
    assert stats1.sim_time == pytest.approx(0.2)
    assert stats1.stop_reason == StopReason.COMPLETED
    state_at_02 = runner.state.copy()

    # Continue from 0.2 to 0.5 without reset
    stats2 = runner.run(0.5)
    assert stats2.steps_completed == 50
    assert stats2.sim_time == pytest.approx(0.5)
    assert stats2.stop_reason == StopReason.COMPLETED

    # Reset runner and verify it restarts cleanly
    runner.reset()
    assert runner.stats.steps_completed == 0
    assert runner.stats.sim_time == 0.0
    assert runner.stats.stop_reason == StopReason.NONE
    assert runner.stats.compute_time == 0.0
    assert runner.stats.sleep_time == 0.0

    stats_rerun = runner.run(0.2)
    assert stats_rerun.steps_completed == 20
    np.testing.assert_allclose(runner.state, state_at_02)


def test_run_rejects_t_final_less_than_current_sim_time():
    runner = _make_runner(duration=0.5, dt=0.01)
    runner.run(0.3)
    with pytest.raises(
        ValueError, match="must not be less than current simulation time"
    ):
        runner.run(0.1)


def test_realtime_clock_default_methods():
    """Verify default Clock methods execute without error."""
    clk = Clock()
    t1 = clk.time()
    assert t1 > 0
    clk.sleep(0.001)
    t2 = clk.time()
    assert t2 >= t1
    # Sleep duration <= 0 is a no-op
    clk.sleep(-0.01)


def test_step_rejects_invalid_step_dt():
    runner = _make_runner(dt=0.01)
    with pytest.raises(ValueError, match="step_dt must be finite and positive"):
        runner.step(-0.01)
    with pytest.raises(ValueError, match="step_dt must be finite and positive"):
        runner.step(0.0)
    with pytest.raises(ValueError, match="step_dt must not exceed the configured dt"):
        runner.step(0.02)


def test_step_noop_when_paused_or_not_running():
    runner = _make_runner(dt=0.01)
    runner.pause()
    runner.step(0.01)
    assert runner.stats.steps_completed == 0

    runner.resume()
    runner.stop("Cancelled")
    runner.step(0.01)
    assert runner.stats.steps_completed == 0


def test_telemetry_callback_invoked():
    telemetry_calls = []

    def on_telemetry(stats, state):
        telemetry_calls.append((stats.steps_completed, state.copy()))

    runner = _make_runner(dt=0.01)
    runner.telemetry = on_telemetry
    runner.step(0.01)
    assert len(telemetry_calls) == 1
    assert telemetry_calls[0][0] == 1


def test_error_propagation_true_on_max_overruns():
    clock = FakeClock()
    x0 = np.array([1.0])

    def slow_deriv(t, x):
        clock.advance(0.05)
        return np.zeros_like(x)

    cfg = RealtimeConfig(
        dt=0.01,
        wallclock_dt=0.01,
        max_overruns=2,
        propagate_errors=True,
    )
    runner = RealtimeRunner(
        slow_deriv, lambda fn, t, x, dt: (fn(t, x), x)[1], x0, cfg, clock=clock
    )

    with pytest.raises(RuntimeError, match="Max consecutive overruns"):
        runner.run(0.1)

    assert not runner.stats.running
    assert runner.stats.stop_reason == StopReason.MAX_OVERRUNS


def test_pause_during_run_stops_loop():
    def on_telemetry(stats, state):
        if stats.steps_completed == 3:
            runner.pause()

    runner = RealtimeRunner(
        lambda t, x: np.zeros_like(x),
        rk4_step,
        make_state(),
        RealtimeConfig(dt=0.01, wallclock_dt=0.0),
        clock=FakeClock(),
        telemetry=on_telemetry,
    )
    stats = runner.run(0.1)
    assert stats.steps_completed == 3
    assert stats.paused
    assert stats.stop_reason == StopReason.PAUSED


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_realtime_config_rejects_invalid_dt(dt):
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        RealtimeConfig(dt=dt)


@pytest.mark.parametrize("wallclock_dt", [-0.1, np.nan, np.inf])
def test_realtime_config_rejects_invalid_wallclock_dt(wallclock_dt):
    with pytest.raises(ValueError, match="wallclock_dt"):
        RealtimeConfig(wallclock_dt=wallclock_dt)


@pytest.mark.parametrize("max_overruns", [0, -1, True, 1.5])
def test_realtime_config_rejects_invalid_max_overruns(max_overruns):
    with pytest.raises(ValueError, match="max_overruns"):
        RealtimeConfig(max_overruns=max_overruns)


@pytest.mark.parametrize("propagate_errors", [1, "true", None, 0.0])
def test_realtime_config_rejects_invalid_propagate_errors(propagate_errors):
    with pytest.raises(TypeError, match="propagate_errors must be a boolean"):
        RealtimeConfig(propagate_errors=propagate_errors)


@pytest.mark.parametrize("t_final", [-0.1, np.nan, np.inf])
def test_run_rejects_invalid_t_final(t_final):
    with pytest.raises(ValueError, match="t_final"):
        _make_runner().run(t_final)
