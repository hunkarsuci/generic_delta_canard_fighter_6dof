"""Soft real-time simulation demonstration.

This is a soft real-time demonstration and does NOT provide deterministic
operating-system scheduling or hardware real-time guarantees.

Run: python examples/run_realtime.py [--deterministic] [--duration 2]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.integrators import rk4_step
from generic_delta_canard_fighter_6dof.realtime import (
    Clock,
    FakeClock,
    RealtimeConfig,
    RealtimeRunner,
    RealtimeStats,
)
from generic_delta_canard_fighter_6dof.state import make_control, make_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Soft real-time 6DOF runner")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run in deterministic mode (fake clock, as-fast-as-possible)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Simulation duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration step size (default: 0.01)",
    )
    args = parser.parse_args(argv)

    geo = create_default_geometry()
    ctrl = make_control()

    x0 = make_state(
        VT=200.0,
        alpha=0.05,
        beta=0.0,
        phi=0.1,
        theta=0.05,
        h=5000.0,
    )

    def deriv(t, x):
        return aircraft_dynamics(t, x, ctrl, geo)

    cfg = RealtimeConfig(
        dt=args.dt,
        wallclock_dt=0.0 if args.deterministic else args.dt,
        max_overruns=10,
    )

    clock = FakeClock() if args.deterministic else Clock()

    def telemetry(stats: RealtimeStats, state: np.ndarray) -> None:
        """Print status every 0.5s of sim time."""
        if stats.steps_completed % max(1, int(0.5 / cfg.dt)) == 0:
            h = float(state[-1]) if len(state) >= 12 else float(state[12])
            print(
                f"t={stats.sim_time:.2f}s  "
                f"wall={stats.wall_time:.3f}s  "
                f"alt={h:.1f}m  "
                f"overruns={stats.overruns}"
            )

    runner = RealtimeRunner(
        derivative_fn=deriv,
        step_fn=rk4_step,
        x0=x0,
        config=cfg,
        clock=clock,
        telemetry=telemetry,
    )

    mode = "deterministic (fake clock)" if args.deterministic else "soft real-time"
    print(f"Starting {mode}, duration={args.duration}s, dt={cfg.dt}s")
    t0 = time.perf_counter()
    stats = runner.run(args.duration)
    elapsed = time.perf_counter() - t0

    print(f"\nCompleted in {elapsed:.3f}s wall time")
    print(f"Sim time: {stats.sim_time:.3f}s")
    print(f"Steps: {stats.steps_completed}")
    print(f"Overruns: {stats.overruns}")
    print(f"Dropped deadlines: {stats.dropped_deadlines}")
    print(f"Final state valid: {np.all(np.isfinite(runner.state))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
