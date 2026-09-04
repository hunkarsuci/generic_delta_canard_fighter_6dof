"""Soft real-time 6DOF simulation with optional live animation.

This is a soft real-time demonstration and does not provide deterministic
operating-system scheduling or hardware real-time guarantees.

Usage:
    python examples/run_realtime.py                           # 15 s soft real-time
    python examples/run_realtime.py --animate                 # 15 s with live plots
    python examples/run_realtime.py --animate --dt 0.005      # finer step
    python examples/run_realtime.py --deterministic           # fake clock, fast
    python examples/run_realtime.py --duration 30             # custom duration
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

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


class AnimatedTelemetry:
    """Live matplotlib animation of aircraft state during real-time simulation.

    Shows four panels: altitude, airspeed, attitude (roll/pitch), and angular rates.
    """

    def __init__(self, history_seconds: float = 10.0, dt: float = 0.01):
        self.dt = dt
        self.buffer_size = max(1, int(history_seconds / dt))

        self.t_history: deque[float] = deque(maxlen=self.buffer_size)
        self.h_history: deque[float] = deque(maxlen=self.buffer_size)
        self.vt_history: deque[float] = deque(maxlen=self.buffer_size)
        self.phi_history: deque[float] = deque(maxlen=self.buffer_size)
        self.theta_history: deque[float] = deque(maxlen=self.buffer_size)
        self.p_history: deque[float] = deque(maxlen=self.buffer_size)
        self.q_history: deque[float] = deque(maxlen=self.buffer_size)
        self.r_history: deque[float] = deque(maxlen=self.buffer_size)

        import matplotlib.pyplot as plt

        self.fig, ((self.ax_alt, self.ax_vt), (self.ax_att, self.ax_rate)) = (
            plt.subplots(2, 2, figsize=(10, 8))
        )
        self.fig.canvas.manager.set_window_title("6DOF Real-Time Simulation")

        (self.line_alt,) = self.ax_alt.plot([], [], "b-", linewidth=1.5)
        self.ax_alt.set_ylabel("Altitude [m]")
        self.ax_alt.set_xlabel("Time [s]")
        self.ax_alt.grid(True, alpha=0.3)

        (self.line_vt,) = self.ax_vt.plot([], [], "r-", linewidth=1.5)
        self.ax_vt.set_ylabel("Airspeed [m/s]")
        self.ax_vt.set_xlabel("Time [s]")
        self.ax_vt.grid(True, alpha=0.3)

        (self.line_phi,) = self.ax_att.plot([], [], "b-", linewidth=1.0, label="Roll")
        (self.line_theta,) = self.ax_att.plot(
            [], [], "r-", linewidth=1.0, label="Pitch"
        )
        self.ax_att.set_ylabel("Attitude [deg]")
        self.ax_att.set_xlabel("Time [s]")
        self.ax_att.legend(loc="upper right", fontsize=8)
        self.ax_att.grid(True, alpha=0.3)

        (self.line_p,) = self.ax_rate.plot(
            [], [], "b-", linewidth=1.0, label="p (roll)"
        )
        (self.line_q,) = self.ax_rate.plot(
            [], [], "r-", linewidth=1.0, label="q (pitch)"
        )
        (self.line_r,) = self.ax_rate.plot([], [], "g-", linewidth=1.0, label="r (yaw)")
        self.ax_rate.set_ylabel("Angular Rate [deg/s]")
        self.ax_rate.set_xlabel("Time [s]")
        self.ax_rate.legend(loc="upper right", fontsize=8)
        self.ax_rate.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.ion()
        self.fig.show()
        self._last_draw = 0.0
        self._draw_interval = 0.1  # redraw every 100 ms

    def __call__(self, stats: RealtimeStats, state: np.ndarray) -> None:
        """Record telemetry and update plots."""
        t = stats.sim_time

        self.t_history.append(t)
        self.h_history.append(float(state[-1]))
        self.vt_history.append(
            float(np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2))
            if len(state) == 13
            else float(state[0])
        )

        if len(state) == 13:
            from generic_delta_canard_fighter_6dof.quat_state import QuatStateIndex
            from generic_delta_canard_fighter_6dof.quaternions import (
                quaternion_to_euler,
            )

            q_vec = state[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
            phi, theta, _ = quaternion_to_euler(q_vec)
            p_val = float(state[QuatStateIndex.P])
            q_val = float(state[QuatStateIndex.Q])
            r_val = float(state[QuatStateIndex.R])
        else:
            from generic_delta_canard_fighter_6dof.state import StateIndex

            phi = float(state[StateIndex.PHI])
            theta = float(state[StateIndex.THETA])
            p_val = float(state[StateIndex.P])
            q_val = float(state[StateIndex.Q])
            r_val = float(state[StateIndex.R])

        self.phi_history.append(np.rad2deg(phi))
        self.theta_history.append(np.rad2deg(theta))
        self.p_history.append(np.rad2deg(p_val))
        self.q_history.append(np.rad2deg(q_val))
        self.r_history.append(np.rad2deg(r_val))

        # Throttle redraws
        if t - self._last_draw >= self._draw_interval:
            self._redraw()
            self._last_draw = t

    def _redraw(self) -> None:
        t_arr = np.array(self.t_history)

        self.line_alt.set_data(t_arr, np.array(self.h_history))
        self.line_vt.set_data(t_arr, np.array(self.vt_history))
        self.line_phi.set_data(t_arr, np.array(self.phi_history))
        self.line_theta.set_data(t_arr, np.array(self.theta_history))
        self.line_p.set_data(t_arr, np.array(self.p_history))
        self.line_q.set_data(t_arr, np.array(self.q_history))
        self.line_r.set_data(t_arr, np.array(self.r_history))

        for ax in [self.ax_alt, self.ax_vt, self.ax_att, self.ax_rate]:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finalize(self) -> None:
        """Switch to blocking mode so the window stays open."""
        import matplotlib.pyplot as plt

        plt.ioff()
        self.fig.show()
        print("\nAnimation window will close when you close the figure.")
        print("Press Ctrl+C in the terminal to exit.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Soft real-time 6DOF simulation runner"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run in deterministic mode (fake clock, as-fast-as-possible)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show live matplotlib animation of aircraft state",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Simulation duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration step size in seconds (default: 0.01)",
    )
    args = parser.parse_args(argv)

    geo = create_default_geometry()
    ctrl = make_control()

    x0 = make_state(
        VT=200.0,
        alpha=0.05,
        beta=0.01,
        phi=0.15,
        theta=0.05,
        psi=0.1,
        p=0.05,
        q=-0.03,
        r=0.02,
        h=5000.0,
    )

    def deriv(t: float, x: np.ndarray) -> np.ndarray:
        return aircraft_dynamics(t, x, ctrl, geo)

    cfg = RealtimeConfig(
        dt=args.dt,
        wallclock_dt=0.0 if args.deterministic else args.dt,
        max_overruns=20,
    )

    clock = FakeClock() if args.deterministic else Clock()

    telemetry = None
    animation = None

    if args.animate:
        animation = AnimatedTelemetry(history_seconds=10.0, dt=args.dt)
        telemetry = animation

        def status_telemetry(stats: RealtimeStats, state: np.ndarray) -> None:
            pass  # animation handles display

    else:

        def status_telemetry(stats: RealtimeStats, state: np.ndarray) -> None:
            if stats.steps_completed % max(1, int(1.0 / cfg.dt)) == 0:
                h = float(state[-1]) if len(state) >= 12 else float(state[12])
                vt = (
                    float(np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2))
                    if len(state) == 13
                    else float(state[0])
                )
                print(
                    f"t={stats.sim_time:6.2f}s  "
                    f"alt={h:8.1f}m  "
                    f"VT={vt:7.1f}m/s  "
                    f"overruns={stats.overruns:3d}"
                )

        telemetry = status_telemetry

    runner = RealtimeRunner(
        derivative_fn=deriv,
        step_fn=rk4_step,
        x0=x0,
        config=cfg,
        clock=clock,
        telemetry=telemetry,
    )

    mode = "deterministic (fake clock)" if args.deterministic else "soft real-time"
    anim_note = " with live animation" if args.animate else ""
    print(f"Starting {mode}{anim_note}, duration={args.duration}s, dt={cfg.dt}s\n")

    t0 = time.perf_counter()
    stats = runner.run(args.duration)
    elapsed = time.perf_counter() - t0

    print(f"\nCompleted in {elapsed:.3f}s wall time")
    print(f"Sim time: {stats.sim_time:.3f}s")
    print(f"Steps: {stats.steps_completed}")
    print(f"Overruns: {stats.overruns}")
    print(f"Dropped deadlines: {stats.dropped_deadlines}")

    state_valid = bool(np.all(np.isfinite(runner.state)))
    print(f"Final state valid: {state_valid}")

    if animation is not None and state_valid:
        try:
            animation.finalize()
            input("Press Enter to close...")
        except (KeyboardInterrupt, EOFError):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
