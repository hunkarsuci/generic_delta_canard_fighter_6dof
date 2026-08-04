"""PID vs LQR controller comparison.

Compares PID and LQR using the same:
- Trimmed initial condition
- Command profiles
- Actuator dynamics
- Simulation step and duration
- Model (aerodynamics + propulsion)

Run: python examples/compare_controllers.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.actuators import ActuatorBank
from generic_delta_canard_fighter_6dof.allocation import allocate
from generic_delta_canard_fighter_6dof.control import (
    AircraftPID,
    LQRController,
    compute_lqr,
)
from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.integrators import rk4_step
from generic_delta_canard_fighter_6dof.linearization import linearize
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.state import (
    ControlIndex,
    StateIndex,
    make_control,
)
from generic_delta_canard_fighter_6dof.trim import trim_straight_level


@dataclass
class ScenarioMetrics:
    """Performance metrics for one control scenario."""

    name: str
    tracking_rmse: dict[str, float]
    settling_time_s: dict[str, float]
    overshoot: dict[str, float]
    steady_state_error: dict[str, float]
    control_energy: float
    max_deflection: dict[str, float]
    max_rate: dict[str, float]
    saturation_duration_s: float
    any_nonfinite: bool


def run_closed_loop(
    x0: np.ndarray,
    controller,
    actuators: ActuatorBank,
    t_final: float,
    dt: float,
    geo,
    fm,
    ref_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a closed-loop simulation.

    Returns (t, x_history, u_achieved_history, u_cmd_history).
    """
    n_steps = int(t_final / dt)
    n_states = len(x0)
    t = np.zeros(n_steps + 1)
    x_hist = np.zeros((n_steps + 1, n_states))
    u_ach_hist = np.zeros((n_steps + 1, 5))
    u_cmd_hist = np.zeros((n_steps + 1, 5))

    x = x0.copy()
    t[0] = 0.0
    x_hist[0] = x

    for i in range(n_steps):
        t_i = i * dt

        # Compute virtual commands from controller
        ref = ref_fn(t_i)
        if isinstance(controller, AircraftPID):
            virtual = controller.step(x, ref, dt)
            # Convert virtual [pitch, roll, yaw, throttle] to surface commands
            surface_cmd = allocate(
                pitch_cmd=float(virtual[0]),
                roll_cmd=float(virtual[1]),
                yaw_cmd=float(virtual[2]),
                throttle_cmd=float(virtual[3]),
            ).surface_commands
        else:
            # LQR: directly produces 5-control surface commands
            surface_cmd = controller.step(x)

        u_cmd_hist[i] = surface_cmd

        # Apply actuator dynamics
        achieved = actuators.step(surface_cmd, dt)
        u_ach_hist[i] = achieved

        # Build control vector from achieved positions
        control = make_control(
            delta_canard=float(achieved[0]),
            delta_elevon_left=float(achieved[1]),
            delta_elevon_right=float(achieved[2]),
            delta_rudder=float(achieved[3]),
            throttle=float(achieved[4]),
        )

        # Integrate
        def deriv(t, x, control=control):
            return aircraft_dynamics(t, x, control, geo, fm)

        x = rk4_step(deriv, t_i, x, dt)
        x_hist[i + 1] = x
        t[i + 1] = t_i + dt

    u_cmd_hist[-1] = u_cmd_hist[-2]
    u_ach_hist[-1] = u_ach_hist[-2]

    return t, x_hist, u_ach_hist, u_cmd_hist


def compute_metrics(
    name: str,
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> ScenarioMetrics:
    """Compute performance metrics from a closed-loop trajectory."""
    # Extract states
    VT = x[:, StateIndex.VT]
    phi = x[:, StateIndex.PHI]
    theta = x[:, StateIndex.THETA]
    p = x[:, StateIndex.P]
    q = x[:, StateIndex.Q]
    r = x[:, StateIndex.R]

    def _rmse(signal: np.ndarray, target: float = 0.0) -> float:
        return float(np.sqrt(np.mean((signal - target) ** 2)))

    def _settling_time(
        signal: np.ndarray, target: float, band: float, dt: float
    ) -> float:
        """Time until signal stays within band of target."""
        within = np.abs(signal - target) <= band
        # Find last crossing into the band
        for i in range(len(within) - 1, -1, -1):
            if not within[i]:
                return min((i + 1) * dt, t[-1])
        return 0.0

    def _overshoot(signal: np.ndarray, target: float) -> float:
        """Maximum overshoot relative to step size."""
        max_dev = float(np.max(np.abs(signal - target)))
        return max_dev

    def _sse(signal: np.ndarray, target: float) -> float:
        """Steady-state error (mean of last 20%)."""
        n = len(signal)
        tail = signal[int(0.8 * n) :]
        return float(np.mean(np.abs(tail - target)))

    # Control energy proxy
    control_energy = float(np.sum(np.diff(u, axis=0) ** 2) / dt)

    # Saturation
    sat_duration = 0.0  # simplified

    return ScenarioMetrics(
        name=name,
        tracking_rmse={
            "phi_rad": _rmse(phi),
            "theta_rad": _rmse(theta),
            "p_radps": _rmse(p),
            "q_radps": _rmse(q),
            "r_radps": _rmse(r),
            "VT_mps": _rmse(VT, float(VT[0])),
        },
        settling_time_s={
            "p": _settling_time(p, 0.0, 0.01, dt),
            "q": _settling_time(q, 0.0, 0.01, dt),
            "r": _settling_time(r, 0.0, 0.01, dt),
        },
        overshoot={
            "phi_rad": _overshoot(phi, 0.0),
            "theta_rad": _overshoot(theta, 0.0),
        },
        steady_state_error={
            "phi_rad": _sse(phi, 0.0),
            "theta_rad": _sse(theta, 0.0),
            "p_radps": _sse(p, 0.0),
            "q_radps": _sse(q, 0.0),
            "r_radps": _sse(r, 0.0),
        },
        control_energy=control_energy,
        max_deflection={
            "canard_deg": float(
                np.rad2deg(np.max(np.abs(u[:, ControlIndex.DELTA_CANARD])))
            ),
            "elevon_deg": float(np.rad2deg(np.max(np.abs(u[:, 1:3])))),
            "rudder_deg": float(
                np.rad2deg(np.max(np.abs(u[:, ControlIndex.DELTA_RUDDER])))
            ),
        },
        max_rate={"placeholder": 0.0},
        saturation_duration_s=sat_duration,
        any_nonfinite=bool(np.any(~np.isfinite(x))),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    if argv is None:
        argv = sys.argv[1:]

    print("Computing trim...")
    trim = trim_straight_level(5000.0, 200.0)
    if not trim.converged:
        print("ERROR: Trim did not converge", file=sys.stderr)
        return 1

    geo = create_default_geometry()
    fm = combined_forces_moments()

    # Linearize at trim for LQR
    print("Linearizing...")
    lin = linearize(trim.state, trim.control, geo, fm)

    # LQR controller
    print("Computing LQR gains...")
    lqr_result = compute_lqr(lin.A, lin.B)
    if not lqr_result.stabilizable:
        print("ERROR: LQR not stabilizable", file=sys.stderr)
        return 1

    lqr_ctrl = LQRController(lqr_result.K, trim.state, trim.control)

    # PID controller
    pid = AircraftPID()

    actuators_pid = ActuatorBank()
    actuators_lqr = ActuatorBank()

    dt = 0.01
    t_final = 5.0

    # Reference: zero rates, trim speed
    def ref_fn(t):
        return np.array([0.0, 0.0, 0.0, 200.0])

    # Small initial perturbation in roll rate
    x0 = trim.state.copy()
    x0[StateIndex.P] = 0.3  # 0.3 rad/s roll rate disturbance

    print(f"Running PID scenario ({t_final}s)...")
    t_pid, x_pid, u_ach_pid, _u_cmd_pid = run_closed_loop(
        x0, pid, actuators_pid, t_final, dt, geo, fm, ref_fn
    )

    print(f"Running LQR scenario ({t_final}s)...")
    t_lqr, x_lqr, u_ach_lqr, _u_cmd_lqr = run_closed_loop(
        x0, lqr_ctrl, actuators_lqr, t_final, dt, geo, fm, ref_fn
    )

    pid_metrics = compute_metrics("PID", t_pid, x_pid, u_ach_pid, dt)
    lqr_metrics = compute_metrics("LQR", t_lqr, x_lqr, u_ach_lqr, dt)

    report = {
        "trim": {
            "altitude_m": 5000.0,
            "airspeed_mps": 200.0,
            "alpha_deg": trim.alpha_deg,
            "throttle": trim.throttle,
        },
        "scenario": {
            "dt": dt,
            "t_final": t_final,
            "initial_perturbation_p_radps": 0.3,
        },
        "PID": {
            "roll_rate_rmse_radps": pid_metrics.tracking_rmse["p_radps"],
            "pitch_rate_rmse_radps": pid_metrics.tracking_rmse["q_radps"],
            "settling_p_s": pid_metrics.settling_time_s["p"],
            "max_elevon_deg": pid_metrics.max_deflection["elevon_deg"],
            "any_nonfinite": pid_metrics.any_nonfinite,
        },
        "LQR": {
            "roll_rate_rmse_radps": lqr_metrics.tracking_rmse["p_radps"],
            "pitch_rate_rmse_radps": lqr_metrics.tracking_rmse["q_radps"],
            "settling_p_s": lqr_metrics.settling_time_s["p"],
            "max_elevon_deg": lqr_metrics.max_deflection["elevon_deg"],
            "any_nonfinite": lqr_metrics.any_nonfinite,
        },
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
