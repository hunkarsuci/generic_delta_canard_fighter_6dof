"""
Deterministic 6DOF nonlinear evaluation entry point.

Run from the repository root:

    python examples/evaluate_6dof.py

Optional arguments:

    --config PATH       JSON configuration file
    --output PATH       Write JSON summary to this file
    --dt FLOAT          Integration step size [s] (default 0.01)
    --t-final FLOAT     Simulation duration [s] (default 10.0)
    --integrator NAME   Integration method: "rk4" (default) or "euler"
    --gravity-only      Use zero non-gravitational forces (default: aero+propulsion)
    --state-only        Print final state and exit (no file output)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from generic_delta_canard_fighter_6dof.atmosphere import (
    isa_atmosphere,
)
from generic_delta_canard_fighter_6dof.constants import GRAVITY_MPS2
from generic_delta_canard_fighter_6dof.equations import (
    aircraft_dynamics,
    zero_forces_moments,
)
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.simulation import simulate
from generic_delta_canard_fighter_6dof.state import (
    StateIndex,
    make_control,
    make_state,
)


@dataclass
class EvalConfig:
    """Deterministic evaluation configuration."""

    dt: float = 0.01
    t_final: float = 10.0
    initial_state: dict[str, float] = field(default_factory=dict)
    control: dict[str, float] = field(default_factory=dict)
    integrator: str = "rk4"
    gravity_only: bool = False


def _default_initial_state() -> dict[str, float]:
    return {
        "VT": 200.0,
        "alpha": 0.05,
        "beta": 0.01,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "phi": 0.1,
        "theta": 0.05,
        "psi": 0.0,
        "x_N": 0.0,
        "y_E": 0.0,
        "h": 5000.0,
    }


def _default_control() -> dict[str, float]:
    return {
        "delta_canard": 0.0,
        "delta_elevon_left": 0.0,
        "delta_elevon_right": 0.0,
        "delta_rudder": 0.0,
        "throttle": 0.0,
    }


def _build_config_from_args(args: list[str]) -> EvalConfig:
    """Parse command-line arguments into an EvalConfig."""
    config = EvalConfig()
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            path = Path(args[i + 1])
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "dt" in data:
                config.dt = float(data["dt"])
            if "t_final" in data:
                config.t_final = float(data["t_final"])
            if "initial_state" in data:
                config.initial_state = data["initial_state"]
            if "control" in data:
                config.control = data["control"]
            if "integrator" in data:
                config.integrator = str(data["integrator"])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            # handled separately
            i += 2
        elif args[i] == "--dt" and i + 1 < len(args):
            config.dt = float(args[i + 1])
            i += 2
        elif args[i] == "--t-final" and i + 1 < len(args):
            config.t_final = float(args[i + 1])
            i += 2
        elif args[i] == "--integrator" and i + 1 < len(args):
            config.integrator = str(args[i + 1])
            i += 2
        elif args[i] == "--state-only":
            i += 1
        elif args[i] == "--gravity-only":
            config.gravity_only = True
            i += 1
        else:
            i += 1

    return config


def _get_output_path(args: list[str]) -> Path | None:
    for i, a in enumerate(args):
        if a == "--output" and i + 1 < len(args):
            return Path(args[i + 1])
    return None


def _compute_metrics(
    t: np.ndarray,
    x: np.ndarray,
    config: EvalConfig,
) -> dict[str, Any]:
    """Compute factual metrics from the simulation result."""
    VT_history = x[:, StateIndex.VT]
    alpha_history = x[:, StateIndex.ALPHA]
    beta_history = x[:, StateIndex.BETA]
    p_history = x[:, StateIndex.P]
    q_history = x[:, StateIndex.Q]
    r_history = x[:, StateIndex.R]
    h_history = x[:, StateIndex.H]
    x_N_final = float(x[-1, StateIndex.X_N])
    y_E_final = float(x[-1, StateIndex.Y_E])
    h_final = float(x[-1, StateIndex.H])
    VT_final = float(x[-1, StateIndex.VT])

    # Quaternion norm check: build quaternion from final Euler angles
    from generic_delta_canard_fighter_6dof.quaternions import euler_to_quaternion

    q_final = euler_to_quaternion(
        float(x[-1, StateIndex.PHI]),
        float(x[-1, StateIndex.THETA]),
        float(x[-1, StateIndex.PSI]),
    )
    quat_norm_deviation = float(abs(np.linalg.norm(q_final) - 1.0))

    # Mechanical energy diagnostic (kinetic + potential, normalized by mass)
    pe_per_mass = GRAVITY_MPS2 * h_history
    ke_per_mass = 0.5 * VT_history**2
    me_per_mass = ke_per_mass + pe_per_mass
    me_initial = float(me_per_mass[0])
    me_final = float(me_per_mass[-1])
    me_change_pct = (me_final - me_initial) / max(abs(me_initial), 1.0) * 100.0

    # Body rate max
    body_rate_history = np.sqrt(p_history**2 + q_history**2 + r_history**2)

    # Aerodynamic validity — check altitude range
    try:
        isa_atmosphere(float(np.min(h_history)))
        isa_atmosphere(float(np.max(h_history)))
        altitude_in_envelope = True
    except ValueError:
        altitude_in_envelope = False

    any_nonfinite = bool(np.any(~np.isfinite(x)))

    return {
        "dt": config.dt,
        "t_final": config.t_final,
        "integrator": config.integrator,
        "final_position_n_m": x_N_final,
        "final_position_e_m": y_E_final,
        "final_altitude_m": h_final,
        "final_speed_mps": VT_final,
        "max_speed_mps": float(np.max(VT_history)),
        "min_speed_mps": float(np.min(VT_history)),
        "max_alpha_deg": float(np.rad2deg(np.max(np.abs(alpha_history)))),
        "max_beta_deg": float(np.rad2deg(np.max(np.abs(beta_history)))),
        "max_body_rate_radps": float(np.max(body_rate_history)),
        "quaternion_norm_deviation": quat_norm_deviation,
        "me_change_percent": me_change_pct,
        "altitude_in_atmosphere_envelope": altitude_in_envelope,
        "any_nonfinite_state": any_nonfinite,
        "num_steps": len(t) - 1,
    }


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Run the deterministic evaluation and return metrics."""
    geo = create_default_geometry()

    # Merge defaults with user config
    init = _default_initial_state()
    init.update(config.initial_state)
    ctrl_dict = _default_control()
    ctrl_dict.update(config.control)

    x0 = make_state(**init)
    control = make_control(**ctrl_dict)

    # Force/moment model: aero + propulsion by default; gravity-only with flag
    if config.gravity_only:
        force_model = zero_forces_moments
    else:
        force_model = combined_forces_moments()

    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return aircraft_dynamics(t, x, control, geo, force_model)

    result = simulate(
        derivative, x0, config.dt, config.t_final, integrator=config.integrator
    )

    metrics = _compute_metrics(result.t, result.x, config)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, nonzero on failure."""
    if argv is None:
        argv = sys.argv[1:]

    state_only = "--state-only" in argv

    try:
        config = _build_config_from_args(argv)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        metrics = run_evaluation(config)
    except (ValueError, RuntimeError) as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 2

    output_path = _get_output_path(argv)

    if state_only:
        print(json.dumps(metrics, indent=2))
        return 0

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results written to {output_path}")
    else:
        # Default: print summary to stdout
        print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
