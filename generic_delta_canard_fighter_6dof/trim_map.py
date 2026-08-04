"""Numerical trim-feasibility map over altitude-airspeed grid.

Extends single-point straight-and-level trim to a configurable sweep.
For every grid point reports convergence, alpha, pitch attitude,
throttle, control deflections, residuals, active bounds, and failure reason.

This is NOT a flight envelope. It is a computational trim map of the
generic placeholder-coefficient model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generic_delta_canard_fighter_6dof.trim import trim_straight_level


@dataclass
class TrimMapPoint:
    """Trim result at one grid point."""

    altitude_m: float
    airspeed_mps: float
    converged: bool
    alpha_deg: float = 0.0
    theta_deg: float = 0.0
    throttle: float = 0.0
    symmetric_elevon_deg: float = 0.0
    force_residual_N: float = 0.0
    moment_residual_Nm: float = 0.0
    scaled_residual_norm: float = 0.0
    nfev: int = 0
    failure_reason: str = ""


def compute_trim_map(
    altitudes_m: list[float],
    airspeeds_mps: list[float],
    *,
    alpha0_deg: float = 5.0,
) -> list[TrimMapPoint]:
    """Compute trim at every altitude-airspeed combination.

    Parameters
    ----------
    altitudes_m:
        List of altitudes [m] to sweep.
    airspeeds_mps:
        List of true airspeeds [m/s] to sweep.
    alpha0_deg:
        Initial guess for angle of attack [deg] at each point.

    Returns
    -------
    list[TrimMapPoint]
        One entry per grid point.
    """
    results: list[TrimMapPoint] = []

    for h in altitudes_m:
        for vt in airspeeds_mps:
            try:
                trim_result = trim_straight_level(h, vt, alpha0_deg=alpha0_deg)
                point = TrimMapPoint(
                    altitude_m=h,
                    airspeed_mps=vt,
                    converged=trim_result.converged,
                    alpha_deg=trim_result.alpha_deg,
                    theta_deg=trim_result.theta_deg,
                    throttle=trim_result.throttle,
                    symmetric_elevon_deg=trim_result.symmetric_elevon_deg,
                    force_residual_N=float(
                        np.linalg.norm(trim_result.force_residual_N)
                    ),
                    moment_residual_Nm=float(
                        np.linalg.norm(trim_result.moment_residual_Nm)
                    ),
                    scaled_residual_norm=trim_result.scaled_residual_norm,
                    nfev=trim_result.nfev,
                    failure_reason="" if trim_result.converged else trim_result.message,
                )
            except ValueError as exc:
                point = TrimMapPoint(
                    altitude_m=h,
                    airspeed_mps=vt,
                    converged=False,
                    failure_reason=str(exc),
                )
            results.append(point)

    return results


def trim_map_summary(results: list[TrimMapPoint]) -> dict[str, Any]:
    """Compute summary statistics for a trim map."""
    converged = [r for r in results if r.converged]
    failed = [r for r in results if not r.converged]

    summary: dict[str, Any] = {
        "total_points": len(results),
        "converged": len(converged),
        "failed": len(failed),
        "convergence_rate": len(converged) / max(len(results), 1),
    }

    if converged:
        alphas = [r.alpha_deg for r in converged]
        throttles = [r.throttle for r in converged]
        elevons = [r.symmetric_elevon_deg for r in converged]
        residuals = [r.scaled_residual_norm for r in converged]

        summary.update(
            {
                "alpha_deg_range": [min(alphas), max(alphas)],
                "throttle_range": [min(throttles), max(throttles)],
                "elevon_deg_range": [min(elevons), max(elevons)],
                "max_scaled_residual": max(residuals),
                "mean_nfev": int(np.mean([r.nfev for r in converged])),
            }
        )

    if failed:
        # Group by failure reason
        reasons: dict[str, int] = {}
        for r in failed:
            reason = r.failure_reason[:80] or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        summary["failure_reasons"] = reasons

    return summary


def write_trim_map_csv(results: list[TrimMapPoint], path: Path) -> None:
    """Write trim map results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "altitude_m",
                "airspeed_mps",
                "converged",
                "alpha_deg",
                "theta_deg",
                "throttle",
                "elevon_deg",
                "force_residual_N",
                "moment_residual_Nm",
                "scaled_residual_norm",
                "nfev",
                "failure_reason",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.altitude_m,
                    r.airspeed_mps,
                    int(r.converged),
                    r.alpha_deg,
                    r.theta_deg,
                    r.throttle,
                    r.symmetric_elevon_deg,
                    r.force_residual_N,
                    r.moment_residual_Nm,
                    r.scaled_residual_norm,
                    r.nfev,
                    r.failure_reason,
                ]
            )
