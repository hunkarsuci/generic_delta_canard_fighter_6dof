"""
Trim diagnostic sweep — maps where straight-and-level trim converges across
the computational envelope.

**All coefficients are synthetic/illustrative placeholder data.** This sweep
uses the existing repository parameters without retuning. Converged points
indicate only that the solver found a solution with the generic model, not
that any real aircraft can trim there. Failed points are reported as
documented trim failures with force/moment residuals.

Usage:
    python examples/trim_sweep.py              # default sweep
    python examples/trim_sweep.py --json       # machine-readable output
"""

from __future__ import annotations

import json
import sys

import numpy as np

from generic_delta_canard_fighter_6dof.trim import trim_straight_level


def _sweep_points() -> list[tuple[float, float]]:
    """Generate (altitude_m, airspeed_mps) sweep points."""
    altitudes = [0, 2000, 5000, 8000, 10000, 12000, 15000]
    airspeeds = [100, 150, 200, 250, 300]
    points = []
    for h in altitudes:
        for vt in airspeeds:
            points.append((float(h), float(vt)))
    return points


def run_trim_sweep() -> list[dict]:
    """Run the trim sweep and return structured results."""
    points = _sweep_points()
    results = []

    for h, vt in points:
        r = trim_straight_level(h, vt)
        entry = {
            "altitude_m": h,
            "airspeed_mps": vt,
            "converged": r.converged,
            "message": r.message,
            "nfev": r.nfev,
            "alpha_deg": round(r.alpha_deg, 4),
            "throttle": round(r.throttle, 6),
            "symmetric_elevon_deg": round(r.symmetric_elevon_deg, 4),
            "force_residual_norm_N": round(
                float(np.linalg.norm(r.force_residual_N)), 4
            ),
            "moment_residual_norm_Nm": round(
                float(np.linalg.norm(r.moment_residual_Nm)), 4
            ),
            "scaled_residual_norm": (
                round(r.scaled_residual_norm, 6) if r.converged else None
            ),
        }
        results.append(entry)

    return results


def _table(results: list[dict]) -> str:
    """Format results as a readable table."""
    lines = []
    header = (
        f"{'h [m]':>7s}  {'VT [m/s]':>8s}  {'conv':>5s}  "
        f"{'alpha [deg]':>10s}  {'thr':>8s}  {'de [deg]':>9s}  "
        f"{'|F| [N]':>9s}  {'|M| [Nm]':>9s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        status = "OK" if r["converged"] else "FAIL"
        lines.append(
            f"{r['altitude_m']:7.0f}  {r['airspeed_mps']:8.0f}  {status:>5s}  "
            f"{r['alpha_deg']:10.4f}  {r['throttle']:8.4f}  "
            f"{r['symmetric_elevon_deg']:9.4f}  "
            f"{r['force_residual_norm_N']:9.4f}  "
            f"{r['moment_residual_norm_Nm']:9.4f}"
        )

    converged = sum(1 for r in results if r["converged"])
    total = len(results)
    lines.append(f"\nConverged: {converged}/{total}")

    failures = [r for r in results if not r["converged"]]
    if failures:
        lines.append("\nTrim failures:")
        for f in failures:
            lines.append(
                f"  h={f['altitude_m']:.0f}m VT={f['airspeed_mps']:.0f}m/s: "
                f"{f['message']}"
            )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    results = run_trim_sweep()

    if "--json" in argv:
        print(json.dumps(results, indent=2))
        return 0

    print(_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
