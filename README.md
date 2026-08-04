# Generic Delta-Canard Fighter 6DOF Dynamics

A Python implementation of a nonlinear six-degree-of-freedom flight dynamics
model for a generic delta-canard fighter aircraft — built for education,
research prototyping, and flight-dynamics software engineering practice.

This is not a model of any real aircraft. All aerodynamic coefficients and
aircraft parameters are generic, configurable placeholders.

## Implementation status

| Capability | Status |
|---|---|
| State/control vectors (12-state, 5-control) | ✓ Implemented |
| Body/NED frame transformations (3-2-1 Euler) | ✓ Validated |
| Quaternion utilities (Hamilton algebra) | ✓ Implemented |
| Wind ⇄ body velocity conversion | ✓ Validated |
| ISA atmosphere (0–20 km) | ✓ Validated |
| Aircraft geometry and mass properties | ✓ Implemented |
| Nonlinear 6DOF equations of motion | ✓ Validated |
| Euler and RK4 integrators | ✓ Verified (4th-order convergence) |
| Generic aerodynamic model | ✓ Implemented (placeholder coefficients) |
| Generic propulsion model | ✓ Implemented (placeholder parameters) |
| Straight-and-level trim | ✓ Validated (residuals < 1e-6 × weight) |
| Numerical linearization | ✓ Validated (nonlinear vs linear < 20% over 0.5 s) |
| Stability/modal analysis | ✓ Implemented |
| Deterministic evaluation CLI | ✓ Implemented |
| CI (lint, format, test, coverage, eval) | ✓ Configured |
| Actuator dynamics | Not implemented |
| PID/LQR control | Not implemented |
| Real-time simulation | Not implemented |

## Reproducible result

```bash
python examples/evaluate_6dof.py --state-only
```

Example output (aero + propulsion model, default controls at zero, generic placeholder coefficients):

```json
{
  "dt": 0.01,
  "t_final": 10.0,
  "final_altitude_m": 5237.03,
  "final_speed_mps": 170.09,
  "max_alpha_deg": 2.86,
  "quaternion_norm_deviation": 0.0,
  "me_change_percent": -4.65,
  "any_nonfinite_state": false
}
```

Use `--gravity-only` for the gravity-only (ballistic) baseline:

## Model architecture

See `docs/model.md` for the full architecture description.

Key points:
- 12-state wind-axis formulation: [VT, α, β, p, q, r, φ, θ, ψ, x_N, y_E, h]
- 5-control vector: [δ_canard, δ_el_L, δ_el_R, δ_rudder, throttle]
- NED navigation frame, body-fixed frame, flat-earth assumption
- Pluggable force/moment models (aerodynamics, propulsion)
- `docs/conventions.md` — full coordinate and sign convention reference

## Aerodynamic model

**All coefficients are generic placeholders** — not representative of any real
aircraft. See `docs/aerodynamic_data.md` for the complete coefficient inventory.

The model uses:
- Linear lift, drag, and pitching moment in α
- Linear side-force and roll/yaw moments in β
- Linear control derivatives
- No Mach, Reynolds, or stall effects

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/hunkarsuci/generic_delta_canard_fighter_6dof.git
cd generic_delta_canard_fighter_6dof
python -m pip install -e ".[dev]"
```

## Running the nonlinear evaluation

```bash
python examples/evaluate_6dof.py --state-only
```

Options:
- `--config PATH` — JSON configuration file
- `--output PATH` — write JSON summary to file
- `--dt FLOAT` — integration step [s] (default 0.01)
- `--t-final FLOAT` — simulation duration [s] (default 10.0)
- `--integrator NAME` — `"rk4"` (default) or `"euler"`
- `--gravity-only` — gravity-only ballistic flight (default: aero + propulsion)
- `--state-only` — print metrics to stdout

## Trim diagnostic sweep

```bash
python examples/trim_sweep.py            # human-readable table
python examples/trim_sweep.py --json     # machine-readable output
```

Sweeps altitude × airspeed and reports convergence, residuals, and control
usage. Uses existing synthetic coefficients without retuning. Any trim
failures are documented with force/moment residuals.

## Computing trim

```bash
python -c "
from generic_delta_canard_fighter_6dof.trim import trim_straight_level
r = trim_straight_level(5000.0, 200.0)
print(f'alpha={r.alpha_deg:.3f} deg, throttle={r.throttle:.4f}, de={r.symmetric_elevon_deg:.3f} deg')
print(f'force residual: {r.force_residual_N}')
print(f'moment residual: {r.moment_residual_Nm}')
"
```

## Linearization and local validation

```bash
python -c "
from generic_delta_canard_fighter_6dof.trim import trim_straight_level
from generic_delta_canard_fighter_6dof.linearization import linearize
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry

r = trim_straight_level(5000.0, 200.0)
assert r.converged
geo = create_default_geometry()
fm = combined_forces_moments()
lin = linearize(r.state, r.control, geo, fm)
print(f'A shape: {lin.A.shape}, B shape: {lin.B.shape}')
print(f'Residual norm: {np.linalg.norm(lin.x_dot0):.2e}')
"
```

## Test and validation approach

```bash
python -m pytest -q                          # Run all tests
python -m pytest --cov=generic_delta_canard_fighter_6dof --cov-report=term-missing
python -m ruff format --check .              # Format check
python -m ruff check .                       # Lint
```

See `docs/validation.md` for the structured validation matrix with
requirement IDs (VAL-FRM-001 through VAL-LIN-006).

## Known limitations

1. **Aerodynamic coefficients are placeholder** — not validated against any real aircraft
2. **Wind-axis singularity** — model cannot handle VT → 0
3. **Euler angle singularity** — theta = ±90° causes failure
4. **No actuator dynamics** — instantaneous control deflection
5. **Flat Earth** — no curvature or rotation effects
6. **No wind/gust modeling**
7. **Linear aerodynamics** — no stall, no compressibility effects
8. **Structural rigidity** — no aeroelastic effects
9. **Simple propulsion** — thrust = T_max × throttle, no Mach/altitude dependence

## Planned work

- Actuator rate and position limits
- PID and LQR control laws
- Real-time simulation support
- Quaternion-based state formulation (removes Euler singularity)
- [OWNER INPUT REQUIRED]: Replace placeholder aerodynamic coefficients

## References

- Stevens, B. L., Lewis, F. L., & Johnson, E. N. — *Aircraft Control and Simulation* (3rd ed.)
- Zipfel, P. H. — *Modeling and Simulation of Aerospace Vehicle Dynamics* (3rd ed.)
- ISO 2533:1975 — *Standard Atmosphere*

## License

This project is licensed under the terms in the LICENSE file.
