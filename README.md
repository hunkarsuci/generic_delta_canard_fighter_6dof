# Generic Delta-Canard Fighter 6DOF Dynamics

A Python implementation of a nonlinear six-degree-of-freedom flight dynamics
model for a generic delta-canard fighter aircraft — built for education,
research prototyping, and flight-dynamics software engineering practice.

This is not a model of any real aircraft. All aerodynamic coefficients and
aircraft parameters are generic, configurable placeholders.

## Implementation status

| Capability | Implementation | Verification level | Evidence | Limitation |
|---|---|---|---|---|
| State/control vectors | Complete | Unit verified | Shape, finiteness, range tests | 12-state Euler; 13-state quat also available |
| Body/NED transforms (3-2-1) | Complete | Numerically verified | Orthonormality, det=+1, geometric 90° checks | Euler singularity at θ=±90° |
| Quaternion utilities | Complete | Numerically verified | Euler↔quat↔DCM round-trips, double-cover | Scalar-first Hamilton convention |
| Wind ⇄ body conversion | Complete | Unit verified | Round-trip and sign-consistency tests | VT→0 singularity returns (0,0,0) |
| ISA atmosphere (0–20 km) | Complete | Numerically verified | Sea-level and tropopause vs ISO 2533:1975 | Sampled points only; geometric altitude used |
| Geometry and mass properties | Complete | Unit verified | Inertia PD check, defaults, validation | CG offsets stored but unused |
| Nonlinear 6DOF EoM | Complete | Numerically verified | Newton's 2nd law, Coriolis, kinematic invariants | β→±90° guarded; gravity hard-wired |
| Euler and RK4 integrators | Complete | Numerically verified | 4th-order convergence via step-halving | Fixed-step only |
| Generic aerodynamic model | Complete | Unit and numerical verified | Dimensional, symmetry, sign, bounds tests | **Synthetic placeholder coefficients** — no wind-tunnel or flight-data validation |
| Generic propulsion model | Complete | Unit and numerical verified | Throttle, thrust-axis tests (indirect) | **Synthetic parameters** — no engine-deck validation |
| Straight-and-level trim | Complete | Model-level validated | Residual < 1e-6·weight, reproducibility, multi-point grid | Valid for generic placeholder model only; not a flight envelope |
| Numerical linearization | Complete | Numerically verified | Nonlinear vs linear agreement over short horizon, perturbation sensitivity | Euler-angle coordinates; not valid near θ=±90° |
| Stability/modal analysis | Complete | Unit verified | Stable/unstable/oscillatory on synthetic systems | Does not label modes or pair complex conjugates |
| 13-state quaternion dynamics | Complete | Numerically verified | Norm preservation, analytic rates, 90° yaw, pitch-90° passage, Euler/quat agreement | Same placeholder aero model |
| Actuator dynamics | Complete | Unit verified | Step response, position/rate saturation, reset | Synthetic default time constants and limits |
| Control allocation | Complete | Unit verified | Symmetric/differential mixing, feasibility, determinism | Fixed linear mapping; not optimal allocation |
| PID stabilization | Complete | Closed-loop model validated | Zero-error, integral, anti-windup, derivative filter, reset | Assumes simulated state availability; synthetic gains |
| LQR stabilization | Complete | Closed-loop model validated | Riccati residual, stable CL eigenvalues at trim, Bryson scaling | Valid near design operating point |
| Trim-feasibility map | Complete | Model-level validated | Grid sweep, convergence rate, residual bounds | Computational map; not a flight envelope |
| Soft real-time runner | Complete | Software timing verified | Fake-clock, overrun detection, step-count, reset | Soft real-time; not a hard real-time system |
| PID vs LQR comparison CLI | Complete | Implemented | `examples/compare_controllers.py` | Single perturbation scenario |

All aerodynamic coefficients and propulsion parameters are **generic placeholders** and are **not physically validated** against any real aircraft data.

## Reproducible result

```bash
python examples/evaluate_6dof.py --state-only
```

Example output (gravity-only, generic aero coefficients):

```json
{
  "dt": 0.01,
  "t_final": 10.0,
  "final_altitude_m": 4508.17,
  "final_speed_mps": 222.81,
  "max_alpha_deg": 28.88,
  "quaternion_norm_deviation": 0.0,
  "me_change_percent": 8.43e-14,
  "any_nonfinite_state": false
}
```

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
- `--state-only` — print metrics to stdout

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
