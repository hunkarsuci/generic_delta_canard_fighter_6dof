# Generic Delta-Canard Fighter 6DOF Dynamics

A Python implementation of a nonlinear six-degree-of-freedom flight dynamics
model for a generic delta-canard fighter aircraft. Built for education,
research prototyping, and flight-dynamics software engineering practice.

This is not a model of any real aircraft. All aerodynamic coefficients and
aircraft parameters are generic, configurable placeholders.

## Implementation status

| Capability | Implementation | Verification level | Evidence | Limitation |
|---|---|---|---|---|
| State/control vectors | Complete | Unit verified | Shape, finiteness, range tests | 12-state Euler; 13-state quaternion also available |
| Body/NED transforms (3-2-1) | Complete | Numerically verified | Orthonormality, det=+1, geometric 90 degree checks | Euler singularity at theta = +/-90 deg |
| Quaternion utilities | Complete | Numerically verified | Euler-quaternion-DCM round-trips, double-cover | Scalar-first Hamilton convention |
| Wind to body conversion | Complete | Unit verified | Round-trip and sign-consistency tests | VT -> 0 singularity |
| ISA atmosphere (0--20 km) | Complete | Numerically verified | Sea-level and tropopause values against ISO 2533:1975 | Sampled points only; geometric altitude used |
| Geometry and mass properties | Complete | Unit verified | Inertia positive-definite check, defaults, validation | CG offsets stored but not consumed by dynamics |
| Nonlinear 6DOF equations of motion | Complete | Numerically verified | Newton's second law, Coriolis, kinematic invariants | Beta -> +/-90 deg guarded; gravity hard-wired |
| Euler and RK4 integrators | Complete | Numerically verified | Fourth-order convergence via step-halving | Fixed-step only |
| Generic aerodynamic model | Complete | Unit and numerical verification | Dimensional, symmetry, sign, and bounds tests | Synthetic placeholder coefficients; no wind-tunnel or flight-data validation |
| Generic propulsion model | Complete | Unit and numerical verification | Throttle and thrust-axis tests (indirect) | Synthetic parameters; no engine-deck validation |
| Straight-and-level trim | Complete | Model-level validated | Residual below 1e-6 times weight, reproducibility, multi-point grid | Valid for generic placeholder model only; not a flight envelope |
| Numerical linearization | Complete | Numerically verified | Nonlinear vs linear agreement over short horizon, perturbation sensitivity | Euler-angle coordinates; invalid near theta = +/-90 deg |
| Stability/modal analysis | Complete | Unit verified | Stable/unstable/oscillatory classification on synthetic systems | Does not label modes or pair complex conjugates |
| 13-state quaternion dynamics | Complete | Numerically verified | Norm preservation, analytic rates, 90 degree yaw, pitch-90 degree passage, Euler/quaternion agreement | Same placeholder aerodynamic model |
| Actuator dynamics | Complete | Unit verified | Step response, position/rate saturation, reset | Synthetic default time constants and limits |
| Control allocation | Complete | Unit verified | Symmetric/differential mixing, feasibility, determinism | Fixed linear mapping; not optimal allocation |
| PID stabilization | Complete | Closed-loop model validated | Zero-error, integral, anti-windup, derivative filter, reset | Assumes simulated state availability; synthetic gains |
| LQR stabilization | Complete | Closed-loop model validated | Riccati residual, stable closed-loop eigenvalues at trim, Bryson scaling | Valid near design operating point |
| Trim-feasibility map | Complete | Model-level validated | Grid sweep, convergence rate, residual bounds | Computational map; not a flight envelope |
| Soft real-time runner | Complete | Software timing verified | Fake-clock, overrun detection, step-count, reset | Soft real-time; not a hard real-time system |
| PID vs LQR comparison CLI | Complete | Implemented | `examples/compare_controllers.py` | Single perturbation scenario |

All aerodynamic coefficients and propulsion parameters are generic placeholders
and are not physically validated against any real aircraft data.

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/hunkarsuci/generic_delta_canard_fighter_6dof.git
cd generic_delta_canard_fighter_6dof
python -m pip install -e ".[dev]"
```

## Reproducible result

```bash
python examples/evaluate_6dof.py --state-only
```

Example output (gravity-only, generic aerodynamic coefficients):

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

- 12-state wind-axis formulation: [VT, alpha, beta, p, q, r, phi, theta, psi, x_N, y_E, h]
- 13-state quaternion formulation: [u, v, w, p, q, r, q0, q1, q2, q3, x_N, y_E, h]
- 5-control vector: [delta_canard, delta_el_L, delta_el_R, delta_rudder, throttle]
- NED navigation frame, body-fixed frame, flat-earth assumption
- Pluggable force/moment models (aerodynamics, propulsion)
- `docs/conventions.md` -- full coordinate and sign convention reference

## Aerodynamic model

All coefficients are generic placeholders and not representative of any real
aircraft. See `docs/aerodynamic_data.md` for the complete coefficient inventory.

The model uses linear lift, drag, and pitching moment in alpha; linear
side-force and roll/yaw moments in beta; and linear control derivatives.
There is no Mach, Reynolds, or stall dependence.

## Running the nonlinear evaluation

```bash
python examples/evaluate_6dof.py --state-only
```

Options:

- `--config PATH` -- JSON configuration file
- `--output PATH` -- write JSON summary to file
- `--dt FLOAT` -- integration step [s] (default 0.01)
- `--t-final FLOAT` -- simulation duration [s] (default 10.0)
- `--state-only` -- print metrics to stdout

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
import numpy as np

r = trim_straight_level(5000.0, 200.0)
assert r.converged
geo = create_default_geometry()
fm = combined_forces_moments()
lin = linearize(r.state, r.control, geo, fm)
print(f'A shape: {lin.A.shape}, B shape: {lin.B.shape}')
print(f'Residual norm: {np.linalg.norm(lin.x_dot0):.2e}')
"
```

## PID and LQR control

The repository includes baseline PID and LQR stabilization controllers.

### PID

```bash
python -c "
from generic_delta_canard_fighter_6dof.control import AircraftPID
from generic_delta_canard_fighter_6dof.state import make_state
import numpy as np

pid = AircraftPID()
x = make_state(VT=200)
ref = np.array([0.0, 0.0, 0.0, 200.0])  # zero rate errors, trim speed
u = pid.step(x, ref, 0.01)
print(f'Virtual commands (pitch, roll, yaw, throttle): {u}')
"
```

### LQR

```bash
python -c "
from generic_delta_canard_fighter_6dof.trim import trim_straight_level
from generic_delta_canard_fighter_6dof.linearization import linearize
from generic_delta_canard_fighter_6dof.control import compute_lqr
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
import numpy as np

r = trim_straight_level(5000.0, 200.0)
lin = linearize(r.state, r.control, create_default_geometry(), combined_forces_moments())
lqr = compute_lqr(lin.A, lin.B)
print(f'Stabilizable: {lqr.stabilizable}')
print(f'K shape: {lqr.K.shape}')
print(f'CL eigenvalues: {np.real(lqr.closed_loop_eigenvalues[:6])}')
"
```

### PID vs LQR comparison

```bash
python examples/compare_controllers.py
```

## Real-time simulation

```bash
# Deterministic (fake clock, as-fast-as-possible)
python examples/run_realtime.py --deterministic --duration 2

# Soft real-time (wall-clock pacing)
python examples/run_realtime.py --duration 5
```

This is a soft real-time demonstration and does not provide deterministic
operating-system scheduling or hardware real-time guarantees.

## Test and validation

```bash
python -m pytest -q
python -m pytest --cov=generic_delta_canard_fighter_6dof --cov-report=term-missing
python -m ruff format --check .
python -m ruff check .
```

See `docs/validation.md` for the structured validation matrix.

## Known limitations

1. Aerodynamic coefficients are placeholders -- not validated against any real aircraft
2. Wind-axis singularity -- model cannot handle VT approaching 0
3. Euler angle singularity -- theta = +/-90 deg causes failure in Euler formulation (quaternion formulation available)
4. Flat Earth -- no curvature or rotation effects
5. No wind or gust modeling
6. Linear aerodynamics -- no stall, no compressibility effects
7. Structural rigidity -- no aeroelastic effects
8. Simple propulsion -- thrust = T_max * throttle, no Mach/altitude dependence
9. Synthetic actuator parameters -- time constants and limits are not based on real actuator data

## References

- Stevens, B. L., Lewis, F. L., & Johnson, E. N. -- *Aircraft Control and Simulation* (3rd ed.)
- Zipfel, P. H. -- *Modeling and Simulation of Aerospace Vehicle Dynamics* (3rd ed.)
- ISO 2533:1975 -- *Standard Atmosphere*

## License

This project is licensed under the terms in the LICENSE file.
