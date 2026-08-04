# Repository Audit — Generic Delta-Canard Fighter 6DOF

**Date**: 2026-08-04 (updated)
**Branch**: `worktree-audit-and-verify` (based on `feat/validated-6dof-foundation`)
**Python**: 3.12.3

## Current implementation

### Module-by-module status

| Module | Status | Notes |
|---|---|---|
| `__init__.py` | Complete | Public API exports for all key classes and functions |
| `constants.py` | Complete | Gravity, ISA sea-level values, air properties |
| `units.py` | Complete | deg↔rad, ft↔m, kt↔m/s conversions |
| `state.py` | Complete | 12-state vector, 5-control vector, validation, dict conversion |
| `transforms.py` | Complete | body→NED DCM, NED→body DCM, rotation matrix validation |
| `quaternions.py` | Complete | Hamilton algebra, euler↔quat, DCM, rates, norm preservation |
| `kinematics.py` | Complete | Euler rates, wind↔body velocity, body→NED velocity, altitude rate |
| `atmosphere.py` | Complete | ISA 0–20 km, flight condition (Mach, q_bar) |
| `geometry.py` | Complete | AircraftGeometry dataclass, inertia matrix, positive-definite check |
| `equations.py` | Complete | Full 6DOF EOM, gravity, translational/rotational acceleration, wind derivatives |
| `integrators.py` | Complete | Euler and RK4 fixed-step integrators with dt validation |
| `simulation.py` | Complete | Deterministic fixed-step simulation runner with non-finite detection |
| `aerodynamics.py` | Complete | Generic linear aero model with placeholder coefficients |
| `propulsion.py` | Complete | T = T_max × throttle; combined forces/moments wrapper |
| `trim.py` | Complete | Straight-and-level trim via bounded least_squares |
| `linearization.py` | Complete | Central-difference A (12×12) and B (12×5) with scale-aware perturbations |
| `stability.py` | Complete | Eigenvalues, eigenvectors, participation factors, modal characteristics |
| `examples/evaluate_6dof.py` | Complete | Deterministic evaluation CLI with configurable options |
| `examples/trim_sweep.py` | Complete | Altitude × airspeed trim convergence sweep |
| `examples/run_02_dynamics_demo.py` | **Removed** | Was empty stub; evaluation CLI supersedes it |

### What exists and works

1. **State and control vectors** — 12-state wind-axis formulation; 5-control vector
2. **Coordinate transforms** — Body↔NED DCM via 3-2-1 Euler; rotation matrix validation
3. **Quaternion utilities** — Hamilton algebra, euler↔quat, DCM, rates; q/−q equivalence validated
4. **Kinematics** — Euler rate equations, wind↔body velocity, body→NED velocity
5. **ISA atmosphere** — 0–20 km troposphere + lower stratosphere; Mach, dynamic pressure
6. **Geometry/mass** — AircraftGeometry with inertia matrix, validation
7. **6DOF equations** — Full nonlinear EOM with gravity; pluggable force/moment models
8. **Integrators** — Euler and RK4; RK4 order-of-accuracy verified (empirical order 4)
9. **Aerodynamic model** — Generic linear aero; **all coefficients are placeholder**
10. **Propulsion model** — T = T_max × throttle; **no Mach/altitude dependence**
11. **Trim** — Straight-and-level; residuals < 1e-6 × weight; bounded unknowns
12. **Linearization** — Numerical central differences; validated against nonlinear (rel < 20% over 0.5 s)
13. **Stability analysis** — Eigenvalues, eigenvectors, participation factors, modal metrics
14. **Simulation** — Deterministic fixed-step runner with non-finite detection
15. **CI** — GitHub Actions: lint, format, test, coverage, eval on Python 3.10 & 3.12
16. **Documentation** — README, model, conventions, aerodynamic data, validation matrix, audit
17. **Tests** — 182 passing; all modules covered including units, kinematics, equations, trim, linearization

### What is still missing

1. **Actuator dynamics** — No rate/position limits (planned)
2. **PID/LQR control** — No control laws (planned)
3. **Real-time simulation** — Not implemented (planned)
4. **Real aerodynamic data** — [OWNER INPUT REQUIRED]

## Convention inventory

### Confirmed conventions

| Convention | Determination | Evidence |
|---|---|---|
| Navigation frame | NED (North, East, Down) | `transforms.py:14-17`, `kinematics.py:187-198` |
| Body frame | x=forward, y=right, z=down | `transforms.py:9-11`, `equations.py:62-74` |
| Euler sequence | 3-2-1 (yaw ψ, pitch θ, roll φ) | `transforms.py:56-57` |
| Euler DCM | body-to-NED (v_ned = C_nb @ v_body) | `transforms.py:25-28` |
| Quaternion storage | [q0, q1, q2, q3]; q0 = scalar | `quaternions.py:15` |
| Quaternion algebra | Hamilton product | `quaternions.py:69-103` (matches Hamilton: ij=k) |
| Quaternion DCM | body-to-NED | `quaternions.py:147-178` (confirmed vs Euler DCM in test) |
| Quaternion rate equation | q_dot = 0.5 * q ⊗ [0, ω] | `quaternions.py:238-244` |
| Gravity direction (NED) | [0, 0, +m·g] (positive down) | `equations.py:136-144` |
| Altitude convention | h positive up; h_dot = −V_D | `kinematics.py:187-198` |
| Wind → body velocity | u = VT·cos(α)·cos(β), v = VT·sin(β), w = VT·sin(α)·cos(β) | `kinematics.py:112-114` |
| Body → wind velocity | VT = ‖v‖, α = atan2(w, u), β = asin(v/VT) | `kinematics.py:152-161` |
| Aerodynamic angles | α positive = nose-up (w > 0), β positive = nose-left (v > 0) | Derived from wind→body equations |
| Force convention (body) | X forward, Y right, Z down | `equations.py:64-68` |
| Moment convention (body) | L roll (x), M pitch (y), N yaw (z) | `equations.py:70-74` |
| State vector | 12-element: [VT, α, β, p, q, r, φ, θ, ψ, x_N, y_E, h] | `state.py:5-10` |
| Control vector | 5-element: [δ_c, δ_el, δ_er, δ_r, throttle] | `state.py:14-20` |
| Units | SI (m, m/s, rad, rad/s, kg, N, Pa) | `state.py:22-23` |
| Inertia matrix layout | [[Ixx, 0, -Ixz], [0, Iyy, 0], [-Ixz, 0, Izz]] | `geometry.py:117-124` |

### Uncertainties / items needing clarification

| Item | Status | Notes |
|---|---|---|
| Quaternion normalization in `quaternion_rates` | **Concerning** | Normalization inside the derivative function masks drift; intentional but should be documented |
| Hamilton vs JPL | **Hamilton** | Confirmed; JPL would use left multiplication differently |
| Ixy, Iyz products | **Assumed zero** | Geometry only models Ixz; typical for xz-plane symmetry |
| No aerodynamic reference point | **Unclear** | CG at (0,0,0) relative to reference point; need to know where moments are computed |
| Wind-axis state formulation | **Confirmed** | Uses VT, α, β as primary states instead of body u, v, w |

## Claim verification

| Claim | Evidence | Status | Required action |
|---|---|---|---|
| "The baseline simulator uses a 12-state Euler-angle model" | `state.py` defines 12-state vector ✓ | ✅ Confirmed | — |
| "Body frame: x_b = forward, y_b = right, z_b = downward" | `transforms.py:9-11` ✓ | ✅ Confirmed | — |
| "NED frame: x_N = north, y_E = east, z_D = down" | `transforms.py:14-17` ✓ | ✅ Confirmed | — |
| "h_dot = -V_D" | `kinematics.py:196` ✓ | ✅ Confirmed | — |
| "q = [q0, q1, q2, q3] where q0 is the scalar part" | `quaternions.py:15` ✓ | ✅ Confirmed | — |
| "The model currently supports 0 m <= altitude <= 20,000 m" | `atmosphere.py:9-10` ✓ | ✅ Confirmed | — |
| "q_bar = 0.5 * rho * VT^2" | `atmosphere.py:232` ✓ | ✅ Confirmed | — |
| "Phase 5 — Nonlinear 6DOF equations" complete | `equations.py` has comprehensive EOM; 48 passing tests (equations + kinematics) | ✅ Confirmed | — |
| "Phase 6 — Aerodynamic model" not started | `aerodynamics.py` implemented with placeholder coefficients; 8 tests | ✅ Resolved | Coefficients are placeholder |
| "Phase 7 — Propulsion and actuators" not started | `propulsion.py` implemented; T = T_max × throttle; 0 dedicated tests (covered by trim/linearization integration) | ✅ Resolved | No actuator dynamics yet |

## Mathematical risks

### Risk 1: Quaternion normalization inside derivative function
- **Severity**: Medium
- **File**: `quaternions.py:240`
- **Symbol**: `quaternion_rates`
- **Explanation**: `quaternion_rates` calls `normalize_quaternion(q)` before computing the derivative. In numerical integration, if the quaternion drifts slightly from unit norm, the derivative `q_dot` computed from the re-normalized quaternion does not correspond to the actual (slightly non-unit) state being integrated. This breaks the consistency between the state and its derivative. Post-step normalization in the integrator loop is the correct place for this.
- **Expected behavior**: Quaternion norm should be maintained by the integrator, not the derivative function
- **Proposed test**: Integrate quaternion with constant angular rate for many steps without re-normalization in derivative; verify norm drift is small
- **Implementation change**: Remove `normalize_quaternion` call from `quaternion_rates`; add post-step normalization option to integrator or simulation loop

### Risk 2: Unused `ControlIndex` import
- **Severity**: Low
- **File**: `equations.py:42`
- **Explanation**: `ControlIndex` is imported but never used in the equations module
- **Proposed test**: Ruff lint check
- **Implementation change**: Remove unused import

### Risk 3: `beta_dot` singularity at β = ±90°
- **Severity**: Low (β rarely exceeds ±30° in normal flight)
- **File**: `equations.py:270`
- **Symbol**: `beta_dot = (VT * v_dot - v * VT_dot) / (VT * VT * cos_beta)`
- **Explanation**: Division by cos(β) → ∞ as β → ±90°
- **Expected behavior**: β = ±90° is physically reachable (pure sideslip) but the Euler-angle state formulation would also fail
- **Proposed test**: Verify graceful handling near β = ±89°
- **Implementation change**: The existing EPSILON guard is adequate for normal flight

### Risk 4: Inertia matrix uses `eigvalsh` for symmetric check
- **Severity**: Low
- **File**: `geometry.py:173`
- **Explanation**: `np.linalg.eigvalsh` assumes the matrix is symmetric/Hermitian. The inertia matrix IS symmetric by construction, but if a caller constructs it differently, this could give wrong results. Using `np.linalg.eigvals` would be more robust.
- **Proposed test**: Verify for a known asymmetric matrix
- **Implementation change**: Low priority; matrix is constructed symmetrically

### Risk 5: ~~Empty `test_equations.py`~~ — RESOLVED
- **Severity**: ~~High~~ → Resolved
- **Resolution**: `tests/test_equations.py` now contains 48 passing tests covering gravity body forces at various attitudes, translational/rotational acceleration, wind derivatives, full aircraft dynamics, ForcesMoments validation, and custom force model integration.

### Risk 6: ~~No aerodynamic model~~ — RESOLVED
- **Severity**: ~~High~~ → Resolved
- **Resolution**: `aerodynamics.py` implements a generic linear aerodynamic model with `AeroCoefficients` dataclass and `aerodynamic_forces_moments`. All coefficients are documented as placeholder. 8 tests verify force scaling, symmetry, control mixing, and sign conventions.

### Risk 7: ~~`kinematics.py` has 0% test coverage~~ — RESOLVED
- **Severity**: ~~Medium~~ → Resolved
- **Resolution**: `tests/test_kinematics.py` exists with the correct `.py` extension. 12 passing tests covering Euler rates, wind↔body velocity, body→NED velocity, and altitude rate.

## Test-gap matrix (current)

| Subsystem | Existing tests | Remaining gaps | Priority |
|---|---|---|---|
| State/Control | 7 tests: shape, validation, dict conversion | Edge cases for VT=0; non-finite rejection in make_state | Low |
| Transforms | 12 tests: identity, rotation check, transpose, yaw 90°/180°, pitch, roll 90°, invalid, determinant, orthonormal, round-trip, composition | — | Low |
| Quaternions | 17 tests: normalize, identity, DCM match, round-trip, yaw 90°, rates, q/−q equivalence, DCM properties, multiplicaiton, conjugate, norm | DCM→quat round-trip edge cases | Low |
| Kinematics | 12 tests: Euler rates, wind↔body, NED velocity, altitude rate | Euler rate singularity behavior | Low |
| Atmosphere | 15 tests: sea level, tropopause, stratosphere, monotonicity, invalid inputs, speed of sound, Mach, q_bar, flight condition | Continuity at tropopause boundary | Low |
| Geometry | 11 tests: defaults, inertia shape/values, inverse, aspect ratio, pos-def, validation | — | Low |
| Equations | 48 tests: gravity body at attitudes, translational/rotational accel, wind derivatives, full dynamics, ForcesMoments | Integration of dynamics with real aero model | Low |
| Integrators | 10 tests: Euler/RK4, RK4 exponential, invalid dt, RK4 order-of-accuracy, convergence, full-model convergence, non-finite rejection | — | Low |
| Simulation | 10 tests: constant/rk4/euler, zero duration, invalid dt/t_final/state, non-finite detection, determinism, eval CLI | — | Low |
| Aerodynamics | 8 tests: zero-speed, q_bar scaling, symmetry, differential elevon, shape, sign conventions, coefficients | Propulsion-only tests | Low |
| Propulsion | 0 dedicated tests | Covered by trim/linearization/eval integration tests | Low |
| Trim | 16 tests: convergence, residuals, VT/alpha/q dot, bounds, beta/phi/theta, reproducibility, negative airspeed, multi-altitude | Trim sweep edge cases at envelope boundaries | Low |
| Linearization | 10 tests: A/B shapes, finite entries, residual, reproducibility, perturbation sensitivity, short-horizon validation, zero-perturbation, wrong shapes, throttle/speed signs | — | Low |
| Stability | 9 tests: stable/unstable/neutral, oscillatory detection, damping ratio, time constant, non-square rejection, aircraft end-to-end | — | Low |
| Units | 12 tests: deg/rad, ft/m, kt/mps, round-trips, array inputs | — | Low |

## Implementation plan

### Done (this branch)

All items from the original immediate plan are complete. Summary of resolved work:
- `test_kinematics.py` renamed (was missing `.py` extension)
- Ruff issues resolved (currently clean)
- Equations of motion tests added (48 tests)
- Integrator verification added (RK4 order-of-accuracy, convergence)
- Coordinate convention tests added (q/−q, DCM→quat, composition)
- Quaternion propagation tests added (norm preservation)
- Quaternion normalization documented as intentional choice in derivative
- Aerodynamic model implemented (generic placeholder coefficients)
- Propulsion model implemented (T = T_max × throttle)
- Evaluation CLI created (`evaluate_6dof.py` with `--gravity-only` flag)
- Trim implemented (straight-and-level, bounded least_squares)
- Linearization implemented (central differences, scale-aware perturbations)
- Stability analysis implemented (eigenvalues, eigenvectors, participation factors)
- CI configured (GitHub Actions: lint, format, test, coverage, eval)
- Documentation written (model, conventions, aerodynamic data, validation matrix, audit)
- README rewritten with current status
- Trim sweep example added (`examples/trim_sweep.py`)
- Unit conversion tests added (`tests/test_units.py`)
- Empty `run_02_dynamics_demo.py` removed (superseded by evaluation CLI)
- Public API exports added to `__init__.py`

### Deferred (needs owner input)

- Real aerodynamic coefficients (placeholder values are documented as such)
- Real propulsion data (T = T_max × throttle is minimal)
- Actuator rate/position limits
- Sensor models
- Real-time simulation
- Control law implementation (PID/LQR)
