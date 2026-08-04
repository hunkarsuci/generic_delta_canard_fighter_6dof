# Repository Audit — Generic Delta-Canard Fighter 6DOF

**Date**: 2026-08-04
**Branch**: `feat/validated-6dof-foundation` (created from `main` at `690af3c`)
**Python**: 3.12.3

## Current implementation

### Module-by-module status

| Module | Status | Lines | Coverage | Notes |
|---|---|---|---|---|
| `__init__.py` | Empty | 0 | — | No public API exports |
| `constants.py` | Complete | 8 | 100% | Gravity, ISA sea-level values, air properties |
| `units.py` | Complete | 14 | 0% | deg↔rad, ft↔m, kt↔m/s; no tests |
| `state.py` | Complete | 72 | 97% | 12-state vector, 5-control vector, validation, dict conversion |
| `transforms.py` | Complete | 20 | 100% | body→NED DCM, NED→body DCM, rotation matrix validation |
| `quaternions.py` | Complete | 53 | 91% | Normalize, conjugate, multiply, euler↔quat, DCM, rates |
| `kinematics.py` | Complete | 36 | 0% | Euler rates, wind↔body velocity, body→NED velocity, altitude rate |
| `atmosphere.py` | Complete | 76 | 100% | ISA 0–20 km, flight condition (Mach, q_bar) |
| `geometry.py` | Complete | 48 | 90% | AircraftGeometry dataclass, inertia matrix, positive-definite check |
| `equations.py` | Complete | 97 | 0% | Full 6DOF EOM, gravity, translational/rotational acceleration, wind derivatives |
| `integrators.py` | Complete | 20 | 100% | Euler and RK4 fixed-step integrators |
| `simulation.py` | **Empty** | 0 | — | No simulation runner implemented |
| `test_equations.py` | **Empty** | 0 | — | No dynamics tests |
| `run_02_dynamics_demo.py` | **Empty** | 0 | — | No dynamics demo |

### What exists and works

1. **State and control vectors** — 12-state wind-axis formulation (VT, α, β, p, q, r, φ, θ, ψ, x_N, y_E, h); 5-control vector (δ_canard, δ_elevon_L, δ_elevon_R, δ_rudder, throttle)
2. **Coordinate transforms** — Body↔NED DCM via 3-2-1 Euler angles; rotation matrix validation
3. **Quaternion utilities** — Hamilton algebra, euler↔quat, DCM conversion, kinematic rates
4. **Kinematics** — Euler rate equations, wind↔body velocity conversion, NED velocity
5. **ISA atmosphere** — 0–20 km troposphere + lower stratosphere; Mach, dynamic pressure
6. **Geometry/mass** — AircraftGeometry with inertia matrix, validation
7. **6DOF equations** — Full nonlinear EOM with gravity; zero-forces-moments placeholder
8. **Integrators** — Euler and RK4 with dt validation
9. **Tests** — 54 passing, 64% coverage

### What is missing

1. **Aerodynamic model** — No lift, drag, side-force, or moment coefficients
2. **Propulsion model** — No thrust model
3. **Actuator dynamics** — No rate/position limits
4. **Simulation runner** — No loop, no output, no scenario configuration
5. **Trim** — Not implemented
6. **Linearization** — Not implemented
7. **Stability analysis** — Not implemented
8. **CI** — No GitHub Actions or other CI
9. **Kinematics tests** — 0% coverage (test file exists but is named `test_kinematics` without `.py`, though it appears to be a Python file)
10. **Equations tests** — 0% coverage (file exists but is empty)
11. **Docs** — Only README

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
| "Phase 5 — Nonlinear 6DOF equations" complete | `equations.py` has EOM but 0% coverage; empty test file | ⚠️ Partial | Add EOM tests |
| "Phase 6 — Aerodynamic model" not started | No aero module exists | ✅ Accurately stated | Implement minimal aero model |
| "Phase 7 — Propulsion and actuators" not started | No propulsion module | ✅ Accurately stated | Implement minimal propulsion |

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

### Risk 5: Empty `test_equations.py`
- **Severity**: High
- **File**: `tests/test_equations.py`
- **Explanation**: The core dynamics module has 0% test coverage. Any equation error is undetected.
- **Proposed test**: Gravity-only ballistic trajectory, constant-velocity check, constant-angular-rate check
- **Implementation change**: Add comprehensive dynamics tests

### Risk 6: No aerodynamic model
- **Severity**: High
- **File**: N/A (missing)
- **Explanation**: Without aerodynamic forces, trim and linearization are impossible. The `zero_forces_moments` placeholder means only gravity acts.
- **Proposed test**: Not applicable until model exists
- **Implementation change**: Add minimal generic aerodynamic model with clearly documented placeholder coefficients

### Risk 7: `kinematics.py` has 0% test coverage
- **Severity**: Medium
- **File**: `tests/test_kinematics`
- **Explanation**: The test file `test_kinematics` (no `.py` extension in the directory listing but is a Python file) exists but doesn't appear in the coverage report as covering kinematics. Actually, looking at the coverage report, `kinematics.py` shows 36/36 missed — the test file exists but its tests apparently aren't being collected or the coverage tool isn't attributing correctly. Let me re-check: the file IS `tests/test_kinematics.py` (read returned it as `tests/test_kinematics` but `file` command says it's a Python script). Wait — there's a bug: the file has no `.py` extension. Let me verify.
- **Actually**: The Glob result and the Bash `ls` show `tests/test_kinematics` without `.py`. The `file` command says it IS a Python script. So the test file is named without the `.py` extension, which means pytest won't discover it. **This is a bug.**
- **Proposed test**: Rename file to `test_kinematics.py`
- **Implementation change**: Rename `tests/test_kinematics` → `tests/test_kinematics.py`

Wait — I need to verify this. The read tool did return content for `tests/test_kinematics`. Let me check the actual filename on disk.

## Test-gap matrix

| Subsystem | Existing tests | Missing verification | Priority |
|---|---|---|---|
| State/Control | 6 tests: shape, validation, dict conversion | Edge cases for VT=0; non-finite rejection in make_state | Low |
| Transforms | 6 tests: identity, rotation check, transpose, yaw 90°, pitch, invalid | Determinant=+1 specifically; composition order; 180° rotations; q and -q equivalence | Medium |
| Quaternions | 7 tests: normalize, identity, DCM match, round-trip, yaw 90°, rates | DCM→quat round-trip; -q equivalence; norm preservation under propagation; invalid dt rejection in rates | Medium |
| Kinematics | **Tests exist but file may be misnamed** (see Risk 7) | Euler rate singularity behavior near limits; wind/body round-trip near zero speed | High |
| Atmosphere | 12 tests: sea level, tropopause, stratosphere, monotonicity, invalid inputs, speed of sound, Mach, q_bar, flight condition | Continuity at tropopause boundary; exact tropopause values; stratosphere exponential form | Low |
| Geometry | 10 tests: defaults, inertia shape/values, inverse, aspect ratio, pos-def, validation | Symmetry of inertia matrix; product of inertia range validation | Low |
| Equations | **None** (file empty) | Gravity-only trajectory; constant velocity/angular rate; force transformation; quaternion norm through integration; state derivative shapes | **Critical** |
| Integrators | 4 tests: Euler/RK4 constant deriv, RK4 exponential, invalid dt | RK4 order-of-accuracy; full-model convergence with dt halving; non-finite state rejection | High |
| Simulation | **None** (module empty) | Deterministic run; output format; config validation; reproducibility | **Critical** |
| Aerodynamics | **None** (no module) | All aerodynamic tests | **Critical** |
| Propulsion | **None** (no module) | All propulsion tests | **Critical** |

## Implementation plan

### Immediate (this branch)

1. **Fix `test_kinematics` filename** — Rename to `test_kinematics.py` so pytest discovers it
2. **Fix ruff issues** — Remove unused imports
3. **Add equations of motion tests** — Gravity-only ballistic, constant velocity, constant angular rate
4. **Add integrator verification** — RK4 order-of-accuracy, convergence study
5. **Add coordinate convention tests** — q/−q equivalence, DCM→quat round-trip, composition order
6. **Add quaternion propagation tests** — Norm preservation, analytical single-axis solution
7. **Fix quaternion_rates** — Remove internal normalize; document choice
8. **Add minimal aerodynamic model** — Generic placeholder coefficients with provenance doc
9. **Add minimal propulsion model** — Simple thrust model
10. **Create nonlinear evaluation CLI** — `examples/evaluate_6dof.py`
11. **Implement trim** — Straight-and-level with bounded solver
12. **Implement numerical linearization** — Central differences with quaternion handling
13. **Add stability analysis** — Eigenvalues, damping, frequencies
14. **Add CI** — GitHub Actions workflow
15. **Write documentation** — model.md, conventions.md, validation.md, aerodynamic_data.md
16. **Rewrite README**

### Deferred (needs owner input)

- Real aerodynamic coefficients
- Real propulsion data
- Actuator rate/position limits
- Sensor models
- Real-time simulation

### OWNER INPUT REQUIRED

1. **Aerodynamic coefficients**: Are there specific coefficient tables, derivatives, or functional forms intended for this project? The current plan uses minimal placeholder values (CLα, CD0, Cmα, etc.) that are sufficient for educational validation but are not representative of any real aircraft.

2. **Propulsion model**: Should thrust be a simple function of throttle and airspeed (e.g., T = T_max * throttle * f(Mach)), or is a more detailed engine model expected?

3. **Reference point for moments**: The CG is at (0, 0, 0) relative to the reference point in `AircraftGeometry`. Is this intentional? Where should aerodynamic moments be computed relative to?

4. **Flight envelope**: What Mach, alpha, beta, and altitude ranges should the aerodynamic model support?

5. **Control surface limits**: What are the deflection limits and rate limits for each surface?
