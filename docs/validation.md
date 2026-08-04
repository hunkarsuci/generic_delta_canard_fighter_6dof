# Validation Matrix

Each entry documents a requirement, the test that verifies it, and the result.

## Coordinate frames and transformations

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-FRM-001 | DCM is orthonormal (det=+1, R@R.T=I) | Fundamental rotation property | `test_body_to_ned_is_rotation_matrix`, `test_dcm_determinant_is_positive_one`, `test_dcm_columns_are_orthonormal` | 1e-12 | Pass |
| VAL-FRM-002 | Body→NED and NED→body are mutual inverses | Round-trip consistency | `test_ned_to_body_is_transpose_inverse`, `test_body_to_ned_and_ned_to_body_round_trip_vector` | 1e-12 | Pass |
| VAL-FRM-003 | Known 90° yaw maps body-x to NED east | Independent geometrical check | `test_yaw_90_degrees_body_forward_points_east` | 1e-12 | Pass |
| VAL-FRM-004 | Positive pitch produces negative NED-z component | Physical: nose-up → above horizon | `test_positive_pitch_body_forward_has_negative_down_component` | 1e-12 | Pass |
| VAL-FRM-005 | Euler→quat→DCM matches direct Euler DCM | Consistency between representations | `test_quaternion_to_dcm_matches_euler_dcm` | 1e-12 | Pass |
| VAL-FRM-006 | q and -q represent the same attitude | Double-cover property of SU(2) | `test_q_and_minus_q_same_dcm`, `test_negative_q_same_euler` | 1e-12 | Pass |

## Atmosphere

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-ATM-001 | Sea-level values match ISA standard | Authoritative reference | `test_isa_atmosphere_sea_level` | rel 1e-3 | Pass |
| VAL-ATM-002 | Tropopause values match ISA standard | Layer boundary verification | `test_isa_atmosphere_tropopause` | rel 2e-3 | Pass |
| VAL-ATM-003 | Pressure and density decrease monotonically | Physical requirement | `test_pressure_decreases_with_altitude`, `test_density_decreases_with_altitude` | — | Pass |
| VAL-ATM-004 | Altitudes outside [0, 20000] m rejected | Input validation | `test_isa_atmosphere_rejects_negative_altitude`, `test_isa_atmosphere_rejects_too_high_altitude` | — | Pass |

## Dynamics

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-DYN-001 | Gravity body force magnitude is m·g | Energy conservation | `test_gravity_body_magnitude_is_mg` | 1e-12 | Pass |
| VAL-DYN-002 | Translational accel without rotation is F/m | Newton's 2nd law | `test_translational_accel_no_rotation` | 1e-12 | Pass |
| VAL-DYN-003 | Coriolis term produces correct w_dot | Kinematic consistency | `test_translational_accel_coriolis` | 1e-12 | Pass |
| VAL-DYN-004 | Rotational accel with diagonal I is I⁻¹M | Euler's equation | `test_rotational_accel_no_gyroscopic` | 1e-12 | Pass |
| VAL-DYN-005 | Ixz coupling produces coupled response | Inertial coupling physics | `test_rotational_accel_with_Ixz_coupling` | 1e-12 | Pass |
| VAL-DYN-006 | Zero angular rates hold attitude constant | Kinematic consistency | `test_dynamics_zero_angular_rates_hold_attitude` | 1e-12 | Pass |
| VAL-DYN-007 | State derivatives are finite at representative condition | Numerical robustness | `test_dynamics_state_derivatives_are_finite` | — | Pass |

## Integrator

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-INT-001 | RK4 exhibits ~4th order convergence | Numerical analysis | `test_rk4_order_of_accuracy` | empirical order ∈ [3, 5] | Pass |
| VAL-INT-002 | Errors decrease monotonically with smaller dt | Consistency | `test_rk4_errors_decrease_with_dt` | — | Pass |
| VAL-INT-003 | Full-model solution converges with dt halving | Numerical verification | `test_full_model_convergence_with_dt_halving` | — | Pass |
| VAL-INT-004 | Invalid dt rejected | Input validation | `test_euler_rejects_invalid_dt`, `test_rk4_rejects_invalid_dt`, `test_rk4_rejects_nonfinite_dt` | — | Pass |

## Trim

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-TRM-001 | Nominal trim converges | Solver functionality | `test_trim_converges_at_nominal_condition` | — | Pass |
| VAL-TRM-002 | Force residuals below 1e-6 × weight | Physical equilibrium | `test_trim_residuals_below_threshold` | 1e-6 × weight | Pass |
| VAL-TRM-003 | Moment residuals below 1 N·m | Physical equilibrium | `test_trim_residuals_below_threshold` | 1.0 | Pass |
| VAL-TRM-004 | VT_dot, alpha_dot, q_dot < 1e-8 | Trim definition | `test_trim_vt_dot_near_zero`, `test_trim_alpha_dot_near_zero`, `test_trim_q_dot_near_zero` | 1e-8 | Pass |
| VAL-TRM-005 | Controls within bounds | Physical limits | `test_trim_controls_within_bounds` | — | Pass |
| VAL-TRM-006 | beta = 0, phi = 0, theta = alpha | Symmetric level flight | `test_trim_beta_zero`, `test_trim_phi_zero_level_wings`, `test_trim_theta_equals_alpha` | — | Pass |
| VAL-TRM-007 | Reproducible | Deterministic solver | `test_trim_reproducible` | 1e-12 | Pass |

## Linearization

| ID | Requirement | Rationale | Test | Tolerance | Result |
|---|---|---|---|---|---|
| VAL-LIN-001 | A is 12×12, B is 12×5 | Correct dimensions | `test_linearize_a_matrix_shape` | — | Pass |
| VAL-LIN-002 | All entries finite | Numerical quality | `test_linearize_a_entries_are_finite` | — | Pass |
| VAL-LIN-003 | Residual near zero at trim | Linearization about equilibrium | `test_linearize_residual_near_zero_at_trim` | 1e-8 | Pass |
| VAL-LIN-004 | Nonlinear vs linear agreement over 0.5 s | Short-horizon validation | `test_nonlinear_vs_linear_short_horizon` | rel 20% | Pass |
| VAL-LIN-005 | Reproducible | Deterministic computation | `test_linearize_reproducible` | 1e-12 | Pass |
| VAL-LIN-006 | Perturbation-size insensitive | Numerical robustness | `test_linearize_perturbation_sensitivity` | rel 10% | Pass |
