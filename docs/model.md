# 6DOF Model Architecture

## State-space formulation

The aircraft is modeled as a rigid body with 12 states and 5 controls in a
flat-earth, non-rotating atmosphere.

### State equations

The dynamics combine:

1. **Wind-axis translational kinematics**: VT, alpha, beta evolve from body accelerations
2. **Euler-angle rotational kinematics**: phi, theta, psi evolve from body angular rates
3. **Rigid-body dynamics**: Body accelerations from Newton-Euler equations
4. **Position kinematics**: NED position from velocity

$$\dot{x} = f(x, u)$$

where the function `f` is implemented in `equations.py:aircraft_dynamics`.

### Coordinate frames

- Navigation: NED (North-East-Down), assumed inertial
- Body: fixed to aircraft CG, x=forward, y=right, z=down
- See `docs/conventions.md` for full conventions.

### Force and moment models

Forces and moments come from pluggable models implementing the `ForceMomentModel`
protocol:

```python
ForceMomentModel = Callable[
    [float, np.ndarray, np.ndarray, AircraftGeometry],
    ForcesMoments,
]
```

Gravity is always applied internally in `aircraft_dynamics`. External models
provide non-gravitational forces only.

Current models:
- `aerodynamics.py` — linear aerodynamic model (generic placeholder coefficients)
- `propulsion.py` — simple thrust model (generic placeholder parameters)
- `equations.py:zero_forces_moments` — zero forces (for testing)

Each model is independent; they can be combined via `propulsion.combined_forces_moments`.

### Quaternion state representation and integration contract

In addition to the wind-axis Euler-angle formulation (12 states), the repository supports a 13-state body-axis quaternion formulation:
$$x_{quat} = [u, v, w, p, q, r, q_0, q_1, q_2, q_3, x_N, y_E, z_D]^T$$
where indices 6–9 contain the unit quaternion attitude ($q_0$ scalar real part, $q_1, q_2, q_3$ vector imaginary parts).

To maintain ODE solver generality:
- The base integrators (`euler_step`, `rk4_step`) in `integrators.py` are strictly generic and do not perform implicit normalization based on array length.
- Integrators accept an optional `post_step: PostStepHook | None` callback.
- Dedicated quaternion integration helpers (`rk4_step_quat`, `euler_step_quat`) and `normalize_quat_state` (in `quat_state.py`) enforce the unit-norm constraint explicitly after each integration step.
- The `simulate()` interface accepts `integrator="rk4_quat"`, `integrator="euler_quat"`, or an explicit `post_step` hook.

### Soft real-time execution semantics and observability

The repository includes a soft real-time execution engine (`realtime.py`) designed for interactive demonstration and simulation pacing.

Key characteristics and observability contracts:
- **Soft real-time pacing**: Uses wall-clock pacing (`time.monotonic()` via injectable `Clock` interface) to pace the simulation step `dt` against target wall-clock step `wallclock_dt`. It does **not** provide hard real-time or RTOS deterministic guarantees.
- **Timing decomposition**: Observability metrics in `RealtimeStats` explicitly decompose elapsed wall time:
  $$\text{wall\_time} = \text{compute\_time} + \text{sleep\_time}$$
  distinguishing numerical solver execution time (`compute_time`) from pacing delay (`sleep_time`). Per-step timings (`last_compute_time`, `last_sleep_time`) are also tracked.
- **Overrun handling & logging**: If step compute time exceeds the target step deadline, an overrun is recorded, dropped deadlines increment, and if `log_overruns=True`, a warning is emitted via the standard `logging` module.
- **Explicit stop reasons**: When the runner halts or pauses, `stats.stop_reason` (`StopReason` enum) explicitly records the cause:
  - `COMPLETED`: Simulation reached target `t_final`.
  - `PAUSED`: Simulation paused by caller.
  - `MAX_OVERRUNS`: Exceeded configured consecutive overrun limit (`max_overruns`).
  - `NON_FINITE_STATE`: Divergence or NaN/Inf state encountered.
  - `RUNTIME_ERROR`: Exception during derivative or step evaluation.
  - `STOPPED`: Explicitly halted via `runner.stop()`.
- **Failure propagation policy**: Configurable via `propagate_errors: bool`. When `False` (default), runtime failures are captured gracefully in `RealtimeStats` with `running=False` and descriptive `stop_message`. When `True`, runtime failures raise exceptions directly to the caller.

## Key limitations

1. **Flat Earth** — no Earth curvature or rotation
2. **Wind-axis state** — VT→0 singularity (aircraft must maintain positive airspeed)
3. **Euler angles** — singularity at theta = ±90° (quaternion representation avoids this singularity)
4. **No wind/gust model** — atmosphere is stationary
5. **No structural flexibility** — rigid body only
6. **Generic aerodynamic coefficients** — not representative of any real aircraft
7. **Linear aerodynamics** — no stall, no compressibility effects
8. **No actuator dynamics** — control deflections are instantaneous
9. **No sensor models** — full state is available
