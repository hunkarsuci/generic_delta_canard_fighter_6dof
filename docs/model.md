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

## Key limitations

1. **Flat Earth** — no Earth curvature or rotation
2. **Wind-axis state** — VT→0 singularity (aircraft must maintain positive airspeed)
3. **Euler angles** — singularity at theta = ±90°
4. **No wind/gust model** — atmosphere is stationary
5. **No structural flexibility** — rigid body only
6. **Generic aerodynamic coefficients** — not representative of any real aircraft
7. **Linear aerodynamics** — no stall, no compressibility effects
8. **No actuator dynamics** — control deflections are instantaneous
9. **No sensor models** — full state is available
