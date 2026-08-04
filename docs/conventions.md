# Coordinate and Sign Conventions

## Frames

### Navigation frame (NED)

```
x_N = North (positive northward)
y_E = East  (positive eastward)
z_D = Down  (positive downward)
```

The NED frame is assumed inertial (non-rotating, flat-earth) at the aircraft's location. This is valid for short-duration, low-speed simulations where Earth curvature and rotation are negligible.

### Body frame

```
x_b = forward through the aircraft nose
y_b = right (starboard) wing
z_b = downward (belly)
```

Origin at the aircraft center of gravity.

### Wind frame

```
x_w = along the relative wind (velocity vector), positive aft
y_w = perpendicular to x_w in the aircraft symmetry plane, positive to the right
z_w = completes the right-handed system, positive downward
```

## Rotations

### Euler angles (3-2-1 sequence)

Rotation order: yaw (ψ) about z, then pitch (θ) about intermediate y, then roll (φ) about final x.

```
C_nb = R_x(φ) · R_y(θ) · R_z(ψ)

v_ned = C_nb · v_body
```

### Quaternions

- **Storage**: `[q0, q1, q2, q3]` — q0 is the scalar part
- **Algebra**: Hamilton (right-handed)
- **Multiplication**: `q_left ⊗ q_right` as defined in `quaternions.py`
- **DCM from quaternion**: `v_ned = C(q) · v_body`
- **Rate equation**: `q_dot = 0.5 · q ⊗ [0, ω_body]`
- **Double cover**: q and -q represent the same attitude (handled by the functions)

## State vector

```
x[0]  = VT    — true airspeed [m/s]
x[1]  = alpha — angle of attack [rad]
x[2]  = beta  — sideslip angle [rad]
x[3]  = p     — body roll rate [rad/s]
x[4]  = q     — body pitch rate [rad/s]
x[5]  = r     — body yaw rate [rad/s]
x[6]  = phi   — roll angle [rad]
x[7]  = theta — pitch angle [rad]
x[8]  = psi   — yaw/heading angle [rad]
x[9]  = x_N   — north position [m]
x[10] = y_E   — east position [m]
x[11] = h     — altitude (positive up) [m]
```

## Control vector

```
u[0] = delta_canard       — canard deflection [rad], +trailing edge down
u[1] = delta_elevon_left  — left elevon deflection [rad], +trailing edge down
u[2] = delta_elevon_right — right elevon deflection [rad], +trailing edge down
u[3] = delta_rudder       — rudder deflection [rad], +trailing edge left
u[4] = throttle           — engine command [0, 1]
```

## Aerodynamic sign conventions

- **CL** positive = lift upward (perpendicular to velocity, positive in -z_w direction)
- **CD** positive = drag aft (parallel to velocity, positive in +x_w direction)
- **CY** positive = side force to the right (+y_w)
- **Cm** positive = nose-up pitching moment
- **Cl** positive = right-wing-down rolling moment
- **Cn** positive = nose-right yawing moment
- **Control deflections**: Positive = trailing edge down (conventional)

## Gravity

- NED frame: `F_grav_ned = [0, 0, m·g]` (positive down)
- Body frame: `F_grav_body = C_bn · F_grav_ned`
- Altitude: `h_dot = -V_D` (altitude positive up, NED down positive down)

## Units

All internal computations use SI units:
- Length: meters [m]
- Mass: kilograms [kg]
- Time: seconds [s]
- Force: Newtons [N]
- Moment: Newton-meters [N·m]
- Angles: radians [rad]
- Angular rates: radians per second [rad/s]
- Pressure: Pascals [Pa]
- Temperature: Kelvin [K]
