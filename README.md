# Generic Delta-Canard Fighter 6DOF Dynamics and Control

This project is an educational Python implementation of a nonlinear 6DOF flight dynamics and control simulator for a generic delta-canard fighter aircraft.

The goal is to build the simulator step by step while learning the engineering behind each part: coordinate frames, aircraft kinematics, atmosphere modeling, rigid-body dynamics, aerodynamics, trim, control design, and real-time simulation.

This is not an official model of any real aircraft. The aircraft parameters and aerodynamic model are generic and configurable. The project is intended for education, research prototyping, and software engineering practice.

---

## Project Status

This repository is being developed phase by phase.

Current progress:

- [x] Phase 0 — Project skeleton
- [x] Phase 1 — Constants, units, state vector, and control vector
- [x] Phase 2 — Body frame, NED frame, kinematics, and quaternion utilities
- [x] Phase 3 — ISA atmosphere model, Mach number, and dynamic pressure
- [ ] Phase 4 — Aircraft geometry and mass properties
- [ ] Phase 5 — Nonlinear 6DOF equations
- [ ] Phase 6 — Aerodynamic model
- [ ] Phase 7 — Propulsion and actuators
- [ ] Phase 8 — Trim
- [ ] Phase 9 — Linearization
- [ ] Phase 10 — PID control
- [ ] Phase 11 — LQR control
- [ ] Phase 12 — Real-time simulation engineering

Development started in June 2026 and the repository is updated gradually.

---

## Educational Scope

The simulator is designed to teach:

- aircraft rigid-body dynamics
- body-fixed and NED coordinate frames
- Euler-angle attitude representation
- quaternion attitude utilities
- wind variables and body-axis velocity conversion
- International Standard Atmosphere modeling
- Mach number and dynamic pressure
- aerodynamic force and moment modeling
- propulsion modeling
- actuator modeling
- trim and equilibrium flight
- numerical linearization
- PID control
- LQR control
- real-time simulation engineering
- testing and GitHub-based engineering workflow

---

## State Vector

The baseline simulator uses a 12-state Euler-angle model:

```text
x = [
    VT, alpha, beta,
    p, q, r,
    phi, theta, psi,
    x_N, y_E, h
]
```

where:

| State | Meaning | Unit |
|---|---|---|
| `VT` | total airspeed | m/s |
| `alpha` | angle of attack | rad |
| `beta` | sideslip angle | rad |
| `p` | body roll rate | rad/s |
| `q` | body pitch rate | rad/s |
| `r` | body yaw rate | rad/s |
| `phi` | roll angle | rad |
| `theta` | pitch angle | rad |
| `psi` | yaw / heading angle | rad |
| `x_N` | north position | m |
| `y_E` | east position | m |
| `h` | altitude | m |

The project uses SI units internally.

Angles are stored in radians, not degrees.

---

## Control Vector

The current control vector is:

```text
u = [
    delta_canard,
    delta_elevon_left,
    delta_elevon_right,
    delta_rudder,
    throttle
]
```

where:

| Control | Meaning | Unit |
|---|---|---|
| `delta_canard` | canard deflection | rad |
| `delta_elevon_left` | left elevon deflection | rad |
| `delta_elevon_right` | right elevon deflection | rad |
| `delta_rudder` | rudder deflection | rad |
| `throttle` | engine command | 0 to 1 |

---

## Coordinate Frame Convention

This project uses a local North-East-Down frame as the navigation frame.

Body frame:

```text
x_b = forward through the aircraft nose
y_b = right wing
z_b = downward
```

NED frame:

```text
x_N = north
y_E = east
z_D = down
```

The state vector stores altitude `h` as positive upward, while the NED frame uses down position as positive downward.

Therefore:

```text
h_dot = -V_D
```

where `V_D` is the NED down velocity.

---

## Quaternion Support

The baseline simulator uses Euler angles because they are intuitive and useful for education.

However, fighter aircraft may perform aggressive maneuvers where Euler angles can become singular near:

```text
theta = +/- 90 degrees
```

For this reason, the project also includes quaternion utilities.

Quaternion convention:

```text
q = [q0, q1, q2, q3]
```

where `q0` is the scalar part.

The quaternion utilities are currently support tools. A full 13-state quaternion-based dynamics model may be added later.

---

## Atmosphere Model

The project currently includes a simplified International Standard Atmosphere model for:

```text
0 m <= altitude <= 20,000 m
```

The atmosphere model computes:

- temperature
- pressure
- density
- speed of sound
- Mach number
- dynamic pressure

The most important aerodynamic input from the atmosphere model is dynamic pressure:

```text
q_bar = 0.5 * rho * VT^2
```

Later, aerodynamic forces and moments will use:

```text
Force  = q_bar * S * C
Moment = q_bar * S * reference_length * C
```

---

## Repository Structure

```text
generic_delta_canard_fighter_6dof/
├── generic_delta_canard_fighter_6dof/
│   ├── __init__.py
│   ├── constants.py
│   ├── units.py
│   ├── state.py
│   ├── transforms.py
│   ├── quaternions.py
│   ├── kinematics.py
│   └── atmosphere.py
│
├── examples/
│   └── run_01_atmosphere_demo.py
│
├── tests/
│   ├── test_state.py
│   ├── test_transforms.py
│   ├── test_quaternions.py
│   ├── test_kinematics.py
│   └── test_atmosphere.py
│
├── README.md
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

This structure will expand as new phases are implemented.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hunkarsuci/generic_delta_canard_fighter_6dof.git
cd generic_delta_canard_fighter_6dof
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package in editable mode with development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

---

## Running Tests

Run all tests from the repository root:

```powershell
pytest
```

The project uses tests to verify each educational phase before moving to the next one.

---

## Running the Atmosphere Demo

Run:

```powershell
python .\examples\run_01_atmosphere_demo.py
```

This demo prints atmosphere values at several altitudes and plots:

- density versus altitude
- speed of sound versus altitude

---

## Development Philosophy

This project is built slowly and deliberately.

Each phase follows the same pattern:

```text
1. Learn the theory
2. Implement the Python code
3. Write tests
4. Run examples
5. Commit the milestone
```

The goal is not only to create a working simulator, but also to understand the flight dynamics and software engineering behind it.
