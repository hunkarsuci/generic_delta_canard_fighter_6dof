# Generic Delta-Canard Fighter 6DOF Dynamics and Control

This project is an educational Python implementation of a nonlinear 6DOF flight dynamics and control simulator for a generic delta-canard fighter aircraft.

The simulator is designed to teach:

- aircraft rigid-body dynamics
- coordinate frames
- aerodynamic modeling
- propulsion modeling
- actuator modeling
- trim
- linearization
- PID control
- LQR control
- real-time simulation engineering

This is not an official model of any real aircraft. The aircraft parameters and aerodynamic model are generic and configurable. The project is intended for education, research prototyping, and software engineering practice.

## Initial 12-State Model

The planned aircraft state vector is:

```text
x = [
    VT, alpha, beta,
    p, q, r,
    phi, theta, psi,
    x_N, y_E, h
]

This repo is to be updated slowly in June
