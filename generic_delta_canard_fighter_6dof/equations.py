"""
Nonlinear 6DOF rigid-body equations of motion.

This module contains the core aircraft dynamics.

The baseline state vector is:

    x = [
        VT, alpha, beta,
        p, q, r,
        phi, theta, psi,
        x_N, y_E, h
    ]

The equations internally convert:

    VT, alpha, beta  <->  u, v, w

because Newton-Euler rigid-body equations are naturally expressed using
body-axis velocity components.

All units are SI units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.constants import EPSILON, GRAVITY_MPS2
from generic_delta_canard_fighter_6dof.geometry import AircraftGeometry
from generic_delta_canard_fighter_6dof.kinematics import (
    altitude_rate_from_down_velocity,
    body_to_wind_angles,
    body_velocity_to_ned_velocity,
    euler_rates,
    wind_to_body_velocity,
)
from generic_delta_canard_fighter_6dof.state import (
    NUM_STATES,
    StateIndex,
    validate_control,
    validate_state,
)
from generic_delta_canard_fighter_6dof.transforms import ned_to_body_dcm


@dataclass
class ForcesMoments:
    """
    Body-axis forces and moments acting on the aircraft.

    Attributes
    ----------
    force_body_N:
        Body-axis force vector [X, Y, Z] in Newtons.
    moment_body_Nm:
        Body-axis moment vector [L, M, N] in Newton-meters.

    Notes
    -----
    Body-axis force convention:

        X = positive forward
        Y = positive right
        Z = positive downward

    Body-axis moment convention:

        L = rolling moment about x_b
        M = pitching moment about y_b
        N = yawing moment about z_b
    """

    force_body_N: np.ndarray
    moment_body_Nm: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate force and moment vectors after object creation.
        """
        self.force_body_N = _as_vector3(self.force_body_N, "force_body_N")
        self.moment_body_Nm = _as_vector3(self.moment_body_Nm, "moment_body_Nm")


ForceMomentModel = Callable[
    [float, np.ndarray, np.ndarray, AircraftGeometry],
    ForcesMoments,
]


def zero_forces_moments(
    t: float,
    x: np.ndarray,
    control: np.ndarray,
    geometry: AircraftGeometry,
) -> ForcesMoments:
    """
    Return zero non-gravitational force and moment.

    This is useful for tests and for demonstrating gravity-only motion.

    Parameters are included so this function has the same interface as later
    aerodynamic and propulsion force/moment models.
    """
    return ForcesMoments(
        force_body_N=np.zeros(3),
        moment_body_Nm=np.zeros(3),
    )


def gravity_force_body(
    phi: float, theta: float, psi: float, mass_kg: float
) -> np.ndarray:
    """
    Compute gravity force expressed in body axes.

    Parameters
    ----------
    phi:
        Roll angle [rad].
    theta:
        Pitch angle [rad].
    psi:
        Yaw angle [rad].
    mass_kg:
        Aircraft mass [kg].

    Returns
    -------
    np.ndarray
        Gravity force in body axes [N].

    Notes
    -----
    In the NED frame, gravity force is:

        F_g_ned = [0, 0, m g]

    because NED z-axis points downward.

    To express gravity in body axes:

        F_g_body = C_bn @ F_g_ned
    """
    gravity_ned_N = np.array([0.0, 0.0, mass_kg * GRAVITY_MPS2], dtype=float)
    C_bn = ned_to_body_dcm(phi, theta, psi)

    return C_bn @ gravity_ned_N


def translational_acceleration_body(
    velocity_body_mps: np.ndarray,
    angular_rates_radps: np.ndarray,
    total_force_body_N: np.ndarray,
    mass_kg: float,
) -> np.ndarray:
    """
    Compute body-axis translational acceleration.

    Parameters
    ----------
    velocity_body_mps:
        Body-axis velocity vector [u, v, w] in [m/s].
    angular_rates_radps:
        Body angular rate vector [p, q, r] in [rad/s].
    total_force_body_N:
        Total body-axis force vector [X, Y, Z] in [N].
    mass_kg:
        Aircraft mass [kg].

    Returns
    -------
    np.ndarray
        Body-axis acceleration vector [u_dot, v_dot, w_dot] in [m/s^2].

    Equation
    --------
    V_dot_body = F_body / m - omega_body x V_body
    """
    velocity_body_mps = _as_vector3(velocity_body_mps, "velocity_body_mps")
    angular_rates_radps = _as_vector3(angular_rates_radps, "angular_rates_radps")
    total_force_body_N = _as_vector3(total_force_body_N, "total_force_body_N")

    if mass_kg <= 0.0:
        raise ValueError("Aircraft mass must be positive.")

    coriolis_term = np.cross(angular_rates_radps, velocity_body_mps)

    return total_force_body_N / mass_kg - coriolis_term


def rotational_acceleration_body(
    angular_rates_radps: np.ndarray,
    total_moment_body_Nm: np.ndarray,
    geometry: AircraftGeometry,
) -> np.ndarray:
    """
    Compute body-axis angular acceleration.

    Parameters
    ----------
    angular_rates_radps:
        Body angular rate vector [p, q, r] in [rad/s].
    total_moment_body_Nm:
        Body-axis moment vector [L, M, N] in [N*m].
    geometry:
        Aircraft geometry and inertia properties.

    Returns
    -------
    np.ndarray
        Angular acceleration vector [p_dot, q_dot, r_dot] in [rad/s^2].

    Equation
    --------
    omega_dot = I^{-1} (M - omega x I omega)
    """
    angular_rates_radps = _as_vector3(angular_rates_radps, "angular_rates_radps")
    total_moment_body_Nm = _as_vector3(total_moment_body_Nm, "total_moment_body_Nm")

    inertia = geometry.inertia_matrix_kg_m2()
    inverse_inertia = geometry.inverse_inertia_matrix_kg_m2()

    angular_momentum = inertia @ angular_rates_radps
    gyroscopic_term = np.cross(angular_rates_radps, angular_momentum)

    return inverse_inertia @ (total_moment_body_Nm - gyroscopic_term)


def body_velocity_derivatives_to_wind_derivatives(
    u: float,
    v: float,
    w: float,
    u_dot: float,
    v_dot: float,
    w_dot: float,
) -> tuple[float, float, float]:
    """
    Convert body-axis velocity derivatives to VT, alpha, and beta derivatives.

    Parameters
    ----------
    u, v, w:
        Body-axis velocity components [m/s].
    u_dot, v_dot, w_dot:
        Body-axis velocity derivatives [m/s^2].

    Returns
    -------
    tuple[float, float, float]
        Tuple (VT_dot, alpha_dot, beta_dot).

    Notes
    -----
    The state stores VT, alpha, beta, but the rigid-body equations produce
    u_dot, v_dot, w_dot. This function connects the two representations.
    """
    VT, _alpha, beta = body_to_wind_angles(u, v, w)

    if VT < EPSILON:
        return 0.0, 0.0, 0.0

    VT_dot = (u * u_dot + v * v_dot + w * w_dot) / VT

    denominator_alpha = u * u + w * w

    if denominator_alpha < EPSILON:
        alpha_dot = 0.0
    else:
        alpha_dot = (u * w_dot - w * u_dot) / denominator_alpha

    cos_beta = np.cos(beta)

    if abs(cos_beta) < EPSILON:
        beta_dot = 0.0
    else:
        beta_dot = (VT * v_dot - v * VT_dot) / (VT * VT * cos_beta)

    return float(VT_dot), float(alpha_dot), float(beta_dot)


def aircraft_dynamics(
    t: float,
    x: np.ndarray,
    control: np.ndarray,
    geometry: AircraftGeometry,
    force_moment_model: ForceMomentModel = zero_forces_moments,
) -> np.ndarray:
    """
    Compute the nonlinear 6DOF state derivative.

    Parameters
    ----------
    t:
        Current simulation time [s].
    x:
        Aircraft state vector.
    control:
        Aircraft control vector.
    geometry:
        Aircraft geometry and mass properties.
    force_moment_model:
        Function that returns non-gravitational body forces and moments.

    Returns
    -------
    np.ndarray
        State derivative x_dot with shape (12,).

    Notes
    -----
    Gravity is always added inside this function.

    The force_moment_model is intended for aerodynamic, propulsion, and actuator
    models that will be added in later phases.
    """
    validate_state(x)
    validate_control(control)

    VT = x[StateIndex.VT]
    alpha = x[StateIndex.ALPHA]
    beta = x[StateIndex.BETA]

    p = x[StateIndex.P]
    q = x[StateIndex.Q]
    r = x[StateIndex.R]

    phi = x[StateIndex.PHI]
    theta = x[StateIndex.THETA]
    psi = x[StateIndex.PSI]

    velocity_body = wind_to_body_velocity(VT, alpha, beta)
    angular_rates = np.array([p, q, r], dtype=float)

    non_gravity = force_moment_model(t, x, control, geometry)

    gravity_body = gravity_force_body(
        phi=phi,
        theta=theta,
        psi=psi,
        mass_kg=geometry.mass_kg,
    )

    total_force_body = non_gravity.force_body_N + gravity_body
    total_moment_body = non_gravity.moment_body_Nm

    body_acceleration = translational_acceleration_body(
        velocity_body_mps=velocity_body,
        angular_rates_radps=angular_rates,
        total_force_body_N=total_force_body,
        mass_kg=geometry.mass_kg,
    )

    angular_acceleration = rotational_acceleration_body(
        angular_rates_radps=angular_rates,
        total_moment_body_Nm=total_moment_body,
        geometry=geometry,
    )

    VT_dot, alpha_dot, beta_dot = body_velocity_derivatives_to_wind_derivatives(
        u=velocity_body[0],
        v=velocity_body[1],
        w=velocity_body[2],
        u_dot=body_acceleration[0],
        v_dot=body_acceleration[1],
        w_dot=body_acceleration[2],
    )

    phi_dot, theta_dot, psi_dot = euler_rates(
        phi=phi,
        theta=theta,
        p=p,
        q=q,
        r=r,
    )

    velocity_ned = body_velocity_to_ned_velocity(
        u=velocity_body[0],
        v=velocity_body[1],
        w=velocity_body[2],
        phi=phi,
        theta=theta,
        psi=psi,
    )

    x_dot = np.zeros(NUM_STATES, dtype=float)

    x_dot[StateIndex.VT] = VT_dot
    x_dot[StateIndex.ALPHA] = alpha_dot
    x_dot[StateIndex.BETA] = beta_dot

    x_dot[StateIndex.P] = angular_acceleration[0]
    x_dot[StateIndex.Q] = angular_acceleration[1]
    x_dot[StateIndex.R] = angular_acceleration[2]

    x_dot[StateIndex.PHI] = phi_dot
    x_dot[StateIndex.THETA] = theta_dot
    x_dot[StateIndex.PSI] = psi_dot

    x_dot[StateIndex.X_N] = velocity_ned[0]
    x_dot[StateIndex.Y_E] = velocity_ned[1]
    x_dot[StateIndex.H] = altitude_rate_from_down_velocity(velocity_ned[2])

    return x_dot


def aircraft_dynamics_quat(
    t: float,
    x_quat: np.ndarray,
    control: np.ndarray,
    geometry: AircraftGeometry,
    force_moment_model: ForceMomentModel = zero_forces_moments,
) -> np.ndarray:
    """Compute the nonlinear 6DOF state derivative in quaternion form.

    Uses a 13-state quaternion formulation:

        x_quat = [u, v, w, p, q, r, q0, q1, q2, q3, x_N, y_E, h]

    This formulation has no Euler-angle singularity and is suitable for
    aggressive-maneuver simulation.

    Parameters
    ----------
    t:
        Current simulation time [s].
    x_quat:
        13-state quaternion aircraft state vector.
    control:
        5-element aircraft control vector.
    geometry:
        Aircraft geometry and mass properties.
    force_moment_model:
        Function returning non-gravitational body forces and moments.

    Returns
    -------
    np.ndarray
        State derivative x_quat_dot with shape (13,).

    Notes
    -----
    Shares the same force/moment models as ``aircraft_dynamics``.
    Aerodynamics and propulsion require wind-axis quantities (VT, alpha, beta)
    which are computed from body u, v, w via the kinematics module.

    Gravity is always added inside this function.

    Quaternion normalization is the caller's responsibility (typically
    done after each integration step, not inside the derivative).
    """
    from generic_delta_canard_fighter_6dof.kinematics import (
        body_to_wind_angles,
    )
    from generic_delta_canard_fighter_6dof.quat_state import (
        NUM_QUAT_STATES,
        QuatStateIndex,
        _validate_quat_state,
    )
    from generic_delta_canard_fighter_6dof.quaternions import (
        quaternion_multiply,
        quaternion_to_dcm,
    )

    _validate_quat_state(x_quat)
    validate_control(control)

    u = float(x_quat[QuatStateIndex.U])
    v = float(x_quat[QuatStateIndex.V])
    w = float(x_quat[QuatStateIndex.W])

    p = float(x_quat[QuatStateIndex.P])
    q = float(x_quat[QuatStateIndex.Q])
    r = float(x_quat[QuatStateIndex.R])

    q_vec = x_quat[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1].copy()

    velocity_body = np.array([u, v, w], dtype=float)
    angular_rates = np.array([p, q, r], dtype=float)

    # -- wind-axis quantities for aero/propulsion models --
    VT, alpha, beta = body_to_wind_angles(u, v, w)

    # Build an Euler-style state for the force/moment model interface.
    # The model interface expects the 12-state Euler vector.
    from generic_delta_canard_fighter_6dof.state import NUM_STATES as _NS

    x_euler = np.zeros(_NS, dtype=float)
    from generic_delta_canard_fighter_6dof.state import StateIndex

    x_euler[StateIndex.VT] = VT
    x_euler[StateIndex.ALPHA] = alpha
    x_euler[StateIndex.BETA] = beta
    x_euler[StateIndex.P] = p
    x_euler[StateIndex.Q] = q
    x_euler[StateIndex.R] = r
    # phi/theta/psi left at 0 (not used by aero/prop, only by gravity)
    x_euler[StateIndex.X_N] = float(x_quat[QuatStateIndex.X_N])
    x_euler[StateIndex.Y_E] = float(x_quat[QuatStateIndex.Y_E])
    x_euler[StateIndex.H] = float(x_quat[QuatStateIndex.H])

    non_gravity = force_moment_model(t, x_euler, control, geometry)

    # -- gravity in body frame using quaternion DCM --
    gravity_ned_N = np.array([0.0, 0.0, geometry.mass_kg * GRAVITY_MPS2], dtype=float)
    C_nb = quaternion_to_dcm(q_vec)
    C_bn = C_nb.T
    gravity_body = C_bn @ gravity_ned_N

    total_force_body = non_gravity.force_body_N + gravity_body
    total_moment_body = non_gravity.moment_body_Nm

    # -- translational acceleration (body frame) --
    body_acceleration = translational_acceleration_body(
        velocity_body_mps=velocity_body,
        angular_rates_radps=angular_rates,
        total_force_body_N=total_force_body,
        mass_kg=geometry.mass_kg,
    )

    # -- rotational acceleration --
    angular_acceleration = rotational_acceleration_body(
        angular_rates_radps=angular_rates,
        total_moment_body_Nm=total_moment_body,
        geometry=geometry,
    )

    # -- quaternion kinematics (no pre-normalization -- caller normalizes post-step) --
    omega_quat = np.array([0.0, p, q, r], dtype=float)
    q_dot = 0.5 * quaternion_multiply(q_vec, omega_quat)

    # -- NED velocity from body velocity --
    velocity_ned = C_nb @ velocity_body

    x_dot = np.zeros(NUM_QUAT_STATES, dtype=float)

    x_dot[QuatStateIndex.U] = body_acceleration[0]
    x_dot[QuatStateIndex.V] = body_acceleration[1]
    x_dot[QuatStateIndex.W] = body_acceleration[2]

    x_dot[QuatStateIndex.P] = angular_acceleration[0]
    x_dot[QuatStateIndex.Q] = angular_acceleration[1]
    x_dot[QuatStateIndex.R] = angular_acceleration[2]

    x_dot[QuatStateIndex.Q0] = q_dot[0]
    x_dot[QuatStateIndex.Q1] = q_dot[1]
    x_dot[QuatStateIndex.Q2] = q_dot[2]
    x_dot[QuatStateIndex.Q3] = q_dot[3]

    x_dot[QuatStateIndex.X_N] = velocity_ned[0]
    x_dot[QuatStateIndex.Y_E] = velocity_ned[1]
    x_dot[QuatStateIndex.H] = altitude_rate_from_down_velocity(velocity_ned[2])

    return x_dot


def _as_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    """
    Convert input to a finite 3-element vector.

    This helper keeps validation consistent across the dynamics module.
    """
    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")

    return vector
