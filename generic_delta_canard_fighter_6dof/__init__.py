"""Generic Delta-Canard Fighter 6DOF — public API."""

from generic_delta_canard_fighter_6dof.aerodynamics import (
    AeroCoefficients,
    aerodynamic_forces_moments,
)
from generic_delta_canard_fighter_6dof.atmosphere import (
    AtmosphereState,
    FlightCondition,
    dynamic_pressure,
    flight_condition,
    isa_atmosphere,
    mach_number,
    speed_of_sound,
)
from generic_delta_canard_fighter_6dof.constants import (
    AIR_GAMMA,
    AIR_GAS_CONSTANT_J_KG_K,
    GRAVITY_MPS2,
    SEA_LEVEL_PRESSURE_PA,
    SEA_LEVEL_TEMPERATURE_K,
)
from generic_delta_canard_fighter_6dof.equations import (
    ForcesMoments,
    aircraft_dynamics,
    body_velocity_derivatives_to_wind_derivatives,
    gravity_force_body,
    rotational_acceleration_body,
    translational_acceleration_body,
    zero_forces_moments,
)
from generic_delta_canard_fighter_6dof.geometry import (
    AircraftGeometry,
    create_default_geometry,
    is_positive_definite,
)
from generic_delta_canard_fighter_6dof.integrators import (
    euler_step,
    rk4_step,
)
from generic_delta_canard_fighter_6dof.kinematics import (
    altitude_rate_from_down_velocity,
    body_to_wind_angles,
    body_velocity_to_ned_velocity,
    euler_rates,
    wind_to_body_velocity,
)
from generic_delta_canard_fighter_6dof.linearization import (
    LinearizationResult,
    linear_prediction,
    linearize,
)
from generic_delta_canard_fighter_6dof.propulsion import (
    PropulsionConfig,
    combined_forces_moments,
    propulsion_forces_moments,
)
from generic_delta_canard_fighter_6dof.quaternions import (
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rates,
    quaternion_to_dcm,
    quaternion_to_euler,
)
from generic_delta_canard_fighter_6dof.simulation import (
    SimulationResult,
    simulate,
)
from generic_delta_canard_fighter_6dof.stability import (
    ModeCharacteristics,
    StabilityAnalysis,
    analyze_stability,
)
from generic_delta_canard_fighter_6dof.state import (
    NUM_CONTROLS,
    NUM_STATES,
    ControlIndex,
    StateIndex,
    control_to_dict,
    make_control,
    make_state,
    state_to_dict,
    validate_control,
    validate_state,
)
from generic_delta_canard_fighter_6dof.transforms import (
    body_to_ned_dcm,
    is_rotation_matrix,
    ned_to_body_dcm,
)
from generic_delta_canard_fighter_6dof.trim import (
    TrimResult,
    trim_straight_level,
)
from generic_delta_canard_fighter_6dof.units import (
    deg_to_rad,
    ft_to_m,
    kt_to_mps,
    m_to_ft,
    mps_to_kt,
    rad_to_deg,
)

__all__ = [
    # aerodynamics
    "AeroCoefficients",
    "aerodynamic_forces_moments",
    # atmosphere
    "AtmosphereState",
    "FlightCondition",
    "dynamic_pressure",
    "flight_condition",
    "isa_atmosphere",
    "mach_number",
    "speed_of_sound",
    # constants
    "AIR_GAMMA",
    "AIR_GAS_CONSTANT_J_KG_K",
    "GRAVITY_MPS2",
    "SEA_LEVEL_PRESSURE_PA",
    "SEA_LEVEL_TEMPERATURE_K",
    # equations
    "ForcesMoments",
    "aircraft_dynamics",
    "body_velocity_derivatives_to_wind_derivatives",
    "gravity_force_body",
    "rotational_acceleration_body",
    "translational_acceleration_body",
    "zero_forces_moments",
    # geometry
    "AircraftGeometry",
    "create_default_geometry",
    "is_positive_definite",
    # integrators
    "euler_step",
    "rk4_step",
    # kinematics
    "altitude_rate_from_down_velocity",
    "body_to_wind_angles",
    "body_velocity_to_ned_velocity",
    "euler_rates",
    "wind_to_body_velocity",
    # linearization
    "LinearizationResult",
    "linear_prediction",
    "linearize",
    # propulsion
    "PropulsionConfig",
    "combined_forces_moments",
    "propulsion_forces_moments",
    # quaternions
    "euler_to_quaternion",
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_rates",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    # simulation
    "SimulationResult",
    "simulate",
    # stability
    "ModeCharacteristics",
    "StabilityAnalysis",
    "analyze_stability",
    # state
    "NUM_CONTROLS",
    "NUM_STATES",
    "ControlIndex",
    "StateIndex",
    "control_to_dict",
    "make_control",
    "make_state",
    "state_to_dict",
    "validate_control",
    "validate_state",
    # transforms
    "body_to_ned_dcm",
    "is_rotation_matrix",
    "ned_to_body_dcm",
    # trim
    "TrimResult",
    "trim_straight_level",
    # units
    "deg_to_rad",
    "ft_to_m",
    "kt_to_mps",
    "m_to_ft",
    "mps_to_kt",
    "rad_to_deg",
]
