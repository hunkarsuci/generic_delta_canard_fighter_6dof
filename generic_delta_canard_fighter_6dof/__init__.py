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
    "AIR_GAMMA",
    "AIR_GAS_CONSTANT_J_KG_K",
    "GRAVITY_MPS2",
    "NUM_CONTROLS",
    "NUM_STATES",
    "SEA_LEVEL_PRESSURE_PA",
    "SEA_LEVEL_TEMPERATURE_K",
    "AeroCoefficients",
    "AircraftGeometry",
    "AtmosphereState",
    "ControlIndex",
    "FlightCondition",
    "ForcesMoments",
    "LinearizationResult",
    "ModeCharacteristics",
    "PropulsionConfig",
    "SimulationResult",
    "StabilityAnalysis",
    "StateIndex",
    "TrimResult",
    "aerodynamic_forces_moments",
    "aircraft_dynamics",
    "altitude_rate_from_down_velocity",
    "analyze_stability",
    "body_to_ned_dcm",
    "body_to_wind_angles",
    "body_velocity_derivatives_to_wind_derivatives",
    "body_velocity_to_ned_velocity",
    "combined_forces_moments",
    "control_to_dict",
    "create_default_geometry",
    "deg_to_rad",
    "dynamic_pressure",
    "euler_rates",
    "euler_step",
    "euler_to_quaternion",
    "flight_condition",
    "ft_to_m",
    "gravity_force_body",
    "is_positive_definite",
    "is_rotation_matrix",
    "isa_atmosphere",
    "kt_to_mps",
    "linear_prediction",
    "linearize",
    "m_to_ft",
    "mach_number",
    "make_control",
    "make_state",
    "mps_to_kt",
    "ned_to_body_dcm",
    "normalize_quaternion",
    "propulsion_forces_moments",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_rates",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    "rad_to_deg",
    "rk4_step",
    "rotational_acceleration_body",
    "simulate",
    "speed_of_sound",
    "state_to_dict",
    "translational_acceleration_body",
    "trim_straight_level",
    "validate_control",
    "validate_state",
    "wind_to_body_velocity",
    "zero_forces_moments",
]
