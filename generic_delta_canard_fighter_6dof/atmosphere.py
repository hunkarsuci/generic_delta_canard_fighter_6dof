"""
International Standard Atmosphere utilities.

This module provides a simple ISA atmosphere model for the first version of
the 6DOF fighter simulator.

The model currently supports:

    0 m <= altitude <= 20,000 m

This covers the troposphere and lower stratosphere, which is enough for the
first educational flight dynamics and control simulations.

All units are SI units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generic_delta_canard_fighter_6dof.constants import (
    AIR_GAMMA,
    AIR_GAS_CONSTANT_J_KG_K,
    GRAVITY_MPS2,
    SEA_LEVEL_PRESSURE_PA,
    SEA_LEVEL_TEMPERATURE_K,
)


TROPOPAUSE_ALTITUDE_M: float = 11_000.0
LOWER_STRATOSPHERE_LIMIT_M: float = 20_000.0
TEMPERATURE_LAPSE_RATE_K_PER_M: float = 0.0065


@dataclass(frozen=True)
class AtmosphereState:
    """
    Atmospheric properties at a given altitude.

    Attributes
    ----------
    temperature_K:
        Static air temperature [K].
    pressure_Pa:
        Static pressure [Pa].
    density_kg_m3:
        Air density [kg/m^3].
    speed_of_sound_mps:
        Local speed of sound [m/s].
    """

    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_mps: float


@dataclass(frozen=True)
class FlightCondition:
    """
    Flight condition at a given altitude and true airspeed.

    Attributes
    ----------
    atmosphere:
        Atmospheric properties at the current altitude.
    mach:
        Mach number [-].
    dynamic_pressure_Pa:
        Dynamic pressure [Pa].
    """

    atmosphere: AtmosphereState
    mach: float
    dynamic_pressure_Pa: float


def isa_atmosphere(altitude_m: float) -> AtmosphereState:
    """
    Compute ISA atmosphere properties at a given altitude.

    Parameters
    ----------
    altitude_m:
        Geometric altitude above mean sea level [m].

    Returns
    -------
    AtmosphereState
        Temperature, pressure, density, and speed of sound.

    Raises
    ------
    ValueError
        If altitude is outside the supported range.

    Notes
    -----
    The model has two regions:

    1. Troposphere:
        0 m <= h <= 11,000 m
        
        Temperature decreases linearly with altitude.

    2. Lower stratosphere:
        11,000 m < h <= 20,000 m

        Temperature is assumed constant.
    """
    altitude_m = float(altitude_m)

    if not np.isfinite(altitude_m):
        raise ValueError("Altitude must be a finite number.")

    if altitude_m < 0.0:
        raise ValueError("Altitude must be non-negative.")

    if altitude_m > LOWER_STRATOSPHERE_LIMIT_M:
        raise ValueError(
            f"Altitude {altitude_m} m is above the supported limit of "
            f"{LOWER_STRATOSPHERE_LIMIT_M} m."
        )

    if altitude_m <= TROPOPAUSE_ALTITUDE_M:
        temperature_K = _troposphere_temperature(altitude_m)
        pressure_Pa = _troposphere_pressure(altitude_m)
    else:
        temperature_K = _tropopause_temperature()
        pressure_Pa = _lower_stratosphere_pressure(altitude_m)

    density_kg_m3 = pressure_Pa / (AIR_GAS_CONSTANT_J_KG_K * temperature_K)
    speed_of_sound_mps = speed_of_sound(temperature_K)

    return AtmosphereState(
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        density_kg_m3=density_kg_m3,
        speed_of_sound_mps=speed_of_sound_mps,
    )


def speed_of_sound(temperature_K: float) -> float:
    """
    Compute local speed of sound.

    Parameters
    ----------
    temperature_K:
        Static air temperature [K].

    Returns
    -------
    float
        Speed of sound [m/s].

    Equation
    --------
    a = sqrt(gamma * R * T)
    """
    temperature_K = float(temperature_K)

    if temperature_K <= 0.0:
        raise ValueError("Temperature must be positive.")

    return float(np.sqrt(AIR_GAMMA * AIR_GAS_CONSTANT_J_KG_K * temperature_K))


def mach_number(true_airspeed_mps: float, speed_of_sound_mps: float) -> float:
    """
    Compute Mach number.

    Parameters
    ----------
    true_airspeed_mps:
        True airspeed [m/s].
    speed_of_sound_mps:
        Local speed of sound [m/s].

    Returns
    -------
    float
        Mach number [-].

    Equation
    --------
    M = V / a
    """
    true_airspeed_mps = float(true_airspeed_mps)
    speed_of_sound_mps = float(speed_of_sound_mps)

    if true_airspeed_mps < 0.0:
        raise ValueError("True airspeed must be non-negative.")

    if speed_of_sound_mps <= 0.0:
        raise ValueError("Speed of sound must be positive.")

    return true_airspeed_mps / speed_of_sound_mps


def dynamic_pressure(density_kg_m3: float, true_airspeed_mps: float) -> float:
    """
    Compute dynamic pressure.

    Parameters
    ----------
    density_kg_m3:
        Air density [kg/m^3].
    true_airspeed_mps:
        True airspeed [m/s].

    Returns
    -------
    float
        Dynamic pressure [Pa].

    Equation
    --------
    q_bar = 0.5 * rho * V^2
    """
    density_kg_m3 = float(density_kg_m3)
    true_airspeed_mps = float(true_airspeed_mps)

    if density_kg_m3 < 0.0:
        raise ValueError("Air density must be non-negative.")

    if true_airspeed_mps < 0.0:
        raise ValueError("True airspeed must be non-negative.")

    return 0.5 * density_kg_m3 * true_airspeed_mps**2


def flight_condition(altitude_m: float, true_airspeed_mps: float) -> FlightCondition:
    """
    Compute atmosphere, Mach number, and dynamic pressure.

    Parameters
    ----------
    altitude_m:
        Altitude [m].
    true_airspeed_mps:
        True airspeed [m/s].

    Returns
    -------
    FlightCondition
        Atmosphere, Mach number, and dynamic pressure.
    """
    atmosphere = isa_atmosphere(altitude_m)

    mach = mach_number(
        true_airspeed_mps=true_airspeed_mps,
        speed_of_sound_mps=atmosphere.speed_of_sound_mps,
    )

    q_bar = dynamic_pressure(
        density_kg_m3=atmosphere.density_kg_m3,
        true_airspeed_mps=true_airspeed_mps,
    )

    return FlightCondition(
        atmosphere=atmosphere,
        mach=mach,
        dynamic_pressure_Pa=q_bar,
    )


def _troposphere_temperature(altitude_m: float) -> float:
    """
    Compute troposphere temperature.

    In the troposphere, temperature decreases linearly with altitude:

        T = T0 - L h
    """
    return SEA_LEVEL_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE_K_PER_M * altitude_m


def _troposphere_pressure(altitude_m: float) -> float:
    """
    Compute troposphere pressure.

    Equation:

        p = p0 * (T / T0)^(g / (R L))
    """
    temperature_K = _troposphere_temperature(altitude_m)

    exponent = GRAVITY_MPS2 / (
        AIR_GAS_CONSTANT_J_KG_K * TEMPERATURE_LAPSE_RATE_K_PER_M
    )

    return SEA_LEVEL_PRESSURE_PA * (temperature_K / SEA_LEVEL_TEMPERATURE_K) ** exponent


def _tropopause_temperature() -> float:
    """
    Return the temperature at the tropopause.
    """
    return _troposphere_temperature(TROPOPAUSE_ALTITUDE_M)


def _tropopause_pressure() -> float:
    """
    Return the pressure at the tropopause.
    """
    return _troposphere_pressure(TROPOPAUSE_ALTITUDE_M)


def _lower_stratosphere_pressure(altitude_m: float) -> float:
    """
    Compute pressure in the lower stratosphere.

    In this simplified region, temperature is constant:

        p = p_11 * exp(-g * (h - h_11) / (R T_11))
    """
    temperature_K = _tropopause_temperature()
    pressure_at_tropopause_Pa = _tropopause_pressure()

    altitude_difference_m = altitude_m - TROPOPAUSE_ALTITUDE_M

    exponent = -GRAVITY_MPS2 * altitude_difference_m / (
        AIR_GAS_CONSTANT_J_KG_K * temperature_K
    )

    return pressure_at_tropopause_Pa * np.exp(exponent)