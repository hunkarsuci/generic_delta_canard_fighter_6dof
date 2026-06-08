"""
Tests for the atmosphere model.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.atmosphere import (
    LOWER_STRATOSPHERE_LIMIT_M,
    TROPOPAUSE_ALTITUDE_M,
    dynamic_pressure,
    flight_condition,
    isa_atmosphere,
    mach_number,
    speed_of_sound,
)


def test_isa_atmosphere_sea_level() -> None:
    atmosphere = isa_atmosphere(0.0)

    assert atmosphere.temperature_K == pytest.approx(288.15)
    assert atmosphere.pressure_Pa == pytest.approx(101_325.0)
    assert atmosphere.density_kg_m3 == pytest.approx(1.225, rel=1.0e-3)
    assert atmosphere.speed_of_sound_mps == pytest.approx(340.294, rel=1.0e-3)


def test_isa_atmosphere_tropopause() -> None:
    atmosphere = isa_atmosphere(TROPOPAUSE_ALTITUDE_M)

    assert atmosphere.temperature_K == pytest.approx(216.65)
    assert atmosphere.pressure_Pa == pytest.approx(22_632.0, rel=2.0e-3)
    assert atmosphere.density_kg_m3 == pytest.approx(0.3639, rel=2.0e-3)


def test_isa_atmosphere_lower_stratosphere() -> None:
    atmosphere = isa_atmosphere(15_000.0)

    assert atmosphere.temperature_K == pytest.approx(216.65)
    assert atmosphere.pressure_Pa > 0.0
    assert atmosphere.density_kg_m3 > 0.0
    assert atmosphere.speed_of_sound_mps > 0.0


def test_density_decreases_with_altitude() -> None:
    sea_level = isa_atmosphere(0.0)
    high_altitude = isa_atmosphere(10_000.0)

    assert high_altitude.density_kg_m3 < sea_level.density_kg_m3


def test_pressure_decreases_with_altitude() -> None:
    sea_level = isa_atmosphere(0.0)
    high_altitude = isa_atmosphere(10_000.0)

    assert high_altitude.pressure_Pa < sea_level.pressure_Pa


def test_isa_atmosphere_rejects_negative_altitude() -> None:
    with pytest.raises(ValueError):
        isa_atmosphere(-1.0)


def test_isa_atmosphere_rejects_too_high_altitude() -> None:
    with pytest.raises(ValueError):
        isa_atmosphere(LOWER_STRATOSPHERE_LIMIT_M + 1.0)


def test_isa_atmosphere_rejects_nonfinite_altitude() -> None:
    with pytest.raises(ValueError):
        isa_atmosphere(np.nan)

    with pytest.raises(ValueError):
        isa_atmosphere(np.inf)


def test_speed_of_sound() -> None:
    a = speed_of_sound(288.15)

    assert a == pytest.approx(340.294, rel=1.0e-3)


def test_speed_of_sound_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        speed_of_sound(0.0)

    with pytest.raises(ValueError):
        speed_of_sound(-10.0)


def test_mach_number() -> None:
    mach = mach_number(true_airspeed_mps=170.0, speed_of_sound_mps=340.0)

    assert mach == pytest.approx(0.5)


def test_mach_number_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        mach_number(true_airspeed_mps=-1.0, speed_of_sound_mps=340.0)

    with pytest.raises(ValueError):
        mach_number(true_airspeed_mps=100.0, speed_of_sound_mps=0.0)


def test_dynamic_pressure() -> None:
    q_bar = dynamic_pressure(density_kg_m3=1.225, true_airspeed_mps=100.0)

    assert q_bar == pytest.approx(6125.0)


def test_dynamic_pressure_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        dynamic_pressure(density_kg_m3=-1.0, true_airspeed_mps=100.0)

    with pytest.raises(ValueError):
        dynamic_pressure(density_kg_m3=1.225, true_airspeed_mps=-100.0)


def test_flight_condition() -> None:
    condition = flight_condition(altitude_m=0.0, true_airspeed_mps=100.0)

    assert condition.atmosphere.density_kg_m3 == pytest.approx(1.225, rel=1.0e-3)
    assert condition.dynamic_pressure_Pa == pytest.approx(6125.0, rel=1.0e-3)
    assert condition.mach == pytest.approx(100.0 / 340.294, rel=1.0e-3)