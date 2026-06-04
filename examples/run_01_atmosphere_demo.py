"""
Atmosphere model demonstration.

Run from the repository root:

    python examples/run_01_atmosphere_demo.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from generic_delta_canard_fighter_6dof.atmosphere import (
    flight_condition,
    isa_atmosphere,
)


def main() -> None:
    """
    Print and plot atmosphere properties.
    """
    altitudes_m = np.linspace(0.0, 20_000.0, 101)

    temperatures_K = []
    pressures_Pa = []
    densities_kg_m3 = []
    speeds_of_sound_mps = []

    for altitude_m in altitudes_m:
        atmosphere = isa_atmosphere(altitude_m)

        temperatures_K.append(atmosphere.temperature_K)
        pressures_Pa.append(atmosphere.pressure_Pa)
        densities_kg_m3.append(atmosphere.density_kg_m3)
        speeds_of_sound_mps.append(atmosphere.speed_of_sound_mps)

    print("Atmosphere examples")
    print("-------------------")

    for altitude_m in [0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0]:
        atmosphere = isa_atmosphere(altitude_m)
        condition = flight_condition(altitude_m=altitude_m, true_airspeed_mps=150.0)

        print(
            f"h = {altitude_m:8.1f} m | "
            f"T = {atmosphere.temperature_K:7.2f} K | "
            f"rho = {atmosphere.density_kg_m3:7.4f} kg/m^3 | "
            f"a = {atmosphere.speed_of_sound_mps:7.2f} m/s | "
            f"M = {condition.mach:5.3f} | "
            f"q = {condition.dynamic_pressure_Pa:8.1f} Pa"
        )

    plt.figure()
    plt.plot(densities_kg_m3, altitudes_m)
    plt.xlabel("Density [kg/m^3]")
    plt.ylabel("Altitude [m]")
    plt.title("ISA Density vs Altitude")
    plt.grid(True)

    plt.figure()
    plt.plot(speeds_of_sound_mps, altitudes_m)
    plt.xlabel("Speed of Sound [m/s]")
    plt.ylabel("Altitude [m]")
    plt.title("Speed of Sound vs Altitude")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()