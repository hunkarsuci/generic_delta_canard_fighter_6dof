"""
Aircraft geometry and mass-property definitions.

This module defines the physical parameters of the generic delta-canard fighter.

The values in this file are not official data for any real aircraft. They are
generic, configurable parameters chosen for educational 6DOF flight dynamics
and control development.

All units are SI units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np 

@dataclass(froze=True)
class AircraftGeometry:
    """
    Aircraft geometry and mass properties.

    Attributes
    ----------
    mass_kg:
        Aircraft mass [kg].
    Ixx_kg_m2:
        Roll moment of inertia about body x-axis [kg*m^2].
    Iyy_kg_m2:
        Pitch moment of inertia about body y-axis [kg*m^2].
    Izz_kg_m2:
        Yaw moment of inertia about body z-axis [kg*m^2].
    Ixz_kg_m2:
        Product of inertia between body x and z axes [kg*m^2].
    wing_area_m2:
        Reference wing area [m^2].
    wingspan_m:
        Reference wingspan [m].
    mean_aerodynamic_chord_m:
        Mean aerodynamic chord [m].
    x_cg_m:
        Center of gravity x-location relative to reference point [m].
    y_cg_m:
        Center of gravity y-location relative to reference point [m].
    z_cg_m:
        Center of gravity z-location relative to reference point [m].
    """
    mass_kg: float

    Ixx_kg_m2: float
    Iyy_kg_m2: float
    Izz_kg_m2: float
    Ixz_kg_m2: float

    wing_area_m2: float
    wingspan_m: float
    mean_aerodynamic_chord_m: float

    x_cg_m: float = 0.0
    y_cg_m: float = 0.0
    z_cg_m: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate geometry and mass-property values after object creation.
        """
        if self.mass_kg <= 0.0:
            raise ValueError("Aircraft mass must be positive.")

        if self.Ixx_kg_m2 <= 0.0:
            raise ValueError("Ixx must be positive.")

        if self.Iyy_kg_m2 <= 0.0:
            raise ValueError("Iyy must be positive.")

        if self.Izz_kg_m2 <= 0.0:
            raise ValueError("Izz must be positive.")

        if self.wing_area_m2 <= 0.0:
            raise ValueError("Wing reference area must be positive.")

        if self.wingspan_m <= 0.0:
            raise ValueError("Wingspan must be positive.")

        if self.mean_aerodynamic_chord_m <= 0.0:
            raise ValueError("Mean aerodynamic chord must be positive.")

        inertia_matrix = self.inertia_matrix_kg_m2()

        if not is_positive_definite(inertia_matrix):
            raise ValueError("Aircraft inertia matrix must be positive definite.")


    def inertia_matrix_kg_m2(self) -> np.ndarray:
        """
        Return the aircraft inertia matrix about the body axes.

        Returns
        -------
        np.ndarray
            3x3 inertia matrix [kg*m^2].

        Notes
        -----
        For many aircraft, the product of inertia Ixz is not zero because the
        aircraft mass distribution is not perfectly symmetric about all axes.

        With the usual aircraft body-axis convention, the inertia matrix is:

            I = [
                [ Ixx,   0, -Ixz],
                [   0, Iyy,    0],
                [-Ixz,   0,  Izz]
            ]
        """
        return np.array(
            [
                [self.Ixx_kg_m2, 0.0, -self.Ixz_kg_m2],
                [0.0, self.Iyy_kg_m2, 0.0],
                [-self.Ixz_kg_m2, 0.0, self.Izz_kg_m2],
            ],
            dtype=float,
        )
    
    def inverse_inertia_matrix_kg_m2(self) -> np.ndarray:
        """
        Return the inverse of the aircraft inertia matrix.

        This will be used later in rotational dynamics:

            omega_dot = I^{-1} (M - omega x I omega)
        """
        return np.linalg.inv(self.inertia_matrix_kg_m2())

    def aspect_ratio(self) -> float:
        """
        Compute wing aspect ratio.

        Equation
        --------
        AR = b^2 / S

        where:

            b = wingspan
            S = wing reference area
        """
        return self.wingspan_m**2 / self.wing_area_m2

def is_positive_definite(matrix: np.ndarray) -> bool:
    """
    Check whether a matrix is positive definite.

    A positive-definite inertia matrix means the rotational kinetic energy is
    positive for every nonzero angular velocity vector.

    Parameters
    ----------
    matrix:
        Matrix to check.

    Returns
    -------
    bool
        True if all eigenvalues are positive.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        return False

    eigenvalues = np.linalg.eigvalsh(matrix)

    return bool(np.all(eigenvalues > 0.0))

def create_default_geometry() -> AircraftGeometry:
    """
    Create a default generic delta-canard fighter geometry.

    Returns
    -------
    AircraftGeometry
        Generic aircraft geometry and mass properties.

    Notes
    -----
    These are educational placeholder values, not official values for any
    specific aircraft.
    """
    return AircraftGeometry(
        mass_kg=9_500.0,
        Ixx_kg_m2=75_000.0,
        Iyy_kg_m2=95_000.0,
        Izz_kg_m2=145_000.0,
        Ixz_kg_m2=2_500.0,
        wing_area_m2=50.0,
        wingspan_m=10.5,
        mean_aerodynamic_chord_m=5.0,
        x_cg_m=0.0,
        y_cg_m=0.0,
        z_cg_m=0.0,
    )