"""
Tests for aircraft geometry and mass properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.geometry import (
    AircraftGeometry,
    create_default_geometry,
    is_positive_definite,
)


def test_create_default_geometry() -> None:
    geometry = create_default_geometry()

    assert geometry.mass_kg > 0.0
    assert geometry.Ixx_kg_m2 > 0.0
    assert geometry.Iyy_kg_m2 > 0.0
    assert geometry.Izz_kg_m2 > 0.0
    assert geometry.wing_area_m2 > 0.0
    assert geometry.wingspan_m > 0.0
    assert geometry.mean_aerodynamic_chord_m > 0.0


def test_inertia_matrix_shape() -> None:
    geometry = create_default_geometry()

    inertia = geometry.inertia_matrix_kg_m2()

    assert inertia.shape == (3, 3)


def test_inertia_matrix_values() -> None:
    geometry = AircraftGeometry(
        mass_kg=10_000.0,
        Ixx_kg_m2=1.0,
        Iyy_kg_m2=2.0,
        Izz_kg_m2=3.0,
        Ixz_kg_m2=0.1,
        wing_area_m2=40.0,
        wingspan_m=10.0,
        mean_aerodynamic_chord_m=4.0,
    )

    expected = np.array(
        [
            [1.0, 0.0, -0.1],
            [0.0, 2.0, 0.0],
            [-0.1, 0.0, 3.0],
        ]
    )

    assert np.allclose(geometry.inertia_matrix_kg_m2(), expected)


def test_inverse_inertia_matrix() -> None:
    geometry = create_default_geometry()

    inertia = geometry.inertia_matrix_kg_m2()
    inverse_inertia = geometry.inverse_inertia_matrix_kg_m2()

    assert np.allclose(inertia @ inverse_inertia, np.eye(3))


def test_aspect_ratio() -> None:
    geometry = AircraftGeometry(
        mass_kg=10_000.0,
        Ixx_kg_m2=1.0,
        Iyy_kg_m2=2.0,
        Izz_kg_m2=3.0,
        Ixz_kg_m2=0.1,
        wing_area_m2=50.0,
        wingspan_m=10.0,
        mean_aerodynamic_chord_m=5.0,
    )

    assert geometry.aspect_ratio() == pytest.approx(2.0)


def test_positive_definite_matrix() -> None:
    matrix = np.eye(3)

    assert is_positive_definite(matrix)


def test_non_positive_definite_matrix() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert not is_positive_definite(matrix)


def test_invalid_matrix_shape_is_not_positive_definite() -> None:
    matrix = np.eye(2)

    assert not is_positive_definite(matrix)


def test_rejects_negative_mass() -> None:
    with pytest.raises(ValueError):
        AircraftGeometry(
            mass_kg=-1.0,
            Ixx_kg_m2=1.0,
            Iyy_kg_m2=2.0,
            Izz_kg_m2=3.0,
            Ixz_kg_m2=0.1,
            wing_area_m2=50.0,
            wingspan_m=10.0,
            mean_aerodynamic_chord_m=5.0,
        )


def test_rejects_invalid_wing_area() -> None:
    with pytest.raises(ValueError):
        AircraftGeometry(
            mass_kg=10_000.0,
            Ixx_kg_m2=1.0,
            Iyy_kg_m2=2.0,
            Izz_kg_m2=3.0,
            Ixz_kg_m2=0.1,
            wing_area_m2=0.0,
            wingspan_m=10.0,
            mean_aerodynamic_chord_m=5.0,
        )


def test_rejects_non_positive_definite_inertia() -> None:
    with pytest.raises(ValueError):
        AircraftGeometry(
            mass_kg=10_000.0,
            Ixx_kg_m2=1.0,
            Iyy_kg_m2=2.0,
            Izz_kg_m2=0.001,
            Ixz_kg_m2=10.0,
            wing_area_m2=50.0,
            wingspan_m=10.0,
            mean_aerodynamic_chord_m=5.0,
        )