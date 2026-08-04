"""
Tests for unit conversion helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.units import (
    deg_to_rad,
    ft_to_m,
    kt_to_mps,
    m_to_ft,
    mps_to_kt,
    rad_to_deg,
)


def test_deg_to_rad_zero() -> None:
    assert deg_to_rad(0.0) == pytest.approx(0.0)


def test_deg_to_rad_180() -> None:
    assert deg_to_rad(180.0) == pytest.approx(np.pi)


def test_rad_to_deg_pi() -> None:
    assert rad_to_deg(np.pi) == pytest.approx(180.0)


def test_deg_rad_roundtrip() -> None:
    for deg in [0.0, 45.0, -30.0, 360.0, -720.0]:
        assert rad_to_deg(deg_to_rad(deg)) == pytest.approx(deg)


def test_ft_to_m_one() -> None:
    assert ft_to_m(1.0) == pytest.approx(0.3048)


def test_m_to_ft_one() -> None:
    assert m_to_ft(1.0) == pytest.approx(1.0 / 0.3048)


def test_ft_m_roundtrip() -> None:
    for ft in [0.0, 100.0, 35000.0]:
        assert m_to_ft(ft_to_m(ft)) == pytest.approx(ft)


def test_kt_to_mps_one() -> None:
    assert kt_to_mps(1.0) == pytest.approx(0.514444)


def test_mps_to_kt_one() -> None:
    assert mps_to_kt(1.0) == pytest.approx(1.0 / 0.514444)


def test_kt_mps_roundtrip() -> None:
    for kt in [0.0, 100.0, 500.0]:
        assert mps_to_kt(kt_to_mps(kt)) == pytest.approx(kt)


def test_array_inputs() -> None:
    """All converters should accept numpy arrays."""
    arr = np.array([0.0, 90.0, 180.0])
    result = deg_to_rad(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result[1] == pytest.approx(np.pi / 2)

    ft_arr = np.array([1.0, 10.0])
    assert ft_to_m(ft_arr).shape == (2,)

    kt_arr = np.array([100.0, 200.0])
    assert kt_to_mps(kt_arr).shape == (2,)
