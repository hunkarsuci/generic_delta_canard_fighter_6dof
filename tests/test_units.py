"""Tests for unit conversion helpers."""

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


def test_deg_to_rad_scalar():
    assert deg_to_rad(180.0) == np.pi


def test_rad_to_deg_scalar():
    assert rad_to_deg(np.pi) == 180.0


def test_deg_rad_round_trip():
    original = 45.0
    assert rad_to_deg(deg_to_rad(original)) == original


def test_deg_to_rad_array():
    result = deg_to_rad(np.array([0.0, 90.0, 180.0]))
    assert np.allclose(result, np.array([0.0, np.pi / 2, np.pi]))


def test_ft_to_m_scalar():
    assert ft_to_m(1.0) == 0.3048


def test_m_to_ft_scalar():
    assert m_to_ft(0.3048) == 1.0


def test_ft_m_round_trip():
    assert m_to_ft(ft_to_m(1000.0)) == 1000.0


def test_kt_to_mps_scalar():
    assert kt_to_mps(1.0) == 0.514444


def test_mps_to_kt_scalar():
    assert mps_to_kt(0.514444) == 1.0


def test_kt_mps_round_trip():
    original = 250.0
    assert mps_to_kt(kt_to_mps(original)) == pytest.approx(original)
