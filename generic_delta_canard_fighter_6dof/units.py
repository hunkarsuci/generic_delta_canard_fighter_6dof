"""
Unit conversion helpers.
The simulator internally uses SI units and radians.
"""

from __future__ import annotations
import numpy as np


def deg_to_rad(degrees: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def rad_to_deg(radians: float | np.ndarray) -> float | np.ndarray:
    """Convert radians to degrees."""
    return np.rad2deg(radians)


def ft_to_m(feet: float | np.ndarray) -> float | np.ndarray:
    """Convert feet to meters."""
    return np.asarray(feet) * 0.3048


def m_to_ft(meters: float | np.ndarray) -> float | np.ndarray:
    """Convert meters to feet."""
    return np.asarray(meters) / 0.3048


def kt_to_mps(knots: float | np.ndarray) -> float | np.ndarray:
    """Convert knots to meters per second."""
    return np.asarray(knots) * 0.514444


def mps_to_kt(mps: float | np.ndarray) -> float | np.ndarray:
    """Convert meters per second to knots."""
    return np.asarray(mps) / 0.514444