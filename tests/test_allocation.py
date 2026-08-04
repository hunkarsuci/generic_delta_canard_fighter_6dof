"""Tests for control allocation and surface mixing."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.allocation import (
    AllocationConfig,
    allocate,
)


def test_pure_pitch_symmetric_elevons():
    """Pure pitch produces symmetric elevon and canard deflection."""
    result = allocate(pitch_cmd=0.1, roll_cmd=0.0, yaw_cmd=0.0, throttle_cmd=0.0)
    surfaces = result.surface_commands
    # Canard = pitch
    assert surfaces[0] == pytest.approx(0.1)
    # Left and right elevons both = pitch (symmetric)
    assert surfaces[1] == pytest.approx(0.1)
    assert surfaces[2] == pytest.approx(0.1)
    # Rudder = 0
    assert surfaces[3] == pytest.approx(0.0)
    assert result.feasible


def test_pure_roll_differential_elevons():
    """Pure roll produces opposite deflections on left/right elevons."""
    result = allocate(pitch_cmd=0.0, roll_cmd=0.15, yaw_cmd=0.0, throttle_cmd=0.0)
    surfaces = result.surface_commands
    # Canard = 0
    assert surfaces[0] == pytest.approx(0.0)
    # Left = -roll, Right = +roll
    assert surfaces[1] == pytest.approx(-0.15)
    assert surfaces[2] == pytest.approx(0.15)
    assert result.feasible


def test_pure_yaw():
    """Pure yaw produces rudder deflection."""
    result = allocate(pitch_cmd=0.0, roll_cmd=0.0, yaw_cmd=0.2, throttle_cmd=0.0)
    assert result.surface_commands[3] == pytest.approx(0.2)
    assert result.feasible


def test_zero_command_neutral():
    result = allocate(pitch_cmd=0.0, roll_cmd=0.0, yaw_cmd=0.0, throttle_cmd=0.0)
    assert np.all(result.surface_commands[:4] == 0.0)
    assert result.surface_commands[4] == 0.0


def test_throttle_passthrough():
    result = allocate(0.0, 0.0, 0.0, throttle_cmd=0.75)
    assert result.surface_commands[4] == pytest.approx(0.75)


def test_mixed_command():
    """Combined pitch + roll + yaw + throttle."""
    result = allocate(pitch_cmd=0.05, roll_cmd=0.1, yaw_cmd=-0.08, throttle_cmd=0.5)
    s = result.surface_commands
    assert s[0] == pytest.approx(0.05)  # canard = pitch
    assert s[1] == pytest.approx(0.05 - 0.1)  # left = pitch - roll
    assert s[2] == pytest.approx(0.05 + 0.1)  # right = pitch + roll
    assert s[3] == pytest.approx(-0.08)  # rudder = yaw
    assert s[4] == pytest.approx(0.5)


def test_infeasible_reported():
    """Commands beyond limits are flagged as infeasible."""
    result = allocate(
        pitch_cmd=1.0,
        roll_cmd=0.0,
        yaw_cmd=0.0,
        throttle_cmd=0.0,
        surface_max=np.array([0.3, 0.3, 0.3, 0.4, 1.0]),
    )
    assert not result.feasible
    # Canard (index 0) and elevons (1,2) should be flagged
    assert 0 in result.infeasible_channels
    assert 1 in result.infeasible_channels
    assert 2 in result.infeasible_channels


def test_deterministic():
    r1 = allocate(0.2, -0.1, 0.05, 0.5)
    r2 = allocate(0.2, -0.1, 0.05, 0.5)
    assert np.all(r1.surface_commands == r2.surface_commands)


def test_config_rejects_negative_gain():
    with pytest.raises(ValueError):
        AllocationConfig(pitch_gain=-1.0)


def test_custom_gains():
    cfg = AllocationConfig(pitch_gain=0.5, roll_gain=2.0, yaw_gain=0.8)
    result = allocate(0.1, 0.05, 0.2, 0.0, config=cfg)
    s = result.surface_commands
    assert s[0] == pytest.approx(0.5 * 0.1)
    assert s[1] == pytest.approx(0.5 * 0.1 - 2.0 * 0.05)
    assert s[2] == pytest.approx(0.5 * 0.1 + 2.0 * 0.05)
    assert s[3] == pytest.approx(0.8 * 0.2)
