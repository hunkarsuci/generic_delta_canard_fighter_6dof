"""Tests for the generic placeholder propulsion model."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.propulsion import PropulsionConfig


def test_propulsion_config_normalizes_thrust_axis() -> None:
    config = PropulsionConfig(thrust_axis_body=np.array([2.0, 0.0, 0.0]))

    assert config.thrust_axis_body == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize("max_thrust", [-1.0, np.nan, np.inf, -np.inf])
def test_propulsion_config_rejects_invalid_max_thrust(max_thrust) -> None:
    with pytest.raises(ValueError, match="max_thrust_N"):
        PropulsionConfig(max_thrust_N=max_thrust)


def test_propulsion_config_allows_zero_max_thrust() -> None:
    config = PropulsionConfig(max_thrust_N=0.0)

    assert config.max_thrust_N == 0.0


@pytest.mark.parametrize(
    "axis",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([np.nan, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0]),
        np.array([1.0, 0.0]),
    ],
)
def test_propulsion_config_rejects_invalid_axis(axis) -> None:
    with pytest.raises(ValueError, match="thrust_axis_body"):
        PropulsionConfig(thrust_axis_body=axis)
