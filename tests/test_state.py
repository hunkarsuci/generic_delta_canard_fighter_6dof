"""
Tests for state and control vector definitions
"""
from __future__ import annotations
import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.state import (
    ControlIndex,
    NUM_CONTROLS,
    NUM_STATES,
    StateIndex,
    control_to_dict,
    make_control,
    make_state,
    state_to_dict,
    validate_control,
    validate_state,
)

# Every test function starts with test_ and 
# this test checks whether make_state() creates a 12 element vector
def test_make_state_shape() -> None:
    x = make_state(VT=120.0, alpha=0.05, h=1000.0)

    assert x.shape == (NUM_STATES,)
    assert x[StateIndex.VT] == pytest.approx(120.0)
    assert x[StateIndex.ALPHA] == pytest.approx(0.05)
    assert x[StateIndex.H] == pytest.approx(1000.0)

def test_make_control_shape() -> None:
    u = make_control(delta_canard=0.01, throttle=0.7)

    assert u.shape == (NUM_CONTROLS,)
    assert u[ControlIndex.DELTA_CANARD] == pytest.approx(0.01)
    assert u[ControlIndex.THROTTLE] == pytest.approx(0.7)


def test_state_validation_rejects_wrong_shape() -> None:
    bad_state = np.zeros(11)

    with pytest.raises(ValueError):
        validate_state(bad_state)


def test_control_validation_rejects_wrong_shape() -> None:
    bad_control = np.zeros(4)

    with pytest.raises(ValueError):
        validate_control(bad_control)


def test_state_validation_rejects_negative_speed() -> None:
    bad_state = make_state(VT=100.0)
    bad_state[StateIndex.VT] = -1.0

    with pytest.raises(ValueError):
        validate_state(bad_state)


def test_control_validation_rejects_invalid_throttle() -> None:
    with pytest.raises(ValueError):
        make_control(throttle=1.5)

    with pytest.raises(ValueError):
        make_control(throttle=-0.1)


def test_state_to_dict() -> None:
    x = make_state(VT=150.0, alpha=0.1, theta=0.2, h=2000.0)
    d = state_to_dict(x)

    assert d["VT"] == pytest.approx(150.0)  # pytest.approx is used for floating point comparison here
    assert d["alpha"] == pytest.approx(0.1)
    assert d["theta"] == pytest.approx(0.2)
    assert d["h"] == pytest.approx(2000.0)


def test_control_to_dict() -> None:
    u = make_control(delta_canard=0.03, delta_rudder=-0.02, throttle=0.8)
    d = control_to_dict(u)

    assert d["delta_canard"] == pytest.approx(0.03)
    assert d["delta_rudder"] == pytest.approx(-0.02)
    assert d["throttle"] == pytest.approx(0.8)
