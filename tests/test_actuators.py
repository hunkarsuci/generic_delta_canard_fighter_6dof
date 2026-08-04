"""Tests for actuator dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.actuators import (
    Actuator,
    ActuatorBank,
    ActuatorConfig,
    default_canard_config,
    default_elevon_config,
    default_rudder_config,
    default_throttle_config,
)

# ── ActuatorConfig validation ────────────────────────────────────────────────


def test_config_valid_defaults():
    cfg = ActuatorConfig()
    assert cfg.tau_s == 0.05
    assert cfg.position_min == -0.5
    assert cfg.position_max == 0.5


def test_config_rejects_negative_tau():
    with pytest.raises(ValueError, match="tau_s"):
        ActuatorConfig(tau_s=-0.01)


def test_config_rejects_zero_tau():
    with pytest.raises(ValueError, match="tau_s"):
        ActuatorConfig(tau_s=0.0)


def test_config_rejects_inverted_position_limits():
    with pytest.raises(ValueError, match="position_min"):
        ActuatorConfig(position_min=0.5, position_max=-0.5)


def test_config_rejects_inverted_rate_limits():
    with pytest.raises(ValueError, match="rate_min"):
        ActuatorConfig(rate_min_radps=5.0, rate_max_radps=-5.0)


# ── Actuator step response ───────────────────────────────────────────────────


def test_zero_command_maintains_initial_position():
    act = Actuator(ActuatorConfig(initial_position=0.0))
    pos = act.step(0.0, 0.01)
    assert pos == pytest.approx(0.0)


def test_step_response_before_saturation():
    """First-order step response: delta(t) = cmd * (1 - exp(-t/tau))."""
    tau = 0.1
    cmd = 0.2
    act = Actuator(
        ActuatorConfig(
            tau_s=tau,
            position_min=-1.0,
            position_max=1.0,
            rate_min_radps=-100.0,
            rate_max_radps=100.0,
            initial_position=0.0,
        )
    )

    dt = 0.001
    t = 0.0
    for _ in range(50):  # 0.05 s
        act.step(cmd, dt)
        t += dt

    # Analytical: position = cmd * (1 - exp(-t/tau))
    expected = cmd * (1.0 - np.exp(-0.05 / tau))
    assert act.position == pytest.approx(expected, rel=0.02)


def test_position_saturation_upper():
    act = Actuator(
        ActuatorConfig(
            tau_s=0.01,
            position_min=-0.3,
            position_max=0.3,
            rate_min_radps=-100.0,
            rate_max_radps=100.0,
            initial_position=0.0,
        )
    )
    # Large command drives into upper saturation immediately
    act.step(1.0, 0.01)
    assert act.position == pytest.approx(0.3)
    assert act.saturated_upper
    assert not act.saturated_lower


def test_position_saturation_lower():
    act = Actuator(
        ActuatorConfig(
            tau_s=0.01,
            position_min=-0.3,
            position_max=0.3,
            rate_min_radps=-100.0,
            rate_max_radps=100.0,
            initial_position=0.0,
        )
    )
    act.step(-1.0, 0.01)
    assert act.position == pytest.approx(-0.3)
    assert act.saturated_lower


def test_rate_limiting():
    """Slow rate limit forces the actuator to track slower than tau would allow."""
    act = Actuator(
        ActuatorConfig(
            tau_s=0.001,  # very fast natural response
            position_min=-1.0,
            position_max=1.0,
            rate_min_radps=-0.5,  # slow rate limit
            rate_max_radps=0.5,
            initial_position=0.0,
        )
    )
    pos = act.step(1.0, 0.01)
    # Rate-limited: max change = 0.5 * 0.01 = 0.005
    assert pos == pytest.approx(0.005, abs=1e-10)
    assert act.rate_saturated


def test_invalid_dt_raises():
    act = Actuator(ActuatorConfig())
    with pytest.raises(ValueError, match="dt"):
        act.step(0.0, -0.01)


def test_nonfinite_command_raises():
    act = Actuator(ActuatorConfig())
    with pytest.raises(ValueError, match="finite"):
        act.step(np.nan, 0.01)


def test_deterministic_repeated_runs():
    cfg = ActuatorConfig(tau_s=0.05, initial_position=0.1)
    traj1 = []
    traj2 = []

    act1 = Actuator(cfg)
    act2 = Actuator(cfg)
    for _ in range(10):
        traj1.append(act1.step(0.5, 0.01))
        traj2.append(act2.step(0.5, 0.01))

    assert traj1 == pytest.approx(traj2)


def test_reset():
    act = Actuator(ActuatorConfig(initial_position=0.0))
    act.step(0.5, 0.01)
    assert act.position != 0.0
    act.reset(0.0)
    assert act.position == 0.0
    assert not act.saturated_upper
    assert not act.saturated_lower


# ── throttle-specific ────────────────────────────────────────────────────────


def test_throttle_stays_within_0_1():
    cfg = default_throttle_config()
    act = Actuator(cfg)
    # Command beyond 1.0 should saturate
    act.step(2.0, 0.1)
    assert 0.0 <= act.position <= 1.0
    act.step(-1.0, 0.1)
    assert 0.0 <= act.position <= 1.0


# ── ActuatorBank ─────────────────────────────────────────────────────────────


def test_bank_default_construction():
    bank = ActuatorBank()
    positions = bank.positions
    assert len(positions) == 5
    assert np.all(np.isfinite(positions))


def test_bank_step_returns_5_vector():
    bank = ActuatorBank()
    cmd = np.array([0.1, 0.0, 0.0, 0.0, 0.5])
    achieved = bank.step(cmd, 0.01)
    assert achieved.shape == (5,)
    assert np.all(np.isfinite(achieved))


def test_bank_rejects_wrong_shape():
    bank = ActuatorBank()
    with pytest.raises(ValueError, match="shape"):
        bank.step(np.zeros(3), 0.01)


def test_bank_reset():
    bank = ActuatorBank()
    bank.step(np.array([0.3, 0.2, 0.1, 0.0, 0.8]), 0.01)
    bank.reset()
    # Throttle default init is 0.0, others at midpoint (0.0 by default for symmetric)
    assert bank.throttle.position == 0.0


# ── Synthetic defaults ───────────────────────────────────────────────────────


def test_default_configs_are_valid():
    for cfg in [
        default_canard_config(),
        default_elevon_config(),
        default_rudder_config(),
        default_throttle_config(),
    ]:
        act = Actuator(cfg)
        pos = act.step(0.0, 0.01)
        assert cfg.position_min <= pos <= cfg.position_max
