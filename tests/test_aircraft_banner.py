"""Physical and geometric checks for the simulation-driven README animation."""

from dataclasses import replace

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
from generic_delta_canard_fighter_6dof.quat_state import euler_state_to_quat_state
from generic_delta_canard_fighter_6dof.state import StateIndex, make_control, make_state
from scripts.render_aircraft_banner import (
    BannerConfig,
    aircraft_mesh,
    display_rotation,
    simulate_flight,
)


@pytest.mark.parametrize("span,area", [(10.5, 50.0), (12.0, 60.0)])
def test_mesh_matches_reference_span_and_wing_area(span, area):
    geometry = replace(create_default_geometry(), wingspan_m=span, wing_area_m2=area)
    mesh = aircraft_mesh(geometry, make_control())
    vertices = np.concatenate([points for points, _ in mesh])
    assert np.ptp(vertices[:, 1]) == pytest.approx(span)
    # Identify the two upper wing panels by their material, independently measure them.
    wing_faces = [p for p, color in mesh if np.array_equal(color, [155, 180, 199])]
    assert len(wing_faces) == 2
    measured_area = sum(
        abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        / 2
        for p in wing_faces
    )
    assert measured_area == pytest.approx(area)


def test_display_preserves_aerospace_pitch_and_bank_signs():
    pitch = euler_state_to_quat_state(make_state(VT=200, theta=np.pi / 6))
    # Positive pitch raises the nose in the upward-positive display frame.
    assert (display_rotation(pitch) @ [1, 0, 0])[2] == pytest.approx(0.5)
    roll = euler_state_to_quat_state(make_state(VT=200, phi=np.pi / 6))
    # Positive bank lowers the right wing, without reversing roll for display.
    assert (display_rotation(roll) @ [0, 1, 0])[2] == pytest.approx(-0.5)


def test_surface_angle_is_preserved_after_geometry_scaling():
    angle = np.deg2rad(15)
    mesh = aircraft_mesh(create_default_geometry(), make_control(delta_canard=angle))
    canards = [p for p, color in mesh if np.array_equal(color, [191, 213, 225])]
    for points in canards:
        trailing_edge = points[2] - points[0]
        assert trailing_edge[2] < 0  # positive command lowers the trailing edge
        assert trailing_edge[2] / trailing_edge[0] == pytest.approx(np.tan(angle))


def test_default_recovery_and_step_size_agreement():
    from generic_delta_canard_fighter_6dof.quat_state import quat_state_to_euler_state

    config = BannerConfig()
    trace = simulate_flight(config)
    refined = simulate_flight(replace(config, integration_dt_s=0.005))
    final = quat_state_to_euler_state(trace.states[-1])
    refined_final = quat_state_to_euler_state(refined.states[-1])
    assert len(trace.time) == 121
    assert trace.time[-1] == pytest.approx(6.0)
    assert np.all(np.isfinite(trace.states))
    assert abs(np.rad2deg(final[StateIndex.PHI])) < 0.1
    assert final[StateIndex.VT] == pytest.approx(200, abs=0.01)
    assert final[StateIndex.H] == pytest.approx(5000, abs=0.05)
    np.testing.assert_allclose(
        np.linalg.norm(trace.states[:, 6:10], axis=1), 1, atol=1e-12
    )
    # Halving dt retains attitude to within 0.01 deg, position to within 2 cm.
    np.testing.assert_allclose(
        final[6:9], refined_final[6:9], atol=np.deg2rad(0.01), rtol=0
    )
    np.testing.assert_allclose(final[9:12], refined_final[9:12], atol=0.02, rtol=0)


def test_custom_geometry_and_scenario_reach_simulator():
    config = BannerConfig(
        altitude_m=4000,
        airspeed_mps=180,
        initial_bank_deg=-12,
        duration_s=1,
        geometry={"wingspan_m": 11, "wing_area_m2": 52},
        aero_coefficients={"CL_alpha": 5.2},
    )
    trace = simulate_flight(config)
    assert trace.geometry.wingspan_m == 11
    assert trace.geometry.wing_area_m2 == 52
    assert trace.states[0, -1] == 4000
    assert np.linalg.norm(trace.states[0, :3]) == pytest.approx(180)
    assert trace.time[-1] == pytest.approx(1)
    assert not np.array_equal(trace.states[0], trace.states[-1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 0},
        {"duration_s": 1.03},
        {"integration_dt_s": 0.03},
        {"fps": 24},
        {"airspeed_mps": float("nan")},
        {"initial_bank_deg": 60},
    ],
)
def test_invalid_scenario_rejected(kwargs):
    with pytest.raises(ValueError):
        BannerConfig(**kwargs)
