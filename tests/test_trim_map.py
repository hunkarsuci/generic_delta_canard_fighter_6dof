"""Tests for numerical trim-feasibility map."""

from __future__ import annotations

import pytest

from generic_delta_canard_fighter_6dof.trim_map import (
    compute_trim_map,
    trim_map_summary,
)


def test_trim_map_nominal_point():
    results = compute_trim_map([5000.0], [200.0])
    assert len(results) == 1
    assert results[0].converged
    assert results[0].alpha_deg > 0
    assert 0.0 <= results[0].throttle <= 1.0


def test_trim_map_small_grid():
    results = compute_trim_map([0.0, 5000.0], [150.0, 200.0])
    assert len(results) == 4
    summary = trim_map_summary(results)
    assert summary["total_points"] == 4
    assert summary["converged"] >= 1


def test_trim_map_summary_convergence_rate():
    results = compute_trim_map([5000.0], [200.0])
    summary = trim_map_summary(results)
    assert summary["convergence_rate"] == 1.0


def test_trim_map_repeatability():
    r1 = compute_trim_map([5000.0], [200.0])
    r2 = compute_trim_map([5000.0], [200.0])
    assert r1[0].alpha_deg == pytest.approx(r2[0].alpha_deg)
