"""
Tests for stability and modal analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.stability import analyze_stability


def test_analyze_stable_system() -> None:
    """A system with all negative real eigenvalues should be fully stable."""
    A = np.diag([-1.0, -2.0, -0.5])
    result = analyze_stability(A)
    assert result.num_stable == 3
    assert result.num_unstable == 0
    assert len(result.modes) == 3


def test_analyze_unstable_system() -> None:
    """A system with a positive eigenvalue should be unstable."""
    A = np.diag([-1.0, 0.5, -0.5])
    result = analyze_stability(A)
    assert result.num_unstable == 1
    assert result.num_stable == 2


def test_analyze_neutral_system() -> None:
    """A zero eigenvalue should be classified as neutral."""
    A = np.diag([0.0, -1.0, -2.0])
    result = analyze_stability(A)
    assert result.num_neutral == 1


def test_oscillatory_mode_detected() -> None:
    """Complex eigenvalues should be flagged as oscillatory."""
    # Underdamped second-order system
    A = np.array([[0.0, 1.0], [-4.0, -0.5]])
    result = analyze_stability(A)
    # Should have at least one oscillatory mode
    assert any(m.is_oscillatory for m in result.modes)


def test_damping_ratio_for_complex_pair() -> None:
    """Verify damping ratio ζ = -Re(λ) / |λ| for a known system."""
    zeta = 0.1
    omega_n = 2.0
    sigma = zeta * omega_n
    A = np.array([[-2 * sigma, -(omega_n**2)], [1.0, 0.0]])
    # This should have eigenvalues: -ζω_n ± j ω_d
    result = analyze_stability(A)
    osc_modes = [m for m in result.modes if m.is_oscillatory]
    assert len(osc_modes) >= 1
    m = osc_modes[0]
    assert m.damping_ratio == pytest.approx(zeta, abs=0.01)
    assert m.natural_frequency_radps == pytest.approx(omega_n, abs=0.01)


def test_time_constant_for_stable_real_pole() -> None:
    """For a real stable pole at λ = -1/τ, time constant should be τ."""
    tau = 0.5
    A = np.array([[-1.0 / tau]])
    result = analyze_stability(A)
    m = result.modes[0]
    assert m.is_stable
    assert not m.is_oscillatory
    assert m.time_constant_s == pytest.approx(tau)


def test_unstable_mode_no_time_constant() -> None:
    """Unstable modes should have no finite time constant."""
    A = np.array([[1.0]])
    result = analyze_stability(A)
    m = result.modes[0]
    assert not m.is_stable
    assert m.time_constant_s is None


def test_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError):
        analyze_stability(np.zeros((3, 2)))


def test_aircraft_linearization_stability() -> None:
    """Analyze stability of the linearized aircraft model at trim."""
    from generic_delta_canard_fighter_6dof.geometry import create_default_geometry
    from generic_delta_canard_fighter_6dof.linearization import linearize
    from generic_delta_canard_fighter_6dof.propulsion import (
        combined_forces_moments,
    )
    from generic_delta_canard_fighter_6dof.trim import trim_straight_level

    trim_result = trim_straight_level(5000.0, 200.0)
    assert trim_result.converged

    geo = create_default_geometry()
    fm = combined_forces_moments()
    lin = linearize(trim_result.state, trim_result.control, geo, fm)

    result = analyze_stability(lin.A)

    # Report counts — don't assert specific values (they depend on
    # placeholder coefficients)
    print("\nAircraft stability at trim (h=5000m, VT=200m/s):")
    print(f"  Stable modes: {result.num_stable}")
    print(f"  Unstable modes: {result.num_unstable}")
    print(f"  Neutral modes: {result.num_neutral}")

    for i, m in enumerate(result.modes):
        lam = m.eigenvalue
        kind = "osc" if m.is_oscillatory else "real"
        stable_str = (
            "stable" if m.is_stable else ("unstable" if not m.is_stable else "neutral")
        )
        tau_str = (
            f"tau={m.time_constant_s:.3f}s" if m.time_constant_s is not None else ""
        )
        freq_str = (
            f"wn={m.natural_frequency_radps:.3f}" if m.natural_frequency_radps else ""
        )
        print(
            f"  mode {i:2d}: {lam.real:+.4f}{lam.imag:+.4f}j [{kind}, {stable_str}] {freq_str} {tau_str}"
        )

    # Basic sanity: total modes should be 12
    assert len(result.modes) == 12
    assert result.num_stable + result.num_unstable + result.num_neutral == 12
