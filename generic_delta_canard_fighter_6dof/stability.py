"""
Eigenvalue and modal analysis for the linearized 6DOF aircraft model.

Reports eigenvalues, natural frequencies, damping ratios, and
time constants for the linearized state matrix A.

Does NOT automatically label modes as short-period, phugoid, Dutch roll, etc.
Such labels should only be applied with state-participation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeCharacteristics:
    """Characteristics of a single real mode or complex pair."""

    eigenvalue: complex
    """The eigenvalue."""

    natural_frequency_radps: float | None
    """Natural frequency ω_n [rad/s], None for real modes."""

    damped_frequency_radps: float | None
    """Damped natural frequency ω_d [rad/s], None for real modes."""

    damping_ratio: float
    """Damping ratio ζ [-]. Negative means unstable."""

    time_constant_s: float | None
    """Approximate time constant τ [s]. ∞ for unstable or very lightly damped."""

    is_stable: bool
    """True if Re(λ) < 0."""

    is_oscillatory: bool
    """True if the mode is complex (oscillatory)."""


@dataclass(frozen=True)
class StabilityAnalysis:
    """Result of eigenvalue-based stability analysis."""

    eigenvalues: np.ndarray
    """All eigenvalues of A, shape (n,)."""

    modes: list[ModeCharacteristics]
    """Characteristics for each eigenvalue / complex pair."""

    num_stable: int
    """Number of eigenvalues with negative real part."""

    num_unstable: int
    """Number of eigenvalues with positive real part."""

    num_neutral: int
    """Number of eigenvalues with approximately zero real part."""

    A: np.ndarray
    """The state matrix that was analyzed."""


def analyze_stability(
    A: np.ndarray,
    *,
    neutral_threshold: float = 1e-6,
) -> StabilityAnalysis:
    """Compute eigenvalue decomposition and modal characteristics.

    Parameters
    ----------
    A:
        State matrix (n×n).
    neutral_threshold:
        |Re(λ)| below this is considered neutral.

    Returns
    -------
    StabilityAnalysis
        Eigenvalues and derived modal properties.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")

    eigvals = np.linalg.eigvals(A)

    # Sort by real part (most stable first)
    idx = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[idx]

    modes = []
    for lam in eigvals_sorted:
        is_osc = abs(lam.imag) > neutral_threshold
        is_stable = lam.real < -neutral_threshold

        if is_osc:
            omega_n = float(abs(lam))
            omega_d = float(abs(lam.imag))
            zeta = float(-lam.real / omega_n) if omega_n > 0 else 0.0
        else:
            omega_n = None
            omega_d = None
            zeta = float("inf") if lam.real >= 0 else float("-inf")
            # Actually: for a real pole at λ = -a, damping ratio is > 1 (overdamped)
            # We don't define ζ for real poles in the classical sense.
            zeta = 1.0 if lam.real < 0 else -1.0

        # Time constant
        if is_stable:
            if is_osc and omega_n > 0 and zeta != 0:
                tau = 1.0 / (zeta * omega_n)
            elif not is_osc:
                tau = 1.0 / abs(lam.real)
            else:
                tau = None
        else:
            tau = None  # unstable

        modes.append(
            ModeCharacteristics(
                eigenvalue=complex(lam),
                natural_frequency_radps=omega_n,
                damped_frequency_radps=omega_d,
                damping_ratio=zeta,
                time_constant_s=tau,
                is_stable=is_stable,
                is_oscillatory=is_osc,
            )
        )

    n_stable = sum(1 for lam in eigvals if lam.real < -neutral_threshold)
    n_unstable = sum(1 for lam in eigvals if lam.real > neutral_threshold)
    n_neutral = len(eigvals) - n_stable - n_unstable

    return StabilityAnalysis(
        eigenvalues=eigvals_sorted,
        modes=modes,
        num_stable=n_stable,
        num_unstable=n_unstable,
        num_neutral=n_neutral,
        A=A.copy(),
    )
