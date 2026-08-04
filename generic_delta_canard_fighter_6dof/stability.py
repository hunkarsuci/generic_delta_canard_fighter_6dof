"""
Eigenvalue and modal analysis for the linearized 6DOF aircraft model.

Reports eigenvalues, natural frequencies, damping ratios, time constants,
right eigenvectors, and state participation factors.

Does NOT automatically label modes as short-period, phugoid, Dutch roll, etc.
Such labels should only be applied with state-participation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Human-readable state labels for eigenvector reporting.
_STATE_LABELS = [
    "VT",
    "alpha",
    "beta",
    "p",
    "q",
    "r",
    "phi",
    "theta",
    "psi",
    "x_N",
    "y_E",
    "h",
]


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

    right_eigenvector: np.ndarray = field(default_factory=lambda: np.array([]))
    """Right eigenvector (complex), shape (n_states,). Normalized to unit length."""

    participation: np.ndarray = field(default_factory=lambda: np.array([]))
    """State participation factors (real), shape (n_states,).
    Element k is |v_k|, normalized so max(|v|) = 1.0.
    Values near 1.0 indicate states that dominate the mode shape."""


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

    right_eigenvectors: np.ndarray
    """Right eigenvector matrix (complex), shape (n, n).
    Column j is the eigenvector for eigenvalue j."""

    A: np.ndarray
    """The state matrix that was analyzed."""

    def dominant_states(
        self, mode_index: int, top_n: int = 3
    ) -> list[tuple[str, float]]:
        """Return the top-N states by participation for a given mode.

        Parameters
        ----------
        mode_index:
            Index into ``modes`` (0-based).
        top_n:
            Number of top states to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (state_label, participation) sorted by decreasing participation.
        """
        m = self.modes[mode_index]
        if len(m.participation) == 0:
            return []
        idx = np.argsort(m.participation)[::-1][:top_n]
        return [
            (_STATE_LABELS[i], float(m.participation[i]))
            for i in idx
            if m.participation[i] > 0.01
        ]


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

    eigvals, right_eigvecs = np.linalg.eig(A)

    # Sort by real part (most stable first)
    idx = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[idx]
    right_eigvecs_sorted = right_eigvecs[:, idx]

    modes = []
    for j, lam in enumerate(eigvals_sorted):
        is_osc = abs(lam.imag) > neutral_threshold
        is_stable = lam.real < -neutral_threshold

        if is_osc:
            omega_n = float(abs(lam))
            omega_d = float(abs(lam.imag))
            zeta = float(-lam.real / omega_n) if omega_n > 0 else 0.0
        else:
            omega_n = None
            omega_d = None
            zeta = 1.0 if lam.real < 0 else -1.0

        # Time constant
        if is_stable:
            if is_osc and omega_n is not None and omega_n > 0 and zeta != 0:
                tau = 1.0 / (zeta * omega_n)
            elif not is_osc:
                tau = 1.0 / abs(lam.real)
            else:
                tau = None
        else:
            tau = None

        # Right eigenvector (normalized to unit length)
        v = right_eigvecs_sorted[:, j].copy()
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v = v / v_norm

        # State participation: normalized magnitude of eigenvector components
        mags = np.abs(v)
        max_mag = np.max(mags)
        participation = mags / max_mag if max_mag > 0 else mags

        modes.append(
            ModeCharacteristics(
                eigenvalue=complex(lam),
                natural_frequency_radps=omega_n,
                damped_frequency_radps=omega_d,
                damping_ratio=zeta,
                time_constant_s=tau,
                is_stable=is_stable,
                is_oscillatory=is_osc,
                right_eigenvector=v.copy(),
                participation=participation.copy(),
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
        right_eigenvectors=right_eigvecs_sorted.copy(),
        A=A.copy(),
    )
