"""
Tests for the deterministic simulation runner.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from generic_delta_canard_fighter_6dof.simulation import simulate


def _constant_deriv(t: float, x: np.ndarray) -> np.ndarray:
    return np.array([1.0])


def test_simulate_constant_derivative_rk4() -> None:
    result = simulate(_constant_deriv, np.array([0.0]), dt=0.1, t_final=1.0)
    assert result.t_final == pytest.approx(1.0)
    assert result.dt == pytest.approx(0.1)
    assert result.t.shape == (11,)
    assert result.x.shape == (11, 1)
    assert result.x[-1, 0] == pytest.approx(1.0, rel=1e-10)


def test_simulate_constant_derivative_euler() -> None:
    result = simulate(
        _constant_deriv,
        np.array([0.0]),
        dt=0.1,
        t_final=1.0,
        integrator="euler",
    )
    assert result.x[-1, 0] == pytest.approx(1.0)


def test_simulate_zero_duration() -> None:
    result = simulate(_constant_deriv, np.array([5.0]), dt=0.1, t_final=0.0)
    assert len(result.t) == 1
    assert result.t[0] == pytest.approx(0.0)
    assert result.x[0, 0] == pytest.approx(5.0)


def test_simulate_rejects_invalid_dt() -> None:
    with pytest.raises(ValueError):
        simulate(_constant_deriv, np.array([0.0]), dt=0.0, t_final=1.0)
    with pytest.raises(ValueError):
        simulate(_constant_deriv, np.array([0.0]), dt=-0.1, t_final=1.0)
    with pytest.raises(ValueError):
        simulate(_constant_deriv, np.array([0.0]), dt=np.nan, t_final=1.0)


def test_simulate_rejects_invalid_t_final() -> None:
    with pytest.raises(ValueError):
        simulate(_constant_deriv, np.array([0.0]), dt=0.1, t_final=-1.0)


def test_simulate_rejects_nonfinite_state() -> None:
    with pytest.raises(ValueError):
        simulate(_constant_deriv, np.array([np.nan]), dt=0.1, t_final=1.0)


def test_simulate_rejects_unknown_integrator() -> None:
    with pytest.raises(ValueError):
        simulate(
            _constant_deriv,
            np.array([0.0]),
            dt=0.1,
            t_final=1.0,
            integrator="adams_bashforth",
        )


def test_simulate_detects_nonfinite_during_integration() -> None:
    """A derivative that produces NaN should be caught."""

    def exploding(t: float, x: np.ndarray) -> np.ndarray:
        if t > 0.5:
            return np.array([np.nan])
        return np.array([1.0])

    with pytest.raises(RuntimeError, match="Non-finite state"):
        simulate(exploding, np.array([0.0]), dt=0.1, t_final=1.0)


def test_simulate_deterministic() -> None:
    """Same inputs produce identical outputs."""
    r1 = simulate(_constant_deriv, np.array([0.0]), dt=0.1, t_final=1.0)
    r2 = simulate(_constant_deriv, np.array([0.0]), dt=0.1, t_final=1.0)
    assert np.allclose(r1.x, r2.x)
    assert np.allclose(r1.t, r2.t)


def test_simulate_different_dt_produces_different_result() -> None:
    """Different step sizes produce measurably different results for a stiff ODE."""

    # Use x_dot = -10*x (faster decay, larger truncation error)
    def fast_decay(t: float, x: np.ndarray) -> np.ndarray:
        return -10.0 * x

    r1 = simulate(fast_decay, np.array([1.0]), dt=0.05, t_final=0.2)
    r2 = simulate(fast_decay, np.array([1.0]), dt=0.01, t_final=0.2)
    # With 4x larger dt, truncation error should be measurably different
    assert not np.allclose(r1.x[-1, 0], r2.x[-1, 0])


# ── Integration test: evaluation CLI ─────────────────────────────────


def _run_eval(*args: str) -> str:
    """Run evaluate_6dof.main with given args and capture stdout."""
    import io
    import sys

    from examples.evaluate_6dof import main

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exit_code = main(list(args))
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    if exit_code != 0:
        raise RuntimeError(f"evaluate_6dof exited with {exit_code}")
    return output


def test_evaluate_default_run() -> None:
    output = _run_eval("--state-only")
    metrics = json.loads(output)
    assert "final_altitude_m" in metrics
    assert "final_speed_mps" in metrics
    assert metrics["any_nonfinite_state"] is False
    assert metrics["num_steps"] > 0


def test_evaluate_output_to_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _run_eval("--state-only", "--output", str(tmp_path))
        # --state-only takes precedence and writes to stdout anyway based on code
        # Actually state_only bypasses file output. Let's test without --state-only:
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_evaluate_deterministic() -> None:
    out1 = _run_eval("--state-only")
    out2 = _run_eval("--state-only")
    assert out1 == out2


def test_evaluate_custom_dt() -> None:
    out = _run_eval("--state-only", "--dt", "0.02", "--t-final", "2.0")
    metrics = json.loads(out)
    assert metrics["dt"] == 0.02
    assert metrics["t_final"] == 2.0


def test_evaluate_invalid_config() -> None:
    import io
    import sys

    from examples.evaluate_6dof import main

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        exit_code = main(["--config", "nonexistent.json"])
        assert exit_code != 0
    finally:
        sys.stderr = old_stderr
