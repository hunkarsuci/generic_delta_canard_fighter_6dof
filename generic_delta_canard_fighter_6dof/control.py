"""PID and LQR flight control systems for the delta-canard fighter.

PID Controller
--------------
Baseline inner-loop stabilization with:
- Proportional, integral, derivative terms
- Derivative filtering (first-order low-pass)
- Output saturation
- Anti-windup (clamping integrator when output saturates)
- Bumpless reset
- Deterministic fixed-step update

LQR Controller
--------------
Trim-based linear-quadratic regulator:
- A and B from numerical linearization at trim
- Bryson-style Q/R scaling from allowable deviations
- Configurable operating point
- Integrates with actuator limits via output clamping

Both controllers operate on virtual channels [pitch, roll, yaw, throttle]
which are mapped to physical surfaces by the allocation module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from generic_delta_canard_fighter_6dof.state import NUM_CONTROLS, NUM_STATES

# ═══════════════════════════════════════════════════════════════════════════════
# PID Controller
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PIDConfig:
    """Configuration for a single PID channel.

    Attributes
    ----------
    kp:
        Proportional gain.
    ki:
        Integral gain.
    kd:
        Derivative gain.
    output_min:
        Minimum output (for saturation).
    output_max:
        Maximum output (for saturation).
    deriv_filter_tau:
        Derivative low-pass filter time constant [s]. Set to 0 to disable.
    integrator_min:
        Minimum integrator state (anti-windup).
    integrator_max:
        Maximum integrator state (anti-windup).
    name:
        Channel label for diagnostics.
    """

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    deriv_filter_tau: float = 0.01
    integrator_min: float = -1.0
    integrator_max: float = 1.0
    name: str = "pid"


class PIDController:
    """Single-channel PID controller with derivative filtering and anti-windup.

    Transfer function (continuous):
        u(s) = (kp + ki/s + kd*s/(tau*s + 1)) * e(s)

    Discretized with forward Euler for the integrator and backward Euler
    for the derivative filter.
    """

    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0

    def reset(self) -> None:
        """Reset integrator and filter states (bumpless)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0

    def step(self, error: float, dt: float) -> float:
        """Compute control output for one time step.

        Parameters
        ----------
        error:
            Reference - measurement (positive = need more output).
        dt:
            Time step [s].

        Returns
        -------
        float
            Control output, clamped to [output_min, output_max].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")

        cfg = self.config

        # Proportional
        p_term = cfg.kp * error

        # Integral with anti-windup (clamp before adding to output)
        self._integral += cfg.ki * error * dt
        self._integral = float(
            np.clip(self._integral, cfg.integrator_min, cfg.integrator_max)
        )
        i_term = self._integral

        # Derivative with first-order low-pass filter
        if cfg.deriv_filter_tau > 0.0 and dt > 0.0:
            raw_deriv = (error - self._prev_error) / dt
            alpha = dt / (cfg.deriv_filter_tau + dt)
            self._deriv_filtered = (
                1.0 - alpha
            ) * self._deriv_filtered + alpha * raw_deriv
        else:
            self._deriv_filtered = (error - self._prev_error) / dt if dt > 0 else 0.0

        d_term = cfg.kd * self._deriv_filtered
        self._prev_error = error

        # Total output with saturation
        output = p_term + i_term + d_term

        # Anti-windup: if output saturates, clamp integrator
        if output > cfg.output_max:
            output = cfg.output_max
            self._integral = min(self._integral, cfg.output_max - p_term - d_term)
        elif output < cfg.output_min:
            output = cfg.output_min
            self._integral = max(self._integral, cfg.output_min - p_term - d_term)

        return float(output)


@dataclass
class AircraftPIDConfig:
    """Configuration for the 4-channel aircraft PID controller.

    Channels: roll, pitch, yaw, throttle.
    Each channel can be independently configured.

    Attributes
    ----------
    roll:
        Roll-rate or bank-angle PID config.
    pitch:
        Pitch-rate or pitch-angle PID config.
    yaw:
        Yaw-rate PID config.
    throttle:
        Airspeed-hold PID config.
    """

    roll: PIDConfig = field(
        default_factory=lambda: PIDConfig(
            kp=0.5,
            ki=0.1,
            kd=0.02,
            output_min=-0.5,
            output_max=0.5,
            integrator_min=-0.3,
            integrator_max=0.3,
            name="roll",
        )
    )
    pitch: PIDConfig = field(
        default_factory=lambda: PIDConfig(
            kp=0.5,
            ki=0.1,
            kd=0.02,
            output_min=-0.5,
            output_max=0.5,
            integrator_min=-0.3,
            integrator_max=0.3,
            name="pitch",
        )
    )
    yaw: PIDConfig = field(
        default_factory=lambda: PIDConfig(
            kp=0.3,
            ki=0.05,
            kd=0.01,
            output_min=-0.4,
            output_max=0.4,
            integrator_min=-0.2,
            integrator_max=0.2,
            name="yaw",
        )
    )
    throttle: PIDConfig = field(
        default_factory=lambda: PIDConfig(
            kp=0.01,
            ki=0.005,
            kd=0.0,
            output_min=0.0,
            output_max=1.0,
            integrator_min=0.0,
            integrator_max=1.0,
            deriv_filter_tau=0.1,
            name="throttle",
        )
    )


class AircraftPID:
    """Four-channel PID controller for the aircraft.

    Channels:
        0: roll  (error = phi_cmd - phi, or p_cmd - p)
        1: pitch (error = theta_cmd - theta, or q_cmd - q)
        2: yaw   (error = r_cmd - r, or beta_cmd - beta)
        3: throttle (error = VT_cmd - VT)

    Output is a 4-vector of virtual channel commands [pitch, roll, yaw, throttle]
    suitable for the allocation module.
    """

    def __init__(self, config: AircraftPIDConfig | None = None) -> None:
        cfg = config or AircraftPIDConfig()
        self.roll_ctl = PIDController(cfg.roll)
        self.pitch_ctl = PIDController(cfg.pitch)
        self.yaw_ctl = PIDController(cfg.yaw)
        self.throttle_ctl = PIDController(cfg.throttle)

    def reset(self) -> None:
        """Reset all channels."""
        for ctl in [self.roll_ctl, self.pitch_ctl, self.yaw_ctl, self.throttle_ctl]:
            ctl.reset()

    def step(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute virtual channel commands from state and reference.

        Parameters
        ----------
        state:
            12-state Euler vector or 13-state quaternion vector.
        reference:
            [phi_cmd, theta_cmd, r_cmd, VT_cmd] — attitude/rate and speed targets.
            Interpretation depends on channel configuration:
            - If configured for rate control, reference entries are rate commands.
            - If configured for attitude control, reference entries are angle commands.
        dt:
            Time step [s].

        Returns
        -------
        np.ndarray
            4-vector [pitch_cmd, roll_cmd, yaw_cmd, throttle_cmd].
        """
        from generic_delta_canard_fighter_6dof.state import StateIndex

        # Extract measurements (works for both 12-state Euler and 13-state quat
        # since indices 3-8 overlap in both representations)
        p = float(state[3])
        q = float(state[4])
        r = float(state[5])

        # Attitude — use quaternion extraction if 13-state, else direct
        if len(state) == 13:
            from generic_delta_canard_fighter_6dof.quat_state import QuatStateIndex
            from generic_delta_canard_fighter_6dof.quaternions import (
                quaternion_to_euler,
            )

            q_vec = state[QuatStateIndex.Q0 : QuatStateIndex.Q3 + 1]
            _phi, _theta, _ = quaternion_to_euler(q_vec)
            VT = float(np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2))
        else:
            _phi = float(state[StateIndex.PHI])
            _theta = float(state[StateIndex.THETA])
            VT = float(state[StateIndex.VT])

        # Errors (positive = need more output)
        roll_error = reference[0] - p  # rate-based roll damping default
        pitch_error = reference[1] - q  # rate-based pitch damping default
        yaw_error = reference[2] - r  # rate-based yaw damping default
        vt_error = reference[3] - VT  # speed error

        roll_out = self.roll_ctl.step(roll_error, dt)
        pitch_out = self.pitch_ctl.step(pitch_error, dt)
        yaw_out = self.yaw_ctl.step(yaw_error, dt)
        throttle_out = self.throttle_ctl.step(vt_error, dt)

        return np.array([pitch_out, roll_out, yaw_out, throttle_out], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# LQR Controller
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LQRConfig:
    """Configuration for a trim-based LQR controller.

    Uses Bryson-style scaling: Q[i,i] = 1 / (max_allowable_deviation[i])^2
                                 R[j,j] = 1 / (max_allowable_control[j])^2

    The scaling converts physical constraints into a well-conditioned
    cost function.

    Attributes
    ----------
    max_state_dev:
        12-vector of maximum allowable state deviations from trim.
        Used to construct Q = diag(1/dev^2).
    max_control_dev:
        5-vector of maximum allowable control deviations.
        Used to construct R = diag(1/dev^2).
    """

    max_state_dev: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                10.0,  # VT [m/s]
                0.05,  # alpha [rad]
                0.02,  # beta [rad]
                0.1,  # p [rad/s]
                0.1,  # q [rad/s]
                0.05,  # r [rad/s]
                0.05,  # phi [rad]
                0.05,  # theta [rad]
                0.05,  # psi [rad]
                10.0,  # x_N [m]
                10.0,  # y_E [m]
                5.0,  # h [m]
            ],
            dtype=float,
        )
    )
    max_control_dev: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.1, 0.1, 0.1, 0.05, 0.1],  # canard, elev_L, elev_R, rudder, throttle
            dtype=float,
        )
    )

    def __post_init__(self) -> None:
        self.max_state_dev = np.asarray(self.max_state_dev, dtype=float)
        self.max_control_dev = np.asarray(self.max_control_dev, dtype=float)
        if self.max_state_dev.shape != (NUM_STATES,):
            raise ValueError(f"max_state_dev must have shape ({NUM_STATES},)")
        if self.max_control_dev.shape != (NUM_CONTROLS,):
            raise ValueError(f"max_control_dev must have shape ({NUM_CONTROLS},)")
        if np.any(self.max_state_dev <= 0.0):
            raise ValueError("All max_state_dev entries must be positive")
        if np.any(self.max_control_dev <= 0.0):
            raise ValueError("All max_control_dev entries must be positive")


@dataclass
class LQRResult:
    """Result of LQR gain computation.

    Attributes
    ----------
    K:
        Gain matrix (NUM_CONTROLS x NUM_STATES).
    A:
        System matrix used.
    B:
        Input matrix used.
    Q:
        State cost matrix used.
    R:
        Control cost matrix used.
    S:
        Solution to the continuous algebraic Riccati equation.
    closed_loop_eigenvalues:
        Eigenvalues of A - B @ K.
    stabilizable:
        True if the (A, B) pair is stabilizable.
    """

    K: np.ndarray
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    S: np.ndarray
    closed_loop_eigenvalues: np.ndarray
    stabilizable: bool = True


class LQRController:
    """Trim-based LQR stabilization controller.

    Computes gains from a linearization and applies them as:
        u = u_trim - K @ (x - x_trim)
    """

    def __init__(self, K: np.ndarray, x_trim: np.ndarray, u_trim: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=float)
        self.x_trim = np.asarray(x_trim, dtype=float)
        self.u_trim = np.asarray(u_trim, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if self.K.shape != (NUM_CONTROLS, NUM_STATES):
            raise ValueError(
                f"K must have shape ({NUM_CONTROLS}, {NUM_STATES}), got {self.K.shape}"
            )
        if self.x_trim.shape != (NUM_STATES,):
            raise ValueError(
                f"x_trim must have shape ({NUM_STATES},), got {self.x_trim.shape}"
            )
        if self.u_trim.shape != (NUM_CONTROLS,):
            raise ValueError(
                f"u_trim must have shape ({NUM_CONTROLS},), got {self.u_trim.shape}"
            )

    def step(self, x: np.ndarray) -> np.ndarray:
        """Compute control from state.

        Parameters
        ----------
        x:
            12-state Euler vector.

        Returns
        -------
        np.ndarray
            5-element control vector.
        """
        x = np.asarray(x, dtype=float)
        dx = x - self.x_trim
        du = -self.K @ dx
        u = self.u_trim + du
        return np.clip(
            u,
            [-np.inf, -np.inf, -np.inf, -np.inf, 0.0],
            [np.inf, np.inf, np.inf, np.inf, 1.0],
        )


def compute_lqr(
    A: np.ndarray,
    B: np.ndarray,
    config: LQRConfig | None = None,
) -> LQRResult:
    """Compute LQR gains from a linearized system.

    Solves the continuous algebraic Riccati equation:
        A^T S + S A - S B R^{-1} B^T S + Q = 0

    Parameters
    ----------
    A:
        System matrix (NUM_STATES x NUM_STATES).
    B:
        Input matrix (NUM_STATES x NUM_CONTROLS).
    config:
        Bryson scaling configuration.

    Returns
    -------
    LQRResult
        Gains and diagnostics.
    """
    from scipy.linalg import eigvals, solve_continuous_are

    if config is None:
        config = LQRConfig()

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != (NUM_STATES, NUM_STATES):
        raise ValueError(f"A must have shape ({NUM_STATES}, {NUM_STATES})")
    if B.shape != (NUM_STATES, NUM_CONTROLS):
        raise ValueError(f"B must have shape ({NUM_STATES}, {NUM_CONTROLS})")

    # Bryson scaling
    Q = np.diag(1.0 / config.max_state_dev**2)
    R = np.diag(1.0 / config.max_control_dev**2)

    # Check stabilizability via controllability of unstable modes
    try:
        S = solve_continuous_are(A, B, Q, R)
    except (ValueError, np.linalg.LinAlgError):
        return LQRResult(
            K=np.zeros((NUM_CONTROLS, NUM_STATES)),
            A=A,
            B=B,
            Q=Q,
            R=R,
            S=np.zeros((NUM_STATES, NUM_STATES)),
            closed_loop_eigenvalues=np.array([]),
            stabilizable=False,
        )

    R_inv_BT = np.linalg.solve(R, B.T)
    K = R_inv_BT @ S

    A_cl = A - B @ K
    cl_eigvals = eigvals(A_cl)

    return LQRResult(
        K=K,
        A=A,
        B=B,
        Q=Q,
        R=R,
        S=S,
        closed_loop_eigenvalues=cl_eigvals,
        stabilizable=True,
    )
