"""Render the README animation from the project's nonlinear flight simulation.

From the repository root: python -m scripts.render_aircraft_banner
The aircraft follows simulated attitude; its detailed mesh remains illustrative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generic_delta_canard_fighter_6dof.actuators import ActuatorBank
from generic_delta_canard_fighter_6dof.aerodynamics import AeroCoefficients
from generic_delta_canard_fighter_6dof.control import LQRController, compute_lqr
from generic_delta_canard_fighter_6dof.equations import aircraft_dynamics_quat
from generic_delta_canard_fighter_6dof.geometry import (
    AircraftGeometry,
    create_default_geometry,
)
from generic_delta_canard_fighter_6dof.integrators import rk4_step
from generic_delta_canard_fighter_6dof.linearization import linearize
from generic_delta_canard_fighter_6dof.propulsion import combined_forces_moments
from generic_delta_canard_fighter_6dof.quat_state import (
    QuatStateIndex as QI,
    euler_state_to_quat_state,
    quat_state_to_euler_state,
)
from generic_delta_canard_fighter_6dof.quaternions import quaternion_to_dcm
from generic_delta_canard_fighter_6dof.state import ControlIndex as CI
from generic_delta_canard_fighter_6dof.state import StateIndex as SI
from generic_delta_canard_fighter_6dof.trim import trim_straight_level

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
WIDTH, HEIGHT, SCALE = 1000, 420, 2
INK = (226, 237, 247)
MUTED = (128, 154, 176)
CYAN = (68, 211, 213)


@dataclass(frozen=True)
class BannerConfig:
    altitude_m: float = 5000.0
    airspeed_mps: float = 200.0
    initial_bank_deg: float = 18.0
    duration_s: float = 6.0
    integration_dt_s: float = 0.01
    fps: int = 20
    geometry: dict[str, float] = field(default_factory=dict)
    aero_coefficients: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        values = [
            self.altitude_m,
            self.airspeed_mps,
            self.initial_bank_deg,
            self.duration_s,
            self.integration_dt_s,
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Scenario values must be finite.")
        if not 0 <= self.altitude_m <= 20000 or self.airspeed_mps <= 0:
            raise ValueError("Use altitude in [0, 20000] m and positive airspeed.")
        if abs(self.initial_bank_deg) > 30:
            raise ValueError("Keep this local LQR demonstration within +/-30 deg bank.")
        if self.duration_s <= 0 or self.integration_dt_s <= 0:
            raise ValueError("Duration and integration step must be positive.")
        if self.fps not in (10, 20, 25, 50):
            raise ValueError("fps must be 10, 20, 25, or 50 for exact GIF timing.")
        ratios = (1 / self.fps / self.integration_dt_s, self.duration_s * self.fps)
        if any(v < 1 or not np.isclose(v, round(v), rtol=0, atol=1e-8) for v in ratios):
            raise ValueError(
                "Frame period must divide into dt steps; duration into frames."
            )
        for overrides in (self.geometry, self.aero_coefficients):
            if not isinstance(overrides, dict) or not all(
                np.isscalar(v) and np.isfinite(v) for v in overrides.values()
            ):
                raise ValueError(
                    "Model overrides must be dictionaries of finite numbers."
                )


@dataclass
class FlightTrace:
    time: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    geometry: AircraftGeometry


def simulate_flight(config: BannerConfig) -> FlightTrace:
    """Sample actual quaternion states and achieved actuator positions for replay."""
    geometry = replace(create_default_geometry(), **config.geometry)
    aero = AeroCoefficients(**config.aero_coefficients)
    trim = trim_straight_level(
        config.altitude_m, config.airspeed_mps, geometry=geometry, aero_coeffs=aero
    )
    if not trim.converged:
        raise ValueError(f"Cannot render this scenario: trim failed: {trim.message}")
    model = combined_forces_moments(aero_coeffs=aero)
    lin = linearize(trim.state, trim.control, geometry, model)
    lqr = compute_lqr(lin.A, lin.B)
    if not lqr.stabilizable:
        raise ValueError("Cannot render this scenario: LQR design failed.")
    controller = LQRController(lqr.K, trim.state.copy(), trim.control)
    actuators = ActuatorBank()
    actuators.reset(trim.control)
    initial = trim.state.copy()
    initial[SI.PHI] = np.deg2rad(config.initial_bank_deg)
    state = euler_state_to_quat_state(initial)
    dt = config.integration_dt_s
    n_steps = round(config.duration_s / dt)
    stride = round(1 / config.fps / dt)
    times, states, controls = [0.0], [state.copy()], [actuators.positions]
    for i in range(n_steps):
        # Trim describes translating level flight, not a fixed NED location.
        reference = trim.state.copy()
        reference[SI.X_N : SI.Y_E + 1] += lin.x_dot0[SI.X_N : SI.Y_E + 1] * (i * dt)
        controller.x_trim = reference
        command = controller.step(quat_state_to_euler_state(state))
        achieved = actuators.step(command, dt)
        state = rk4_step(
            lambda t, x: aircraft_dynamics_quat(t, x, achieved, geometry, model),
            i * dt,
            state,
            dt,
        )
        if not np.all(np.isfinite(state)):
            raise ValueError(f"Non-finite simulation state at {(i + 1) * dt:.3f} s.")
        if (i + 1) % stride == 0:
            times.append((i + 1) * dt)
            states.append(state.copy())
            controls.append(achieved.copy())
    return FlightTrace(np.array(times), np.array(states), np.array(controls), geometry)


def wing_scale(geometry: AircraftGeometry) -> np.ndarray:
    """Match the drawn wing's projected area and tip-to-tip span to the model."""
    plan = np.array([(1.8, 0.43), (-3.45, 4.35), (-3.65, 1.6), (-3.7, 0.48)])
    # Shoelace gives twice one half-wing's area, i.e. both visible half-wings.
    area = abs(
        np.dot(plan[:, 0], np.roll(plan[:, 1], 1))
        - np.dot(plan[:, 1], np.roll(plan[:, 0], 1))
    )
    span_scale = geometry.wingspan_m / (2 * 4.35)
    length_scale = geometry.wing_area_m2 / (area * span_scale)
    return np.array([length_scale, span_scale, geometry.mean_aerodynamic_chord_m / 5])


def display_rotation(state: np.ndarray) -> np.ndarray:
    """Map the mesh's x-forward/y-right/z-up coordinates through body/NED axes."""
    flip_z = np.diag([1, 1, -1])
    attitude = quaternion_to_dcm(state[QI.Q0 : QI.Q3 + 1])
    return flip_z @ attitude @ flip_z


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Use local fonts without fetching external assets."""
    candidates = (
        ["C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size * SCALE)
        except OSError:
            continue
    return ImageFont.load_default(size=size * SCALE)


def line(draw, points, fill, width=1):
    draw.line(
        [(int(x * SCALE), int(y * SCALE)) for x, y in points],
        fill=fill,
        width=width * SCALE,
    )


def text(draw, xy, value, size, fill=INK, bold=False):
    draw.text(
        (xy[0] * SCALE, xy[1] * SCALE),
        value,
        font=font(size, bold),
        fill=fill,
    )


def aircraft_mesh(geometry: AircraftGeometry, control: np.ndarray):
    """Illustrative mesh scaled to reference dimensions, with achieved deflections."""
    faces = []
    scale = wing_scale(geometry)

    def face(points, color):
        faces.append((np.array(points, dtype=float), np.array(color)))

    def surface(points, color, angle, hinge_x):
        # Positive trailing-edge-down deflection in the mesh's z-up frame.
        # Rotate after dimension scaling so the physical angle is preserved.
        points = np.array(points, dtype=float) * scale
        hinge_x *= scale[0]
        offset = points[:, 0] - hinge_x
        points[:, 0] = hinge_x + offset * math.cos(angle)
        points[:, 2] += offset * math.sin(angle)
        face(points / scale, color)

    def hull(stations, color, segments=12):
        rings = [
            [
                (x, ry * math.cos(a), z + rz * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)
            ]
            for x, ry, rz, z in stations
        ]
        for first, second in zip(rings, rings[1:]):
            for i in range(segments):
                j = (i + 1) % segments
                face([first[i], first[j], second[j], second[i]], color)
        face(list(reversed(rings[0])), color)
        face(rings[-1], color)

    # Round fuselage, with a pointed nose and dark engine nozzle.
    hull(
        [
            (-4.3, 0.40, 0.40, 0),
            (-3.5, 0.64, 0.58, 0),
            (-1.0, 0.70, 0.59, 0),
            (1.8, 0.51, 0.49, 0.03),
            (3.6, 0.35, 0.34, 0.02),
            (5.7, 0.015, 0.015, 0.02),
        ],
        (173, 193, 209),
    )
    hull([(-4.55, 0.34, 0.34, 0), (-4.25, 0.41, 0.41, 0)], (47, 65, 82))
    for side in (-1, 1):
        # Thin solid wings: forward swept leading edge, broad delta trailing edge.
        plan = [(1.8, 0.43), (-3.45, 4.35), (-3.65, 1.6), (-3.7, 0.48)]
        top = [(x, side * y, 0.02) for x, y in plan]
        bottom = [(x, side * y, -0.13) for x, y in plan]
        face(top, (155, 180, 199))
        face(list(reversed(bottom)), (104, 130, 151))
        for i in range(4):
            j = (i + 1) % 4
            face([top[i], bottom[i], bottom[j], top[j]], (112, 146, 166))
        # Distinct forward canards.
        surface(
            [
                (3.35, side * 0.34, 0.19),
                (2.1, side * 1.9, 0.19),
                (1.45, side * 1.8, 0.19),
                (2.15, side * 0.44, 0.19),
            ],
            (191, 213, 225),
            control[CI.DELTA_CANARD],
            3.35,
        )
        # Elevon panels and restrained turquoise wingtip markings.
        surface(
            [
                (-2.75, side * 1.10, 0.04),
                (-3.26, side * 3.85, 0.04),
                (-3.45, side * 3.60, 0.04),
                (-3.48, side * 1.10, 0.04),
            ],
            (121, 153, 174),
            control[CI.DELTA_ELEVON_LEFT if side == -1 else CI.DELTA_ELEVON_RIGHT],
            -2.75,
        )
        face(
            [
                (-3.08, side * 4.01, 0.045),
                (-3.45, side * 4.35, 0.045),
                (-3.48, side * 3.91, 0.045),
            ],
            (55, 187, 194),
        )
        # Side intakes.
        face(
            [
                (1.15, side * 0.54, -0.1),
                (0.78, side * 0.95, -0.13),
                (0.78, side * 0.85, -0.48),
                (1.15, side * 0.48, -0.4),
            ],
            (28, 49, 64),
        )
    # Tall swept fin with two shaded sides.
    for side in (-1, 1):
        face(
            [
                (-1.65, side * 0.06, 0.45),
                (-2.92, side * 0.045, 2.30),
                (-3.65, side * 0.045, 2.30),
                (-3.90, side * 0.06, 0.39),
            ],
            (137, 168, 189) if side == -1 else (107, 140, 165),
        )
    # Blue glass canopy, raised above the fuselage.
    hull(
        [
            (0.8, 0.20, 0.06, 0.52),
            (1.5, 0.32, 0.27, 0.53),
            (2.65, 0.27, 0.29, 0.43),
            (3.5, 0.04, 0.035, 0.34),
        ],
        (40, 117, 151),
        segments=10,
    )
    return [(points * scale, color) for points, color in faces]


def background(geometry: AircraftGeometry):
    img = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT * SCALE):
        v = y / (HEIGHT * SCALE)
        draw.line(
            (0, y, WIDTH * SCALE, y),
            fill=(10 + int(4 * v), 20 + int(6 * v), 33 + int(8 * v)),
        )
    # Fine graph-paper grid, confined to the illustration area.
    for x in range(370, 1000, 35):
        line(draw, [(x, 36), (x, 362)], (22, 39, 54))
    for y in range(47, 365, 35):
        line(draw, [(370, y), (966, y)], (22, 39, 54))
    line(draw, [(36, 42), (65, 42)], CYAN, 3)
    text(draw, (77, 31), "GENERIC AIRCRAFT / 6DOF", 12, CYAN, True)
    text(draw, (36, 70), "DELTA", 48, INK, True)
    text(draw, (36, 126), "+ CANARD", 36, INK, True)
    text(draw, (39, 193), "LQR BANK RECOVERY", 13, CYAN, True)
    text(draw, (39, 229), "AIRSPEED", 10, MUTED)
    text(draw, (182, 229), "ALTITUDE", 10, MUTED)
    text(draw, (39, 281), "ROLL RESPONSE", 10, MUTED)
    line(draw, [(39, 354), (316, 354)], (44, 67, 85))
    line(draw, [(36, 381), (964, 381)], (38, 60, 78))
    text(
        draw,
        (37, 392),
        f"SPAN {geometry.wingspan_m:g} m  /  WING {geometry.wing_area_m2:g} m²",
        10,
        MUTED,
    )
    text(draw, (380, 392), "13-STATE QUATERNION  /  RK4  /  ACTUATORS", 10, MUTED)
    text(draw, (810, 392), "SYNTHETIC AERODYNAMICS", 10, MUTED)
    return img


def camera_basis():
    camera = np.array([0.72, -1.25, 1.90])
    camera /= np.linalg.norm(camera)
    right = np.cross([0, 0, 1], camera)
    right /= np.linalg.norm(right)
    return camera, right, np.cross(camera, right)


def fit_view(trace: FlightTrace):
    """Fit every sampled attitude at one fixed scale, preserving motion amplitude."""
    _, right, up = camera_basis()
    projected = []
    for state, control in zip(trace.states, trace.controls):
        mesh = aircraft_mesh(trace.geometry, control)
        points = (
            np.concatenate([points for points, _ in mesh]) @ display_rotation(state).T
        )
        projected.append(np.column_stack((points @ right, -points @ up)))
    points = np.concatenate(projected)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = min(530 / (high[0] - low[0]), 252 / (high[1] - low[1]))
    return scale, (high + low) / 2


def render_frame(base, trace, index, view, euler_states):
    img = base.copy()
    draw = ImageDraw.Draw(img)
    state, control = trace.states[index], trace.controls[index]
    mesh = aircraft_mesh(trace.geometry, control)
    rotation = display_rotation(state)
    camera, right, up = camera_basis()
    scale, center = view

    def project(points):
        planar = np.column_stack((points @ right, -points @ up))
        return (planar - center) * scale + [670, 207]

    transformed = [(points @ rotation.T, color) for points, color in mesh]
    transformed.sort(key=lambda item: np.mean(item[0] @ camera))
    light = np.array([0.25, -0.45, 0.85])
    light /= np.linalg.norm(light)
    for points, color in transformed:
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length
        if normal @ camera < 0:
            normal = -normal
        shade = 0.58 + 0.42 * max(0, normal @ light)
        fill = tuple(np.clip(color * shade, 0, 255).astype(int))
        vertices = [tuple(p * SCALE) for p in project(points)]
        draw.polygon(vertices, fill=fill)
        draw.line(
            vertices + vertices[:1], fill=tuple(int(c * 0.86) for c in fill), width=1
        )
    euler = euler_states[index]
    text(draw, (39, 246), f"{euler[SI.VT]:.1f} m/s", 19, INK, True)
    text(draw, (182, 246), f"{euler[SI.H]:.1f} m", 19, INK, True)
    text(draw, (235, 281), f"{np.rad2deg(euler[SI.PHI]):+.2f}°", 12, CYAN, True)
    # Plot the same roll samples used to orient the aircraft, with a moving marker.
    rolls = np.rad2deg(euler_states[:, SI.PHI])
    low, high = min(-1, rolls.min()), max(1, rolls.max())
    xy = np.column_stack(
        (
            39 + 277 * trace.time / trace.time[-1],
            346 - 37 * (rolls - low) / (high - low),
        )
    )
    line(draw, xy, (44, 67, 85))
    if index:
        line(draw, xy[: index + 1], CYAN, 2)
    px, py = xy[index] * SCALE
    draw.ellipse(
        (px - 3 * SCALE, py - 3 * SCALE, px + 3 * SCALE, py + 3 * SCALE), fill=CYAN
    )
    text(draw, (39, 357), "0 s", 9, MUTED)
    text(draw, (290, 357), f"{trace.time[-1]:g} s", 9, MUTED)
    text(draw, (394, 43), "NONLINEAR MODEL / LQR", 11, MUTED)
    text(draw, (820, 43), f"t = {trace.time[index]:04.2f} s", 13, CYAN, True)
    text(draw, (396, 344), f"ROLL {np.rad2deg(euler[SI.PHI]):+06.2f}°", 11, MUTED)
    text(draw, (563, 344), f"PITCH {np.rad2deg(euler[SI.THETA]):+06.2f}°", 11, MUTED)
    text(draw, (739, 344), f"YAW {np.rad2deg(euler[SI.PSI]):+06.2f}°", 11, MUTED)
    if index == len(trace.time) - 1:
        text(draw, (831, 67), "REPLAY", 10, CYAN)
    return img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def write_trace(trace: FlightTrace, config: BannerConfig, output: Path):
    """Export all rendered samples so the animation can be independently inspected."""
    with (output / "canard-flight.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "u_mps",
                "v_mps",
                "w_mps",
                "p_radps",
                "q_radps",
                "r_radps",
                "q0",
                "q1",
                "q2",
                "q3",
                "north_m",
                "east_m",
                "altitude_m",
                "canard_rad",
                "elevon_left_rad",
                "elevon_right_rad",
                "rudder_rad",
                "throttle",
            ]
        )
        for t, state, control in zip(trace.time, trace.states, trace.controls):
            writer.writerow([f"{value:.12g}" for value in [t, *state, *control]])
    euler = np.array([quat_state_to_euler_state(x) for x in trace.states])
    report = {
        "scenario": asdict(config),
        "resolved_geometry": asdict(trace.geometry),
        "resolved_aero_coefficients": asdict(
            AeroCoefficients(**config.aero_coefficients)
        ),
        "sample_count": len(trace.time),
        "final_time_s": float(trace.time[-1]),
        "initial_bank_deg": float(np.rad2deg(euler[0, SI.PHI])),
        "final_bank_deg": float(np.rad2deg(euler[-1, SI.PHI])),
        "max_quaternion_norm_error": float(
            np.max(
                np.abs(np.linalg.norm(trace.states[:, QI.Q0 : QI.Q3 + 1], axis=1) - 1)
            )
        ),
        "note": "Model simulation; synthetic coefficients. Mesh details are illustrative.",
    }
    (output / "canard-flight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return euler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=OUTPUT / "banner_config.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Export flight samples without rendering images.",
    )
    args = parser.parse_args(argv)
    try:
        config = BannerConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        trace = simulate_flight(config)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    euler = write_trace(trace, config, output)
    print(
        f"Simulated {trace.time[-1]:g} s: bank {config.initial_bank_deg:g} -> {np.rad2deg(euler[-1, SI.PHI]):.3f} deg"
    )
    if args.simulate_only:
        return
    base, view = background(trace.geometry), fit_view(trace)
    frames = [render_frame(base, trace, i, view, euler) for i in range(len(trace.time))]
    frames[0].save(output / "canard-aircraft.png", optimize=True)
    # One shared palette prevents color flicker between GIF frames.
    samples = Image.new("RGB", (WIDTH, HEIGHT * 8))
    for i in range(8):
        samples.paste(frames[round(i * (len(frames) - 1) / 7)], (0, i * HEIGHT))
    palette = samples.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    path = output / "canard-aircraft.gif"
    durations = [1000 // config.fps] * len(frames)
    durations[-1] = 900  # Explicit end hold before replay; never reverse the physics.
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        f"Rendered {len(frames)} frames: {path} ({path.stat().st_size / 1024:.0f} KiB)"
    )


if __name__ == "__main__":
    main()
