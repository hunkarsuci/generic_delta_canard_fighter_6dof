# Aircraft animation driven by the project

The README GIF replays a six-second bank recovery computed by the repository's
own model. The camera follows the aircraft's position so its attitude remains
easy to see. Attitude is shown at its actual amplitude, with a fixed camera
orientation and scale throughout the replay.

The default scenario uses:

- `create_default_geometry()`: 10.5 m span, 50 m² reference wing area, 9,500 kg mass.
- `trim_straight_level()`: trim at 5,000 m and 200 m/s.
- `linearize()` and `compute_lqr()`: feedback gains at that trim point.
- `ActuatorBank`: achieved controls, initialized at trim.
- `aircraft_dynamics_quat()`: 13-state nonlinear dynamics with the existing
  aerodynamic and propulsion models, plus gravity.
- `rk4_step()`: 0.01 s integration, sampled at 20 frames per second.

The initial bank is 18 degrees. All displayed angles, airspeed, altitude, and
the roll plot come from the computed state history. Quaternion attitude is
converted through the project's body/NED conventions into the renderer's
upward-positive coordinates. No decorative attitude oscillations are added.
The controller's position reference advances along the trimmed flight path.

## Customize and regenerate

From the repository root:

```bash
python -m pip install -e .
python -m pip install "Pillow>=10.1"
python -m scripts.render_aircraft_banner
```

Edit [banner_config.json](banner_config.json) to change altitude, airspeed,
initial bank, duration, integration step, or frame rate. Empty override objects
use the package's current defaults, so changes to the model are picked up on
regeneration. For example:

```json
{
  "altitude_m": 4000.0,
  "airspeed_mps": 180.0,
  "initial_bank_deg": 12.0,
  "duration_s": 6.0,
  "integration_dt_s": 0.01,
  "fps": 20,
  "geometry": {"wingspan_m": 11.0, "wing_area_m2": 52.0},
  "aero_coefficients": {"CL_alpha": 5.2}
}
```

Overrides are passed to `AircraftGeometry` and `AeroCoefficients`, then used by
the simulation; geometry also scales the renderer. Unrecognized fields are
rejected. This local LQR example limits the initial bank to ±30 degrees; a custom
configuration must still produce a converged trim and successful controller
design.

Supported frame rates are 10, 20, 25, and 50 fps. Each frame interval must contain
a whole number of integration steps, and duration must contain a whole number
of frame intervals. This avoids rounding the requested final time.

```bash
# Render a different scenario without replacing the README assets:
python -m scripts.render_aircraft_banner --config my_scenario.json --output-dir /tmp/aircraft-preview

# Export flight data without rendering:
python -m scripts.render_aircraft_banner --simulate-only --output-dir /tmp/aircraft-data
```

On Windows, use an appropriate output directory such as
`$env:TEMP/aircraft-preview` in place of `/tmp/aircraft-preview`.

## Generated files

| File | Contents |
|---|---|
| `canard-aircraft.gif` | 1000 × 420 animation; actual-time playback plus a 0.9 s end hold |
| `canard-aircraft.png` | Still image of the initial state |
| `canard-flight.csv` | Every rendered time, 13-state vector, and five achieved controls, with units in headers |
| `canard-flight.json` | Resolved configuration, geometry, aerodynamic coefficients, and summary checks |

The final frame says **REPLAY** before the initial condition resets. States are
never reversed or blended to invent a seamless trajectory. Commit the generated
assets along with configuration/script changes to update the GitHub README.
The GIF is precomputed; GitHub does not run the simulator for each visitor.

## Visual assumptions

The repository contains reference geometry, not a CAD airframe. The procedural
mesh matches tip-to-tip span and the summed projected area of the two drawn wing
panels to the model's reference values. Fuselage, canopy, fin, canard placement,
and surface hinge locations are illustrative. Mean aerodynamic chord sets the
mesh's vertical scale; the mesh is not used to calculate aerodynamic forces.
Canard and elevon deflections use the achieved actuator values without angular
exaggeration; the rudder is not separately articulated in the mesh.

The existing aerodynamic model gives the canard channel no force or moment
authority, so the animation does not manufacture canard-driven maneuvers. The
aircraft and engine parameters remain synthetic and are not real-aircraft data.

The script uses Segoe UI, DejaVu Sans, or Pillow's default font, depending on
availability. Fonts can change text layout slightly. All procedural artwork is
covered by the repository's MIT license.
