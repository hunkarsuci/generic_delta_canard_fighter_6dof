# Aerodynamic Model: Coefficient Inventory

**IMPORTANT**: All coefficient values in this document are GENERIC PLACEHOLDERS.
They do NOT represent any real aircraft. They are chosen to produce plausible
trim solutions for validating the simulation, trim, linearization, and stability
analysis infrastructure. Replace with actual aircraft data before using for any
flight dynamics study.

## Coefficient table

### Longitudinal

| Symbol | Value | Source | Independent variables | Notes |
|---|---|---|---|---|
| CL0 | 0.12 | Placeholder | — | Zero-alpha lift coefficient |
| CL_alpha | 5.0 /rad | Placeholder | alpha | Lift-curve slope |
| CL_de | 0.4 /rad | Placeholder | symmetric elevon | Control derivative |
| CD0 | 0.022 | Placeholder | — | Zero-lift drag |
| K | 0.08 | Placeholder | CL² | Induced drag factor |
| Cm0 | 0.01 | Placeholder | — | Zero-alpha pitching moment |
| Cm_alpha | -0.55 /rad | Placeholder | alpha | Pitch stiffness (negative for stability) |
| Cm_de | -1.2 /rad | Placeholder | symmetric elevon | Pitch control power |

### Lateral-directional

| Symbol | Value | Source | Independent variables | Notes |
|---|---|---|---|---|
| CY_beta | -1.0 /rad | Placeholder | beta | Side-force derivative |
| Cl_beta | -0.08 /rad | Placeholder | beta | Dihedral effect |
| Cl_da | 0.06 /rad | Placeholder | differential elevon | Roll control power |
| Cl_dr | 0.012 /rad | Placeholder | rudder | Roll due to rudder |
| Cn_beta | 0.12 /rad | Placeholder | beta | Weathercock stability |
| Cn_da | -0.01 /rad | Placeholder | differential elevon | Adverse yaw |
| Cn_dr | -0.06 /rad | Placeholder | rudder | Yaw control power |

## Reference values

| Parameter | Value | Unit |
|---|---|---|
| Wing reference area (S) | 50.0 | m² |
| Wingspan (b) | 10.5 | m |
| Mean aerodynamic chord (c̄) | 5.0 | m |
| Aircraft mass | 9,500 | kg |

## Model limitations

1. **No Mach number dependence** — coefficients are constant with Mach
2. **No Reynolds number dependence**
3. **Linear aerodynamics** — no stall modeling, no nonlinear CL(alpha)
4. **No unsteady effects** — no hysteresis, no rate derivatives
5. **No ground effect**
6. **No high-alpha modeling** — valid only for small to moderate alpha
7. **Linear control derivatives** — no saturation or nonlinear mixing
8. **Symmetric aircraft** — lateral coefficients assume symmetry

## Control mixing

- Symmetric elevon: `de = (delta_elevon_left + delta_elevon_right) / 2`
- Differential elevon: `da = (delta_elevon_right - delta_elevon_left) / 2`

## Extrapolation behavior

The coefficient model uses linear extrapolation outside the documented range. There is no clamping, rate limiting, or saturation. The caller is responsible for staying within the valid envelope.

[OWNER INPUT REQUIRED]: Replace all placeholder coefficients with actual aircraft data. Specify valid alpha, beta, Mach, and control deflection ranges. Add rate derivatives if needed for dynamic stability fidelity.
