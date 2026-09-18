# Repository improvement checklist

Reviewed on 2026-09-18. This is a practical maintenance plan based on the current
source, tests, examples, documentation, packaging, and CI configuration.

Baseline: 269 tests pass on Python 3.11.15; package statement coverage is 95%.
Ruff lint and formatting checks pass. CI already tests Python 3.10–3.13 and
enforces 90% coverage. Preserve these checks during cleanup.

Work through the priorities below in order. Keep behavior fixes separate from
file moves so changes remain easy to review.

## 1. Fix behavior and misleading documentation

- [ ] **Resolve the inactive canard control.** `aerodynamics.py` never reads
  `DELTA_CANARD`, although `allocation.py` commands it and `actuators.py` models
  it. Add configurable, explicitly synthetic canard derivatives and tests for
  their force/moment effects, or clearly document the channel as unsupported.
  Update `docs/aerodynamic_data.md` and the README accordingly.
- [ ] **Make simulation end times consistent.** In `simulation.py`, `dt=0.3`
  and `t_final=1.0` produce a final sample at 1.2 while reporting 1.0. Define a
  final-step policy and apply it consistently to `realtime.py` and
  `examples/compare_controllers.py`, which currently use different rounding.
  Test zero duration, durations shorter than one step, and non-divisible times.
- [ ] **Make quaternion normalization explicit.** `integrators.py` normalizes
  indices 6–9 whenever a state has length 13. A generic 13-variable system with
  zero derivatives therefore changes unexpectedly. Use an explicit option,
  post-step callback, or aircraft-specific wrapper; preserve quaternion tests.
- [ ] **Strengthen configuration validation.** `PropulsionConfig` currently
  accepts negative thrust and NaN thrust axes. Validate finite values, physical
  bounds, vector shapes, and timing settings across `propulsion.py`,
  `geometry.py`, `allocation.py`, `actuators.py`, `control.py`, and `realtime.py`.
  Add focused regression tests for invalid inputs.
- [ ] **Refresh current documentation and archive old audits.** Move
  `docs/audit.md` and `docs/completion_audit.md` into a clearly dated historical
  section. Some findings remain valid, but many implementation and CI claims
  are obsolete. Update `docs/model.md` for quaternion dynamics, actuators,
  controllers, and real-time execution; extend `docs/validation.md` to cover
  these features. Reconcile the wind-axis wording and rotation formula in
  `docs/conventions.md` with `kinematics.py` and `transforms.py`.

## 2. Improve repository structure and everyday use

- [ ] **Keep a simple, documented layout.** Add a small repository map to the
  README and a documentation index. Keep reusable behavior in the package,
  runnable demonstrations in `examples/`, and verification in `tests/`.
  Consider a `src/generic_delta_canard_fighter_6dof/` layout when adding installed
  package checks; it is optional, not a prerequisite for these fixes.
- [ ] **Move reusable scenario code into the package.** Extract the closed-loop
  runner and reusable metrics from `examples/compare_controllers.py`. Keep CLI
  parsing and plotting in examples. Split PID/LQR code or Euler/quaternion
  dynamics into smaller modules only where it clarifies responsibilities;
  preserve existing imports through compatibility exports in `__init__.py`.
- [ ] **Finish and standardize examples.** Implement or remove the empty
  `examples/run_02_dynamics_demo.py`; use one naming convention for all examples.
  Add sample JSON configurations. Retain the gravity-only evaluation as an
  analytical check and add a documented aero-plus-propulsion scenario.
- [ ] **Use consistent CLI handling.** Replace manual argument scanning in
  `examples/evaluate_6dof.py` with `argparse`, already used by `run_realtime.py`.
  Reject unknown flags, missing values, and invalid configuration keys. Define
  config/CLI precedence and report output-file errors cleanly.
- [ ] **Consolidate dependencies and tool settings.** Make `pyproject.toml` the
  dependency source of truth; remove duplication in `requirements.txt` or make
  it reference the package. Remove unused `pydantic`, consider a plotting extra
  for `matplotlib`, and record tested dependency constraints. Put shared Ruff
  and coverage settings in `pyproject.toml`.
- [ ] **Improve contributor guidance and housekeeping.** Add `CONTRIBUTING.md`
  with setup, checks, conventions, and example commands. Trim unrelated
  template sections from `.gitignore` while retaining useful protections, and
  choose an ignored directory for generated plots and simulation results.
  Keep `LICENSE` intact.

## 3. Strengthen tests, APIs, and maintenance

- [ ] **Close meaningful test gaps.** Add dedicated propulsion tests and cover
  trim-map failure/CSV paths, real-time pacing/overruns, and linearization
  fallback paths. Current module coverage is 78% for `trim_map.py`, 80% for
  `realtime.py`, and 87% for `linearization.py`. Test behavior rather than
  targeting a percentage alone.
- [ ] **Tighten controller verification.** In `tests/test_control.py`, nominal
  trim/linearization failures should fail instead of skipping the integration
  test. Requiring only some negative eigenvalue real parts does not establish
  closed-loop stability; assert the intended stability criterion and explicitly
  account for any neutral modes. Add a deterministic closed-loop scenario.
- [ ] **Clarify real-time failures and telemetry.** `realtime.py` currently
  suppresses `RuntimeError` in `run()` and has a no-op `log_overruns` setting.
  Provide an observable stop reason or documented exception policy, implement
  logging, and distinguish compute time from elapsed wall time. Test repeated
  runs and pause/resume behavior against documented semantics.
- [ ] **Standardize public interfaces.** Review exports in `__init__.py`, add
  missing callback/config type annotations, and replace aircraft-specific raw
  indices with `StateIndex`, `QuatStateIndex`, and `ControlIndex`. Keep units,
  shapes, exceptions, and Euler/quaternion support explicit in docstrings.
  Clarify unused CG offsets in `geometry.py` and whether trim-map results should
  expose active bounds, as its module description promises.
- [ ] **Test the distributable package in CI.** Build a wheel, install it in a
  clean environment, and run an import/smoke check outside the checkout. Current
  editable installs and pytest's `pythonpath = ["."]` can conceal packaging
  problems. Add a Windows smoke job and automated noninteractive example checks.

Suggested first batch: fix end-time reporting, make normalization explicit,
validate propulsion inputs, and reconcile documentation. Handle larger file
moves after those behavior changes have regression coverage.
