# Changelog

All notable changes to xpcsjax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the authoritative changelog. The Sphinx page at
`docs/source/changelog.rst` surfaces the same milestones for users browsing
the rendered documentation.

## [Unreleased]

### Added

- **Mode-agnostic fit-quality metric `nrmse`** (`xpcsjax/service/fit_quality.py`),
  attached by `service.fit.run_fit` (CLI and GUI) under
  `nlsq_diagnostics["fit_quality"]` and surfaced as top-level `nrmse` /
  `sigma_source` in `nlsq_result.json`. `reduced_chi_squared` is not comparable
  across modes (homodyne: `sum((r/sigma)^2)/dof` over the full matrix with a
  constant `sigma = 0.01` placeholder when the data carry no uncertainties;
  `two_component`: unweighted SSR over the off-diagonal/t>0 mask divided by a
  far-lag noise-variance estimate). `nrmse = sqrt(SSR/n_valid) / std(c2_data)`
  over the same `t>0`, off-diagonal mask for every mode, evaluated on the
  fit-time model; the denominator is a data statistic, so a collapsing fitted
  contrast cannot improve it. Advisory only — `quality_flag` still follows
  `reduced_chi_squared`. `sigma_source` names what `reduced_chi_squared` was
  normalized with (`data` / `default_constant_0.01` / `far_lag_estimate`).

### Changed

- **`xpcsjax.cli.config_template.validate_config` renamed to
  `validate_config_file` and returns a `ValidationReport`** (was a `bool`
  that also printed). Printing now happens only in the `config_generator`
  command layer; `xpcsjax.service.config.validate_config(dict)` is
  unchanged. (codebase review item F2)

- **GUI: cancelling a fit no longer blocks the UI thread.**
  (`xpcsjax/gui/ipc/handle.py`, `controllers/fit_queue.py`, item A10)
  `WorkerHandle.cancel()` signals the worker and drives the
  join → SIGKILL escalation from a `QTimer` (poll 250 ms, escalate 5 s, give
  up 7 s), emitting `reaped`; the previous fully synchronous sequence
  (up to ~9 s per worker) survives only for the atexit/close-event path,
  where no event loop runs. Cancelling a worker that has just exited still
  reaps it and emits `reaped`.

- **Heterodyne config classes renamed** `ParameterManager` →
  `HeterodyneParameterManager`, `ParameterSpace` →
  `HeterodyneParameterSpace`, `ValidationResult` →
  `HeterodyneValidationResult` (`xpcsjax/config/heterodyne_*.py`, item F1).
  The old bare names remain importable as aliases for one release.

- **Retired the "xpcsjax is NLSQ-only; Bayesian sampling is permanently out of
  scope; use the upstream `homodyne` package" scope statement.** Removed from
  the four config templates, the `xpcsjax --help` / `xpcsjax-config` text, the
  module docstrings that restated it, and the user / developer docs. The
  ``method != "nlsq"`` guard in `xpcsjax/data/optimization.py` stays (its
  message now reads `Unsupported optimization method: ...; only 'nlsq' is
  implemented.`).

- **`repair_nan_values` (quality control) now defaults to `False`.**
  (`xpcsjax/data/quality_controller.py`) Median-filling non-finite `c2_exp`
  values reaches the fit indistinguishable from measured data; it must now
  be enabled explicitly via `quality_control.repair_nan_values: true`. When
  enabled, each repair now logs at WARNING with the count of values
  replaced, instead of being silent at DEBUG. None of the four shipped
  config templates set this key, so they all pick up the new default. Also
  removed `_repair_scaling_issues` (and its `repair_scaling_issues` config
  field) entirely: its mean-based heuristic divided/multiplied the whole
  `c2_exp` stack by 10/100, which its own comment already documented as
  capable of corrupting valid raw-count data — no safe setting existed.

- **Non-finite floats (`NaN`/`+-Inf`) in persisted JSON now always encode as
  `null`, never the strings `"Infinity"`/`"-Infinity"`.** (`xpcsjax/io/json_utils.py`)
  `io.json_safe`/`json_serializer` previously encoded `+-Inf` as those strings
  while `service.persist` (a separate, now-deleted copy of the same sanitizer)
  encoded both `NaN` and `+-Inf` as `null` — so a diverged/degenerate fit's
  `nlsq_result.json` (via persist) and its `parameters.json`/
  `analysis_results_nlsq.json`/`convergence_metrics.json` siblings (via
  `io.nlsq_writers`, in the same output directory) disagreed on the same
  value. `io.json_safe` is now the single sanitizer (`service.persist`
  delegates to it); any downstream JSON reader that special-cased the
  `"Infinity"` string convention should treat `null` as the non-finite
  sentinel for both NaN and +-Inf.

- **`core/` and `config/` dead-code and duplication cleanup (codebase review,
  no numerical or CLI-facing behavior change unless noted).** Removed
  unreachable `except ImportError` shims for in-package/hard-dependency
  modules (`config/parameter_manager.py`'s physics-validator fallback and its
  dead `_validate_physical_constraints_fallback`, `config/manager.py`'s
  logging fallback, `core/diagonal_correction.py`'s JAX/scipy shims,
  `core/physics_factors.py`'s `jnp = np` fallback) and dead diagnostics
  (`core/jax_backend.py`'s always-zero `_fallback_stats`). Consolidated the
  homodyne/heterodyne physics-validator machinery
  (`ConstraintSeverity`/`PhysicsViolation`/`ConstraintRule`/severity-priority
  ordering/the non-finite predicate) into a new shared
  `config/physics_validation_base.py`; the two sibling modules keep their own
  constraint tables and differing defaults. Added
  `AnalysisMode.try_parse()` (non-raising sibling of `.parse()`) and a
  `config/types.py::dict_section()` helper, replacing ~10 independently
  reimplemented mode-string / config-section-normalization sites across
  `config/manager.py`, `config/parameter_manager.py`, `core/models.py`,
  `core/homodyne_model.py`. Extracted `ConfigManager._load_scaling_values`
  / `_filter_active_parameters` / `_drop_fixed_parameters` from
  `get_initial_parameters`, and `heterodyne_parameter_space.py::_validate_tie`
  from `_apply_tied_parameters`'s per-tie loop, to reduce cyclomatic
  complexity; both keep every prior code path and error/warning message
  string identical. Renamed the heterodyne-specific
  `ParameterManager`/`ParameterSpace`/`ValidationResult` classes (which share
  a name but not a contract with their homodyne counterparts) to
  `HeterodyneParameterManager`/`HeterodyneParameterSpace`/
  `HeterodyneValidationResult`, keeping the old names as module-level aliases
  for one release. One small behavior change:
  `heterodyne_parameter_space.py`'s `parameter_space` section lookup
  (previously a bare `config_dict.get("parameter_space") or {}` with no
  type guard) now goes through `dict_section()`, so a wrong-type
  `parameter_space` value logs a warning and degrades to `{}` instead of
  raising `AttributeError` on the first `.get()` call downstream.

### Removed

- **`xpcsjax/data/performance_engine.py` (`PerformanceEngine`,
  `AdaptiveChunker`, `MultiLevelCache`, `MemoryMapManager`) — never
  constructed anywhere in the package.** Also removed the dead `MemoryPool`
  class and unused `AdvancedMemoryManager` methods, and the
  `quality_control.repair_scaling_issues` and
  `performance.performance_engine_enabled` config keys that gated the
  deleted code; the only remaining knob under `performance` is
  `memory_pressure_monitoring`.

### Fixed

- **APS-U loader no longer mislabels data after a bad bin index.**
  (`xpcsjax/data/hdf5_readers.py`, codebase review 2026-09-15 item A1) An
  out-of-range correlation-matrix bin index used to skip the matrix but not
  its `(q, phi)` pair, so every later matrix carried the next pair's labels,
  and the follow-up count mismatch truncated on a warning. Both now raise
  `XPCSDataFormatError` (the APS-old reader already did).

- **NPZ cache is keyed to its source HDF5 file; a changed source is a cache
  miss, not an error.** (`xpcsjax/data/npz_cache.py`, `xpcs_loader.py`;
  items A2 + review follow-up) `cache_metadata` now records
  `source_file`/`source_size`/`source_mtime_ns`. A name or size mismatch
  raises `CacheStaleError` internally, which the loader turns into a WARNING,
  an HDF5 re-read and a cache rewrite; an mtime-only mismatch (a
  content-preserving `cp`/`rsync`/restore) warns and serves the cache.
  Caches written before this release lack the keys and are served with a
  warning. A cache reused as the `data_file_name` (`.npz` override) is not
  validated against itself.

- **CLI `--initial-*` overrides can no longer be silently dropped.**
  (`xpcsjax/cli/config_handling.py`, item A11) When the active-parameter
  resolver fails, the CLI raises `ValueError("cannot apply --initial-*
  overrides ...")` instead of warning and running the fit on the YAML values
  the user overrode.

- **L3's CV penalty is now differentiable at uniform per-angle groups (and
  L2's gradient sees live L5 shear weights).**
  (`xpcsjax/optimization/nlsq/adaptive_regularization.py`,
  `strategies/hybrid_streaming.py`) The L3 CV² regularizer went through
  `jnp.std`, whose gradient is `0/0 = NaN` for a uniform per-angle group —
  every group is uniform at the quantile-seeded x0, so that block's gradient
  was NaN from the first L-BFGS step and those parameters never moved on the
  streaming L2 path. CV² is now `var/mean²` (identical value, smooth
  gradient). (Also fixed in this release: the jitted `value_and_grad` had
  captured the shear weighter's weight table as a trace-time constant,
  freezing the phi0 feedback at iteration 0; the weights are now a traced
  argument.)

- **Homodyne (`static_*` / `laminar_flow`) uncertainties follow the same
  one-rule covariance contract as heterodyne; three Critical audit findings
  closed.** (F1) The laminar wrapper never honoured `covariance_is_placeholder`:
  the hybrid-streaming L2 and stratified-LS accepted-L2 branches returned an
  identity covariance that was expanded and shipped as `sigma = 1.0` for every
  parameter. (F2) `recovery.safe_uncertainties_from_pcov` floored every
  singular / null-space / non-finite variance to `1e-5` on every wrapper path
  (a fabricated "known" uncertainty on exactly the directions the solver could
  not determine). (F3) The early `OUT_OF_CORE` branch (first strategy check)
  handed the compact `[contrast, offset | physics]` x0 to the per-angle kernel,
  which sliced physics slots as scaling (chi2 ~ 1e13, negative covariance
  diagonal). Now: every homodyne strategy (`strategies/stratified_ls.py`,
  `strategies/out_of_core.py`, `strategies/hybrid_streaming.py`, the wrapper's
  standard path) applies `covariance.finalize_covariance` to the REDUCED solver
  covariance before fixed-slot restore / scaling expansion and sets
  `info["covariance_is_placeholder"]`; the wrapper honours it (all-NaN
  uncertainties + `nlsq_diagnostics["covariance_is_placeholder"]` on every
  `OptimizationResult`, including the out-of-core direct returns), and
  otherwise reports `sqrt(diag)` with NaN for non-finite / negative variances
  and exact `0.0` only for structural rows (fixed physical slots, frozen
  constant-mode scaling) — no floor. The stratified-LS host recompute no longer
  pseudo-inverts a singular JᵀJ; the streaming plain branch no longer
  substitutes an identity for a missing / singular `pcov`; the failed-fit
  sentinels (`core.py` adapter+wrapper failure, `_cmaes_failed_result`) and
  `multistart.py`'s absent-best-covariance case return NaN + flag instead of
  zeros + identity; the sequential per-angle path marks zero-Jacobian
  directions NaN before the inverse-variance combination (excluded, not
  weighted as infinitely precise) and reports NaN for a parameter unknown at
  every angle. The early `OUT_OF_CORE` branch expands the compact x0 with the
  same `expand_per_angle_parameters` the `>= 1 M` recheck branch uses.
  `tests/parity/_golden/laminar_flow_end_to_end.npz` had only its `diag_keys`
  field updated for the new diagnostics key (recorded numerics untouched).
- **Heterodyne `two_component` uncertainties are now computed by one estimator and
  one failure rule on every path.** Three divergences found by the 2026-09-15
  covariance audit are closed via the new shared `optimization/nlsq/covariance.py`:
  (1) the in-memory engine route (`fit_two_component_via_engine`, the default
  `< 1 M`-point path) let nlsq scale its covariance by the PADDED,
  diagonal-masked residual length, so its uncertainties were smaller than
  `fit_nlsq_multi_phi`'s for the same problem by
  `sqrt((n_valid − p)/(ysize − p))` (2.7 % on the test fixture, ~1/(2 n_t) plus
  chunk padding on real data); the covariance is now rescaled post-solve onto
  the `n_phi (n_t−1)(n_t−2)` valid observations (`rescale_covariance_dof`; the
  solve itself is untouched) and `nlsq_diagnostics["covariance_dof"]` records
  the counts. (2) A non-real covariance — nlsq's all-`inf` singular marker (was
  shipped as `inf` sigma with no flag on the in-memory paths), a pseudo-inverse
  whose unidentified directions read as exactly `0.0` variance (hybrid-streaming
  plain branch, stratified-LS plain recompute), or a missing streaming `pcov`
  (was silently replaced by an identity) — is now all-NaN with
  `nlsq_diagnostics["covariance_is_placeholder"] = True` everywhere
  (`finalize_covariance`, applied in `build_result_from_nlsq`, the engine route,
  the streaming plain branch and every stratified-LS branch). (3) The
  stratified-LS host recompute (accepted L2/L3 and adapter-returned-no-covariance)
  now robust-scales the residual/Jacobian on `config.loss` exactly as nlsq's
  `curve_fit` does (`gauss_newton_covariance`, verified against nlsq's own
  `scale_for_robust_loss_function` for all five losses), so an accepted-layer
  popt and an adapter popt are judged by the same estimator; the pinv fallback
  is gone from that branch too. This is a consistency alignment, not a
  reproduced numeric error: on the synthetic fixtures (unweighted residuals
  ~1e-3 ≪ `f_scale = 1`) the robust scaling is the identity to ~1e-6 relative,
  and it only departs from the plain-SSR estimator when scaled residuals reach
  O(1) (e.g. sigma-weighted fits) — unverified on real data. The
  hybrid-streaming path minimises the plain SSR by design, so its
  `pinv(JᵀJ)·SSR/(n−p)` is the matching estimator for its own objective (its
  null-space / missing-`pcov` cases now fall under rule (2)). Homodyne paths
  are unchanged by this entry (their own audit findings are tracked separately).
- **Homodyne angle-stratified fits (auto for ≥100 k multi-angle points) no longer
  fit the c2 diagonal.** `wrapper.py`'s per-point (full-copy) stratified model
  branch evaluated the theory at `t1 == t2` as `offset + contrast` and compared
  it with the loader's diagonal-corrected data, while the non-stratified branch
  applies `apply_diagonal_correction` to the theory and the engine residual
  (`strategies/residual_jit.py`) masks `t1 == t2`. The optimizer distorted the
  physics to chase those lag-free points (120 k-point synthetic: off-diagonal
  SSR 5.45 vs 0.03 on the standard path, `D0` 1319 vs 1000). The stratified
  branch now zeroes the diagonal residual, matching `residual_jit`; reported
  `chi_squared` on that path is therefore the off-diagonal SSR (the masked
  points still count in `n_data`, so `reduced_chi_squared`'s dof is unchanged,
  as on the `residual_jit` path). This changes
  `laminar_flow` / `static_*` fit results on large datasets (it is a deliberate
  departure from the upstream homodyne behaviour, which fit the diagonal).
- **Heterodyne fitted/residual plots were evaluated one `dt` off the fit-time
  grid.** `viz.nlsq_plots._evaluate_c2_per_angle` fed the loader's `t1` (origin
  0) to the adapter kernel, while the heterodyne fit runs the stateful
  `HeterodyneModel` at `t_start = dt`. Negligible at ~1000 frames, decisive on
  short grids. The evaluator now builds the same model the fit ran; the NRMSE
  metric reuses it for every mode. `HeterodyneModel.from_config` also tolerates
  present-but-null YAML sections (`scattering: null`).
- **Heterodyne `two_component` ≥1 M stratified-LS path reported `uncertainty = 1.0`
  for every parameter when the `execute_layers` L2 hierarchical candidate was
  accepted.** Two defects: the driver tagged the identity placeholder only under
  `nlsq_diagnostics` (nested), so the result builder's NaN guard never saw it;
  and the branch used an identity placeholder at all. The builder now honours the
  flag in both locations, and the accepted L2 branch computes the same
  host-`jacfwd` Gauss-Newton covariance (`s² (JᵀJ)⁻¹` at the L2 `popt`,
  data-only residual) the L3-only branch already used, so `result.uncertainties`
  are real. `_chunked_jacfwd_dense` now fills a preallocated Jacobian instead of
  concatenating column blocks, halving the host-memory peak of that recompute.
  Two behaviour changes on the accepted-layer covariance recompute branches
  (accepted L2 *and* accepted L3-only): a singular `JᵀJ` no longer falls back
  to `np.linalg.pinv` (which reports unidentified directions as exactly zero
  variance), and the recompute is skipped on every adapter-less branch when the
  dense Jacobian would exceed the `select_nlsq_strategy` memory budget; both
  cases return all-NaN uncertainties tagged `covariance_is_placeholder=True`
  instead of a number. Large-N L3 fits that previously received (pinv or
  full-budget) uncertainties therefore now report NaN with the flag set;
  parameters are unaffected. The plain adapter-returned-no-covariance fallback
  keeps its pinv fallback (mirrors laminar's `strategies/stratified_ls.py`).
- **`NLSQWrapper.fit`'s two out-of-core (`OUT_OF_CORE`, >75% RAM) return
  paths — the initial strategy decision and the post-stratification
  recheck — had drifted on `reduced_chi_squared`'s DOF for a static
  (`static_anisotropic` / `static_isotropic`) analysis mode with an explicit
  `per_angle_mode: "constant"` token.** The initial branch resolved the token
  with the plain (non-pinned) resolver, reporting DOF = `n_physical` (3 for a
  3-physical-parameter fixture); the recheck branch already used
  `resolve_per_angle_mode_static_pinned`, which pins static modes to the dense
  `individual` layout regardless of the requested token, reporting
  DOF = `n_physical + 2*n_phi` (13 for `n_phi=5`) for the identical dense
  `popt`. Both branches are now one function
  (`wrapper_out_of_core_route.run_out_of_core_route`) using the pinned
  resolver, so this can't drift again; the initial branch's DOF for this case
  moves from 3 to 13 (the bug), matching the recheck branch's pre-existing
  (correct) value. Also re-raises `MemoryError` from the >=1 M stratified-LS
  route instead of falling through to the dense in-memory `curve_fit_large`
  path, which needs strictly *more* memory than the route that just OOM'd.

## [0.1.7] - 2026-09-04

### Changed

- **NLSQ floor raised to `nlsq>=0.7.6`** (from `>=0.6.10`) across
  `pyproject.toml`, `.pre-commit-config.yaml`, the runtime dependency check in
  `runtime/utils/system_validator.py`, the conda recipes under
  `conda-recipe/`, and the documentation that quotes the pin. `uv.lock`
  re-resolved. Follows the bump procedure in
  `docs/source/development/nlsq_integration.rst`.
- **`conda-recipe/nlsq/meta.yaml` updated to nlsq 0.7.6.** The 0.7.5 step in
  this line added the `optimistix >=0.1.0` core dependency (introduced
  upstream in 0.7.4) and restored the upstream `pyside6 >=6.10.0` floor now
  that conda-forge ships PySide6 6.11.2 (the recipe previously relaxed it to
  `>=6.4.0` because conda-forge topped out at 6.9.3). 0.7.6 changes neither
  the dependency set nor the `nlsq` / `nlsq-gui` entry points, so only the
  version, source URL, and sha256 moved.
- **`make verify` now reports zero warnings** (was 10). Three upstream nlsq
  warnings were filtered at the narrowest scope that is honest for each:
  `RuntimeWarning: Optimization may be stagnant` is a global
  `filterwarnings` entry, because
  `ConvergenceMonitor.detect_stagnation` fires exactly when a trust-region
  solve has converged (`relative_variance < threshold or grad_stagnation`) and
  so is never actionable; the two `NumericalStabilityGuard` warnings
  (`Ill-conditioned Jacobian`, `Could not compute SVD for condition number`)
  are scoped to `tests/optimization/test_fixed_parameters_integration.py`
  via `pytestmark`, since pinning a parameter collapses its bounds to zero
  width and leaves a structurally-zero Jacobian column by design — an
  ill-conditioned Jacobian remains a real signal everywhere else.

### Notes

- `optimistix` now resolves into `uv.lock` transitively via nlsq, so the
  long-deferred `jaxopt` → `optimistix` migration for the L2 hierarchical
  warm-start (`optimization/nlsq/hierarchical.py`) no longer requires adding a
  new dependency — only a direct declaration in `pyproject.toml`. `jaxopt`
  remains the code path in use; nothing was migrated.

## [0.1.6] - 2026-08-30

### Fixed

- **Negative-dominant `D_offset` now detected in physics validators.** The
  `D_offset`/`D0` overfitting checks only compared `ratio > 0.5`, missing the
  symmetric case where a large-magnitude *negative* offset dominates `D0`
  (e.g. `D_offset_ref/D0_ref = -0.503` in a real C045 `two_component` fit,
  implying a negative diffusion coefficient at early times). Both checks now
  compare `abs(ratio) > 0.5` (#75).
- Resolved mypy hard-gate failures on main (CI run 33300386957): explicit
  type annotations for union-typed locals in `fallback_chain.py`,
  `recovery.py`, and `validation.py`, plus a new `AngleGroup` `TypedDict` in
  `chunking.py` replacing an overly-loose `dict[str, Any]` (#73).
- **Duplicate `HeterodyneModel` class name.** Two unrelated classes shared
  the name — `core/heterodyne_model.py`'s `PhysicsModelBase` adapter and
  `core/heterodyne_model_stateful.py`'s public stateful model — so
  `from xpcsjax.core.heterodyne_model import HeterodyneModel` and
  `from xpcsjax.core import HeterodyneModel` silently returned different
  classes. Renamed the adapter to `HeterodynePhysicsAdapter` across every
  real call site; the public `xpcsjax.core.HeterodyneModel` export is
  unaffected (#71).
- **Stratified-LS (>=1M point) diagonal-exclusion mask** compared indices
  into two independently-built `np.unique()` arrays instead of the actual
  `t1`/`t2` values, wrongly zeroing real off-diagonal points and keeping
  true diagonal ones (#71).
- **Hybrid-streaming `per_angle_mode="constant"` fallback index
  misalignment.** In the quantile-estimation-failure fallback, L3's
  regularization group indices and L4's gradient-collapse-monitor watched
  indices were resolved via the literal `per_angle_mode_actual` string
  instead of the real 2-param `[contrast_mean, offset_mean, *physical]`
  layout the fallback actually uses, watching the scaling head as if it
  were physics (#71).
- **`coerce_finite_float` no longer silently coerces a stray YAML boolean**
  (e.g. a typo'd `max: true`) to `1.0`/`0.0` at the shared bounds/values
  validation boundary (#71).
- **Degenerate 1-frame Siegert-ceiling check no longer reads the excluded
  `tau=0` diagonal spike.** The `matrix.shape[1] <= 1` fallback branch in
  `data/filtering_utils.py` and `data/validation.py` read `diagonal[0]`
  into the check instead of skipping it, reintroducing the anti-pattern the
  surrounding code otherwise avoids (#71).
- 19 correctness gaps closed in the NLSQ fitting workflow and 6 more across
  config/core/viz, found via independent step-by-step audits that traced the
  NLSQ workflow and `docs/diagrams/architecture.md` against the real code.

### Removed

- ~7800 lines of confirmed-dead code removed repo-wide (ponytail-audit).

### Documentation

- Config examples across the documentation now match the shipped YAML
  templates exactly — bounds format, key nesting, and the heterodyne
  physics/scaling parameter layout — instead of several invented shapes
  that a reader could not actually load (#70).
- Theory prose terminology and notation cleaned up: dropped the "diffusion
  integral" label for `D(t1,t2)` (referred to by symbol only, alongside
  `J(t)`), and aligned Sphinx theory notation with the README (#69, #70).

### Dependencies

- `nlsq` bumped 0.7.1 → 0.7.4 and `evosax` bumped 0.2.0 → 0.3.1 in
  `uv.lock` and the conda recipe (`conda-recipe/xpcsjax/meta.yaml`); host
  `setuptools` floor raised to `>=77` to match evosax 0.3.1's build
  requirement (#74).

## [0.1.5] - 2026-08-25

### Added

- **`fixed_parameters` / `active_parameters` support across every NLSQ execution
  tier** (#55, #57, #59). A user can now freeze named physical parameters or
  restrict the optimized set for `NLSQWrapper.fit()`, `NLSQAdapter.fit()`, the
  CMA-ES escape, and the sequential, out-of-core, stratified-LS, and
  hybrid-streaming tiers alike — `resolve_optimized_physical_parameters` computes
  a single mask-based strip/restore descriptor threaded through all of them, so
  the free-parameter subset stays consistent regardless of which tier a fit
  dispatches to. Unknown names in either config key raise `ValueError` instead of
  silently narrowing or ignoring the request.
- **`tied_parameters` is now homodyne-invalid, not silently ignored.** Homodyne's
  `ParameterSpace`/`ParameterManager` have no tie/DOF-reduction mechanism (it is
  a `two_component`-only feature); configuring `tied_parameters` on
  `static_anisotropic`/`static_isotropic`/`laminar_flow` now raises `ValueError`
  at construction instead of silently producing two fully independent free
  parameters.
- **Heterodyne hybrid-streaming L2 now reports a real Hessian-based covariance**
  (`2*sigma^2*H^-1`, pseudo-inverse fallback on a singular Hessian) in place of
  the previous identity placeholder, mirroring the `laminar_flow` fix. Falls back
  to identity with `covariance_is_placeholder=True` only when the Hessian itself
  is non-finite.

### Fixed

- **`tied_parameters` alias/canonical name collisions are now rejected, not
  silently last-writer-wins**, across six config-load sites: the heterodyne
  `_apply_tied_parameters`, `_apply_initial_parameters`, `_apply_fixed_parameters`,
  `_apply_parameter_space_bounds`, and the homodyne `ParameterManager`'s bounds
  loader (#67). A config that names both a parameter's public alias (e.g.
  `v_beta`) and its internal canonical name (`beta`) as separate entries — for
  example `tied_parameters: {beta: v_beta, v_beta: D0_ref}` — previously
  collapsed to whichever entry a dict comprehension processed last, dropping one
  tie/bound/fixed-value with no diagnostic; all now raise `ValueError` on
  collision.
- **`active_parameters` typos now raise, matching `fixed_parameters`.**
  `resolve_optimized_physical_parameters` validated `fixed_parameters` names
  strictly but had no equivalent check for `active_parameters`, so a typo'd name
  silently narrowed the optimized set instead of erroring.
- **Heterodyne stratified-LS no longer attempts a doomed OOM allocation before
  falling back to the Layer-2 escape.** At `per_angle_mode=individual` with
  enough angles, the unconditional baseline `trf` solve's dense Jacobian sizing
  didn't account for the per-angle scaling tail, so a fit that was always going
  to need the hierarchical escape first tried (and failed) a multi-tens-of-GB
  allocation, wasting time and logging a misleading error. A pre-flight memory
  estimate now skips straight to the escape when the baseline solve would exceed
  budget and the escape can pick up from `p0_full` anyway.
- **Cached `t1`/`t2` arrays are now always recomputed from the current `dt`
  convention** instead of trusted verbatim from a `.npz` cache. The dt
  fingerprint check only fired when the cache carried dt metadata, so a
  cache built by an older xpcsjax version or an external converter (no dt key)
  passed silently and could desync frame-0 alignment against a cache-free run.
- **Scalar `per_angle_scaling` contrast/offset values are no longer dropped.**
  `get_initial_parameters()` only injected list-typed contrast/offset values,
  so a `two_component` config writing a bare scalar (documented as equivalent to
  a length-1 list) hit a `KeyError` in simulated-data plot rendering; scalars are
  now normalized to length-1 lists at both injection sites.
- **CI stability**: retry workflow hardened against starvation and
  false-positive cancellations, the runner-cancellation label mismatch (GitHub
  reports a runner-killed step as `failure`, not `cancelled`) is now handled,
  and the Ubuntu test matrix is sharded to cut per-shard wall time (#65, #66).

## [0.1.4] - 2026-08-13

### Fixed

- **`nlsq_result.npz` now carries `c2_exp`/`c2_fitted`/`residuals`/`t1`/`t2`/`phi_angles`/`wavevector_q`**.
  `save_results_npz` previously wrote only scalars, parameters, and
  covariance — the raw `(n_phi, n_t1, n_t2)` correlation surfaces (and the
  scattering wavevector needed to interpret them) a user needs for
  downstream re-analysis were only ever written to
  `plots/simulated_data/c2_fitted_data.npz`, and only when plotting was
  enabled. `service/persist.py::merge_fitted_c2()` now folds those
  already-computed arrays into the primary NPZ (best-effort, mtime-guarded
  against stale leftovers from a previous run in the same output directory)
  after plotting runs, for both the CLI and the GUI worker.
- **Whole-codebase module-review sweep: 18 findings fixed across 12 modules**
  (2 confirmed HIGH blockers + 16 advisory). Highlights: `ParameterSpace.from_config`
  crashed on a blank `parameter_space:` YAML section; public `xpcsjax.HeterodyneModel`
  resolved to the no-arg `PhysicsModelBase` adapter instead of the documented
  stateful model used by every production heterodyne path and the test suite.
  Advisory fixes span CLI persistence logging, config parameter-name aliasing,
  path-traversal-check de-duplication, an OOM-risk unbounded residuals scatter
  plot, and more.

## [0.1.3] - 2026-08-09

### Fixed

- **Deep-RCA whole-codebase debug audit: 21 confirmed bugs across 24 files** (#49, #50). Highlights:
  - `wrapper.py` / `strategies/stratified_ls.py`: a DOF-correction truthiness bug understated reduced-$\chi^2$/`quality_flag` when the `anti_degeneracy` config section is absent but averaged/individual scaling is still active.
  - `heterodyne_core.py`: the CMA-ES warm-start auto-skip compared raw (non noise-normalized) SSR/dof against a threshold meant for noise-normalized reduced $\chi^2$, silently defeating the global escape on genuinely poor basins; added a missing weights/data shape guard in `_fit_cmaes`.
  - `jax_backend.py`: `validate_backend()`'s self-test used `jax.grad` on a matrix-output function, so `gradient_support` was always `False` even on a working JAX install; fixed to use `jacobian`, resolving the downstream always-`ImportError` fallout in `model_mixins.py`.
  - `anti_degeneracy_controller.py`: Layer 2 (`HierarchicalOptimizer`) was constructed mode-blind, reporting active even in `averaged`/`constant` mode where its hard-coded individual-mode layout is unusable; construction is now gated on the resolved per-angle mode.
  - `hierarchical.py`: outer-loop convergence was gated on absolute parameter change instead of the already-computed relative change, making convergence unreachable for parameters spanning 6+ orders of magnitude.
  - `adaptive_regularization.py`: the CV-safe-divide fix sanitized the numerator as well as the denominator, silently changing the near-zero-mean fallback value and killing the L3 regularization penalty exactly where it is needed; corrected to sanitize only the denominator (applied to all three CV sites: `adaptive_regularization.py`, `heterodyne_core.py`, `heterodyne_stratified_ls.py`).
  - `strategies/sequential.py`: `combine_angle_results` used a scalar per-angle weight for `combined_params` while `combined_cov` used an unmasked per-parameter inverse-variance weight, an internally inconsistent pairing; both now share the same masked/floored weight matrix.
  - `data/performance_engine.py`: `MultiLevelCache` disk writes were not serialized per key (unsynchronized `"wb"` opens race); fixed via atomic temp-file + `os.replace`.
  - `data/memory_manager.py`: `cleanup_virtual_memory()` deleted files by a shared filename-prefix scan instead of tracking self-created files, risking deletion of another live instance's mmap-backed file.
  - `config.py`: `NLSQConfig.from_dict()` crashed (instead of degrading to default) on an explicit YAML `null` for non-`Optional` numeric fields.
  - GUI: `project_panel.py` / `result_presenter.py` rendered a NaN parameter with the same "—" sentinel as "run has no such parameter," silently defeating the comparison-view diff marker; both now render `"NaN"` distinctly.
  - Smaller fixes: an unguarded `end_frame` `KeyError` in `HeterodyneModel.from_config`, an unbounded `lscpu` subprocess call (hang risk) in `device/cpu.py`, an uncommented empty `except FileNotFoundError` flagged by CodeQL, and unlogged `fast_chi2_mode` subsampling in `out_of_core.py`.
  - Verification: ruff clean, mypy clean (0 issues, 198 files), full suite green (2628 passed, +6 new regression tests, 0 failures).

## [0.1.2] - 2026-08-06

### Added

- **Heterodyne `tied_parameters` equality constraints** (#27, #30). Support for
  user-configured parameter tying across components and angles in `two_component`
  fits. Tied parameters are constraint-reduced during NLSQ optimization and
  automatically expanded back to full 14-parameter blocks (`expand_reduced_result`)
  across all execution paths (`individual`, `averaged`, `constant`, `stratified-LS`,
  `hybrid-streaming`).
- **`XPCSDataLoader` context manager support** (#40). `load_xpcs_data` now uses
  a `with` statement for `XPCSDataLoader`, ensuring proper cleanup of underlying
  HDF5 handles, `PerformanceEngine`, and `AdvancedMemoryManager` background threads on completion or failure.

### Changed

- **`datashader` is now a required dependency**, enabling fast visualization paths by default without requiring the optional `[viz-fast]` extra.
- **Removed unused dependencies** (`scikit-learn`, `cloudpickle`, `interpax`) from `pyproject.toml` (#26).

### Fixed

- **Deep-RCA multi-agent correctness & stability audit across core modules**:
  - **`xpcsjax/optimization/` (29 confirmed bugs, #37, #38)**: Fixed residual calculations, per-angle $\chi^2$ decomposition, parameter scaling, and state synchronization across NLSQ strategies.
  - **`xpcsjax/data/` (12 confirmed bugs, #33, #35)**: Fixed an inert allocation guard on `.npz` cache reads (where `mmap_mode` rendered size checks ineffective), resolved a memory leak caused by a circular reference between `AdvancedMemoryManager` and `MemoryPressureMonitor`, and ensured background thread shutdown on exit.
  - **`xpcsjax/core/` (6 confirmed bugs, #34)**: Fixed edge-case numerical issues and parameter bound checks in model evaluators.
  - **Silent failure & code review sweep (13 confirmed bugs, #36, #39)**: Resolved silent exception swallowing, fallback bugs, and missing edge-case handling.
- **Silenced expected `RuntimeWarning`s in degenerate-input guard paths** when handling edge cases in statistical diagnostics and fitting guards.
- **CI / Mypy / CodeQL**: Fixed pre-existing mypy hard-gate failures and CodeQL security alerts (#352-#357, #359).
- **CMA-ES escape no longer reports `converged`/`good` on a refinement-only
  success** (#25). The `two_component` joint CMA-ES escape's "cmaes"
  kept-branch left its success flag at the default, so `solve_success` fell
  back to `global_escape is not None` — always `True` once the CMA-ES vector
  beat the warm start, even when the global search itself exhausted its
  restart budget (`reason=max_restarts`) and only a post-search NLSQ polish
  produced the improvement. This let a degenerate, physically-collapsed
  result print `status=converged`/`quality=good` with all-NaN
  uncertainties — observed on a real C045 `two_component` fit. The real
  `nlsq` backend's convergence-reason vocabulary (`"xtol"` only — the
  previous `CMAES_CONVERGED_REASONS` constant was pycma-era dead code that
  never matched anything) now gates the success flag.
- **Heterodyne multistart candidates were silently re-solving the same
  starting point** (#24). `heterodyne_multistart`'s worker only called
  `update_values()`, which never moves the frozen initial-values snapshot
  `fit_nlsq_multi_phi` actually reads from — so every LHS-sampled candidate,
  and the final best-start re-fit, started from the same config values,
  defeating multistart entirely. Added
  `ParameterManager.reseed_initial_values()` to move both the live values
  and the frozen snapshot together.
- **`adjust_covariance_for_transforms` call-site arg mismatch** (#22). The
  sequential per-angle result-assembly path in `wrapper.py` passed 4
  positional args to a 3-param function (an unused solver-space params
  array leaked in), which would raise `TypeError` on any `laminar_flow`
  sequential-per-angle fit with an active shear-parameter transform.
  Caught by mypy; no prior test exercised the branch, now covered.
- **Redundant `JAX_PLATFORMS` warning silenced** (#23). `device.cpu` warned
  unconditionally whenever the JAX backend was already live, even though
  xpcsjax's own `__init__.py` pre-sets `JAX_PLATFORMS=cpu` before any JAX
  import — the common case. Now only warns when the live backend actually
  diverges from `cpu`.
- **Heterodyne per-angle `contrast`/`offset` bounds override was silently
  ignored.** The `individual`/`averaged`/`constant` scaling-first bounds
  builders in `heterodyne_core.py`, `heterodyne_engine_route.py`, and
  `heterodyne_constant_mode.py` pulled `contrast`/`offset` bounds from the
  static `ParameterRegistry` defaults (`[0,1]`/[0.5,1.5]`) instead of the
  config-resolved `ParameterManager`, so a tightened
  `parameter_space.bounds` override for `contrast`/`offset` had no effect on
  the NLSQ solve even though physics params respected it correctly.
- **GUI design-critique findings addressed** (#11). Cancel now asks for
  confirmation and gets its own toolbar separator from Run (previously one
  misclick discarded a possibly multi-minute fit with no undo); every
  toolbar/File-menu action gained a keyboard shortcut and tooltip; Edit
  Config validates YAML syntax before writing instead of silently saving
  invalid YAML; the Comparison dock renders a real side-by-side table with a
  `≠` marker on disagreeing rows; the per-phi results grid gained a
  color-bar legend (values were previously auto-scaled per tile with no
  legend, so identical colors could mean different numbers on different
  tiles) and a "Jump to φ" navigator above 8 angles; log lines and the
  FIT FAILED header are now severity color-coded; a persistent status-bar
  label now names which dataset Run targets; and the previously-orphaned
  HDF5/C₂ inspector (`data_inspect.py`) is now reachable via a new
  "Inspect Data File…" File-menu action. Follow-up bugfixes from the same PR:
  `ComparisonView` no longer crashes on a `None` chi-squared value, the
  color-bar's colormap resolution is now guarded the same way the sibling
  plot-colormap call already was, and `DataInspectDialog` catches the wider
  `RuntimeError`/`KeyError`/`ValueError` surface a corrupted-but-valid HDF5
  file can raise (previously only `OSError` was caught).
- **PyInstaller spec was missing Pillow.** `pillow>=12.3.0` (matplotlib's
  transitive image-backend dependency) was absent from both
  `xpcsjax-gui.spec`'s `collect_all()` list and the freeze-safety test's
  covered-dependency allowlist, failing
  `test_pyinstaller_spec_covers_runtime_deps` on every push. Added a real
  `collect_all("PIL")` entry (Pillow ships a compiled extension and
  dynamically-loaded format plugins that static analysis misses) plus the
  `pillow` → `PIL` dist/import-name alias.
- **Whole-codebase debug audits: 168 confirmed bugs across all 12 modules**,
  fixed module-by-module via a discover → adversarially-verify workflow,
  each round closed with a 4-agent adversarial re-review of the fix diff
  itself (145 bugs, reconciled with the concurrent docs PR #18; then a
  further 23 confirmed across `cli`/`config`/`data`/`device`/`gui`/`io`/
  `optimization`/`runtime`/`service`/`utils`/`viz`). Representative fixes:
  per-angle plot filename collisions when two phi angles round to the same
  integer; inverted (empty) `parameter_space.bounds` YAML silently accepted
  instead of rejected; unlocked deque iteration racing a background memory
  monitor thread; integer-dtype `c2_exp` silently truncating the
  negative-correlation repair floor; OS-core-reservation math zeroing out at
  exact 16/32-core boundaries; malformed numeric YAML config values accepted
  without clear errors; `REQUIRED_DEPENDENCIES`/`PUBLIC_API_SYMBOLS` drift
  from `pyproject.toml`/`__all__`; `logging.configure()` crash when all
  level args are `None`. All prior audits' verified non-defects
  (heterodyne `final_cost` path-dependence, shear double-radians port,
  Gap A/L5 design choices, C044 degeneracy, pointwise-evaluator
  non-wiring, etc.) were re-confirmed correct and left untouched.

### Security

- **Bumped `cryptography` dependency to `50.0.0`** to resolve CVE-2026-69247.

### Documentation

- Added pre-release software disclaimer banner to `README.md`.
- Documented heterodyne `tied_parameters` configuration in user guide and theory docs (#30).
- Documented `XPCSDataLoader` context-manager usage (#40).
- Added `make test-viz` target to developer commands reference (#41).
- Added missing API pages for the `device`/`io`/`utils` modules — the
  `api/index.rst` toctree only covered 9 of the 12 top-level packages (#18).
  Added `tests/test_docs_structure.py`, a structural doc-coverage check
  requiring every top-level `xpcsjax` submodule to have a
  `docs/source/api/{name}.rst` page (page-existence only, not per-symbol
  coverage — see `docs/adr/0001-automated-structural-doc-coverage-check.md`).

### Testing / Internal

- **Graphify dead-code cleanup**: Removed 17 unused/dead-code symbols identified via graphify knowledge graph audit (#32).
- Consolidated `_decompose_chi2_per_angle` import path (#28).
- Closed test-coverage gaps flagged by the debug-audit reviews: `cmaes_sigma0`
  value-level checks at both joint-escape sites, the `_fit_cmaes` DOF clamp
  preventing negative `reduced_chi_squared` on tiny matrices, per-angle
  `phi_index` threading in `postfit.py`, `nlsq_plots.py` null-config-section
  guards, and the heterodyne streaming path's own `auto_tune_lambda` call
  site (#17, #21).
- CI: fixed Windows-only failures unrelated to the above code changes — bare
  `bash` on `windows-latest` now resolves to the WSL launcher stub instead of
  Git-for-Windows' `bash.exe` (added explicit resolution), and a `Path.stat`
  test monkeypatch was broadened/scoped to tolerate `follow_symlinks=` and
  avoid masking `get_safe_output_dir`'s unrelated `exists()` call.
- `.gitignore` cleanup and lockfile update (`uv.lock`).
- Hardened the PyInstaller spec's `collect_all()` drift-guard extraction
  against comment text — a `):`-shaped sequence or a quoted word inside any
  spec comment could corrupt the regex-extracted dependency list; extraction
  now strips comments first via a shared `_extract_collect_all_names()`
  helper.
- Added a repo-level write-time Python quality gate (#7).
- Bumped GitHub Actions to their Node 24 majors (`checkout` v5,
  `setup-python` v6, `upload/download-artifact` v5).

## [0.1.1] - 2026-06-26

Maintenance release. No user-facing behavioural or API changes; the fitting
results, public API, and config formats are identical to 0.1.0.

### Changed

- **System memory detection unified and simplified** in the optimization
  memory-strategy layer — a single detection path replaces the previous
  duplicated logic. Behaviour-preserving.
- Removed obsolete adapter metadata methods that were no longer referenced by
  any live path.

### Internal / CI

- Mirror pushes to the OSTI GitLab instance (`wchen/xpcsjax`) via a new
  `.github/workflows/mirror.yml`.
- Repinned `pypa/gh-action-pypi-publish` to a valid `v1.14.0` commit SHA in the
  release workflow.
- Rebuilt the graphify codebase knowledge graph and refreshed the README
  badges.

## [0.1.0] - 2026-06-22

### Added

- Initial consolidated release. xpcsjax v0.1 ports the homodyne and
  heterodyne NLSQ pipelines into a single JAX-native package.
- **Unified public API** — seven lazy-loaded symbols:
  `xpcsjax.data.xpcs_loader.load_xpcs_data`,
  `xpcsjax.optimization.nlsq.fit_nlsq`,
  `xpcsjax.config.ConfigManager`,
  `xpcsjax.core.HomodyneModel`,
  `xpcsjax.core.HeterodyneModel`,
  `xpcsjax.optimization.nlsq.results.OptimizationResult`,
  `xpcsjax.viz.nlsq_plots.generate_nlsq_plots`.
- **JAX-first with float64.** `JAX_ENABLE_X64=1` is set at package
  import time; parameters span 6+ orders of magnitude and float32 is
  unsafe.
- **Homodyne parity oracle.** Characterisation tests pin xpcsjax's
  homodyne output to upstream `homodyne` results at `rtol=1e-10`.
- **Heterodyne multi-angle.** Joint fitting across φ angles with
  χ²-exact residuals using the per-angle scaling layouts `constant` /
  `individual` / `auto` (`auto` resolves to `averaged` at `n_phi ≥ 3`,
  else `individual`); returns a single `OptimizationResult`.
- **NLSQ engine split.** xpcsjax owns strategy routing, the 5-layer
  anti-degeneracy controller, CMA-ES escape, LHS multistart,
  angle-stratified chunking, and shear weighting. NLSQ owns the
  `CurveFit` JIT cache and the trust-region solve.
- **Anti-degeneracy controller** with five composable layers: per-angle
  reparameterisation, hierarchical optimisation, adaptive
  cross-validation regularisation, gradient-collapse monitoring, and
  shear-sensitivity weighting.
- **Memory-aware strategy selection** via
  `xpcsjax.optimization.nlsq.select_nlsq_strategy` — picks between
  in-memory, hybrid-streaming, and out-of-core paths based on dataset
  size and available RAM. (Angle-stratified least squares is a separate
  ≥1M-point dispatch path.)
- **Visualization module** (`xpcsjax.viz`) — three public plot
  functions (`plot_nlsq_fit` 3-panel comparison, `plot_residual_map`
  4-panel diagnostic, `plot_simulated_data` single-panel theoretical
  heatmap), orchestrated by `generate_nlsq_plots`. Artifacts are
  serialised as LZMA-compressed NPZ + JSON under
  `output_dir/simulated_data/`. Optional Datashader fast path (5–10×
  per-call speedup; install via `pip install 'xpcsjax[viz-fast]'`)
  with transparent matplotlib fallback. Parallel multi-process
  rendering via `multiprocessing.Pool(spawn)`.
  `xpcsjax.viz.diagnostics.compute_diagonal_overlay_stats` extracts
  the t₁ = t₂ diagonal from experimental and fitted c² surfaces.
- **Desktop analysis workbench (GUI)** (`xpcsjax/gui/`). A PySide6 graphical
  front-end registered as the `xpcsjax-gui` / `xj-gui` console script
  (`xpcsjax.gui.app:main`; recognises `--help` / `--version`, forwards the rest
  to Qt). Config-first, toolbar-driven workflow (Create Config → Edit Config →
  Load Config → Run → Cancel → Export Figure, no tabs), an Inspector dock
  (params / uncertainties / diagnostics), a streaming Fitting-Process log
  dock, interactive PyQtGraph per-phi result/residual plots, and a
  datasets→runs project sidebar with side-by-side comparison. Sessions persist
  to `.xpcsproj` JSON (atomic writes, per-run output dirs).
  **Architectural invariant:** the GUI process never imports JAX — every fit
  runs in a separate `spawn` worker (`xpcsjax/gui/ipc/`) that lazily imports
  JAX + the service layer and streams structured `FitEvent`s back to the UI.
  Optional install: `pip install -e ".[gui]"`. PyInstaller freeze support via
  `.[packaging]` + `packaging/xpcsjax-gui.spec` (one-dir bundle,
  `multiprocessing.freeze_support()` first). Documentation:
  `docs/source/user_guide/gui.rst`.
- **Headless core-service layer** (`xpcsjax/service/`). The argparse-free,
  Qt-free orchestration seam shared by the CLI and the GUI worker:
  `service.config` (JAX-free `load_config` / `validate_config` / `available_modes`
  / `template_dict`), `service.events` (JAX-free streamed `FitEvent` schema),
  and the worker-side `service.data` / `service.fit` / `service.plots`, plus
  `service.persist` (result serialisation, relocated from `cli.result_saving`).
  Documentation: `docs/source/api/service.rst`.
- **Command-line interface** (`xpcsjax/cli/`). Console scripts registered in
  `pyproject.toml`, each with an `xj` short alias:
    - `xpcsjax` / `xj` — single flat flag-driven command for NLSQ fits;
      `--plot-experimental-data` / `--plot-simulated-data` switch to a
      standalone-plot path that skips optimisation.
    - `xpcsjax-config` / `xj-config` — generate, `--show-template`,
      `--validate`, or `--interactive`ly build a YAML config from the four
      mode templates.
    - `xpcsjax-config-xla` / `xj-config-xla` — inspect/print the CPU
      `XLA_FLAGS`.
    - `xpcsjax-validate` / `xj-validate` — validate the installation.
    - `xpcsjax-post-install` / `xpcsjax-cleanup` (+ `xj-` aliases) — install
      and remove shell completion + XLA activation scripts.
    - `xjexp` / `xjsim` — plotting shortcuts (experimental QC / simulated C₂),
      mirroring upstream heterodyne's `hexp` / `hsim`.
- **Runtime utilities** (`xpcsjax/runtime/`). System validator
  (`xpcsjax.runtime.utils.system_validator`) checking environment, dependency
  versions, JAX/float64 config, and template / public-API integrity
  (NLSQ-only — no Bayesian/MCMC probes), plus the bash/zsh/fish completion and
  XLA activation shell assets under `xpcsjax.runtime.shell`.
- Documentation: `docs/source/user_guide/cli.rst`, `docs/source/api/cli.rst`,
  and `docs/source/api/runtime.rst`.
- **Heterodyne joint global escapes (parity gap C closed).** The joint
  CMA-ES (`enable_cmaes=True`) and joint multistart (`multistart=True`)
  escapes in `heterodyne_core.py` are now real global escapes over the full
  `[physics | scaling]` vector — seed-pinned, keep-better vs. the plain NLSQ
  joint fit, and best-effort fall back on failure (reusing the shared
  `fit_with_cmaes` / `run_multistart_nlsq`). Escape results are tagged
  `nlsq_diagnostics["global_escape"]` and carry NaN covariance /
  uncertainties with `n_iterations=0` by construction. See
  `docs/source/theory/heterodyne_anti_degeneracy.rst`.
- **Dedicated heterodyne logging module**
  (`optimization/nlsq/heterodyne_logging.py`). The `two_component` core and
  stratified-LS paths now route progress/banner logging through it for
  structured-logging parity with the `laminar_flow` pipeline (resolves the
  multi-minute stratified-LS silence).

### Changed

- **Heterodyne streaming anti-degeneracy (parity gap D closed).** The
  `two_component` STREAMING tier no longer freezes the quantile-estimated
  per-angle scaling. It now optimizes the scaling tail (contrast + offset)
  and runs L1–L4, reaching mechanism parity with `laminar_flow` streaming.
  The scaling treatment is selected by `anti_degeneracy_config.per_angle_mode`,
  with `"auto"` as the default — including when `anti_degeneracy_config` is
  absent/`None` (no "freeze when unconfigured" special case). `"auto"`
  resolves to `auto_averaged` at `n_phi ≥ constant_scaling_threshold`
  (default 3), else `individual`; `per_angle_mode="constant"` is the explicit
  frozen-scaling opt-out. See
  `docs/source/theory/heterodyne_memory_strategy.rst`.
- **Symmetric anti-degeneracy diagnostics.** Both `laminar_flow` and
  `two_component` now emit the same top-level `nlsq_diagnostics` activation
  keys (`hierarchical_active`, `regularization_active`, `shear_weighting`,
  plus `gradient_monitor` when L4 ran) via the shared
  `assemble_anti_degeneracy_diagnostics` across every dataset-size path.
  `shear_weighting` is reported inactive for heterodyne by design (L5 is
  `laminar_flow`-only).

- **Deprecation — `analysis_mode` taxonomy.** The bare value
  `analysis_mode: static` is deprecated. It was ambiguous between
  `static_isotropic` (angle-collapsed) and `static_anisotropic`
  (angle-resolved) and silently collapsed downstream. The canonical set
  is now exactly four modes:
    - `static_isotropic`
    - `static_anisotropic`
    - `laminar_flow`
    - `two_component` (with `heterodyne` accepted as a case-insensitive
      synonym, normalised to `two_component` at config load time)

  `ConfigManager._normalize_analysis_mode` still accepts the legacy bare
  value but rewrites it to `static_anisotropic` (the drop-in replacement
  that preserves angle resolution) and emits a deprecation warning at
  config-load time. Migrate old configs to one of the canonical modes to
  silence the warning.

  See `docs/source/user_guide/analysis_modes.rst` for the full mode
  reference.

### Removed

- **Per-angle-mode unification (Phase 7) — the `fourier` per-angle scaling
  mode is gone.** Both `laminar_flow` and `two_component` now expose a single
  canonical vocabulary of three per-angle scaling layouts — `constant`,
  `averaged`, and `individual` (resolved from the `auto` default) — across all
  eight execution paths. The Fourier-reparameterized scaling tail and its
  `independent` sibling were deleted along with the
  `optimization/nlsq/fourier_reparam.py` module, the `FourierReparameterizer`
  class, the `fourier_order` / `fourier_auto_threshold` config keys (all four
  mode templates plus the homodyne and heterodyne config dataclasses), and the
  `fourier_basis_dim` diagnostics key. **Breaking:** configs that set
  `per_angle_mode: fourier` (or `independent`) are rejected at resolve time —
  use `individual` for free per-angle scaling or `averaged` for the
  angularly-averaged two-DOF tail. Verified no-worse-SSR on the laminar
  CMA-ES / stratified / streaming paths; the homodyne `rtol=1e-10` parity
  baselines were regenerated against explicit `individual`.
- **Dead-code cleanup** — removed code that was unreachable, superseded, or
  never wired into the NLSQ pipeline. No behavioural change; verified by the
  full suite (1253 passed, 8 skipped).
    - `xpcsjax/core/theory.py` — the unused `TheoryEngine` module, ported
      verbatim from homodyne but never imported by any live path.
    - Deprecated streaming shims in
      `optimization/nlsq/strategies/hybrid_streaming.py`
      (`fit_with_streaming_optimizer_deprecated`,
      `fit_with_streaming_optimizer_stratified_deprecated`) together with their
      only (also dead) caller `NLSQWrapper._fit_with_streaming_optimizer`.
    - `StratifiedResidualFunction._compute_chunk_residuals_raw` — raised
      `RuntimeError` on call; the live path is `_call_jax_vectorized`.
    - The duplicate `compute_g2_scaled_with_factors` in
      `core/physics_nlsq.py` (the live copy lives in `core/jax_backend.py`).
    - Unused symbols `NLSQCheckpointError`, `FallbackInfo`,
      `cli/xla_config.auto_configure`,
      `heterodyne_parameter_names.get_param_index`, and the module-level
      `heterodyne_parameter_space.clamp_to_open_interval`.
    - The dangling `xpcsjax.core.theory` Sphinx autodoc stub in
      `docs/source/api/modules.rst`.

### Fixed

- Removed dangling Sphinx autodoc reference to
  `xpcsjax.config.parameter_space.PriorDistribution` (the class was
  deleted during the Phase-7 CMC cleanup but the autodoc directive
  survived, producing a build warning).
- **Per-angle heterodyne CMA-ES escape now honours its config.** `_fit_cmaes`
  previously dropped `seed` (non-reproducible), `cmaes_warmstart_auto_skip` /
  `*_skip_threshold` (paid for a full global search even when the NLSQ
  warm-start was already good), and `cmaes_sigma0` (always used step `0.5`).
  Each angle is now seed-pinned (`_PER_ANGLE_CMAES_SEED + angle_idx`), skips
  below threshold, and threads `sigma0` through — and tags
  `metadata["global_escape"]` for diagnostics-contract symmetry with the
  joint escapes.
- **Heterodyne joint global escapes honour the resolved per-angle scaling
  mode** instead of forcing Fourier — `effective_mode` is resolved before the
  global-escape gate, so enabling CMA-ES / multistart never changes the
  scaling layout (`auto → averaged` at `n_phi ≥ 3`, else `individual`).
- **Flow-parameter units corrected** in `ParameterRegistry`: `gamma_dot` and
  `v_offset` were labelled `nm/frame`; the physically correct unit is `Å/s`
  (metadata-only — bounds, defaults, and log-space flags unchanged).
- **Noise-normalized stratified-LS reduced χ².**
  `build_hybrid_streaming_result` now computes `SSR / (σ²_noise · dof)` (the
  driver threads an estimated far-lag photon-noise variance via
  `info["sigma2_noise"]`) instead of raw `SSR/dof`, which collapsed to
  ~0.0024 on normalized C₂ data and mislabelled fits as "good".
- **Heterodyne visualization reads physics-first parameters.** Heterodyne
  `result.parameters` is laid out `[physics | contrast | offset]` (vs.
  homodyne's scaling-first layout); the viz unpackers and uncertainty
  slicing now use the heterodyne layout, fixing a C044 fitted-C₂ heatmap that
  rendered ~1e6.
- **Robustness guards** against non-finite values and a JIT-cache thread race
  on the heterodyne paths.

### Documentation

- Added `make docs` target that runs Sphinx with `-W` (warnings treated
  as errors), so dangling autodoc references and broken cross-refs fail
  the build instead of accumulating silently. Wired into `make ci-full`.
- Reconciled the `api/modules.rst` "All modules" page with the live package:
  removed the deleted `xpcsjax.core.heterodyne_physics`, registered the
  modules added since (CLI, runtime, viz, and the new heterodyne NLSQ
  modules), and fixed stale `:mod:`/`:func:`/`:class:` cross-references in the
  user-guide and theory pages so the strict `-W` build is warning-clean.

### Out of scope (v0.x series)

- Bayesian sampling — NumPyro, BlackJAX, NUTS, HMC, CMC (Consensus
  Monte Carlo), ArviZ, parallel tempering. Use the upstream `homodyne`
  / `heterodyne` packages for Bayesian XPCS analysis.
- GPU support. v0.1 sets `NLSQ_SKIP_GPU_CHECK=1` and runs CPU-only;
  GPU paths are planned for v0.2+.

[Unreleased]: https://github.com/imewei/xpcsjax/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/imewei/xpcsjax/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/imewei/xpcsjax/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/imewei/xpcsjax/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/imewei/xpcsjax/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/imewei/xpcsjax/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/imewei/xpcsjax/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/imewei/xpcsjax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imewei/xpcsjax/releases/tag/v0.1.0
