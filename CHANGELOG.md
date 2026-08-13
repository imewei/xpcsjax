# Changelog

All notable changes to xpcsjax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the authoritative changelog. The Sphinx page at
`docs/source/changelog.rst` surfaces the same milestones for users browsing
the rendered documentation.

## [Unreleased]

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

[Unreleased]: https://github.com/imewei/xpcsjax/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/imewei/xpcsjax/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/imewei/xpcsjax/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/imewei/xpcsjax/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/imewei/xpcsjax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imewei/xpcsjax/releases/tag/v0.1.0
