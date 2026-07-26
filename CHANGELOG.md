# Changelog

All notable changes to xpcsjax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the authoritative changelog. The Sphinx page at
`docs/source/changelog.rst` surfaces the same milestones for users browsing
the rendered documentation.

## [Unreleased]

## [0.1.2] - 2026-07-25

### Fixed

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

### Internal / CI

- Hardened the PyInstaller spec's `collect_all()` drift-guard extraction
  against comment text — a `):`-shaped sequence or a quoted word inside any
  spec comment could corrupt the regex-extracted dependency list; extraction
  now strips comments first via a shared `_extract_collect_all_names()`
  helper.
- Added a repo-level write-time Python quality gate (#7).
- Bumped GitHub Actions to their Node 24 majors (`checkout` v5,
  `setup-python` v6, `upload/download-artifact` v5).
- Rebuilt the graphify codebase knowledge graph.

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

[Unreleased]: https://github.com/imewei/xpcsjax/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/imewei/xpcsjax/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/imewei/xpcsjax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imewei/xpcsjax/releases/tag/v0.1.0
