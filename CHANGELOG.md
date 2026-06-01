# Changelog

All notable changes to xpcsjax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the authoritative changelog. The Sphinx page at
`docs/source/changelog.rst` surfaces the same milestones for users browsing
the rendered documentation.

## [Unreleased]

### Added

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

- **Breaking — `analysis_mode` taxonomy.** The bare value
  `analysis_mode: static` is no longer accepted. It was ambiguous between
  `static_isotropic` (angle-collapsed) and `static_anisotropic`
  (angle-resolved) and silently collapsed downstream. The canonical set
  is now exactly four modes:
    - `static_isotropic`
    - `static_anisotropic`
    - `laminar_flow`
    - `two_component` (with `heterodyne` accepted as a case-insensitive
      synonym, normalised to `two_component` at config load time)

  `ConfigManager._normalize_analysis_mode` now raises `ValueError` at
  config-load time when it sees the legacy bare value, with an explicit
  migration hint. The error message names `static_anisotropic` as the
  recommended drop-in replacement (preserves angle resolution).

  See `docs/source/user_guide/analysis_modes.rst` for the full mode
  reference and `docs/MIGRATION.md` for the migration table.

### Fixed

- Removed dangling Sphinx autodoc reference to
  `xpcsjax.config.parameter_space.PriorDistribution` (the class was
  deleted during the Phase-7 CMC cleanup but the autodoc directive
  survived, producing a build warning).

### Documentation

- Added `make docs` target that runs Sphinx with `-W` (warnings treated
  as errors), so dangling autodoc references and broken cross-refs fail
  the build instead of accumulating silently. Wired into `make ci-full`.

## [0.1.0]

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
- **Heterodyne multi-angle.** Joint Fourier-reparameterised fitting
  across φ angles with χ²-exact residuals; returns one `NLSQResult` per
  angle.
- **NLSQ engine split.** xpcsjax owns strategy routing, the 5-layer
  anti-degeneracy controller, CMA-ES escape, LHS multistart,
  angle-stratified chunking, and shear weighting. NLSQ owns the
  `CurveFit` JIT cache and the trust-region solve.
- **Anti-degeneracy controller** with five composable layers: Fourier /
  constant reparameterisation, hierarchical optimisation, adaptive
  cross-validation regularisation, gradient-collapse monitoring, and
  shear-sensitivity weighting.
- **Memory-aware strategy selection** via
  `xpcsjax.optimization.nlsq.select_nlsq_strategy` — picks between
  in-memory, stratified-least-squares, hybrid-streaming, and
  out-of-core paths based on dataset size and available RAM.
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

### Out of scope (v0.x series)

- Bayesian sampling — NumPyro, BlackJAX, NUTS, HMC, CMC (Consensus
  Monte Carlo), ArviZ, parallel tempering. Use the upstream `homodyne`
  / `heterodyne` packages for Bayesian XPCS analysis.
- GPU support. v0.1 sets `NLSQ_SKIP_GPU_CHECK=1` and runs CPU-only;
  GPU paths are planned for v0.2+.
