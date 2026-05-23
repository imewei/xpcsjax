# Changelog

All notable changes to xpcsjax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the authoritative changelog. The Sphinx page at
`docs/source/changelog.rst` surfaces the same milestones for users browsing
the rendered documentation.

## [Unreleased]

### Changed

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
