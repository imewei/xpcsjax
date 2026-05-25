Changelog
=========

The authoritative changelog lives in the top-level ``CHANGELOG.md`` of the
repository. This page surfaces the major user-facing milestones for the
current release line.

Unreleased
----------

**Command-line interface.** xpcsjax now ships console scripts (with ``xj``
short aliases): ``xpcsjax`` runs flag-driven NLSQ fits and standalone
QC/simulation plots; ``xpcsjax-config`` generates, prints, and validates
configs from the four mode templates; ``xpcsjax-validate`` checks the
installation; ``xjexp`` / ``xjsim`` are plotting shortcuts; and
``xpcsjax-post-install`` / ``xpcsjax-cleanup`` manage shell completion and XLA
activation scripts. See :doc:`/user_guide/cli`, :doc:`/api/cli`, and
:doc:`/api/runtime`.

**Runtime utilities.** New :mod:`xpcsjax.runtime` package providing system
validation (CPU, RAM, JAX, dependency, template/public-API integrity checks —
NLSQ-only, no Bayesian probes) and the bash/zsh/fish completion and XLA
activation assets.

**Breaking change — ``analysis_mode`` taxonomy.** The bare value
``analysis_mode: static`` is no longer accepted. It was ambiguous
between ``static_isotropic`` (angle-collapsed) and ``static_anisotropic``
(angle-resolved) and silently collapsed downstream. The canonical set
is now exactly four modes:

* ``static_isotropic``
* ``static_anisotropic``
* ``laminar_flow``
* ``two_component`` (with ``heterodyne`` accepted as a case-insensitive
  synonym, normalised to ``two_component`` at config load time)

Old configs using ``analysis_mode: static`` must be migrated. The
recommended drop-in default is ``static_anisotropic`` (preserves
angular resolution). See :doc:`/user_guide/analysis_modes` for the
full description of each mode and the data-preparation distinction.

v0.1.0 — initial consolidated release
-------------------------------------

xpcsjax v0.1 ports the homodyne and heterodyne NLSQ pipelines into a single
JAX-native package. Highlights:

* **Unified public API** — seven lazy-loaded symbols (:func:`xpcsjax.data.xpcs_loader.load_xpcs_data`,
  :func:`xpcsjax.optimization.nlsq.fit_nlsq`, :class:`xpcsjax.config.ConfigManager`,
  :class:`xpcsjax.core.HomodyneModel`, :class:`xpcsjax.core.HeterodyneModel`,
  :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`,
  :func:`xpcsjax.viz.nlsq_plots.generate_nlsq_plots`).
* **JAX-first with float64.** ``JAX_ENABLE_X64=1`` is set at package import
  time; parameters span 6+ orders of magnitude and float32 is unsafe.
* **Homodyne parity oracle.** Characterisation tests pin xpcsjax's homodyne
  output to upstream ``homodyne`` results at ``rtol=1e-10``.
* **Heterodyne multi-angle.** Joint Fourier-reparameterised fitting across
  φ angles with χ²-exact residuals; returns one ``NLSQResult`` per angle.
* **NLSQ engine split.** xpcsjax owns strategy routing, the 5-layer
  anti-degeneracy controller, CMA-ES escape, LHS multistart, angle-stratified
  chunking, and shear weighting. NLSQ owns the ``CurveFit`` JIT cache and
  the trust-region solve.
* **Anti-degeneracy controller** with five composable layers: Fourier /
  constant reparameterisation, hierarchical optimisation, adaptive
  cross-validation regularisation, gradient-collapse monitoring, and
  shear-sensitivity weighting.
* **Memory-aware strategy selection** via
  :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy` — picks between
  in-memory, stratified-least-squares, hybrid-streaming, and out-of-core
  paths based on dataset size and available RAM.
* **Visualization module** (``xpcsjax.viz``) — three public plot functions
  (:func:`~xpcsjax.viz.nlsq_plots.plot_nlsq_fit` 3-panel comparison,
  :func:`~xpcsjax.viz.nlsq_plots.plot_residual_map` 4-panel diagnostic,
  :func:`~xpcsjax.viz.nlsq_plots.plot_simulated_data` single-panel theoretical
  heatmap), orchestrated by :func:`~xpcsjax.viz.nlsq_plots.generate_nlsq_plots`.
  Artifacts are serialized as LZMA-compressed NPZ + JSON under
  ``output_dir/simulated_data/``. Optional Datashader fast path (5–10× per-call
  speedup; install via ``pip install 'xpcsjax[viz-fast]'``) with transparent
  matplotlib fallback. Parallel multi-process rendering via
  ``multiprocessing.Pool(spawn)``. Diagnostic helper
  :func:`~xpcsjax.viz.diagnostics.compute_diagonal_overlay_stats` extracts the
  t₁ = t₂ diagonal from experimental and fitted c² surfaces.

Out of scope for v0.1 (and the v0.x series):

* Bayesian sampling — NumPyro, BlackJAX, NUTS, HMC, CMC (Consensus Monte
  Carlo), ArviZ, parallel tempering. Use the upstream ``homodyne`` /
  ``heterodyne`` packages for Bayesian XPCS analysis.

* GPU support. v0.1 sets ``NLSQ_SKIP_GPU_CHECK=1`` and runs CPU-only;
  GPU paths are planned for v0.2+.
