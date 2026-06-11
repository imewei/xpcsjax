Changelog
=========

The authoritative changelog lives in the top-level ``CHANGELOG.md`` of the
repository. This page surfaces the major user-facing milestones for the
current release line.

Unreleased
----------

**Heterodyne config bounds overrides (``parameter_space.bounds``).** The
``two_component`` config loader now honors list-format
``parameter_space.bounds`` overrides — ``ParameterSpace.from_config`` applies
them through its ``_apply_parameter_space_bounds`` helper, reaching parity with
homodyne's ``ParameterManager._load_config_bounds``.
Previously the heterodyne path silently ignored config bounds and fell back to
registry defaults, so a narrow default window could clamp a valid warm-start
(e.g. the C044 creep-flow fit needs ``v_beta ≈ -0.43``, outside the conservative
``[0, 2]`` registry default). Template/alias names (``v_beta``, ``phi0_het``)
are translated to their canonical kernel entries (``beta``, ``phi0``) so the
override lands on the right registry parameter. The registry default for
``v_beta`` stays ``[0, 2]`` by design — widening it destabilised the non-convex
engine-route single-angle solve, so configs needing negative exponents must opt
in explicitly. See :ref:`Overriding bounds (parameter_space.bounds)
<parameter_space_bounds>`.

**Heterodyne streaming anti-degeneracy (parity gap D closed).** The
``two_component`` STREAMING tier previously froze the quantile-estimated
per-angle scaling and ran no anti-degeneracy layers. It now **optimizes** the
scaling tail (contrast + offset) and runs **L1–L4**, reaching mechanism parity
with ``laminar_flow`` streaming. The scaling treatment is selected by
``anti_degeneracy_config.per_angle_mode``, with ``"auto"`` as the default —
including when ``anti_degeneracy_config`` is absent or ``None`` (no
"freeze when unconfigured" special case). ``"auto"`` resolves to
``auto_averaged`` at ``n_phi ≥ constant_scaling_threshold`` (default 3), else
``individual``; ``per_angle_mode="constant"`` is the explicit frozen-scaling
opt-out. See :ref:`Streaming anti-degeneracy <streaming_antidegeneracy>`.

**Heterodyne joint global escapes (parity gap C closed).** The heterodyne
joint CMA-ES (``enable_cmaes=True``) and joint multistart (``multistart=True``)
escapes are now **real global escapes** over the full ``[physics | scaling]``
vector — seed-pinned, **keep-better** vs. the plain NLSQ joint fit, and
**best-effort fall back** to the plain joint fit on failure (reusing the shared
``fit_with_cmaes`` / ``run_multistart_nlsq``). An escape result is tagged
``nlsq_diagnostics["global_escape"]`` and, by construction, carries NaN
covariance / uncertainties and ``n_iterations=0``. See
:doc:`/theory/heterodyne_anti_degeneracy`.

**Symmetric anti-degeneracy diagnostics.** Both ``laminar_flow`` and
``two_component`` now emit the same top-level ``nlsq_diagnostics`` activation
keys (``hierarchical_active``, ``regularization_active``, ``shear_weighting``,
plus ``gradient_monitor`` when L4 ran) via the shared
``assemble_anti_degeneracy_diagnostics`` across every dataset-size path, with
honest per-path values. ``shear_weighting`` is reported as inactive for
heterodyne by design (L5 is ``laminar_flow``-only — heterodyne's velocity/flow
term is structurally different from a shear rate).

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

**Internal dead-code cleanup.** Removed code that was unreachable, superseded,
or never wired into the NLSQ pipeline — the unused ``xpcsjax.core.theory``
module, the deprecated streaming shims and their dead wrapper caller, a dead
``_compute_chunk_residuals_raw`` path, a duplicate
``compute_g2_scaled_with_factors`` (the live copy stays in
``xpcsjax.core.jax_backend``), and a handful of unused symbols. No behavioural
change; the full test suite passes. See ``CHANGELOG.md`` for the itemised list.

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
