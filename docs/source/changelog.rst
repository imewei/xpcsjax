Changelog
=========

The authoritative changelog lives in the top-level ``CHANGELOG.md`` of the
repository. This page surfaces the major user-facing milestones for the
current release line.

Unreleased
----------

v0.1.6 — heterodyne naming fix, stratified-LS/hybrid-streaming correctness fixes, dead-code cleanup
-----------------------------------------------------------------------------------------------------

*Released 2026-08-30.*

* **Negative-dominant ``D_offset`` now detected in physics validators** — the
  ``D_offset``/``D0`` overfitting checks compared ``ratio > 0.5`` only,
  missing the symmetric case where a large-magnitude negative offset
  dominates ``D0``; both checks now compare ``abs(ratio) > 0.5``.
* mypy hard-gate failures on main resolved with explicit type annotations
  and a new ``AngleGroup`` ``TypedDict`` replacing a loose ``dict[str, Any]``.
* ``nlsq`` bumped to 0.7.4 and ``evosax`` bumped to 0.3.1 (``uv.lock`` and the
  conda recipe).
* **Duplicate ``HeterodyneModel`` class name resolved.** The
  ``PhysicsModelBase`` adapter in ``core/heterodyne_model.py`` shared its name
  with the public stateful ``core/heterodyne_model_stateful.py`` class, so the
  two import paths silently returned different classes. The adapter is
  renamed to ``HeterodynePhysicsAdapter``; the public
  ``xpcsjax.core.HeterodyneModel`` export is unaffected.
* **Stratified-LS (≥1M point) diagonal-exclusion mask fixed** — it compared
  indices into two independently-built ``np.unique()`` arrays instead of the
  actual ``t1``/``t2`` values, wrongly zeroing real off-diagonal points and
  keeping true diagonal ones.
* **Hybrid-streaming ``per_angle_mode="constant"`` fallback index
  misalignment fixed** — the quantile-estimation-failure fallback resolved
  L3's regularization groups and L4's gradient-collapse-monitor indices from
  the literal mode string instead of the real 2-param scaling-head layout it
  actually falls back to.
* **``coerce_finite_float`` no longer silently coerces a stray YAML boolean**
  (e.g. a typo'd ``max: true``) to ``1.0``/``0.0``.
* **The degenerate 1-frame Siegert-ceiling check no longer reads the excluded
  ``tau=0`` diagonal spike** in its fallback branch.
* 19 correctness gaps closed in the NLSQ fitting workflow and 6 more across
  config/core/viz, found via independent step-by-step audits that traced the
  NLSQ workflow and ``docs/diagrams/architecture.md`` against the real code.
* ~7800 lines of confirmed-dead code removed repo-wide.
* Docs: config examples across the documentation now match the shipped YAML
  templates exactly (bounds format, key nesting, heterodyne parameter
  layout); theory prose terminology and notation cleaned up.

v0.1.5 — fixed/active/tied parameter hardening and heterodyne covariance fix
------------------------------------------------------------------------------

*Released 2026-08-25.*

* **``fixed_parameters``/``active_parameters`` now honored across every NLSQ
  execution tier** — sequential, out-of-core, stratified-LS, hybrid-streaming,
  and the CMA-ES escape all thread the same mask-based strip/restore
  descriptor, and unknown names in either key now raise instead of silently
  narrowing or ignoring the request.
* **``tied_parameters`` alias/canonical name collisions now raise
  ``ValueError``** instead of silently last-writer-wins across six config-load
  sites, and ``tied_parameters`` on a non-``two_component`` mode (which has no
  tie mechanism) is now rejected instead of silently accepted and ignored.
* **Heterodyne hybrid-streaming L2 reports a real Hessian-based covariance**
  in place of the previous identity placeholder.
* Fixed a stale-cache ``t1``/``t2`` desync risk, a dropped-scalar
  ``per_angle_scaling`` bug, and a doomed pre-escape OOM allocation attempt on
  the heterodyne stratified-LS path.

v0.1.4 — NPZ result-file completeness and module-review sweep
---------------------------------------------------------------

*Released 2026-08-13.*

* **``nlsq_result.npz`` now carries ``c2_exp``/``c2_fitted``/``residuals``**
  (plus ``t1``/``t2``/``phi_angles``/``wavevector_q``): the primary result
  file previously held only scalars, parameters, and covariance, so the raw
  correlation surfaces (and the scattering wavevector needed to interpret
  them) were only reachable via the plots directory, and only when plotting
  ran. ``service.persist.merge_fitted_c2()`` folds them in (best-effort,
  mtime-guarded) after plotting, for both the CLI and the GUI worker.
* **Whole-codebase module-review sweep: 18 findings fixed across 12 modules**
  (2 confirmed HIGH blockers + 16 advisory), including a crash on a blank
  ``parameter_space:`` YAML section and public ``xpcsjax.HeterodyneModel``
  resolving to the wrong adapter class.

v0.1.3 — deep-RCA whole-codebase debug audit
---------------------------------------------

*Released 2026-08-09.*

* **21 confirmed bugs fixed across 24 files** (#49, #50), adversarially
  verified via a 12-shard find-then-verify audit. Highlights: a DOF-correction
  truthiness bug that understated reduced-:math:`\chi^2`/``quality_flag`` when
  the ``anti_degeneracy`` config section is absent; a CMA-ES warm-start
  auto-skip that compared raw (non noise-normalized) SSR/dof against a
  noise-normalized threshold, silently defeating the global escape on poor
  basins; ``validate_backend()``'s self-test using ``jax.grad`` on a
  matrix-output function so ``gradient_support`` was always ``False``; Layer 2
  (``HierarchicalOptimizer``) constructed mode-blind in ``averaged``/
  ``constant`` per-angle modes; a CV-safe-divide fix that over-sanitized the
  numerator and silently killed the L3 regularization penalty at the
  near-zero-mean case it exists to catch (three call sites); and an
  inconsistent weighting pairing between ``combined_params`` and
  ``combined_cov`` in ``combine_angle_results``.
* Verification: ruff clean, mypy clean (0 issues, 198 files), full suite
  green (2628 passed, +6 new regression tests, 0 failures).

v0.1.2 — tied parameters, deep-RCA audits, and GUI/packaging fixes
--------------------------------------------------------------------

*Released 2026-08-06.*

* **Heterodyne ``tied_parameters`` equality constraints** (#27, #30):
  user-configured parameter tying across components and angles in
  ``two_component`` fits, constraint-reduced during optimization and
  expanded back to full 14-parameter blocks across every execution path.
* **``XPCSDataLoader`` context-manager support** (#40): ``load_xpcs_data``
  now uses a ``with`` statement, ensuring HDF5 handles and background
  threads are cleaned up on completion or failure.
* **Deep-RCA multi-agent audits: 197 confirmed bugs fixed** across all 12
  top-level modules (``optimization`` 29, ``data`` 12, ``core`` 6, plus a
  168-bug whole-codebase sweep), each round closed with an independent
  4-agent adversarial re-review of the fix diff itself. Representative
  fixes: a CMA-ES escape reporting ``converged``/``good`` on a
  refinement-only success with all-NaN uncertainties (#25); heterodyne
  multistart silently re-solving the same starting point instead of
  moving the frozen initial-values snapshot (#24); heterodyne per-angle
  ``contrast``/``offset`` bounds overrides being silently ignored; a
  mypy-caught ``adjust_covariance_for_transforms`` arg mismatch (#22); a
  redundant ``JAX_PLATFORMS`` warning (#23).
* **GUI design-critique findings addressed** (#11): a Cancel confirmation
  dialog, keyboard shortcuts/tooltips on every toolbar/File-menu action,
  YAML-validating Edit Config, a real side-by-side Comparison table, a
  color-bar legend + "Jump to φ" navigator on the per-phi results grid,
  severity-colored log/failure text, a persistent Run-target status label,
  and a new "Inspect Data File…" action wiring the previously-orphaned
  HDF5/C₂ inspector into the UI — plus follow-up crash/exception-handling
  fixes on the same surfaces.
* Packaging: the PyInstaller spec was missing a real ``collect_all("PIL")``
  entry for Pillow (matplotlib's transitive image-backend dependency),
  failing the freeze-safety test on every push.
* Security: bumped ``cryptography`` to ``50.0.0`` to resolve CVE-2026-69247.
* Dependencies: ``datashader`` is now required (no more ``[viz-fast]``
  extra needed for the fast visualization path); removed unused
  ``scikit-learn``, ``cloudpickle``, ``interpax``.
* Documentation: added missing API pages for the ``device``/``io``/``utils``
  modules plus a structural doc-coverage test requiring every top-level
  submodule to have a docs page (#18); documented ``tied_parameters`` and
  the ``XPCSDataLoader`` context manager.
* Internal / CI: hardened the PyInstaller spec's drift-guard extraction
  against comment text, added a repo-level write-time Python quality gate,
  bumped GitHub Actions to their Node 24 majors, and rebuilt the knowledge
  graph.

See ``CHANGELOG.md`` for the itemised list.

v0.1.1 — maintenance release
----------------------------

*Released 2026-06-26.*

A maintenance release with no user-facing behavioural or API changes — fitting
results, the public API, and config formats are identical to v0.1.0.

* **System memory detection** in the optimization memory-strategy layer was
  unified and simplified into a single detection path (behaviour-preserving),
  and obsolete adapter metadata methods were removed.
* Internal / CI: GitHub→GitLab mirror workflow (``wchen/xpcsjax``), repinned
  ``pypa/gh-action-pypi-publish`` to a valid ``v1.14.0`` SHA, and refreshed the
  knowledge graph and README badges.

See ``CHANGELOG.md`` for the itemised list.

v0.1.0 — initial consolidated release
-------------------------------------

*Released 2026-06-22.*

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
* **Heterodyne multi-angle.** Joint per-angle-reparameterised fitting across
  φ angles with χ²-exact residuals; returns a single ``OptimizationResult`` with
  per-angle detail under ``nlsq_diagnostics``.
* **NLSQ engine split.** xpcsjax owns strategy routing, the 5-layer
  anti-degeneracy controller, CMA-ES escape, LHS multistart, angle-stratified
  chunking, and shear weighting. NLSQ owns the ``CurveFit`` JIT cache and
  the trust-region solve.
* **Anti-degeneracy controller** with five composable layers: per-angle
  reparameterisation, hierarchical optimisation, adaptive
  cross-validation regularisation, gradient-collapse monitoring, and
  shear-sensitivity weighting.
* **Memory-aware strategy selection** via
  :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy` — picks between
  in-memory, hybrid-streaming, and out-of-core paths based on dataset
  size and available RAM. (Angle-stratified least squares is a separate
  :math:`\geq` 1M-point dispatch path.)
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

This release also includes:

**Desktop analysis workbench (GUI).** A PySide6 graphical front-end launched
with ``xpcsjax-gui`` (alias ``xj-gui``): a config-first, toolbar-driven
workflow (Create Config → Edit Config → Load Config → Run → Cancel → Export
Figure, no tabs), an Inspector dock, a streaming Fitting-Process log, interactive
PyQtGraph per-phi result/residual plots, and a datasets→runs project sidebar
that persists to ``.xpcsproj`` files. The GUI process never imports JAX —
every fit runs in a separate ``spawn`` worker that streams structured progress
events back to the UI. Optional install: ``uv pip install -e ".[gui]"``.
See :doc:`/user_guide/gui`.

**Headless core-service layer (``xpcsjax.service``).** The argparse-free,
Qt-free orchestration seam shared by the CLI and the GUI worker — config
loading / validation / templates (JAX-free), the streamed fit-event schema
(JAX-free), and the worker-side data / fit / plot / persistence services. See
:doc:`/api/service`.

**Command-line interface.** xpcsjax ships console scripts (with ``xj``
short aliases): ``xpcsjax`` runs flag-driven NLSQ fits and standalone
QC/simulation plots; ``xpcsjax-config`` generates, prints, and validates
configs from the four mode templates; ``xpcsjax-validate`` checks the
installation; ``xjexp`` / ``xjsim`` are plotting shortcuts; and
``xpcsjax-post-install`` / ``xpcsjax-cleanup`` manage shell completion and XLA
activation scripts. See :doc:`/user_guide/cli`, :doc:`/api/cli`, and
:doc:`/api/runtime`.

**Runtime utilities.** New :mod:`xpcsjax.runtime` package providing system
validation (CPU, RAM, JAX, dependency, template/public-API integrity checks —
NLSQ-only, no Bayesian probes) and the bash/zsh completion (fish is a non-fatal
no-op) and bash/zsh/fish XLA activation assets.

**Heterodyne config bounds overrides (``parameter_space.bounds``).** The
``two_component`` config loader honors list-format
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
``two_component`` emit the same top-level ``nlsq_diagnostics`` activation
keys (``hierarchical_active``, ``regularization_active``, ``shear_weighting``,
plus ``gradient_monitor`` when L4 ran) via the shared
``assemble_anti_degeneracy_diagnostics`` across every dataset-size path, with
honest per-path values. ``shear_weighting`` is reported as inactive for
heterodyne by design (L5 is ``laminar_flow``-only — heterodyne's velocity/flow
term is structurally different from a shear rate).

**Deprecation — ``analysis_mode`` taxonomy.** The bare value
``analysis_mode: static`` is deprecated. It was ambiguous
between ``static_isotropic`` (angle-collapsed) and ``static_anisotropic``
(angle-resolved) and silently collapsed downstream. The canonical set
is now exactly four modes:

* ``static_isotropic``
* ``static_anisotropic``
* ``laminar_flow``
* ``two_component`` (with ``heterodyne`` accepted as a case-insensitive
  synonym, normalised to ``two_component`` at config load time)

Configs using ``analysis_mode: static`` are still accepted: the loader
rewrites the value to ``static_anisotropic`` (preserves angular
resolution) and emits a deprecation warning. Migrate to one of the
canonical modes to silence it. See :doc:`/user_guide/analysis_modes`
for the full description of each mode and the data-preparation
distinction.

**Internal dead-code cleanup.** Removed code that was unreachable, superseded,
or never wired into the NLSQ pipeline — the unused ``xpcsjax.core.theory``
module, the deprecated streaming shims and their dead wrapper caller, a dead
``_compute_chunk_residuals_raw`` path, a duplicate
``compute_g2_scaled_with_factors`` (the live copy stays in
``xpcsjax.core.jax_backend``), and a handful of unused symbols. No behavioural
change; the full test suite passes. See ``CHANGELOG.md`` for the itemised list.

Out of scope for v0.1 (and the v0.x series):

* Bayesian sampling — NumPyro, BlackJAX, NUTS, HMC, CMC (Consensus Monte
  Carlo), ArviZ, parallel tempering. Use the upstream ``homodyne`` /
  ``heterodyne`` packages for Bayesian XPCS analysis.

* GPU support. v0.1 sets ``NLSQ_SKIP_GPU_CHECK=1`` and runs CPU-only;
  GPU paths are planned for v0.2+.
