.. _heterodyne_memory_strategy:

Heterodyne Memory Strategy and Angle Stratification
=====================================================

Starting with the Task-14 branch, the heterodyne (``two_component``) NLSQ
path mirrors homodyne's memory-aware angle-stratification mechanism.  This
page documents the decision flow, the activation thresholds, the intentional
deviation from homodyne's 100k–1 M regime, the default-on shuffle, and the
configuration knobs.

Overview
--------

xpcsjax routes every NLSQ call through ``select_nlsq_strategy``, which
classifies the dataset into one of three memory tiers (``STANDARD``,
``LARGE``, ``STREAMING``).  For heterodyne the dispatch in
``_fit_nlsq_heterodyne`` (``xpcsjax/optimization/nlsq/__init__.py``) is:

1. **CMA-ES escape** — highest precedence; triggered when the anti-degeneracy
   controller raises the CMA-ES flag.
2. **Multi-start** — LHS multistart when ``config.multistart.enabled``.
3. **Hybrid streaming** — when ``tier ∈ {LARGE, STREAMING}`` and
   ``optimization.hybrid_streaming.enable = true``; this path is inherently
   stratified by angle chunk.  The streaming path **optimizes** the per-angle
   scaling tail (contrast + offset) via the ``per_angle_mode`` dispatch
   (``anti_degeneracy_config.per_angle_mode``) and runs L1–L4 anti-degeneracy
   layers.  See :ref:`Streaming anti-degeneracy <streaming_antidegeneracy>`
   below.
4. **Stratified least-squares** — when the tier is ``STANDARD`` and
   ``should_use_stratification(n_points, n_phi, per_angle=True, imbalance)``
   returns ``True`` **and** ``n_points ≥ 1 000 000``; uses
   ``fit_heterodyne_stratified_least_squares``
   (``xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py``).
5. **In-memory joint fit** — the existing batched-by-angle solver for all
   remaining cases.

Stratified least-squares module
--------------------------------

``fit_heterodyne_stratified_least_squares`` implements angle stratification
for the heterodyne model via three building blocks:

* ``make_scaling_expander`` — constructs the per-angle scaling expansion for
  the active ``per_angle_mode`` (``averaged``, ``individual``, or
  ``fourier``).
* ``build_joint_pointwise_residual`` — assembles the full residual vector
  as a flat concatenation of per-angle residuals.
* ``strategies/chunking.py`` — model-agnostic data-layout helper shared with
  homodyne; handles the angle-stratified chunking of the ``(n_phi, N, N)``
  data arrays.

Activation gate: ≥ 1 M points
------------------------------

Heterodyne stratification engages **only when** ``n_points ≥ 1 000 000``.

This differs from homodyne, which has an additional 100k–1 M "shuffle-only"
regime where it reorders + shuffles data while still using the standard solver.
That regime does **not** transfer to heterodyne because heterodyne's sub-1 M
solver is batched by angle (``(n_phi, N, N)`` tensors) rather than a flat
point list; there is no point list to shuffle.  Below 1 M points heterodyne
uses the existing in-memory joint fit unchanged.

.. list-table:: Stratification activation by point count
   :header-rows: 1
   :widths: 25 35 40

   * - Point count
     - Homodyne behaviour
     - Heterodyne behaviour
   * - < 100 000
     - Standard solver
     - In-memory joint fit
   * - 100 000 – 1 000 000
     - Shuffle-only (reorder, standard solver)
     - In-memory joint fit (no shuffle)
   * - ≥ 1 000 000
     - Stratified LS
     - Stratified LS (this feature)

Default-on seed-42 pre-shuffle
-------------------------------

When stratified-LS activates, a seed-42 pre-shuffle of the flattened point
list is applied before chunk assignment.  This is **default-on** and mirrors
homodyne's local-minimum-avoidance practice.  The shuffle is
**objective-invariant**: reordering residual elements does not change the sum
of squared residuals, so the SSR reported by the stratified path equals the
SSR that the in-memory joint fit would report for the same converged
parameters.  The equivalence test (``tests/heterodyne/``) verifies this
invariant at rtol 1e-9.

.. note::
   Set ``optimization.stratification.enabled: false`` to disable the shuffle
   and stratified-LS entirely, reverting to the in-memory joint fit at all
   point counts.

Configuration
-------------

All stratification knobs live under ``optimization.stratification`` in the
mode YAML (e.g. ``xpcsjax_two_component.yaml``):

.. code-block:: yaml

   optimization:
     stratification:
       enabled: auto           # "auto" (default-on) | true | false
       target_chunk_size: 100000
       max_imbalance_ratio: 5.0
       force_sequential_fallback: false
       check_memory_safety: true
       use_index_based: false

``enabled: auto`` (default) activates the ``should_use_stratification``
heuristic; ``enabled: false`` disables entirely; ``enabled: true`` forces it
on regardless of the heuristic.  The remaining keys match homodyne's
stratification config and control how angle chunks are balanced and whether
memory-safety checks are enforced before chunk assembly.

.. _streaming_antidegeneracy:

Streaming anti-degeneracy (Gap D closed)
-----------------------------------------

The STREAMING tier previously froze the quantile-estimated per-angle scaling
inside the JIT closure and ran no anti-degeneracy layers.  This has been
closed: heterodyne streaming now **optimizes** the scaling tail and runs
**L1–L4**, reaching mechanism parity with ``laminar_flow`` streaming.

Per-angle mode dispatch
~~~~~~~~~~~~~~~~~~~~~~~

The scaling treatment is selected by ``anti_degeneracy_config.per_angle_mode``:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - ``per_angle_mode`` value
     - Optimized params
     - Notes
   * - ``"fixed_constant"``
     - none (scaling frozen)
     - Explicit opt-out via ``per_angle_mode="constant"``; freezes scaling (no L1/L2/L3).
   * - ``"auto"`` → ``"auto_averaged"`` / ``"individual"``
     - 2 (mean) at ``n_phi ≥ threshold``; else 2·n_phi
     - **THE DEFAULT**, including when ``anti_degeneracy_config`` is absent or ``None`` (mirrors laminar — no "freeze when unconfigured" special case). Resolves to ``auto_averaged`` at ``n_phi ≥ constant_scaling_threshold`` (default 3), else ``individual``.
   * - ``"individual"``
     - 2 × n_phi
     - Per-angle contrast + offset optimized jointly.
   * - ``"fourier"``
     - 2 × (2K+1) Fourier coeffs
     - Smooth angular variation.  Falls back silently to ``"individual"`` when n_phi < 1+2K; the effective mode is surfaced via ``meta["fourier_effective_mode"]``.

Layer activation on the STREAMING path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **L1** — active for all optimized modes (``auto_averaged``, ``individual``,
  ``fourier``); skipped for ``fixed_constant`` (no tail to reparameterize).
- **L2** — active for ``individual`` and ``fourier`` only, gated identically to
  ``laminar_flow`` streaming (``not use_constant``).  ``auto_averaged`` and
  ``fixed_constant`` have ≤ 2 per-angle DoF so hierarchical alternation is not
  needed.  On the L2 branch the ``[physics | scaling]`` vector is permuted to
  the ``[per_angle | physics]`` layout expected by
  :class:`~xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer` and
  un-permuted on exit; covariance is an identity placeholder on this branch.
- **L3** — active when ``anti_degeneracy_config.regularization.enable`` is
  ``True`` (default) and there is a scaling tail (``n_scaling > 0``).  Uses
  group-variance config on the plain branch and
  ``compute_regularization_jax`` inside the hierarchical loss.
- **L4** — always active when there is a scaling tail.  The gradient-collapse
  monitor is wired via ``callback=`` on the plain branch and via
  ``_hier_grad`` on the L2 branch.  Strictly observational: monitor-on ==
  monitor-off objective.
- **L5** — omitted by design (``laminar_flow``-only); the diagnostics block
  reports the ``'laminar_flow_inactive'`` sentinel.

Diagnostics
~~~~~~~~~~~

The streaming path emits the symmetric ``info["anti_degeneracy"]`` block via
the shared
:func:`~xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics.assemble_anti_degeneracy_diagnostics`
assembler.  Keys: ``hierarchical_active``, ``regularization_active``,
``shear_weighting`` (always ``'laminar_flow_inactive'``), ``gradient_monitor``
(when L4 ran), and ``per_angle_mode``.  The SSR of the optimized solution is
available at ``info["ssr"]``; the frozen-scaling baseline SSR at
``info["ssr_frozen_baseline"]`` (``ssr <= ssr_frozen_baseline`` is the
objective-parity invariant).

Parity notes
------------

* **Mechanism parity, not numerical parity.** The stratified-LS path fits the
  same heterodyne model as the in-memory joint fit; results may differ
  slightly due to landing in different local minima (the ~0.17 % χ² spread
  documented for the C044 ``two_component`` degeneracy case is normal).  The
  objective is conserved: ``sum(chi2_per_angle) == chi_squared`` holds exactly.
* **SSR conservation.** The cross-evaluation invariant is verified in the
  equivalence test: evaluating either path's parameters through the other
  path's residual function gives the same SSR at rtol 1e-9.
* **Streaming parity contract.** The STREAMING anti-degeneracy contract is
  mechanism + objective parity with ``laminar_flow`` streaming (not
  ``rtol=1e-10`` — that gate is homodyne-specific).  See
  :ref:`heterodyne_anti_degeneracy` for details.
* **L5 not applicable.** As described in :ref:`heterodyne_anti_degeneracy`,
  L5 shear-sensitivity weighting is ``laminar_flow``-only and does not apply
  to the heterodyne two-component model; this remains true on all heterodyne
  paths including stratified-LS and streaming.

Memory profile and known limitations
-------------------------------------

The ≥1M stratified-LS path solves **all** real data points (no subsampling —
silent downsampling is prohibited).  At the C044 scale (23 angles × 1001 × 1001
≈ 23M points, 16 joint parameters) the dominant resident cost during the solve
is the **dense Jacobian** that ``trf`` forms each iteration: ``N × n_params``
float64 (≈ 2.9 GB at 23M × 16) plus the forward-mode AD tangents threaded
through the pointwise kernel.  On a memory-tight host this transient drives the
``AdvancedMemoryManager`` pressure monitor across its 75 % / 90 % thresholds; the
pressure is **live working-set arrays, not a leak**, and recovers between
iterations.

What the package does about it:

* **The per-iteration Jacobian is owned by the external ``nlsq`` library**
  (``CurveFit`` with ``x_scale="jac"``); the adapter passes no analytic
  Jacobian.  It is **not** reducible from xpcsjax without changing ``x_scale`` /
  ``method`` or modifying ``nlsq`` — either of which would perturb the
  non-convex ``trf`` + ``soft_l1`` solve and risk a different (worse) local
  basin, with no ``rtol`` gate to catch the regression.  It is therefore left
  unchanged by design.
* **The post-solve covariance Jacobian *is* owned by xpcsjax** and is computed
  with a column-blocked forward-mode JVP (``_chunked_jacfwd_dense`` in
  ``heterodyne_stratified_ls.py``) that is byte-identical to ``jax.jacfwd``
  (verified at ``rtol ≤ 1e-12``) but caps the AD-tangent width, cutting the one
  xpcsjax-controlled multi-GB spike.  This affects covariance only, never the
  fit trajectory.
* **Pressure logging is regime-aware.** When GC repeatedly frees nothing — the
  signal that pressure is live JAX/NumPy arrays — the monitor demotes the
  warning to DEBUG and the critical event to a reframed WARNING ("live arrays,
  best-effort cleanup only; not a leak"), and gates ``jax.clear_caches()``
  (which would otherwise force XLA recompiles mid-solve) behind that same
  signal plus a cooldown.  The 90 %+ state stays *visible* in case of a genuine
  OOM climb, but stops reading as an actionable defect during a normal hot
  solve.

Future work (not implemented — each alters results and needs parity verification):

* **Workload-level reduction of the dense-Jacobian peak.** The only way to cut
  the ≈ 2.9 GB / iteration peak without basin risk is to feed fewer points to
  the dense solve — e.g. an out-of-core tier below the in-memory stratified-LS
  path, or a coarser ``stratification.target_chunk_size``.  Both change the
  numerics and must be validated against the C044 objective oracle
  (``tests/heterodyne/test_two_component_real_data.py``).  Subsampling the data
  is **not** an option (no silent downsampling).
* **Upstream chunked/blockwise Jacobian.** A memory-bounded Jacobian that
  accumulates ``JᵀJ`` blockwise belongs in ``nlsq`` itself (or in a future
  solver robust to the pointwise evaluator).  Tracked alongside the
  pointwise-evaluator basin-fragility note in the heterodyne engine-route
  documentation.
