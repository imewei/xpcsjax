Heterodyne Anti-Degeneracy System (4 layers)
=============================================

Heterodyne uses a 4-layer anti-degeneracy defense, mirroring homodyne's
system minus L5 (shear-sensitivity weighting).

Per-Angle Modes
---------------

xpcsjax heterodyne supports the same four ``per_angle_mode`` values as
homodyne (see
https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html
for full theory):

================ ============================== ===========================
Mode             Optimizer params (K=2, n_phi)  Notes
================ ============================== ===========================
``constant``     ``n_physics``                  beta, offset pre-estimated
                                                from quantile and frozen
``auto``         depends on n_phi:              recommended default
                 n_phi < 3 -> constant
                 3 <= n_phi < 6 -> averaged
                 n_phi >= 6 -> fourier
``fourier``      ``n_physics + 2(2K+1)``        truncated Fourier basis
                                                for beta(phi), offset(phi)
``individual``   ``n_physics + 2*n_phi``        free per-angle scaling
================ ============================== ===========================

Joint global escapes
--------------------

Beyond the per-angle defense layers, heterodyne provides two **joint-fit
global escapes** that search the full ``[physics | scaling]`` vector. Both are
**real global escapes** (no longer the Phase-6 minimal stubs):

``_fit_joint_cmaes_multi_phi`` (``enable_cmaes=True``)
  A **seed-pinned** CMA-ES global search over the joint vector, reusing the
  shared :func:`xpcsjax.optimization.nlsq.cmaes_wrapper.fit_with_cmaes`.

``_fit_joint_multistart`` (``multistart=True``)
  A **seed-pinned** Latin-Hypercube multistart sweep over the joint vector,
  reusing the shared ``run_multistart_nlsq`` (each start re-runs the plain
  joint fit seeded at ``x_start``).

Both escapes are **keep-better** — they are accepted only if their data-only
SSR beats the plain NLSQ joint fit — and **best-effort fall back** to the plain
joint fit on failure. This is the joint-fit global escape; the per-angle
escapes were already real. Together these closed the joint-escape parity gap
with ``laminar_flow``. The escapes are strategy-level and do **not** touch the
anti-degeneracy controller.

An escape result is tagged ``nlsq_diagnostics["global_escape"]`` (``"cmaes"`` or
``"multistart"``; the key is absent on a plain joint fit) and, **by
construction**, carries NaN ``covariance`` / ``parameter_uncertainties`` and
``n_iterations=0`` — the escape returns a pre-accepted (compared-and-kept)
vector with no covariance solve, so consumers needing uncertainties should
detect an escape result via the ``global_escape`` tag.

Defense Layers
--------------

L1: Mode-level reparameterization
  Selected by ``per_angle_mode``. Removes the flat optimization direction
  algebraically.

L2: Hierarchical optimization
  Activated by ``config.enable_hierarchical``. Runs as a two-stage solve:
  stage 1 fits the physics-only parameters with quantile-frozen scaling
  (the same path the standalone ``constant`` mode uses); stage 2 warm-
  starts the joint solve from the stage-1 estimate. Stage-1 χ² and the
  stage-1/stage-2 χ² ratio are recorded in
  ``result.nlsq_diagnostics['hierarchical']``. Note this is an inline
  two-stage implementation, not a delegation to
  ``xpcsjax.optimization.nlsq.hierarchical.fit_hierarchical_two_stage`` —
  see the follow-up tracking item for unifying with homodyne's helper.

L3: Adaptive CV regularization
  Activated by ``config.regularization_mode != 'none'``. Instantiates
  ``adaptive_regularization.AdaptiveRegularizer`` keyed to the per-angle
  scaling groups (contrast + offset) and appends JAX-traceable penalty
  rows to the augmented residual. The ``'adaptive'`` mode uses the
  relative/CV penalty branch; ``'absolute'`` uses the absolute branch.
  Active group indices and penalty weights are recorded in
  ``result.nlsq_diagnostics['regularization']``.

L4: Gradient collapse monitor
  Activated by ``config.enable_gradient_monitoring``. Implemented as a
  **per-iteration** gradient-collapse monitor **shared with homodyne**:
  ``build_gradient_collapse_callback`` builds an NLSQ ``curve_fit``
  callback that, each iteration, feeds the physical/per-angle gradient
  ratio to a :class:`GradientCollapseMonitor`. The monitor is **strictly
  diagnostic** — it observes the solve and never mutates it, so a fit with
  monitoring enabled is **bit-identical** to one with it disabled
  (including the homodyne ``rtol=1e-10`` characterization baselines).

  When the solver callback never fires (e.g. a solve path that does not
  forward per-iteration callbacks), L4 falls back to a **post-solve
  covariance-condition** check: the joint solver's covariance matrix is
  decomposed via SVD and the condition number ``σ_max / σ_min`` is used as
  a proxy for gradient collapse. The two paths are distinguished by the
  ``mechanism`` field of the diagnostics block (see below).

  ``config.gradient_consecutive_triggers`` is now **effective**: it sets
  the number of consecutive low-ratio iterations required before
  ``collapse_detected`` is raised.

  The L4 block lives at ``result.nlsq_diagnostics['gradient_monitor']``
  and carries these keys:

  ``collapse_detected`` : bool
    Whether gradient collapse was confirmed.
  ``trigger_count`` : int
    Number of recorded collapse events.
  ``min_gradient_ratio`` / ``max_gradient_ratio`` : float
    Extremes of the observed physical/per-angle gradient ratio
    (``nan`` when no observations were recorded).
  ``n_observations`` : int
    Number of per-iteration observations the callback recorded.
  ``ratio_threshold`` : float
    The configured collapse ratio threshold.
  ``consecutive_triggers`` : int
    The configured consecutive-trigger count
    (``config.gradient_consecutive_triggers``).
  ``mechanism`` : str
    Which path produced the block — ``'per_iteration_gradient_ratio'``
    when the callback recorded ≥ 1 observation, or
    ``'post_solve_fallback'`` when it recorded none and the post-solve
    covariance-condition check ran instead.

L5: Shear-sensitivity weighting (laminar_flow ONLY)
  L5 up-weights data near the flow direction phi0 to exploit *laminar_flow's*
  shear-sensitivity peak (``d g1_shear / d gamma_dot ~ cos(phi0 - phi)``). It is
  tied to laminar_flow's shear-rate (``gamma_dot``) kernel and is active for
  ``laminar_flow`` only — the static modes have no flow term, so L5 is gated
  off for them too (see ``_LAYER_GATES`` in ``anti_degeneracy_controller.py``).

  The heterodyne two-component model has its OWN velocity/flow term
  (``v0``, ``v_offset``, ``phi0_het``), but it is **structurally different** from
  laminar_flow's shear rate, so laminar_flow's shear-sensitivity weighting does
  not transfer: the relative weight of the two components already varies with
  phi in an informative way, so an additional ``cos(phi0 - phi)`` weight has no
  physics motivation. xpcsjax heterodyne therefore omits L5; see
  ``result.nlsq_diagnostics['shear_weighting'] == 'not_applicable_heterodyne'``
  for the explicit marker.

  This is a structural decision, not a TODO.

Symmetric diagnostics contract
------------------------------

Both ``laminar_flow`` and ``two_component`` now emit the same top-level
``nlsq_diagnostics`` activation keys —
``{hierarchical_active, regularization_active, shear_weighting}``, plus
``gradient_monitor`` when L4 ran — via the shared assembler
``xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics.assemble_anti_degeneracy_diagnostics``.
The ``*_active`` flags are **always present**, taking the value ``False`` when
the corresponding layer did not run, so a caller can read the same key set
regardless of mode. The ``shear_weighting`` value is mode-appropriate:
``'not_applicable_heterodyne'`` for ``two_component`` and
``'laminar_flow_inactive'`` for ``laminar_flow``'s in-memory path.

This was a **diagnostics-only** unification: the L2/L3 solve code was already
shared between the two modes, and only the emission of the diagnostics keys was
made symmetric. Both characterization baselines remain bit-identical.

These flat top-level activation keys are now emitted on **every** laminar path
— in-memory, HYBRID_STREAMING, stratified-LS (≥1 M points), sequential, and
out-of-core — as well as on all heterodyne paths, via the shared assembler. The
values are honest per path: HYBRID_STREAMING reports the real active L2/L3/L5 it
runs, while stratified-LS, sequential, and out-of-core report inactive markers
(``hierarchical_active=False``, ``regularization_active=False``,
``shear_weighting='laminar_flow_inactive'``) because those layers do not run on
those paths. Activation is never fabricated.
