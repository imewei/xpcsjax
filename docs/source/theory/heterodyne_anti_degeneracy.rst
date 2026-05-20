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
  **post-solve** diagnostic: the joint solver's covariance matrix is
  decomposed via SVD and the condition number ``σ_max / σ_min`` is used
  as a proxy for gradient collapse. If the ratio exceeds
  ``config.gradient_ratio_threshold`` (or the covariance is rank-
  deficient), ``collapse_detected=True`` is reported in
  ``result.nlsq_diagnostics['gradient_monitor']``. This is **not** the
  per-iteration ``GradientCollapseMonitor`` callback used by homodyne;
  ``config.gradient_consecutive_triggers`` is parsed and surfaced in
  diagnostics but currently has no effect — it is preserved for a
  future per-iteration implementation (host_callback or a custom solver
  wrapper). See the follow-up tracking item for the callback migration.

L5: Shear-sensitivity weighting (NOT APPLICABLE to heterodyne)
  Homodyne's L5 up-weights data near phi=0 to exploit the laminar-flow
  shear-sensitivity peak. The heterodyne two-component model does not
  exhibit this peak — the relative weight of the two components varies
  with phi in a way that's already informative, so additional weighting
  has no physics motivation. xpcsjax heterodyne deliberately omits L5;
  see ``result.nlsq_diagnostics['shear_weighting'] == 'not_applicable_heterodyne'``
  for the explicit marker.

  This is a structural decision, not a TODO.
