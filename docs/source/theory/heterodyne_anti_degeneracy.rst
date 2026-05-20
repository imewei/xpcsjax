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
  Activated by ``config.enable_hierarchical``. Two stages: physics-only
  with fixed scaling, then jointly refined. Currently MVP-wired
  (diagnostic contract populated); full integration with NLSQ residual
  semantics deferred to a future phase.

L3: Adaptive CV regularization
  Activated by ``config.regularization_mode != 'none'``. Penalizes
  solutions with anomalous per-angle residual variance. Currently
  MVP-wired (diagnostic contract populated); full residual wrapping
  with AdaptiveRegularizer deferred to a future phase.

L4: Gradient collapse monitor
  Activated by ``config.enable_gradient_monitoring``. Detects flat
  landscapes via gradient-norm collapse and (in future phases) restarts
  with perturbed initialization. Currently MVP-wired (diagnostic
  contract populated); full GradientCollapseMonitor callback integration
  with NLSQ solver iterations deferred to a future phase.

L5: Shear-sensitivity weighting (NOT APPLICABLE to heterodyne)
  Homodyne's L5 up-weights data near phi=0 to exploit the laminar-flow
  shear-sensitivity peak. The heterodyne two-component model does not
  exhibit this peak — the relative weight of the two components varies
  with phi in a way that's already informative, so additional weighting
  has no physics motivation. xpcsjax heterodyne deliberately omits L5;
  see ``result.nlsq_diagnostics['shear_weighting'] == 'not_applicable_heterodyne'``
  for the explicit marker.

  This is a structural decision, not a TODO.
