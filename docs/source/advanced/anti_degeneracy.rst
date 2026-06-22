Anti-degeneracy Controller
==========================

XPCS fits — especially in ``laminar_flow`` and ``two_component``
modes — exhibit parameter degeneracies that a vanilla trust-region
Levenberg-Marquardt solver handles poorly. The shear sub-space has
near-flat directions, the contrast/offset pair is weakly identified,
and at large data scales the gradient can collapse before the
solution is reached.

xpcsjax addresses this with a five-layer controller in
:mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`. The
controller is constructed by the classmethod
:meth:`AntiDegeneracyController.from_config` (internally via
``AntiDegeneracyConfig.from_dict``) from the ``anti_degeneracy``
section of the YAML config, and is consulted on every iteration of the
fit.

The five layers
---------------

Layer 1 — Per-angle reparameterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.per_angle_mode.PerAngleScalingPlan`

- **What it does.** Resolves how the per-angle scaling tail
  (contrast/offset) is parameterised before the trust-region solve.
  The three resolved modes are ``constant`` (a single frozen scaling
  shared across all angles), ``averaged`` (two averaged scaling
  parameters spanning all angles), and ``individual`` (a separate
  contrast/offset pair per angle). Choosing the smallest adequate
  layout keeps weakly-constrained scaling directions out of the
  trust-region step so it shrinks rather than thrashes.
- **When it activates.** Engaged automatically for
  ``laminar_flow`` and ``two_component`` modes; the resolved mode is
  selected from ``anti_degeneracy.per_angle_mode`` in the YAML (the
  ``auto`` default resolves to ``averaged`` for ``n_phi >= 3`` else
  ``individual``).
- **What it costs.** A one-shot host-side layout decision per fit,
  amortised by the NLSQ JIT cache. No per-iteration overhead.

Layer 2 — Hierarchical optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer`

- **What it does.** Two-stage *alternating* (block-coordinate)
  optimisation between the physical-parameter block and the per-angle
  scaling block: stage 1 fits the physical parameters with the
  per-angle scaling frozen, stage 2 fits the per-angle scaling with the
  physical parameters frozen, iterating to convergence. The split is
  physics-vs-per-angle, not diffusion-vs-shear.
- **When it activates.** Default-active for all modes (it is not gated
  in ``_LAYER_GATES``; only L5 is mode-gated).
- **What it costs.** Roughly 2× the number of trust-region
  iterations on simple problems; the savings on degenerate problems
  justify it.

Layer 3 — Adaptive regularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer`

- **What it does.** Adaptive CV-based (coefficient-of-variation)
  regularisation that penalises the relative spread of the per-angle
  parameters: ``L_reg = lambda * CV^2 * MSE * n_points`` with
  ``CV = std(params) / abs(mean(params))``. ``lambda`` is auto-tuned as
  ``target_contribution / target_cv^2``.
- **When it activates.** When ``regularization`` is enabled and there
  is a per-angle scaling tail to constrain. Configured via
  ``regularization.{mode, lambda, target_cv, target_contribution,
  max_cv}``.
- **What it costs.** A small per-iteration penalty term; negligible.

Layer 4 — Gradient collapse monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor`

- **What it does.** Watches the *ratio*
  ``norm(grad_physical) / norm(grad_per_angle)`` against
  ``gradient_ratio_threshold`` (default ``0.01``) over consecutive
  iterations. On detection it applies a configurable response — one of
  ``warn`` / ``hierarchical`` / ``reset`` / ``abort`` (default
  ``hierarchical``). It is strictly diagnostic and does **not** trigger
  the CMA-ES escape (a separate mechanism, see :doc:`cma_es_escape`).
- **When it activates.** Checked at every iteration where a solver
  callback fires; otherwise a post-solve covariance-condition fallback.
- **What it costs.** O(n_params) per iteration for the gradient ratio;
  no extra solve.

Layer 5 — Shear sensitivity weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`

- **What it does.** Computes a per-angle weight
  ``w(phi) = w_min + (1 - w_min) * |cos(phi0 - phi)|^alpha`` (defaults
  ``alpha = 1``, ``w_min = 0.3``). Angles near ``phi0``
  (parallel/antiparallel to the flow) carry the most shear-sensitivity
  information and are *up-weighted*; flow-perpendicular angles
  (``|cos| ~ 0``) are damped toward ``w_min``. The ``w_min`` floor keeps
  every angle contributing.
- **When it activates.** ``laminar_flow`` **only** (gated in
  ``_LAYER_GATES``), when ``anti_degeneracy.shear_weighting.enable`` is
  true. ``two_component`` has a structurally different velocity term, so
  L5 does not transfer to it.
- **What it costs.** One ``cos`` evaluation per angle per fit;
  trivial.

Activation flow
---------------

The controller runs as a sequence:

.. code-block:: text

    [start of fit]
      │
      ▼
    PerAngleScalingPlan.resolve()         (one-shot scaling layout)
      │
      ▼
    HierarchicalOptimizer.wrap()          (block-split residual)
      │
      ▼
    [each LM iteration]
      │
      ├─► AdaptiveRegularizer.lambda_for(J)
      │
      ├─► residuals weighted by ShearSensitivityWeighting (laminar only)
      │
      └─► GradientCollapseMonitor.check(grad_ratio)   (diagnostic only)

The order is fixed in :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`; do not
reorder without re-running the parity tests
(:doc:`parity_testing`).

Configuration
-------------

The relevant YAML block:

.. code-block:: yaml

    optimization:
      nlsq:
        anti_degeneracy:
          enable: true
          per_angle_mode: auto
          hierarchical:
            enable: true
            max_outer_iterations: 10
          regularization:
            enable: true
            target_cv: 0.10
          gradient_monitoring:
            enable: true
            ratio_threshold: 0.01
            response: hierarchical
          shear_weighting:
            enable: true

Setting ``anti_degeneracy.enable: false`` disables all five layers.
This is useful for parity tests against the unmodified trust-region
solve but not recommended for production fits.

Reading the audit
-----------------

Each layer logs an entry to
:attr:`~xpcsjax.optimization.nlsq.results.OptimizationResult.recovery_actions`.
A typical sequence on a ``laminar_flow`` fit:

.. code-block:: text

    per_angle_mode: resolved (averaged)
    hierarchical: outer_step=0  inner_converged_in=12
    adaptive_reg: cond=2.1e+09  lambda=3.4e-05
    shear_weight: applied (phi0=12.34)
    hierarchical: outer_step=1  inner_converged_in=8
    ...

Cross-references
----------------

- :doc:`cma_es_escape` — the escape path triggered by Layer 4.
- :doc:`/theory/anti_degeneracy` — derivation of the degeneracies
  and motivation for each layer.
- :doc:`parity_testing` — the regression tests that pin the
  controller's behaviour.
