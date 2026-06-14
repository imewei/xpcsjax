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
controller is constructed by ``make_controller`` from the
``anti_degeneracy`` section of the YAML config and is consulted on
every iteration of the fit.

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

- **What it does.** Splits the parameter vector into a "fast"
  inner block (diffusion coefficients) and a "slow" outer block
  (shear coefficients, contrast/offset). Solves the inner problem
  to convergence at each outer step, similar in spirit to a
  block-coordinate descent.
- **When it activates.** When the controller detects that the
  active parameter set spans both physics and scaling sub-spaces
  (typically heterodyne).
- **What it costs.** Roughly 2× the number of trust-region
  iterations on simple problems; the savings on degenerate problems
  justify it.

Layer 3 — Adaptive regularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer`

- **What it does.** Adds a Tikhonov-style regularisation term whose
  strength is set by the smallest singular value of the local
  Jacobian. As the fit progresses and the Jacobian becomes
  better-conditioned, the regularisation strength decays.
- **When it activates.** Whenever the Jacobian condition number
  exceeds a configurable threshold (default 1e8).
- **What it costs.** Negligible at well-conditioned iterations; one
  extra small linear solve at ill-conditioned ones.

Layer 4 — Gradient collapse monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor`

- **What it does.** Watches the norm of the cost-function gradient
  across iterations. If it drops below a threshold without the
  parameter step converging, the controller declares "gradient
  collapse" and triggers a recovery action — most commonly the
  CMA-ES escape (see :doc:`cma_es_escape`).
- **When it activates.** Anytime; checked at every iteration.
- **What it costs.** O(n_params) per iteration for the norm; no
  extra solve.

Layer 5 — Shear sensitivity weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`

- **What it does.** Computes a per-angle weight proportional to the
  squared shear-mode amplitude ``cos^2(2(phi - phi0))``. Angles
  near ``phi0`` carry little shear information and are down-weighted
  in the residual so they do not drown the contribution from
  flow-perpendicular angles.
- **When it activates.** For ``laminar_flow`` and ``two_component``
  modes when ``anti_degeneracy.shear_weighting.enabled`` is true.
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
      ├─► residuals weighted by ShearSensitivityWeighting
      │
      └─► GradientCollapseMonitor.check(grad_norm)
            │
            └─[collapse]─► request CMA-ES escape

The order is fixed in :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`; do not
reorder without re-running the parity tests
(:doc:`parity_testing`).

Configuration
-------------

The relevant YAML block:

.. code-block:: yaml

    anti_degeneracy:
      enabled: true
      per_angle_mode: auto
      hierarchical:
        enabled: true
        inner_max_iters: 50
      adaptive_regularization:
        enabled: true
        cond_threshold: 1.0e8
      gradient_collapse:
        enabled: true
        grad_norm_threshold: 1.0e-8
      shear_weighting:
        enabled: true

Setting ``anti_degeneracy.enabled: false`` disables all five layers.
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
