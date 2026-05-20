CMA-ES Escape
=============

When the trust-region Levenberg-Marquardt solve plateaus on a
non-convex landscape, xpcsjax can escape via the Covariance Matrix
Adaptation Evolution Strategy (CMA-ES). The implementation wraps
``evosax``'s CMA-ES with a BIPOP restart schedule and is exposed
through three small types in
:mod:`xpcsjax.optimization.nlsq.cmaes_wrapper`:

- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper` —
  the wrapper itself.
- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig`
  — frozen dataclass of tuning knobs.
- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESResult` —
  the return value, mirroring the
  :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`
  shape so the result builder can fuse it with the LM run.

When CMA-ES fires
-----------------

The escape is governed by the gradient-collapse monitor (Layer 4 of
the :doc:`anti_degeneracy` controller). The trigger is conservative:

1. The Levenberg-Marquardt iteration count exceeds a configurable
   minimum (default ``min_lm_iters`` on
   :class:`xpcsjax.optimization.nlsq.CMAESWrapperConfig`).
2. The cost-function gradient norm drops below
   ``grad_norm_threshold`` *without* the parameter step satisfying
   ``xtol``.
3. The reduced chi-squared remains above the user's plateau
   threshold.

When all three conditions hold, the LM run is paused, its best
parameter vector becomes the CMA-ES centroid, and the escape begins.

.. note::

   The escape is **not** a replacement for the trust-region solve.
   CMA-ES is a global-search method with sub-linear local
   convergence; once it finds a better basin, control returns to LM
   for the final quadratic refinement.

Mode defaults
-------------

The default-on/default-off behaviour reflects the empirical
difficulty of each analysis mode:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Analysis mode
     - CMA-ES default
     - Rationale
   * - ``static_isotropic``
     - off
     - 3 parameters, well-conditioned in practice; LM converges
       cleanly.
   * - ``laminar_flow``
     - off
     - 7 parameters, but the anti-degeneracy controller usually
       prevents collapse. Multistart (:doc:`/examples/multistart_robust_fit`)
       is the primary defence.
   * - ``two_component`` (heterodyne)
     - on
     - Per-angle contrast/offset multipliers make the landscape
       genuinely multi-modal; CMA-ES escape pays for itself.

Override the default by setting ``optimization.nlsq.cmaes_escape.enabled``
in the YAML config.

BIPOP restart strategy
----------------------

The wrapper uses the BIPOP-CMA-ES restart schedule
(Hansen, 2009): two interleaved restart regimes — a population of
``lambda_default = 4 + 3 * floor(log(d))`` and a larger population
``lambda_large`` — run alternately under a shrinking budget until
the total function-evaluation budget is exhausted.

BIPOP performs noticeably better than plain restarts on multi-modal
problems because the larger-population regime escapes local basins
that the default population cannot.

The schedule is implemented inside
:class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper` and
governed by the following ``CMAESWrapperConfig`` fields:

.. code-block:: python

    CMAESWrapperConfig(
        sigma0=0.3,                # initial step size in transformed coords
        lambda_default=None,       # auto: 4 + 3*floor(log(d))
        lambda_large=None,         # auto: 2 * lambda_default
        max_evals=2000,            # total function-evaluation budget
        max_restarts=4,            # cap on BIPOP restarts
        seed=0,
        grad_norm_threshold=1.0e-8,
        min_lm_iters=20,
        early_stop_on_chi2_red=None,
    )

All fields have defaults; the dataclass is frozen.

Calling the wrapper directly
----------------------------

The wrapper is normally driven by the LM-with-escape loop inside
:func:`~xpcsjax.optimization.nlsq.fit_nlsq`, but it can be used standalone for
diagnostics:

.. code-block:: python

    import numpy as np

    from xpcsjax.optimization.nlsq.cmaes_wrapper import (
        CMAESWrapper, CMAESWrapperConfig,
    )

    def cost_fn(params: np.ndarray) -> float:
        # User-supplied scalar cost; xpcsjax wraps the LM residual^2 sum.
        ...

    wrapper = CMAESWrapper(
        cost_fn=cost_fn,
        bounds_lo=np.array([1.0, -2.0, 0.0]),
        bounds_hi=np.array([1.0e6, 2.0, 1.0e4]),
        x0=np.array([1.0e3, -1.5, 1.0e2]),
        config=CMAESWrapperConfig(max_evals=1000, seed=42),
    )
    result = wrapper.run()

The returned :class:`xpcsjax.optimization.nlsq.CMAESResult` carries the best ``x``, the best
cost, the number of evaluations used, and a flag indicating whether
the BIPOP schedule completed or stopped on the budget.

Hand-off back to LM
-------------------

When CMA-ES finishes, its best ``x`` becomes the new initial guess
for a final LM polish. This step is essential — CMA-ES has no
quadratic-convergence regime, so without the polish the reported
uncertainties (which come from the LM Jacobian) would be missing.
The polish is run with a tightened ``ftol``/``xtol``/``gtol`` and
appears as a separate ``recovery_actions`` entry on the
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult`:

.. code-block:: text

    cma_es: triggered (lm_iter=37, grad_norm=4.2e-09)
    cma_es: completed (evals=1840, restarts=2, best_cost=1.23e-04)
    lm_polish: converged (iters=11, ftol satisfied)

Configuration
-------------

The YAML block recognised by xpcsjax:

.. code-block:: yaml

    optimization:
      nlsq:
        cmaes_escape:
          enabled: true
          sigma0: 0.3
          max_evals: 2000
          max_restarts: 4
          seed: 0
          early_stop_on_chi2_red: 1.0e-4

The block is read by the heterodyne adapter and by the homodyne
adapter from their respective ``optimization.nlsq`` sections; see
:doc:`/examples/heterodyne_multiangle` for the heterodyne layout.

Cross-references
----------------

- :doc:`anti_degeneracy` — the gradient-collapse monitor that
  triggers the escape.
- :doc:`/examples/multistart_robust_fit` — a complementary global
  search that runs *outside* the LM loop.
- :doc:`memory_routing` — the strategy decision that bounds how
  much CMA-ES can evaluate per call.
