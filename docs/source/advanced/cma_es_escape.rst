CMA-ES Escape
=============

When the trust-region Levenberg-Marquardt solve plateaus on a
non-convex landscape, xpcsjax can escape via the Covariance Matrix
Adaptation Evolution Strategy (CMA-ES). The implementation wraps NLSQ's
``CMAESOptimizer`` (``nlsq.global_optimization``, which itself uses the
``evosax`` backend) and runs CMA-ES as a standalone global search,
optionally followed by an NLSQ trust-region refinement. It is exposed
through three types in
:mod:`xpcsjax.optimization.nlsq.cmaes_wrapper`:

- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper` —
  the wrapper itself.
- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig`
  — dataclass of tuning knobs.
- :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESResult` —
  the return value (fields ``parameters``, ``covariance``,
  ``chi_squared``, ``success``, ``diagnostics``, ``method_used``,
  ``nlsq_refined``, ``message``).

When CMA-ES fires
-----------------

CMA-ES is **not** triggered by the gradient-collapse monitor. It is
enabled from config (``optimization.nlsq.cmaes.enable: true``), and
``fit_nlsq`` then dispatches to ``fit_nlsq_cmaes``. Whether it actually
engages is decided by :meth:`CMAESWrapper.should_use_cmaes`, which
returns ``True`` when the parameter-bounds scale ratio
(``max_range / min_range``) is at or above ``scale_threshold`` (default
``1000``) and ``evosax`` is installed. XPCS flow fits routinely exceed
this because the diffusion and shear parameters span many decades.

.. note::

   The escape is **not** a replacement for the trust-region solve.
   CMA-ES is a global-search method with sub-linear local convergence;
   after it finds a better basin, an optional NLSQ trust-region (TRF)
   refinement (``refine_with_nlsq``, default on) polishes the result and
   recovers the covariance.

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

Override the default by setting ``optimization.nlsq.cmaes.enable``
in the YAML config (heterodyne uses the flat key ``enable_cmaes``).

BIPOP restart strategy
----------------------

Setting ``restart_strategy="bipop"`` (the default) selects NLSQ's
BIPOP-CMA-ES restart schedule (Hansen, 2009): two interleaved restart
regimes — a small default population and a larger one — run alternately
until the restart budget (``max_restarts``) is exhausted.

BIPOP performs noticeably better than plain restarts on multi-modal
problems because the larger-population regime escapes local basins
that the default population cannot.

The schedule is implemented inside
:class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper` and
governed by the following ``CMAESWrapperConfig`` fields:

.. code-block:: python

    CMAESWrapperConfig(
        preset="cmaes",            # "cmaes-fast" | "cmaes" | "cmaes-global"
        sigma=0.5,                 # initial step (fraction of bound range)
        restart_strategy="bipop",  # "none" | "bipop"
        max_restarts=9,            # cap on BIPOP restarts
        popsize=None,              # None = auto (4 + 3*ln(n))
        max_generations=None,      # None = preset + adaptive scaling
        refine_with_nlsq=True,     # NLSQ-TRF polish after CMA-ES
        normalize=True,            # bounds-based [0, 1] normalization
    )

All fields have defaults (these are a representative subset).

Calling the wrapper directly
----------------------------

The wrapper is normally driven from
:func:`~xpcsjax.optimization.nlsq.fit_nlsq`, but it can be used
standalone for diagnostics. The constructor takes only an optional
config; the data and bounds are passed to :meth:`CMAESWrapper.fit`:

.. code-block:: python

    import numpy as np

    from xpcsjax.optimization.nlsq.cmaes_wrapper import (
        CMAESWrapper, CMAESWrapperConfig,
    )

    wrapper = CMAESWrapper(config=CMAESWrapperConfig(seed=42))
    result = wrapper.fit(
        model_func,                       # callable (xdata, *params) -> ydata
        xdata, ydata,
        p0=np.array([1.0e3, -1.5, 1.0e2]),
        bounds=(np.array([1.0, -2.0, 0.0]),
                np.array([1.0e6, 2.0, 1.0e4])),
    )

The returned
:class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESResult` carries
``parameters``, ``covariance``, ``chi_squared``, ``success``,
``diagnostics``, ``method_used``, ``nlsq_refined``, and ``message``.

Optional NLSQ refinement
------------------------

When ``refine_with_nlsq`` is true (the default), CMA-ES's best vector
seeds a final NLSQ trust-region (TRF) refinement run with tightened
``refinement_ftol`` / ``refinement_xtol`` / ``refinement_gtol``. This
recovers the covariance — and hence the uncertainties — which the
CMA-ES search itself does not produce. The result reports
``nlsq_refined`` and ``method_used`` in its ``diagnostics``.

Configuration
-------------

The YAML block recognised by xpcsjax:

.. code-block:: yaml

    optimization:
      nlsq:
        cmaes:
          enable: true
          preset: cmaes-global
          sigma: 0.5
          restart_strategy: bipop
          max_restarts: 10
          scale_threshold: 1000.0
          normalize: true
          refine_with_nlsq: true

Both the homodyne and heterodyne adapters read this same nested
``optimization.nlsq.cmaes`` YAML block (``enable``, ``sigma``,
``max_generations``, ...); see :doc:`/examples/heterodyne_multiangle` for
the heterodyne layout. Internally, heterodyne's ``NLSQConfig.from_dict``
unpacks that block into flat dataclass fields named ``enable_cmaes`` /
``cmaes_sigma0`` / ``cmaes_max_iterations`` — those are Python attribute
names, not YAML keys, so they never appear in the config file itself.

Cross-references
----------------

- :doc:`anti_degeneracy` — the five-layer controller (its
  gradient-collapse monitor is diagnostic and does **not** trigger
  CMA-ES).
- :doc:`/examples/multistart_robust_fit` — a complementary global
  search that runs *outside* the LM loop.
- :doc:`memory_routing` — the strategy decision that bounds how
  much CMA-ES can evaluate per call.
