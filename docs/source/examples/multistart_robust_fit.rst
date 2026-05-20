Robust Multi-start Fitting
==========================

.. currentmodule:: xpcsjax


Trust-region least-squares is locally quadratically convergent but
globally bound to the basin of attraction of its initial guess. For
the seven-parameter ``laminar_flow`` model and the heterodyne models,
non-convex landscapes are common enough that a single starting point
is unsafe. The
:func:`~xpcsjax.optimization.nlsq.core.fit_nlsq_multistart` wrapper draws
multiple starts via Latin Hypercube Sampling (LHS) and returns the
best run.

When to use multistart
----------------------

- The objective is known or suspected to be multi-modal.
- The user-supplied initial guess is uninformed (e.g. order-of-magnitude
  only).
- A previous single-start fit converged but ``result.quality_flag``
  flagged the solution as suspect.
- Heterodyne fits where the contrast/offset sub-space introduces
  near-flat directions.

If only one of the listed conditions holds, a single
:func:`~xpcsjax.optimization.nlsq.fit_nlsq` call followed by inspection of
``recovery_actions`` is usually sufficient. The CMA-ES escape (see
:doc:`/advanced/cma_es_escape`) will also engage automatically when
the trust-region solve plateaus above a threshold, so an explicit
multistart is *not* the only line of defence.

Calling fit_nlsq_multistart
---------------------------

The multistart entry point lives one level deeper than the public
:func:`~xpcsjax.optimization.nlsq.fit_nlsq` wrapper:

.. code-block:: python

    from pathlib import Path

    from xpcsjax import ConfigManager, load_xpcs_data
    from xpcsjax.optimization.nlsq import fit_nlsq_multistart

    config_path = Path("config_laminar_flow.yaml")

    data = load_xpcs_data(str(config_path))
    cm = ConfigManager(str(config_path))
    cm.load_config()

    result = fit_nlsq_multistart(
        data,
        config=cm,
        n_starts=16,
    )

The signature accepts the same data/config pair as
:func:`~xpcsjax.optimization.nlsq.fit_nlsq`, plus ``n_starts`` (the number of LHS
samples to draw). The returned object is the best-of-N
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` —
"best" is decided by minimum unweighted residual sum of squares on
converged starts.

How the LHS sampling works
--------------------------

For each free parameter ``p_i`` with bound ``[lo_i, hi_i]``, the
sampler:

1. Builds a Latin Hypercube over the unit cube ``[0, 1]^d`` for ``d``
   active parameters and ``n_starts`` rows.
2. Maps each column ``i`` to ``[lo_i, hi_i]`` (linear by default; log
   for parameters whose bounds span more than three decades, e.g.
   ``D0``).
3. Optionally seeds the first row with the user-supplied initial
   guess so that single-start behaviour is recovered as ``n_starts``
   shrinks to one.

Each row becomes one independent fit; the wrapper reuses the same
NLSQ ``CurveFit`` JIT cache so per-start compile cost is amortised.

.. note::

   ``fit_nlsq_multistart`` does **not** parallelise across starts in
   v0.1 — the runs are serial. This keeps RAM pressure predictable
   under the same memory router that
   :func:`~xpcsjax.optimization.nlsq.fit_nlsq` uses (see :doc:`/advanced/memory_routing`).

Inspecting per-start outcomes
-----------------------------

The returned :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`
is the single best run; the per-start audit lives on
``result.recovery_actions`` and ``result.nlsq_diagnostics``. To
inspect every start, call the lower-level helper:

.. code-block:: python

    from xpcsjax.optimization.nlsq.multistart import run_lhs_starts

    all_results, best_idx = run_lhs_starts(
        data, config=cm, n_starts=16, return_all=True,
    )
    for i, r in enumerate(all_results):
        marker = "*" if i == best_idx else " "
        print(
            f"{marker} start={i:2d}  "
            f"chi2_red={float(r.reduced_chi_squared): .4e}  "
            f"success={r.success}"
        )

(The ``return_all=True`` form is intended for diagnostics; production
pipelines should call :func:`~xpcsjax.optimization.nlsq.core.fit_nlsq_multistart`
and trust the picked best.)

Choosing n_starts
-----------------

A useful default scaling is ``n_starts = 4 * d`` for ``d`` free
parameters, capped at ``32``. For the 7-parameter laminar-flow model
this yields ``n_starts = 28``. For static isotropic (3 parameters)
``n_starts = 12`` is usually enough.

Larger values cost compute roughly linearly but the marginal benefit
flattens beyond ``8 * d``.

Interaction with CMA-ES escape
------------------------------

Inside any given start, if the trust-region solve plateaus, the
CMA-ES escape (:class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper`)
can still fire — multistart and CMA-ES are independent layers. In
practice:

- For homodyne fits, CMA-ES escape defaults to off; multistart is the
  primary defence against bad initial guesses.
- For heterodyne fits, CMA-ES escape defaults to on; combining it
  with multistart is conservative but rarely necessary.

See :doc:`/advanced/cma_es_escape` for the trigger condition and the
``CMAESWrapperConfig`` knobs.

Next steps
----------

- :doc:`/advanced/cma_es_escape` — independent global-search layer.
- :doc:`/advanced/anti_degeneracy` — what fires inside each start.
- :doc:`/advanced/memory_routing` — how multistart shares the memory
  router with single-start fits.
