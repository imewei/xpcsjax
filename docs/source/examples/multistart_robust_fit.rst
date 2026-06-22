Robust Multi-start Fitting
==========================

.. currentmodule:: xpcsjax


Trust-region least-squares is locally quadratically convergent but
globally bound to the basin of attraction of its initial guess. For
the seven-parameter ``laminar_flow`` model and the heterodyne models,
non-convex landscapes are common enough that a single starting point
is unsafe. The
:func:`~xpcsjax.optimization.nlsq.core.fit_nlsq_multistart` wrapper draws
multiple starts via Latin Hypercube Sampling (LHS) and returns a
:class:`~xpcsjax.optimization.nlsq.multistart.MultiStartResult`
aggregating every start, with the winning run on its ``best`` field.

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

    ms_result = fit_nlsq_multistart(data, cm)
    best = ms_result.best                         # winning SingleStartResult
    result = ms_result.to_optimization_result()   # winner as an OptimizationResult

``fit_nlsq_multistart(data, config)`` takes the same data/config pair as
:func:`~xpcsjax.optimization.nlsq.fit_nlsq`. The number of LHS samples is
**not** a call argument — it is read from the configuration
(``optimization.nlsq.multi_start.n_starts``), and multistart must be
enabled there (``optimization.nlsq.multi_start.enable: true``) or the
call raises ``ValueError``. The return value is a
:class:`~xpcsjax.optimization.nlsq.multistart.MultiStartResult`; its
``best`` field is the winning start ("best" is decided by minimum
unweighted residual sum of squares on converged starts) and
``to_optimization_result()`` repacks that winner as an ordinary
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.

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

   ``fit_nlsq_multistart`` runs the starts **in parallel by default**:
   with the default config (``multi_start.n_workers: 0`` → auto =
   ``min(os.cpu_count(), n_starts)``) it dispatches each start to a
   spawn-context ``ProcessPoolExecutor``. It falls back to serial only
   for large datasets (> 500,000 points when ``n_workers > 1``) or when
   ``n_workers == 1``. Set ``multi_start.n_workers: 1`` to force serial
   execution and keep RAM pressure predictable.

Inspecting per-start outcomes
-----------------------------

The :class:`~xpcsjax.optimization.nlsq.multistart.MultiStartResult` keeps
every start, not just the winner. Iterate ``all_results`` to audit the
basin structure:

.. code-block:: python

    ms_result = fit_nlsq_multistart(data, cm)

    for i, r in enumerate(ms_result.all_results):
        marker = "*" if r is ms_result.best else " "
        print(
            f"{marker} start={i:2d}  "
            f"chi2_red={float(r.reduced_chi_squared): .4e}  "
            f"success={r.success}"
        )
    print(f"{ms_result.n_successful} converged, "
          f"{ms_result.n_unique_basins} distinct basins")

Each entry is a
:class:`~xpcsjax.optimization.nlsq.multistart.SingleStartResult`; the
aggregate counters (``n_successful``, ``n_unique_basins``,
``degeneracy_detected``) summarise the run.

Choosing n_starts
-----------------

Set ``optimization.nlsq.multi_start.n_starts`` in the config. A useful
default scaling is ``n_starts = 4 * d`` for ``d`` free parameters,
capped at ``32``. For the 7-parameter laminar-flow model this yields
``n_starts = 28``. For static isotropic (3 parameters) ``n_starts = 12``
is usually enough.

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
