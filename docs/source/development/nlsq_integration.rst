NLSQ Integration
================

xpcsjax depends on the upstream ``nlsq>=0.6.10`` package for its
trust-region solver core and JIT-cache infrastructure, and provides
its own routing, anti-degeneracy, escape, multistart, stratification,
and weighting layers on top. This page describes the ownership split,
the APIs that have been removed upstream and must not be reintroduced
locally, and the procedure for bumping the NLSQ floor.

The contract
------------

The dependency pin lives in :file:`pyproject.toml`:

.. code-block:: text

   dependencies = [
       ...
       "nlsq>=0.6.10",     # JAX-native trf + CMA-ES + memory routing
       "evosax>=0.2.0",    # CMA-ES JAX backend (BIPOP restart)
       ...
   ]

NLSQ's role is narrow: it provides the trust-region (Levenberg-
Marquardt) solver and the surrounding JIT machinery. Everything
above the solver — strategy selection, anti-degeneracy, escape
recovery, multistart, stratification, weighting, bounds — lives in
xpcsjax.

Ownership split
---------------

The split is intentional and tight. Treat it as a contract, not a
suggestion.

NLSQ provides
~~~~~~~~~~~~~

``nlsq.CurveFit``
    The JIT-cached entry point that compiles the residual function
    once and reuses it across calls. xpcsjax instantiates
    ``CurveFit`` directly and feeds it the residual and Jacobian
    closures.

``nlsq.curve_fit``
    The functional wrapper around ``CurveFit`` that matches SciPy's
    ``curve_fit`` signature. xpcsjax uses this when the caller does
    not need to keep a long-lived ``CurveFit`` instance.

The trust-region (TRF) solve
    Levenberg-Marquardt with a trust-region radius update, implemented
    on top of JAX. This is the algorithmic core; xpcsjax does not
    re-implement it.

``nlsq.OptimizationGoal``
    The three-tier preset: ``FAST`` (loose tolerances, few iterations),
    ``ROBUST`` (default), and ``QUALITY`` (tight tolerances). xpcsjax
    selects the goal based on the routed strategy.

xpcsjax provides
~~~~~~~~~~~~~~~~

:func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`
    The memory-aware strategy router. Inspects available system RAM
    (via ``psutil``), the dataset shape, and the chunk-budget config
    to choose between ``DIRECT``, ``STRATIFIED_LS``, ``SEQUENTIAL``,
    ``OUT_OF_CORE``, and ``HYBRID_STREAMING``. NLSQ's own memory
    selector is not used (see below).

The 5-layer anti-degeneracy controller
    Implemented in :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`.
    Detects and breaks degenerate fitting regimes — flat residual
    landscapes, parameter collinearity, near-zero Jacobian columns —
    before they collapse the trust-region step. Each layer addresses
    a different failure mode; the controller composes them.

The CMA-ES escape
    Implemented in :mod:`xpcsjax.optimization.nlsq.cmaes_wrapper`.
    Auto-triggered above a configurable degeneracy threshold;
    delegates to the ``evosax`` BIPOP-CMA-ES backend for global
    search, then hands the best point back to the trust-region solve
    for local refinement.

LHS multistart
    Implemented in :mod:`xpcsjax.optimization.nlsq.multistart`.
    Latin hypercube sampling over the parameter prior to spawn
    independent NLSQ fits, retaining the best by final cost.

Angle-stratified chunking
    Implemented in :mod:`xpcsjax.optimization.nlsq.strategies.stratified_ls`.
    For large datasets, partition the data by ``phi`` angle so that
    each chunk fits in memory and the strata are statistically
    representative.

Shear weighting
    Implemented in :mod:`xpcsjax.optimization.nlsq.shear_weighting`.
    Per-residual weights that compensate for the shear-dependent
    noise structure observed in real XPCS data.

Bounds and parameter transforms
    Implemented in :mod:`xpcsjax.optimization.nlsq.transforms`. Maps
    physically bounded parameters (positive diffusion coefficients,
    angles modulo :math:`2\pi`, etc.) to the unbounded space the
    trust-region solver operates on.

Why xpcsjax routes memory itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xpcsjax's strategy router has access to information NLSQ's generic
budget selector does not — specifically, the phi-stratified chunking
plan, the heterodyne two-component structure, and the CMA-ES escape
budget. Centralising the routing in
:func:`xpcsjax.optimization.nlsq.select_nlsq_strategy` keeps that
plan in one place. Calling NLSQ's memory selector in addition would
make two independent decisions, and the second one would always be
inferior because it lacks the higher-level context.

Removed upstream APIs to avoid
------------------------------

The following NLSQ APIs were either removed in v0.6.0 or are
superseded by xpcsjax's own layers. They must not appear in new
xpcsjax code.

``nlsq.WorkflowSelector``
    **Removed in NLSQ v0.6.0.** The upstream package no longer exposes
    a unified workflow selector; consumers are expected to drive
    ``CurveFit`` directly with the strategy of their choice. xpcsjax
    drives ``CurveFit`` from
    :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy` and the
    per-strategy modules under
    :mod:`xpcsjax.optimization.nlsq.strategies`.

    If you find a reference to ``WorkflowSelector`` in xpcsjax code,
    it is a port artefact and should be deleted.

``nlsq.MemoryBudgetSelector``
    **Not used.** xpcsjax routes memory itself via
    :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`; calling
    NLSQ's budget selector would compete with the xpcsjax router.

``nlsq.fit``
    **Not used.** This is NLSQ's high-level unified fit API. xpcsjax
    calls the lower-level ``CurveFit`` directly so that the strategy,
    bounds, transforms, and weighting hooks all land in the right
    place. Do not call ``nlsq.fit`` from xpcsjax.

.. warning::

   These three names are tripwires. If a code review surfaces any
   of them in new xpcsjax code, the patch must be reworked to use
   the xpcsjax-owned equivalents listed above.

Calling convention inside the engine
------------------------------------

When working inside :mod:`xpcsjax.optimization.nlsq`, the convention
is:

1. Build the residual and Jacobian closures from the xpcsjax model
   (homodyne or heterodyne).
2. Apply bounds and parameter transforms via
   :mod:`xpcsjax.optimization.nlsq.transforms` so the trust-region
   solver sees an unbounded parameter vector.
3. Apply shear weights via
   :mod:`xpcsjax.optimization.nlsq.shear_weighting`.
4. Hand the closures to ``nlsq.CurveFit`` with the
   ``OptimizationGoal`` chosen by the strategy router.
5. Wrap the result with the xpcsjax
   :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`
   adapter so the public API does not leak NLSQ types.

The wrapper layer is what keeps NLSQ a swappable dependency. xpcsjax
public types do not transitively expose NLSQ; the adapter modules
under :mod:`xpcsjax.optimization.nlsq` are the only place NLSQ types
appear in xpcsjax's import graph.

Bumping the NLSQ floor
----------------------

When a new ``nlsq`` release lands, the following procedure raises the
floor in :file:`pyproject.toml` safely.

1. **Inspect the upstream changelog.**

   Pay particular attention to:

   - Removed or renamed APIs (especially anything around the
     workflow selector, memory selector, or ``CurveFit`` signature).
   - Changes to the trust-region default tolerances.
   - Changes to the ``OptimizationGoal`` presets.

2. **Edit the dependency pin.**

   In :file:`pyproject.toml`:

   .. code-block:: text

      dependencies = [
          ...
          "nlsq>=0.6.NN",   # bumped from 0.6.10
          ...
      ]

   Re-resolve the lockfile:

   .. code-block:: shell

      uv sync

3. **Run the integration smoke check.**

   .. code-block:: shell

      make verify-nlsq

   This confirms that :func:`xpcsjax.optimization.nlsq.fit_nlsq`,
   ``nlsq``, and ``evosax`` all import cleanly. A failure here
   typically means the new ``nlsq`` release renamed or moved a
   symbol xpcsjax depends on.

4. **Run the pre-push gate.**

   .. code-block:: shell

      make verify

   Lint, advisory mypy, and the smoke test suite. If any
   optimization-shard test fails, the bump introduced a behaviour
   change that needs to be either absorbed (by updating xpcsjax) or
   reverted.

5. **Run the characterisation gate.**

   .. code-block:: shell

      XPCSJAX_RUN_CHARACTERIZATION=1 make test-characterization

   This is the strongest check available short of running the full
   characterisation matrix. If a parity test starts failing at
   ``rtol=1e-10``, the NLSQ release changed solver behaviour in a way
   that drifts xpcsjax off the homodyne baseline. Investigate before
   merging the bump; do not loosen the tolerance.

6. **Update the changelog.**

   Note the floor bump in the project changelog with the upstream
   release link and a one-line description of what changed.

.. important::

   The NLSQ floor only goes up, never down. If a bump introduces a
   parity regression, the correct response is to fix xpcsjax, not
   to roll back the floor.

Related modules and references
------------------------------

The NLSQ-facing surface in xpcsjax lives entirely under
:mod:`xpcsjax.optimization.nlsq`. The directory is split into:

- **Wiring**: :file:`__init__.py`, :file:`wrapper.py`, :file:`core.py`,
  :file:`config.py`, :file:`adapter.py`, :file:`adapter_base.py`.
- **Heterodyne wiring**: :file:`heterodyne_*.py` (see
  :doc:`porting_notes` for status).
- **Anti-degeneracy and recovery**:
  :file:`anti_degeneracy_controller.py`,
  :file:`adaptive_regularization.py`, :file:`fallback_chain.py`,
  :file:`gradient_monitor.py`, :file:`recovery.py`.
- **Escape**: :file:`cmaes_wrapper.py`.
- **Multistart and stratification**: :file:`multistart.py`,
  :file:`hierarchical.py`, :file:`strategies/`.
- **Parameter handling**: :file:`transforms.py`, :file:`fourier_reparam.py`,
  :file:`parameter_index_mapper.py`, :file:`parameter_utils.py`.
- **Numerics support**: :file:`jacobian.py`, :file:`fit_computation.py`,
  :file:`parallel_accumulator.py`, :file:`memory.py`,
  :file:`data_prep.py`, :file:`shear_weighting.py`.
- **Results**: :file:`results.py`, :file:`result_builder.py`,
  :file:`progress.py`.

See also:

- :doc:`/advanced/architecture` for the cross-module data flow.
- :doc:`porting_notes` for how the heterodyne adapter modules slot
  into the engine.
- :doc:`testing` for the optimization and characterisation test
  shards that gate this code.
