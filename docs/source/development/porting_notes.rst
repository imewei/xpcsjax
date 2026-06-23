Porting Notes
=============

xpcsjax is a port. Its two upstream sources are the ``homodyne`` and
``heterodyne`` Python packages, which xpcsjax merges into a single
JAX-native NLSQ pipeline. This page documents the relationship to
both upstreams, the parity coverage that gates the port, the current
state of the heterodyne migration, what xpcsjax intentionally drops
from the upstreams, and the workflow to follow when porting a new
module.

Relationship to the upstream packages
-------------------------------------

``homodyne``
    The reference implementation for the homodyne XPCS model. The
    physics kernels, NLSQ engine structure, and anti-degeneracy
    controller in xpcsjax derive from homodyne's implementation.

``heterodyne``
    The reference implementation for the two-component heterodyne
    XPCS model. The port is complete (Phase 6):
    :class:`xpcsjax.core.HeterodyneModel` is a fully public model with
    per-angle-mode parity.

The parity contract
-------------------

.. note::

   The real-data / upstream parity oracles were removed from the
   repository. xpcsjax no longer ships the generated homodyne
   baselines, the upstream-homodyne equivalence test, or the
   real-data heterodyne C044 oracle. The remaining parity coverage is
   the **synthetic golden / engine-preservation** tests under
   :file:`tests/parity/` — most importantly
   :file:`tests/parity/test_homodyne_engine_preservation.py` (golden
   tests in :file:`tests/parity/_golden/`),
   :file:`tests/parity/test_engine_heterodyne_fit_parity.py`, and
   :file:`tests/parity/test_engine_route_result_contract.py`. These run
   from data committed in the repository and need no external upstream
   package or dataset.

Heterodyne port status
----------------------

The heterodyne migration is complete (Phase 6).
:class:`xpcsjax.core.HeterodyneModel` is a public lazy export (it is in
``_LAZY_EXPORTS`` / ``__all__`` and exercised — not ``xfail``-marked —
by :file:`tests/test_lazy_imports.py`), and the two-component model has
full per-angle-mode parity with homodyne. The remaining heterodyne work
is architectural cleanup (procedural-parity convergence), not a missing
capability.

Heterodyne modules in xpcsjax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under :mod:`xpcsjax.core` (physics):

- :mod:`xpcsjax.core.heterodyne_jax_backend`
- :mod:`xpcsjax.core.heterodyne_model`
- :mod:`xpcsjax.core.heterodyne_model_stateful`
- :mod:`xpcsjax.core.heterodyne_models`
- :mod:`xpcsjax.core.heterodyne_physics_factors`
- :mod:`xpcsjax.core.heterodyne_physics_kernel`
- :mod:`xpcsjax.core.heterodyne_physics_utils`
- :mod:`xpcsjax.core.heterodyne_scaling_utils`

Under :mod:`xpcsjax.optimization.nlsq` (engine):

- :mod:`xpcsjax.optimization.nlsq.heterodyne_adapter`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_adapter_base`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_config`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_core`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_data_prep`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_memory`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_result_builder`
- :mod:`xpcsjax.optimization.nlsq.heterodyne_results`

Tests under :file:`tests/heterodyne/` cover the two-component model
via the smoke variant (:file:`test_two_component_smoke.py`) and the
config unwrap path (:file:`test_config_unwrap.py`).

How heterodyne parity is guarded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heterodyne fits a different model from homodyne, so it is verified by
**mechanism + objective** parity rather than a byte-exact
``rtol=1e-10`` gate:

1. The smoke variant :file:`tests/heterodyne/test_two_component_smoke.py`
   exercises the pipeline on tiny synthetic data with no external
   dependencies.
2. Engine-route parity on committed synthetic data is asserted by
   :file:`tests/parity/test_engine_heterodyne_fit_parity.py` and
   :file:`tests/parity/test_engine_route_result_contract.py`.
3. The ``auto → averaged`` default no-worse-SSR contract is asserted
   by the synthetic test in
   :file:`tests/parity/test_phase5_default_no_worse.py`.

The NLSQ-only filter: what xpcsjax intentionally omits
------------------------------------------------------

xpcsjax v0.1 is NLSQ-only by design. Several substantial subsystems
present in the upstream ``homodyne`` and ``heterodyne`` packages are
**intentionally absent** from xpcsjax. New contributors will trip
over their stale references if they don't know to expect this.

Intentionally absent
~~~~~~~~~~~~~~~~~~~~

The upstream packages provide a parallel sampling pipeline alongside
their NLSQ pipeline. xpcsjax keeps only the NLSQ side. Specifically,
the following are **out of scope** for v0.1 and should not be
reintroduced:

- The CMC (Consensus Monte Carlo) pipeline.
- NUTS and HMC samplers from NumPyro.
- Posterior-based diagnostics (R-hat, ESS, BFMI via ArviZ).
- Parallel tempering and any other replica-exchange sampler.
- BlackJAX samplers.

The homodyne port's CMC/MCMC machinery (``get_cmc_config``,
``_get_default_cmc_config``, and the ``"mcmc"`` config block) has
already been **removed** — those symbols no longer exist anywhere in
the package. What remains are a handful of **defensive guards** that
*name* Bayesian sampling only to reject it as out of scope (for example
the ``ValueError`` in :file:`xpcsjax/data/optimization.py` that rejects
non-NLSQ methods). Those guards reject invalid input; they are not dead
code, so keep them.

.. warning::

   - Do not add new call sites that introduce a CMC / MCMC pathway.
   - Do not write new tests that exercise a Bayesian path.
   - Keep the existing defensive guards that reject Bayesian methods.

Users who need Bayesian XPCS analysis should use the upstream
``homodyne`` or ``heterodyne`` packages directly; that capability is
permanently out of scope for xpcsjax.

Single optimisation pathway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is exactly one optimisation entry point in xpcsjax:
:func:`xpcsjax.optimization.nlsq.fit_nlsq`. Underneath it,
:func:`xpcsjax.optimization.nlsq.core.fit_nlsq_jax` and
:func:`xpcsjax.optimization.nlsq.core.fit_nlsq_multistart` are the two
internal variants; the strategy router
(:func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`) picks the
right one based on memory budget.

There is no second optimiser pathway to "fall back to". If a port
needs a new optimiser-level feature, it goes into the NLSQ engine; do
not introduce a parallel path (for example, calling SciPy's
``least_squares`` directly).

Porting workflow for new modules
--------------------------------

When porting a new module from upstream — homodyne or heterodyne —
follow this order:

1. **Write or extend a synthetic parity test first.**

   If the new module participates in an end-to-end path, add or
   extend a synthetic test under :file:`tests/parity/` that pins the
   expected behaviour on data committed in the repository (golden
   tests live in :file:`tests/parity/_golden/`). Without a failing
   test to drive the port, regressions accumulate silently.

2. **Port the code.**

   Place physics in :mod:`xpcsjax.core`; place engine code in
   :mod:`xpcsjax.optimization.nlsq`. Reuse the existing wiring
   (anti-degeneracy controller, CMA-ES escape, multistart, memory
   routing) rather than reimplementing it. See :doc:`nlsq_integration`
   for the ownership split.

3. **Run the full pre-push gate.**

   .. code-block:: shell

      make verify

   This catches lint and smoke regressions, and runs the synthetic
   parity tests under :file:`tests/parity/`.

4. **Update documentation.**

   If the new module exposes a public symbol, add it to
   ``_LAZY_EXPORTS`` and the literal ``__all__`` in
   :mod:`xpcsjax`, and document it under :doc:`/api/index`.
   If the module is part of the heterodyne push, update the status
   list above.
