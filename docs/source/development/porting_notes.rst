Porting Notes
=============

xpcsjax is a port. Its two upstream sources are the ``homodyne`` and
``heterodyne`` Python packages, which xpcsjax merges into a single
JAX-native NLSQ pipeline. This page documents the relationship to
both upstreams, the parity contract that gates the port, the current
state of the heterodyne migration, what xpcsjax intentionally drops
from the upstreams, and the workflow to follow when porting a new
module.

Relationship to the upstream packages
-------------------------------------

``homodyne``
    The reference implementation for the homodyne XPCS model. xpcsjax
    consumes this package in two distinct ways:

    1. As a **port source**. The physics kernels, NLSQ engine
       structure, and anti-degeneracy controller in xpcsjax derive
       from homodyne's implementation.
    2. As a **parity oracle**. The characterisation test suite runs
       homodyne against a canonical fixture set, serialises the
       output, and asserts that xpcsjax reproduces it exactly.

``heterodyne``
    The reference implementation for the two-component heterodyne
    XPCS model. The port is complete (Phase 6):
    :class:`xpcsjax.core.HeterodyneModel` is a fully public model with
    per-angle-mode parity. Parity is guarded by the availability-gated
    real-data oracle (:file:`tests/heterodyne/test_two_component_real_data.py`,
    the C044 dataset) rather than a byte-exact characterisation fixture.

The dual role of homodyne — port source *and* parity oracle — is the
strongest correctness guarantee xpcsjax has. Any commit that breaks
parity is a bug in the port, by construction.

The parity contract
-------------------

The homodyne parity contract is encoded in a single file:
:file:`tests/characterization/test_homodyne_equivalence.py`. The
contract has three components:

1. **Tolerance.** Bit-comparable output at ``rtol=1e-10``. This is
   tight enough to catch every algorithmic drift the port can plausibly
   introduce while permitting last-bit float64 reductions ordering
   differences.
2. **Baselines.** Serialised under
   :file:`tests/characterization/fixtures/`. Each baseline is the
   output of running the upstream ``homodyne`` package against a
   canonical fixture set. The script that regenerates them is
   :file:`scripts/generate_homodyne_baselines.py`.
3. **Gating.** End-to-end paths are env-gated on
   ``XPCSJAX_RUN_CHARACTERIZATION=1`` so local smoke runs stay fast.
   CI sets the env var; local pre-push runs do not, by default.

Regeneration discipline
~~~~~~~~~~~~~~~~~~~~~~~

There is exactly one situation in which baselines should be
regenerated: **the upstream ``homodyne`` package itself changed**.
Concretely:

- A homodyne release fixed a bug whose effect is observable in the
  fixture set.
- A homodyne release deliberately changed an algorithm (for example,
  a new Jacobian scaling) and you have explicit confirmation from
  the upstream maintainers that the change is intentional.

In every other situation — a failing characterisation test means
xpcsjax has a regression. Fix the xpcsjax code; do not regenerate
the baseline.

To regenerate when the situation genuinely calls for it:

.. code-block:: shell

   make run-example
   # which is equivalent to:
   uv run python scripts/generate_homodyne_baselines.py

Then run the characterisation gate to confirm the fresh baselines
match the (unchanged) xpcsjax implementation:

.. code-block:: shell

   XPCSJAX_RUN_CHARACTERIZATION=1 make test-characterization

Commit the regenerated fixtures and the corresponding upstream version
bump in a single commit, with a message that names the upstream
release that triggered the regeneration.

.. warning::

   Do not regenerate baselines to make a failing parity test pass.
   The contract is one-directional: xpcsjax tracks homodyne, not the
   other way around. A drift detected by the gate is a port bug, and
   the correct response is to fix the port.

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
on real data (:file:`test_two_component_real_data.py`), the smoke
variant (:file:`test_two_component_smoke.py`), and the config
unwrap path (:file:`test_config_unwrap.py`).

How heterodyne parity is guarded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heterodyne fits a different model from homodyne, so it is verified by
**mechanism + objective** parity rather than the byte-exact
``rtol=1e-10`` characterisation gate used for homodyne:

1. The availability-gated real-data oracle
   :file:`tests/heterodyne/test_two_component_real_data.py` runs the
   two-component fit against the C044 dataset whenever that data is
   present and skips cleanly otherwise.
2. The smoke variant :file:`tests/heterodyne/test_two_component_smoke.py`
   exercises the pipeline on tiny synthetic data with no external
   dependencies.
3. Per-angle-mode parity (``constant`` / ``averaged`` / ``individual``)
   is asserted by the no-worse-SSR contracts described in
   :doc:`/advanced/parity_testing`.

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

1. **Generate a fresh baseline first, before changing xpcsjax.**

   If the new module participates in an end-to-end path covered by
   the characterisation harness, regenerate baselines from the
   current upstream release before you touch xpcsjax code:

   .. code-block:: shell

      make run-example

   This pins the parity target. Without this step you cannot
   distinguish "xpcsjax has a port bug" from "upstream changed".

2. **Write the parity test next.**

   Add the corresponding test under :file:`tests/characterization/`
   if it doesn't already exist. The test should load the baseline
   and assert ``rtol=1e-10`` against the xpcsjax output.

3. **Port the code.**

   Place physics in :mod:`xpcsjax.core`; place engine code in
   :mod:`xpcsjax.optimization.nlsq`. Reuse the existing wiring
   (anti-degeneracy controller, CMA-ES escape, multistart, memory
   routing) rather than reimplementing it. See :doc:`nlsq_integration`
   for the ownership split.

4. **Run the gate.**

   .. code-block:: shell

      XPCSJAX_RUN_CHARACTERIZATION=1 make test-characterization

   Iterate on the xpcsjax code until the test passes at
   ``rtol=1e-10``. Do not loosen the tolerance.

5. **Run the full pre-push gate.**

   .. code-block:: shell

      make verify

   This catches lint and smoke regressions outside the parity
   contract.

6. **Update documentation.**

   If the new module exposes a public symbol, add it to
   ``_LAZY_EXPORTS`` and the literal ``__all__`` in
   :mod:`xpcsjax`, and document it under :doc:`/api/index`.
   If the module is part of the heterodyne push, update the status
   list above.

.. note::

   The "baseline first, test second, port third" order is what
   converts the port from an open-ended translation exercise into a
   bounded one. Without the baseline in place, there is no failing
   test to drive the port, and regressions accumulate silently.
