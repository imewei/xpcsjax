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
    XPCS model. xpcsjax consumes this only as a port source for now;
    the heterodyne parity oracle will follow in Phase 6 (see below).

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

The heterodyne migration is mid-port. The public API gate currently
``xfail``-marks :class:`xpcsjax.core.HeterodyneModel` (see
:file:`tests/test_lazy_imports.py`). The physics and engine modules
are present, but the end-to-end Phase 6 parity gate has not yet
landed.

Heterodyne modules already in xpcsjax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under :mod:`xpcsjax.core` (physics):

- :mod:`xpcsjax.core.heterodyne_jax_backend`
- :mod:`xpcsjax.core.heterodyne_model`
- :mod:`xpcsjax.core.heterodyne_model_stateful`
- :mod:`xpcsjax.core.heterodyne_models`
- :mod:`xpcsjax.core.heterodyne_physics`
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

What still has to happen for Phase 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the heterodyne port reaches end-to-end parity:

1. Add a heterodyne characterisation harness alongside
   :file:`tests/characterization/test_homodyne_equivalence.py`,
   loading baselines from a new
   :file:`tests/characterization/fixtures/heterodyne/` directory.
2. Extend :file:`scripts/generate_homodyne_baselines.py` (or add a
   sibling) to drive the upstream ``heterodyne`` package end-to-end.
3. Flip the ``xfail`` marker in :file:`tests/test_lazy_imports.py`
   to ``xpass`` / passing.
4. Confirm the ``HeterodyneModel`` re-export from
   :mod:`xpcsjax.core` is exercised in the lazy-import test.
5. Update :doc:`/api/index` to drop the xfail caveat on
   :class:`xpcsjax.core.HeterodyneModel`.

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

Stale references survive in a handful of places from the port:

- :file:`xpcsjax/config/manager.py` — ``get_cmc_config``,
  ``_get_default_cmc_config``, and the ``"mcmc"`` config block.
- Scattered string literals in :mod:`xpcsjax.core`,
  :mod:`xpcsjax.data`, and :mod:`xpcsjax.utils.logging`.

Treat all of these as **scheduled-for-removal dead code**:

.. warning::

   - Do not add new call sites that depend on the stale CMC / MCMC
     entries.
   - Do not write new tests that exercise these code paths.
   - When you find a stale reference incidentally, remove it as part
     of the surrounding change rather than working around it.

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
