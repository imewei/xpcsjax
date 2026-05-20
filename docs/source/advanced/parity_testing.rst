Parity Testing Against Upstream Homodyne
========================================

xpcsjax v0.1 is a port of the upstream ``homodyne`` and
``heterodyne`` Python packages. The port is held to a strict
bit-equivalence contract: on the same configuration and data,
xpcsjax must produce results that match the upstream ``homodyne``
package within a relative tolerance of ``1e-10``. This is the
**parity oracle** that catches regressions.

The two halves of the contract
------------------------------

The baseline generator
~~~~~~~~~~~~~~~~~~~~~~

``scripts/generate_homodyne_baselines.py`` is a CLI that:

1. Imports the upstream ``homodyne`` package.
2. Runs the upstream NLSQ pipeline on a curated set of configs that
   cover the supported analysis modes.
3. Serialises each result (best-fit parameters, covariance,
   chi-squared, c2 prediction) under
   ``tests/characterization/fixtures/`` using a deterministic file
   layout keyed by the config hash.

Run it via:

.. code-block:: shell

    make run-example

or equivalently:

.. code-block:: shell

    uv run python scripts/generate_homodyne_baselines.py

.. note::

   The script is idempotent: re-running it overwrites the fixtures
   only if the upstream homodyne output changes. The Git diff on the
   ``tests/characterization/fixtures/`` tree is the canonical signal
   that a regeneration moved a baseline.

The characterisation test
~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/characterization/test_homodyne_equivalence.py`` loads each
baseline, re-runs the same config through xpcsjax, and asserts:

.. code-block:: python

    np.testing.assert_allclose(
        xpcsjax_params, homodyne_params, rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        xpcsjax_chi2, homodyne_chi2, rtol=1e-10,
    )
    np.testing.assert_allclose(
        xpcsjax_c2_pred, homodyne_c2_pred, rtol=1e-10,
    )

The tolerance is deliberately tight. Any divergence above ``1e-10``
indicates a real algorithmic change, not numerical noise — float64
arithmetic on identical inputs is reproducible to ``1e-15`` in
practice, and the ``1e-10`` margin only covers the JIT cache and
sum-reduction ordering differences.

Running parity tests
--------------------

Parity tests are env-gated to keep CI fast. They run only when:

.. code-block:: shell

    XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest tests/characterization/

Or, equivalently, via the Makefile:

.. code-block:: shell

    XPCSJAX_RUN_CHARACTERIZATION=1 make test-characterization

The default ``make test`` and ``make verify`` do **not** run the
characterisation suite — they would otherwise require the upstream
``homodyne`` package installed in the dev environment.

.. important::

   The env gate is intentional. Skipping the parity suite locally
   is fine for unrelated refactors. It is **not** fine for changes
   that touch any of the following:

   - :mod:`xpcsjax.core` physics kernels.
   - :mod:`xpcsjax.optimization.nlsq` adapter or
     anti-degeneracy controller.
   - :mod:`xpcsjax.config` parameter resolution.

   For those, set ``XPCSJAX_RUN_CHARACTERIZATION=1`` before pushing.

When a parity test fails
------------------------

A parity failure is a hard signal. There are two legitimate
responses, in order of priority:

1. **Identify the regression in xpcsjax.** The most common cause is
   an unintentional change in residual ordering (summation order
   matters at ``1e-10``), a misconfigured Fourier reparameterisation
   harmonic count, or an accidental float32 cast on an intermediate
   buffer. Fix the regression; the parity test passes; ship.

2. **Acknowledge an upstream change in ``homodyne``.** If the
   upstream package has been updated and its outputs have moved,
   the baselines must be regenerated. Run
   ``make run-example`` again, commit the new fixtures, and
   re-run the parity suite. The Git diff on the fixtures becomes
   the audit trail of "upstream changed by X".

What you must **not** do:

.. warning::

   Do not loosen the ``rtol``. The ``1e-10`` tolerance is the
   contract. Loosening it silently accepts algorithmic drift and
   destroys the value of the parity oracle. If a real change in
   numerical behaviour is justified, regenerate the baselines so
   the diff is visible in code review.

Heterodyne parity
-----------------

Heterodyne baselines are produced by the same script when the
upstream ``heterodyne`` package is installed in the environment.
The characterisation tests for two-component fits compare the
returned ``list[NLSQResult]`` entry-by-entry. The same ``1e-10``
``rtol`` applies, although the per-angle chi-squared may differ in
the least-significant digit because of the angle-stratified chunking
in :mod:`xpcsjax.optimization.nlsq.strategies` — this is captured
explicitly in the test with an ``atol=1e-12``.

Heterodyne parity is currently gated by the same
``XPCSJAX_RUN_CHARACTERIZATION=1`` flag and additionally by
``XPCSJAX_RUN_HETERODYNE=1`` to allow homodyne-only baseline runs
during the mid-port phase.

Workflow summary
----------------

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────┐
    │ scripts/generate_homodyne_baselines.py                       │
    │   ↓                                                          │
    │ tests/characterization/fixtures/   (committed to Git)        │
    │   ↓                                                          │
    │ tests/characterization/test_homodyne_equivalence.py          │
    │   ↓                                                          │
    │ XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest tests/...       │
    └──────────────────────────────────────────────────────────────┘

Cross-references
----------------

- :doc:`architecture` — what xpcsjax owns vs what upstream NLSQ
  owns; the boundary that parity tests pin down.
- :doc:`anti_degeneracy` — the layer most likely to perturb
  numerical output if reordered.
- :doc:`memory_routing` — the strategy router's effect on
  summation order, which is why ``atol=1e-12`` is used alongside
  ``rtol``.
