Testing
=======

xpcsjax tests are sharded by domain rather than by pyramid layer
(unit / integration / e2e). Each shard corresponds to a top-level
directory under :file:`tests/` and a dedicated Makefile target. This
page describes the layout, the per-shard ``make test-*`` invocations,
the smoke and verify gates, and the JAX environment that pytest
configures automatically.

Test layout
-----------

The :file:`tests/` directory is sharded by domain. The most-used shards
(the full tree also has ``cli/``, ``config/``, ``data/``, ``gui/``,
``integration/``, ``parity/``, ``runtime/``, ``service/``, ``viz/`` and
several top-level ``test_*.py`` modules) are:

.. code-block:: text

   tests/
   |- benchmarks/         # opt-in performance regression suite
   |- characterization/   # residual-level synthetic parity checks
   |- core/               # physics models (homodyne + heterodyne)
   |- heterodyne/         # heterodyne end-to-end fits
   |- optimization/       # NLSQ engine, anti-degeneracy, CMA-ES
   |- property/           # Hypothesis property-based invariants
   |- ...                 # cli/, config/, data/, gui/, integration/, etc.
   |- test_lazy_imports.py

Each shard owns a specific class of guarantee:

``tests/core/``
    Physics-model unit tests. Covers :class:`xpcsjax.core.HomodyneModel`,
    the heterodyne model variants, and the kernel functions in
    :mod:`xpcsjax.core.physics` and :mod:`xpcsjax.core.heterodyne_physics_kernel`.

``tests/optimization/``
    NLSQ engine tests. Covers
    :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`, the
    5-layer anti-degeneracy controller, the CMA-ES escape trigger,
    Jacobian utilities, memory routing, and the streaming-strategy
    smoke path. This is the largest shard and changes most often.

``tests/heterodyne/``
    Heterodyne end-to-end fits on tiny synthetic data
    (:file:`test_two_component_smoke.py`) plus the config-unwrap path.

``tests/characterization/``
    Residual-level parity checks against committed synthetic baselines
    (:file:`test_heterodyne_residual_parity.py`). See
    :doc:`porting_notes` for the parity coverage overview.

``tests/property/``
    `Hypothesis <https://hypothesis.readthedocs.io/>`_ property-based
    tests. Covers diagonal-correction invariants and cross-cutting
    parameter-registry properties (monotonicity, clipping, bounds
    handling).

``tests/benchmarks/``
    Wall-clock regression suite. **Opt-in** via the
    ``XPCSJAX_RUN_BENCHMARKS=1`` environment variable; not part of the
    default test runs. See :ref:`testing-benchmarks` below.

``tests/test_lazy_imports.py``
    Top-level smoke check that the public API in
    :mod:`xpcsjax` resolves correctly through its lazy
    ``__getattr__`` mechanism. ``HeterodyneModel`` is exercised here as a
    public lazy export (Phase 6 complete), not ``xfail``-marked; see
    :doc:`porting_notes`.

``tests/test_docs_structure.py``
    Structural check that every top-level ``xpcsjax`` submodule has a
    matching :file:`docs/source/api/{name}.rst` page (page existence
    only, not content — adding a new top-level package without a page
    fails this test). See
    ``docs/adr/0001-automated-structural-doc-coverage-check.md`` for why
    symbol-level and content checks are deliberately *not* automated
    here.

Running the test shards
-----------------------

The Makefile exposes one target per shard plus a small set of
aggregate targets:

.. code-block:: shell

   make test-core              # tests/core
   make test-optimization      # tests/optimization (alias: make test-nlsq)
   make test-heterodyne        # tests/heterodyne
   make test-characterization  # tests/characterization
   make test-property          # tests/property
   make test-viz               # tests/viz (pytest-mpl snapshot comparison)

Each shard target is a thin wrapper around:

.. code-block:: shell

   uv run pytest tests/<shard> -v --tb=short

``make test-viz`` is the one exception: it runs
``uv run pytest tests/viz -v --mpl`` instead, since the shard compares
rendered plots against committed baseline images via
`pytest-mpl <https://github.com/matplotlib/pytest-mpl>`_ rather than
asserting on return values.

You can pass extra pytest options on the command line instead. To run
a single file:

.. code-block:: shell

   uv run pytest tests/optimization/test_anti_degeneracy_layers.py -v

To run a single test by node id:

.. code-block:: shell

   uv run pytest tests/optimization/test_jacobian.py::test_some_name -v

To run everything (sequentially):

.. code-block:: shell

   make test

To run everything in parallel (recommended on multi-core machines):

.. code-block:: shell

   make test-parallel          # tests/ with -n auto
   make test-parallel-fast     # tests/ with -n auto and -m "not slow"

Smoke and verify gates
----------------------

Two targets exist for the day-to-day inner loop and the pre-push
checkpoint.

``make test-smoke``
    Runs the full ``tests/`` tree in parallel under ``-x`` (fail
    fast) and ``-q`` (quiet output), with the heavy/flaky nodes listed
    in the Makefile's ``HEAVY_NODES`` deselected (currently a CMA-ES
    escape test and a GUI worker-handle test). Equivalent to:

    .. code-block:: shell

       uv run pytest tests -n auto -v --tb=short -x -q \
         --deselect "tests/optimization/test_heterodyne_joint_escapes.py::test_individual_cmaes_escape_returns_scaling_first" \
         --deselect "tests/gui/test_worker_handle.py::test_handle_synthesizes_died_on_abnormal_exit"

    Run ``make test-smoke`` directly rather than copying this by hand —
    the Makefile's ``HEAVY_NODES``/``PARALLEL_DESELECT`` variables are the
    source of truth and this list drifts if either changes. This is the
    same command ``make verify`` uses for its test step.

``make verify``
    The pre-push gate. Runs three steps in order:

    1. ``ruff check`` on ``xpcsjax/`` and ``tests/`` — hard fail.
    2. ``mypy xpcsjax/`` in **advisory** mode — output truncated to
       the summary line; does not block the gate.
    3. ``make test-smoke`` — hard fail.

    Use ``make verify-fast`` to skip the smoke test step (lint +
    advisory mypy only).

``make quick``
    Format + smoke, for iterative work. Equivalent to
    ``make format && make test-smoke``.

``make test-quick``
    Fastest variant: ``pytest tests -v -x --tb=no -q``. Useful for
    seeing only failures while iterating.

JAX environment for tests
-------------------------

The pytest invocation does not require any manual environment setup.
:file:`pyproject.toml` configures the JAX runtime through pytest's
``env`` block:

.. code-block:: toml

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   env = [
       "JAX_ENABLE_X64=1",
   ]

This guarantees float64 arithmetic for every test, matching the
production code path (xpcsjax sets the same flag in
:mod:`xpcsjax` before any JAX import).

.. note::

   The XLA flags that :mod:`xpcsjax` sets — including the
   host platform device count and the constant-folding disable —
   are inherited by tests automatically because the test process
   imports :mod:`xpcsjax` (directly or transitively).

Coverage
--------

Coverage is computed against the unit-runnable surface only. Engine
orchestration files that are only reachable through large end-to-end
fits are excluded in :file:`pyproject.toml` under
``[tool.coverage.run].omit``:

.. code-block:: toml

   omit = [
       "xpcsjax/optimization/nlsq/core.py",
       "xpcsjax/optimization/nlsq/wrapper.py",
       "xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py",
       "xpcsjax/optimization/nlsq/strategies/out_of_core.py",
       "xpcsjax/optimization/nlsq/strategies/sequential.py",
       "xpcsjax/optimization/nlsq/strategies/stratified_ls.py",
       "xpcsjax/optimization/nlsq/recovery.py",
       "xpcsjax/optimization/nlsq/progress.py",
       "xpcsjax/optimization/nlsq/result_builder.py",
       "xpcsjax/data/performance_engine.py",
       "xpcsjax/data/memory_manager.py",
       "xpcsjax/utils/path_validation.py",
   ]

The current v0.1 floor is 55 % across the non-engine-orchestration
surface (``fail_under = 55`` in :file:`pyproject.toml`, raised from 50
after a unit-test campaign). Run coverage locally with:

.. code-block:: shell

   make test-coverage           # serial, generates htmlcov/
   make test-coverage-parallel  # parallel, faster on multi-core

The HTML report lands in :file:`htmlcov/index.html`.

Characterisation tests
----------------------

The characterisation shard holds residual-level parity checks against
committed synthetic baselines:

.. code-block:: shell

   make test-characterization

It runs :file:`tests/characterization/test_heterodyne_residual_parity.py`,
which loads the committed :file:`heterodyne_residuals.json` baseline and
asserts that xpcsjax reproduces it.

.. note::

   The real-data / upstream parity oracles were removed from the
   repository, so there is nothing here that depends on the upstream
   ``homodyne`` package or an external dataset. The remaining
   engine-preservation and engine-route parity coverage lives under
   :file:`tests/parity/`; see :doc:`porting_notes`.

.. _testing-benchmarks:

Performance regression suite
----------------------------

The benchmark suite is opt-in. Set ``XPCSJAX_RUN_BENCHMARKS=1`` to
enable it. The Makefile provides two convenience targets:

.. code-block:: shell

   make perf-baseline   # pin current wall-clocks as the baseline
   make perf-compare    # run and fail on >25% mean regression

Results land in :file:`.benchmarks/`. The compare target fails the
build if any benchmark regresses by more than 25 % against the pinned
baseline.

Property-based tests
--------------------

The property shard uses Hypothesis to generate inputs and assert
cross-cutting invariants. Current coverage includes the diagonal
correction (:file:`tests/property/test_diagonal_correction.py`) and
the parameter registry (:file:`tests/property/test_parameter_invariants.py`).

When adding a new property test:

- Prefer narrow numeric strategies (``hypothesis.strategies.floats``
  with explicit ``min_value`` / ``max_value``) over the defaults.
  Unbounded float strategies hit NaN and ``inf`` paths the production
  code does not handle.
- Mark slow generators with ``@settings(deadline=None)`` only when the
  property genuinely requires expensive inputs; otherwise tighten the
  strategy.
