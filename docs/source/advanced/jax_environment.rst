JAX Environment Setup
=====================

xpcsjax configures JAX at package import time, before any JAX module
is loaded. The configuration lives at the top of
:mod:`xpcsjax` and is applied via the ``os.environ`` dictionary.
Changing any of these settings *after* JAX has been imported has no
effect; they must be in place before the first ``import jax``.

The three settings
------------------

JAX_ENABLE_X64
~~~~~~~~~~~~~~

.. code-block:: python

    os.environ["JAX_ENABLE_X64"] = "1"

XPCS fit parameters span six or more orders of magnitude (e.g.
``D0`` in the ``[1, 1e6]`` range, ``D_offset`` in the ``[0, 1e4]``
range, ``alpha`` near ``-1.5``). Float32 has only seven decimal
digits of precision, which is insufficient to express a Jacobian
that mixes those scales — the trust-region solve would lose
significant digits in column scaling and the reported uncertainties
would be wrong by orders of magnitude.

``JAX_ENABLE_X64=1`` forces float64 throughout. The cost is roughly
2× memory and a smaller compute hit on CPU. v0.1 is CPU-only so the
trade-off is straightforward.

.. note::

   ``pyproject.toml`` also sets ``JAX_ENABLE_X64=1`` under
   ``[tool.pytest.ini_options]``. Tests therefore see the same
   precision as a normal import; no further configuration is
   required.

XLA_FLAGS
~~~~~~~~~

.. code-block:: python

    os.environ["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=4 "
        "--xla_disable_hlo_passes=constant_folding"
    )

Two flags, each with a specific motivation.

``--xla_force_host_platform_device_count=4``
    Forces XLA to expose four logical CPU devices regardless of the
    physical core count. xpcsjax uses this for parallel-path
    decisions inside the multistart and parallel-accumulator code:
    when the strategy is ``OUT_OF_CORE`` (see
    :doc:`memory_routing`), J^T J accumulation can vmap across the
    four virtual devices. Setting this to less than four disables
    that parallelism.

``--xla_disable_hlo_passes=constant_folding``
    Disables XLA's constant-folding HLO pass. On the
    ``HYBRID_STREAMING`` strategy, the compiled function ingests an
    index array that is mathematically constant but practically
    enormous (over 23 million entries on large datasets). With
    constant folding enabled, XLA tries to inline the entire array
    into the compiled program, triggering JAX's "slow compile"
    warning and adding more than one second of compile latency per
    call. Disabling the pass shifts the index array to a runtime
    argument and removes the warning.

NLSQ_SKIP_GPU_CHECK
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    os.environ["NLSQ_SKIP_GPU_CHECK"] = "1"

The upstream NLSQ library probes for a GPU at import time. xpcsjax
v0.1 is CPU-only — GPU support is scheduled for v0.2 — and the GPU
probe slows package import by several hundred milliseconds even when
no GPU is present. ``NLSQ_SKIP_GPU_CHECK=1`` tells NLSQ to skip the
probe.

.. seealso::

   :doc:`/development/cpu_gpu_decision` records *why* v0.1 pins the CPU
   backend: the fit is compile-dominated, so a GPU (which only speeds the
   ~1.6 % warm numeric loop) buys almost nothing, and float64 erases its
   edge on consumer cards. The deferral is a workload decision, not a
   solver-rewrite barrier — the NLSQ trust-region solve is already pure JAX.

Overriding the defaults
-----------------------

xpcsjax sets each variable using::

    os.environ.setdefault("JAX_ENABLE_X64", "1")

(or the equivalent direct assignment). The ``setdefault`` form means
a value set in the calling shell takes precedence. For example, to
expose eight logical devices instead of four:

.. code-block:: shell

    export XLA_FLAGS="--xla_force_host_platform_device_count=8 \
                      --xla_disable_hlo_passes=constant_folding"
    uv run python my_fit_script.py

.. important::

   These variables must be set **before** the first ``import
   xpcsjax`` in the process. Once xpcsjax is imported, JAX itself is
   imported, and the JAX configuration is frozen. Setting the
   variables afterwards is silently ignored.

   This is why pytest reads ``JAX_ENABLE_X64`` from
   ``pyproject.toml`` rather than from a conftest fixture: by the
   time a fixture would run, the test module's ``import xpcsjax``
   has already executed.

Verifying the configuration
---------------------------

The simplest check is to inspect JAX's compile cache after import:

.. code-block:: python

    import jax
    import xpcsjax  # noqa: F401  -- triggers the env setup

    print(jax.config.jax_enable_x64)         # True
    print(jax.devices())                      # 4 CPU devices

If ``jax_enable_x64`` is ``False`` or ``len(jax.devices()) != 4``,
something in the environment imported JAX before xpcsjax got the
chance.

Adding more settings
--------------------

If you need to add a new environment variable (for example, an
``XLA_PYTHON_CLIENT_PREALLOCATE`` override when v0.2 introduces GPU
support), do so at the **top of** :mod:`xpcsjax`, before
any direct or transitive JAX import. Adding the same line in a
submodule will not work — the submodule is imported lazily, and by
then JAX is already initialised.
