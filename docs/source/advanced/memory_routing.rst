Memory Routing
==============

xpcsjax routes large-scale NLSQ fits across three optimization
strategies, chosen automatically by
:func:`~xpcsjax.optimization.nlsq.select_nlsq_strategy` based on the
estimated peak memory of the problem against the available system
RAM. The decision is encoded as an
:class:`~xpcsjax.optimization.nlsq.memory.NLSQStrategy` value carried
on a :class:`~xpcsjax.optimization.nlsq.memory.StrategyDecision`
dataclass.

The three strategies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Strategy
     - Description
     - Engages when
   * - ``NLSQStrategy.STANDARD``
     - In-memory full Jacobian. Standard NLSQ trust-region path.
     - ``peak_memory_gb <= threshold`` and the int64 index array
       fits.
   * - ``NLSQStrategy.OUT_OF_CORE``
     - Chunk-wise J^T J accumulation. Builds the normal equations in
       streamed blocks, never materialising the full Jacobian.
     - ``peak_memory_gb > threshold`` but the int64 index array
       still fits.
   * - ``NLSQStrategy.HYBRID_STREAMING``
     - L-BFGS warmup followed by a streaming Gauss-Newton refinement.
       Avoids ever materialising even an index array of fit-point
       row indices.
     - ``index_memory_gb > threshold`` — the extreme-scale regime
       where indices alone would not fit.

The string values are stable identifiers (``"standard"``,
``"out_of_core"``, ``"hybrid_streaming"``) and may be referenced
from configuration files where applicable.

The decision tree
-----------------

:func:`~xpcsjax.optimization.nlsq.select_nlsq_strategy` implements a
pure memory-based selector:

1. Compute ``index_memory_gb = 8 * n_points / 1024**3`` (int64
   indices, dataset-shape dependent).
2. Compute ``peak_memory_gb = estimate_peak_memory_gb(n_points,
   n_params)``. This estimates Jacobian + normal-equations matrices
   plus a small overhead.
3. Compute ``threshold_gb = memory_fraction *
   detect_total_system_memory()``. By default ``memory_fraction``
   is the package default; it can be overridden per call.
4. If ``index_memory_gb > threshold_gb`` → ``HYBRID_STREAMING``.
5. Elif ``peak_memory_gb > threshold_gb`` → ``OUT_OF_CORE``.
6. Else → ``STANDARD``.

The resulting :class:`xpcsjax.optimization.nlsq.StrategyDecision` carries the chosen strategy,
the threshold used, both memory estimates, and a human-readable
``reason`` string explaining why the branch was taken. That ``reason``
is surfaced on
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` via
``result.streaming_diagnostics`` and ``result.stratification_diagnostics``.

Implementation pieces
---------------------

:class:`~xpcsjax.optimization.nlsq.NLSQMemoryManager`
    The owner of the threshold computation and the strategy
    selector. One per process; obtained via
    :func:`~xpcsjax.optimization.nlsq.get_memory_manager`.

:func:`~xpcsjax.optimization.nlsq.detect_total_system_memory`
    Wraps ``psutil.virtual_memory().total`` with a fallback to a
    conservative default if ``psutil`` is unavailable. Returns
    bytes; helpers convert to GB.

:func:`~xpcsjax.optimization.nlsq.estimate_peak_memory_gb`
    Computes the expected peak working set for the full-Jacobian
    path. Inputs are ``n_points`` and ``n_params`` (the active
    parameters from
    :meth:`~xpcsjax.config.ConfigManager.get_active_parameters`).
    The estimate is the dominant term — Jacobian plus J^T J — with
    a small margin for the residual buffer.

Why xpcsjax routes memory itself
--------------------------------

The upstream NLSQ library exposes its own ``MemoryBudgetSelector``,
but xpcsjax does not call it. The reasons are concrete:

- xpcsjax knows the *shape* of an XPCS problem (number of angles,
  number of time-lag pairs, contrast/offset multipliers) and can
  compute ``n_points`` deterministically before any fit begins.
  NLSQ's selector treats the residual function as a black box.
- xpcsjax integrates the strategy decision with the angle-stratified
  chunking under :mod:`xpcsjax.optimization.nlsq.strategies` and the
  parallel accumulator paths. NLSQ's selector cannot reach into
  those layers.
- The ``HYBRID_STREAMING`` strategy is xpcsjax-specific: NLSQ has
  no equivalent for it, and forcing NLSQ's selector to act would
  inappropriately route extreme-scale problems to ``OUT_OF_CORE``.

.. note::

   The upstream NLSQ library still owns the actual trust-region
   solve once xpcsjax has assembled the (possibly streamed)
   Jacobian or normal equations. The split is **strategy** versus
   **solver**, not memory versus compute.

Tuning ``memory_fraction``
--------------------------

The default ``memory_fraction`` is clamped to
``[MIN_MEMORY_FRACTION, MAX_MEMORY_FRACTION]`` in
:mod:`xpcsjax.optimization.nlsq.memory`. Common adjustments:

- On a shared workstation, set a smaller fraction to leave room for
  other processes:

  .. code-block:: python

      from xpcsjax.optimization.nlsq import select_nlsq_strategy

      decision = select_nlsq_strategy(
          n_points=n_points,
          n_params=n_params,
          memory_fraction=0.25,
      )

- On a dedicated fitting node, a larger fraction keeps the
  ``STANDARD`` path active longer:

  .. code-block:: python

      decision = select_nlsq_strategy(
          n_points=n_points, n_params=n_params, memory_fraction=0.6,
      )

The clamps prevent denial-of-service either way: values below
``MIN_MEMORY_FRACTION`` would force every fit into
``HYBRID_STREAMING``; values above ``MAX_MEMORY_FRACTION`` would
risk OOM.

Inspecting the choice
---------------------

The chosen strategy is reported on the returned
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult`. The full audit lives at:

.. code-block:: python

    result.streaming_diagnostics       # streaming/HYBRID_STREAMING details
    result.stratification_diagnostics  # angle-stratified chunking details
    result.nlsq_diagnostics            # upstream NLSQ metadata

If a fit unexpectedly takes the ``HYBRID_STREAMING`` branch on a
small problem, the ``reason`` field in those diagnostics usually
points to a misconfigured ``memory_fraction`` or a process running
under another tool's memory limit.
