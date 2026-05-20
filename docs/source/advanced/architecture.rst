Architecture Overview
=====================

xpcsjax is a JAX-native NLSQ port of the upstream ``homodyne`` and
``heterodyne`` XPCS packages. The architecture is shaped by four
constraints that show up everywhere in the codebase:

1. **JAX must be configured before it is imported.** Environment
   variables set at the top of :mod:`xpcsjax` control float precision,
   XLA passes, and device-count emulation; if any code imports JAX
   before they are set, those settings cannot be undone.
2. **Importing the package must stay cheap.** JAX itself is slow to
   import. The six public symbols are lazy-loaded.
3. **NLSQ owns the trust-region solve; xpcsjax owns the strategy.**
   The upstream ``nlsq`` library provides ``CurveFit``; xpcsjax
   provides memory routing, anti-degeneracy, multistart, CMA-ES
   escape, and bounds/transforms.
4. **NLSQ-only by design.** There is no Bayesian sampling pathway in
   xpcsjax. Users who need posterior inference should use the upstream
   ``homodyne``/``heterodyne`` packages directly.

Subpackage layout
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Subpackage
     - Responsibility
   * - :mod:`xpcsjax.config`
     - YAML loading, parameter bounds, active-parameter resolution.
       Hosts :class:`~xpcsjax.config.ConfigManager`.
   * - :mod:`xpcsjax.core`
     - Physics models. :class:`~xpcsjax.core.HomodyneModel` and the
       heterodyne kernels under ``core/heterodyne_*``.
   * - :mod:`xpcsjax.data`
     - Data loading and validation;
       :func:`~xpcsjax.data.xpcs_loader.load_xpcs_data` and shape/dtype/NaN gates.
   * - :mod:`xpcsjax.optimization.nlsq`
     - The fitting engine: trust-region adapter, memory router,
       anti-degeneracy controller, CMA-ES wrapper, multistart, result
       builder.
   * - :mod:`xpcsjax.device`
     - JAX device discovery, CPU-only enforcement for v0.1,
       diagnostics for ``device_info`` on
       :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.
   * - :mod:`xpcsjax.io`
     - File-format adapters (NPZ, HDF5) sitting upstream of
       :mod:`xpcsjax.data`.
   * - :mod:`xpcsjax.utils`
     - Structured logging, small numeric helpers, no domain logic.

Lazy public API
---------------

:mod:`xpcsjax` exports exactly six names. They are listed in
``_LAZY_EXPORTS`` and resolved on first attribute access by a
module-level ``__getattr__``:

.. code-block:: python

    _LAZY_EXPORTS = {
        "load_xpcs_data":     "xpcsjax.data",
        "fit_nlsq":           "xpcsjax.optimization.nlsq",
        "ConfigManager":      "xpcsjax.config",
        "HomodyneModel":      "xpcsjax.core",
        "HeterodyneModel":    "xpcsjax.core",
        "OptimizationResult": "xpcsjax.optimization.nlsq.results",
    }

This is described in detail in :doc:`lazy_api`.

End-to-end call graph
---------------------

A typical homodyne fit walks the following path. Heterodyne is similar
but returns ``list[NLSQResult]`` and consults the
``optimization.nlsq`` sub-block of the YAML.

.. code-block:: text

    user
      │
      ▼
    xpcsjax.fit_nlsq(data, config)
      │
      ├─► xpcsjax.config.ConfigManager        (parameter bounds, active params)
      │
      ├─► xpcsjax.optimization.nlsq.select_nlsq_strategy
      │     │
      │     ├─ STANDARD          (in-memory full Jacobian)
      │     ├─ OUT_OF_CORE       (chunk-wise J^T J accumulation)
      │     └─ HYBRID_STREAMING  (L-BFGS warmup + streaming GN)
      │
      ├─► AntiDegeneracyController            (FourierReparam, Hierarchical,
      │                                        AdaptiveRegularizer,
      │                                        GradientCollapseMonitor,
      │                                        ShearSensitivityWeighting)
      │
      ├─► nlsq.CurveFit                       (trust-region LM solve)
      │     │
      │     └─[plateau]─► CMAESWrapper        (BIPOP restart escape)
      │
      └─► OptimizationResult                  (built by result_builder)

Each block has a dedicated advanced page:

- :doc:`jax_environment` for the env-var setup.
- :doc:`lazy_api` for the public-symbol lazy resolution.
- :doc:`memory_routing` for ``select_nlsq_strategy``.
- :doc:`anti_degeneracy` for the controller layers.
- :doc:`cma_es_escape` for the global-search fallback.
- :doc:`parity_testing` for the upstream-homodyne equivalence contract.

NLSQ engine split
-----------------

The boundary between xpcsjax and the upstream NLSQ library is fixed:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Owned by ``nlsq>=0.6.10``
     - Owned by xpcsjax
   * - ``CurveFit`` JIT cache
     - :func:`~xpcsjax.optimization.nlsq.select_nlsq_strategy`
   * - ``curve_fit()`` entry point
     - :class:`~xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`
   * - Trust-region Levenberg-Marquardt solve
     - :class:`~xpcsjax.optimization.nlsq.cmaes_wrapper.CMAESWrapper`
   * - Jacobian column scaling
     - LHS multistart via
       :func:`~xpcsjax.optimization.nlsq.core.fit_nlsq_multistart`
   * - Convergence criteria (``ftol``, ``xtol``, ``gtol``)
     - Bounds + parameter transforms in
       :mod:`xpcsjax.optimization.nlsq.transforms`
   * - —
     - Angle-stratified chunking and shear-weighting

.. note::

   ``WorkflowSelector`` was removed in NLSQ v0.6.0. xpcsjax calls
   ``CurveFit`` directly. Any documentation or code referring to
   ``WorkflowSelector`` predates the NLSQ v0.6 cut and should be
   updated.

.. warning::

   The upstream NLSQ library also exposes ``fit()`` and a
   ``MemoryBudgetSelector``. xpcsjax does not use them — xpcsjax
   routes memory itself (see :doc:`memory_routing`). Do not call them
   from within the xpcsjax codebase.

The graphify knowledge graph
----------------------------

A graphify-generated knowledge graph lives at ``graphify-out/`` in the
repository root. ``GRAPH_REPORT.md`` lists the community hubs (e.g.
"Performance Engine", "Config Manager", "Heterodyne Model"); the
wiki under ``graphify-out/wiki/`` provides per-symbol pages with the
extracted and inferred edges. For cross-module navigation prefer
``graphify query``, ``graphify path``, or ``graphify explain`` over
plain ``grep``.
