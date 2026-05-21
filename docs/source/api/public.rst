Public API
==========

The seven symbols exposed on the top-level :mod:`xpcsjax` namespace. All seven
are lazy-loaded: the package's ``__getattr__`` only resolves a name on first
access, which keeps the JAX import out of the path until you actually need
it.

.. currentmodule:: xpcsjax

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Symbol
     - Kind
     - Backing module
   * - :func:`xpcsjax.data.xpcs_loader.load_xpcs_data`
     - function
     - :mod:`xpcsjax.data`
   * - :func:`xpcsjax.optimization.nlsq.fit_nlsq`
     - function
     - :mod:`xpcsjax.optimization.nlsq`
   * - :class:`xpcsjax.config.ConfigManager`
     - class
     - :mod:`xpcsjax.config`
   * - :class:`xpcsjax.core.HomodyneModel`
     - class
     - :mod:`xpcsjax.core`
   * - :class:`xpcsjax.core.HeterodyneModel`
     - class
     - :mod:`xpcsjax.core`
   * - :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`
     - dataclass
     - :mod:`xpcsjax.optimization.nlsq.results`
   * - :func:`xpcsjax.viz.nlsq_plots.generate_nlsq_plots`
     - function
     - :mod:`xpcsjax.viz`

The mapping above is the verbatim ``_LAZY_EXPORTS`` table in
``xpcsjax/__init__.py``. Adding a public symbol means (a) adding to
``_LAZY_EXPORTS``, (b) adding to the literal ``__all__``, and (c) making sure
the target submodule actually exposes the name. A runtime ``assert`` catches
drift between (a) and (b).

Detailed documentation lives at the canonical module paths:

* :doc:`data` for :func:`xpcsjax.data.xpcs_loader.load_xpcs_data`.
* :doc:`optimization` for :func:`xpcsjax.optimization.nlsq.fit_nlsq` and the
  per-strategy submodules.
* :doc:`config` for :class:`xpcsjax.config.ConfigManager`, the parameter
  registry, and the validators.
* :doc:`core` for :class:`xpcsjax.core.HomodyneModel` and
  :class:`xpcsjax.core.HeterodyneModel`.
* :doc:`results` for :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`
  and the related result dataclasses.
* :doc:`viz` for :func:`xpcsjax.viz.nlsq_plots.generate_nlsq_plots`, the three
  low-level plot functions, :class:`xpcsjax.viz.diagnostics.DiagonalOverlayResult`,
  :func:`xpcsjax.viz.diagnostics.compute_diagonal_overlay_stats`, and the optional
  Datashader backend.
