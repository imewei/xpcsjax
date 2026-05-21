xpcsjax — JAX-native XPCS NLSQ fitting
======================================

.. only:: html

   .. image:: https://img.shields.io/badge/JAX-native-2EA44F
      :alt: JAX-native

   .. image:: https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white
      :alt: Python 3.12+

   .. image:: https://img.shields.io/badge/license-see%20LICENSE-informational
      :alt: License

**xpcsjax** is a unified JAX-native package for non-linear least-squares (NLSQ)
fitting of X-ray Photon Correlation Spectroscopy (XPCS) data. v0.1 consolidates
the homodyne and heterodyne NLSQ pipelines from the upstream ``homodyne`` and
``heterodyne`` packages into a single JAX-first codebase with a small,
lazy-loaded public API.

.. admonition:: Scope of v0.1
   :class: scope-warning

   xpcsjax is **NLSQ-only by design**. Bayesian sampling — NumPyro, BlackJAX,
   ArviZ, CMC (Consensus Monte Carlo), NUTS, HMC, parallel tempering — is
   **out of scope** for this package and will not be added. Users needing
   Bayesian XPCS analysis should use the upstream ``homodyne`` or
   ``heterodyne`` packages.

A 30-second tour
----------------

.. code-block:: python

   from xpcsjax import load_xpcs_data, fit_nlsq

   data   = load_xpcs_data("config.yaml")
   result = fit_nlsq(data, "config.yaml")

   print(result.parameters)
   print(result.reduced_chi_squared)

The same two-function workflow drives both homodyne and heterodyne modes; the
:class:`~xpcsjax.config.ConfigManager` decides which physics model and which
NLSQ strategy to use, based on the ``analysis_mode`` field in your YAML.

Where to go next
----------------

.. grid:: 2
   :gutter: 2
   :class-container: feature-grid

   .. grid-item-card:: Install & first fit
      :link: quickstart
      :link-type: doc

      Set up a uv-managed environment and run the homodyne quickstart against
      the bundled example config.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      Data loading, analysis modes, NLSQ strategies, and how to read an
      :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.

   .. grid-item-card:: Theory
      :link: theory/index
      :link-type: doc

      Homodyne and heterodyne models, anti-degeneracy, transport-coefficient
      formalism, and the rationale behind the 5-layer defence system.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      The seven lazy public symbols, plus the submodule surface that backs
      them. Auto-generated from the live source.


Public API at a glance
----------------------

The package exposes seven symbols, all lazy-loaded via a module-level
``__getattr__``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Purpose
   * - :func:`xpcsjax.data.xpcs_loader.load_xpcs_data`
     - Read XPCS HDF5 + YAML config into a homodyne/heterodyne-ready dict.
   * - :func:`xpcsjax.optimization.nlsq.fit_nlsq`
     - Run the NLSQ fit. Dispatches to homodyne or heterodyne path based on
       ``analysis_mode``.
   * - :class:`xpcsjax.config.ConfigManager`
     - Load, validate, and query a YAML config; canonical source of bounds
       and active-parameter lists.
   * - :class:`xpcsjax.core.HomodyneModel`
     - Hybrid stateful + JIT model for static / laminar-flow homodyne XPCS.
   * - :class:`xpcsjax.core.HeterodyneModel`
     - Two-component (sample + reference) heterodyne model with 14 physics
       parameters.
   * - :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`
     - Dataclass holding fitted parameters, covariance, χ², diagnostics, and
       quality flags.
   * - :func:`xpcsjax.viz.nlsq_plots.generate_nlsq_plots`
     - Generate diagnostic plots and serialize NPZ + JSON artifacts for every
       φ angle after fitting. Supports homodyne and heterodyne (individual
       per-angle mode). Optional Datashader fast path + parallel rendering.

See :doc:`api/public` for the full public surface and :doc:`api/index` for
the per-submodule autodoc reference.


Documentation map
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced topics

   advanced/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
