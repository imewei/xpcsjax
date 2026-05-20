xpcsjax.core
============

Physics models and the JAX kernels that back them.

.. currentmodule:: xpcsjax.core

Homodyne model
--------------

.. autoclass:: xpcsjax.core.HomodyneModel
   :members:

The homodyne model uses a *hybrid* architecture — stateful storage of
configuration and pre-computed physics factors, with high-level methods that
delegate to JIT-compiled functional cores. ``HomodyneModel.__init__`` accepts
a config dict with ``analyzer_parameters.temporal/scattering/geometry`` keys
(matching the YAML schema).

The primary user-facing method, ``compute_c2``, takes a parameter vector and
an array of φ angles and returns the C2 correlation stack:

.. code-block:: python

   import numpy as np
   from xpcsjax import HomodyneModel

   model = HomodyneModel(config_dict)

   params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
   phi    = np.array([0, 30, 45, 60, 90])
   c2     = model.compute_c2(params, phi, contrast=0.5, offset=1.0)
   # c2.shape == (len(phi), n_time, n_time)

Heterodyne model
----------------

.. autoclass:: xpcsjax.core.HeterodyneModel
   :members:

The heterodyne model implements the two-component reference-plus-sample
formalism with 14 physics parameters. The fit path uses the stateful
variant at :mod:`xpcsjax.core.heterodyne_model_stateful`, constructed via
``HeterodyneModel.from_config`` — see the dispatch in
:func:`xpcsjax.optimization.nlsq.fit_nlsq`.

Underlying components
---------------------

.. autoclass:: xpcsjax.core.models.DiffusionModel

.. autoclass:: xpcsjax.core.models.CombinedModel

.. autoclass:: xpcsjax.core.models.PhysicsModelBase

.. autoclass:: xpcsjax.core.physics_factors.PhysicsFactors

.. autoclass:: xpcsjax.core.physics.ValidationResult

.. autoclass:: xpcsjax.core.model_mixins.BenchmarkingMixin

.. autoclass:: xpcsjax.core.model_mixins.GradientCapabilityMixin

.. autoclass:: xpcsjax.core.model_mixins.OptimizationRecommendationMixin

.. autofunction:: xpcsjax.core.heterodyne_jax_backend.compute_c2_heterodyne

.. autofunction:: xpcsjax.core.heterodyne_physics_kernel.compute_c2_unified

The :mod:`xpcsjax.core.heterodyne_jax_backend` module mirrors
:mod:`xpcsjax.core.jax_backend` for the two-component model. The
``HeterodyneModel`` orchestrates calls to the heterodyne backend the same
way ``HomodyneModel`` orchestrates calls to ``jax_backend``.
