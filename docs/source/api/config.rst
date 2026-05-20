xpcsjax.config
==============

Configuration loading, parameter registry, parameter manager, and physics
validators.

.. currentmodule:: xpcsjax.config

ConfigManager
-------------

.. autoclass:: xpcsjax.config.ConfigManager
   :members:

   The constructor accepts ``config_file`` (path string, default
   ``"xpcsjax_config.yaml"``) and optional ``config_override`` (a dict that
   is shallow-merged into the loaded YAML). Both YAML and JSON files are
   supported via auto-detection.

   Calling :func:`xpcsjax.optimization.nlsq.fit_nlsq` with a path-like argument constructs a
   ``ConfigManager`` for you. Pre-build one yourself when you need to
   override parts of the configuration in code:

   .. code-block:: python

      from xpcsjax import ConfigManager, fit_nlsq

      cfg = ConfigManager("config.yaml")
      cfg.config["optimization"]["nlsq"]["max_iterations"] = 500
      result = fit_nlsq(data, cfg)

Parameter registry and space
----------------------------

.. autoclass:: xpcsjax.config.parameter_registry.ParameterRegistry



.. autoclass:: xpcsjax.config.parameter_space.ParameterSpace



.. autoclass:: xpcsjax.config.parameter_space.PriorDistribution



.. autoclass:: xpcsjax.config.parameter_registry.ParameterInfo



.. autoclass:: xpcsjax.config.physics_validators.PhysicsViolation



.. autoclass:: xpcsjax.config.types.HomodyneConfig



.. autoclass:: xpcsjax.config.types.BoundDict



.. autoclass:: xpcsjax.config.types.ExperimentalDataConfig



.. autoclass:: xpcsjax.config.types.InitialParametersConfig



.. autoclass:: xpcsjax.config.types.OptimizationConfig



.. autoclass:: xpcsjax.config.types.ParameterSpaceConfig



.. autoclass:: xpcsjax.config.types.StreamingConfig



.. autoclass:: xpcsjax.config.types.StratificationConfig



.. autoclass:: xpcsjax.config.types.SequentialConfig



.. autoclass:: xpcsjax.config.types.NLSQValidationConfig



.. autoclass:: xpcsjax.config.parameter_manager.ParameterManager



Physics validators
------------------

The :mod:`xpcsjax.config.physics_validators` module enforces cross-parameter
constraints (e.g. positivity, bound consistency, mode-specific subset
requirements). The most-called entry points:

.. autofunction:: xpcsjax.config.physics_validators.validate_all_parameters
   :noindex:

.. autofunction:: xpcsjax.config.physics_validators.validate_cross_parameter_constraints
   :noindex:

These run automatically during ``ConfigManager.load_config``; they raise
``ValueError`` with the full failed-constraint list when something is wrong.

Heterodyne-specific configuration
---------------------------------

The heterodyne path uses a parallel parameter manager rooted at
:mod:`xpcsjax.config.heterodyne_parameter_manager`. It provides the 14
physics parameters and 2 scaling parameters specific to the two-component
model. Consult the module source for the registry layout.
