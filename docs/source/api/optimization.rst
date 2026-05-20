xpcsjax.optimization.nlsq
=========================

The NLSQ optimisation subpackage. xpcsjax owns the *strategy* and the
*physics-aware controllers*; the upstream NLSQ library owns the trust-region
solve and the JIT cache. See :doc:`/development/nlsq_integration` for the
contract between the two.

.. currentmodule:: xpcsjax.optimization.nlsq

Public dispatch
---------------

.. autofunction:: xpcsjax.optimization.nlsq.fit_nlsq

   The single-entry public wrapper. Dispatches between homodyne and
   heterodyne paths based on ``config["analysis_mode"]``.

Lower-level fit functions
-------------------------

.. autofunction:: xpcsjax.optimization.nlsq.core.fit_nlsq_jax

.. autofunction:: xpcsjax.optimization.nlsq.core.fit_nlsq_multistart

.. autofunction:: xpcsjax.optimization.nlsq.core.fit_nlsq_cmaes

Strategy selection
------------------

The xpcsjax strategy selector decides which solver path to take based on
dataset size and available RAM — *not* the NLSQ default
``MemoryBudgetSelector``.

.. autofunction:: xpcsjax.optimization.nlsq.select_nlsq_strategy

.. autoclass:: xpcsjax.optimization.nlsq.NLSQStrategy



.. autoclass:: xpcsjax.optimization.nlsq.StrategyDecision



.. autofunction:: xpcsjax.optimization.nlsq.detect_total_system_memory

.. autofunction:: xpcsjax.optimization.nlsq.estimate_peak_memory_gb

.. autofunction:: xpcsjax.optimization.nlsq.get_adaptive_memory_threshold

Anti-degeneracy controller
--------------------------

.. autoclass:: xpcsjax.optimization.nlsq.AntiDegeneracyController



The 5-layer defence system is composed of:

* :class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`
* :class:`~xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer`
* :class:`~xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer`
* :class:`~xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor`
* :class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`

See :doc:`/advanced/anti_degeneracy` for the layering rationale.

Layer classes
~~~~~~~~~~~~~

.. autoclass:: xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer



.. autoclass:: xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer



.. autoclass:: xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer



.. autoclass:: xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor



.. autoclass:: xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting



.. autoclass:: xpcsjax.optimization.nlsq.shear_weighting.ShearWeightingConfig



.. autoclass:: xpcsjax.optimization.nlsq.hierarchical.HierarchicalConfig



.. autoclass:: xpcsjax.optimization.nlsq.gradient_monitor.GradientMonitorConfig



.. autoclass:: xpcsjax.optimization.nlsq.parameter_index_mapper.ParameterIndexMapper



.. autoclass:: xpcsjax.optimization.nlsq.multistart.MultiStartResult



.. autoclass:: xpcsjax.optimization.nlsq.multistart.MultiStartConfig



.. autoclass:: xpcsjax.optimization.nlsq.multistart.SingleStartResult



.. autoclass:: xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig



.. autoclass:: xpcsjax.optimization.nlsq.fourier_reparam.FourierReparamConfig



.. autoclass:: xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig



.. autoclass:: xpcsjax.optimization.nlsq.hierarchical.HierarchicalResult



.. autofunction:: xpcsjax.optimization.nlsq.core.fit_nlsq_multistart
   :no-index:

NLSQResult and NLSQ-side configs
--------------------------------

.. autoclass:: xpcsjax.optimization.nlsq.core.NLSQResult



.. autoclass:: xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult



CMA-ES escape
-------------

.. autoclass:: xpcsjax.optimization.nlsq.CMAESWrapper



.. autoclass:: xpcsjax.optimization.nlsq.CMAESWrapperConfig



.. autoclass:: xpcsjax.optimization.nlsq.CMAESResult



Configuration objects
---------------------

.. autoclass:: xpcsjax.optimization.nlsq.NLSQConfig



.. autoclass:: xpcsjax.optimization.nlsq.HybridRecoveryConfig



Memory management
-----------------

.. autoclass:: xpcsjax.optimization.nlsq.NLSQMemoryManager



.. autofunction:: xpcsjax.optimization.nlsq.get_memory_manager

Availability flags
------------------

The subpackage exposes several module-level booleans that report whether
optional dependencies were importable at package import time:

* ``NLSQ_CURVEFIT_AVAILABLE`` — ``True`` when the NLSQ ``CurveFit`` JIT
  cache is present (requires ``nlsq>=0.6.10``).
* ``NLSQ_AVAILABLE`` — alias for ``NLSQ_CURVEFIT_AVAILABLE``.
* ``JAX_AVAILABLE`` — ``True`` when JAX imports cleanly.
* ``NLSQ_GLOBAL_OPT_AVAILABLE`` — ``True`` when the NLSQ global-optimisation
  module is present.
* ``NLSQ_GOAL_AVAILABLE`` — ``True`` when ``OptimizationGoal`` is importable
  (NLSQ 0.6.4+).
