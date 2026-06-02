:orphan:

All modules
===========

This page registers every xpcsjax submodule as a :mod: target so that
internal cross-references resolve in the rendered documentation. It is
intentionally light: each ``py:module`` entry is members-suppressed because
the user-visible API surface is covered by the dedicated pages under
:doc:`index`. Modules already documented with ``automodule`` on a dedicated
page (the ``xpcsjax.cli`` submodules and ``xpcsjax.runtime``) are intentionally
omitted here to avoid duplicate object descriptions.

.. py:module:: xpcsjax

.. py:module:: xpcsjax.cli

.. py:module:: xpcsjax.cli.config_generator

.. py:module:: xpcsjax.cli.main

.. py:module:: xpcsjax.cli.xla_config

.. py:module:: xpcsjax.config

.. py:module:: xpcsjax.config.heterodyne_parameter_manager

.. py:module:: xpcsjax.config.heterodyne_parameter_names

.. py:module:: xpcsjax.config.heterodyne_parameter_space

.. py:module:: xpcsjax.config.heterodyne_physics_validators

.. py:module:: xpcsjax.config.manager

.. py:module:: xpcsjax.config.parameter_manager

.. py:module:: xpcsjax.config.parameter_names

.. py:module:: xpcsjax.config.parameter_registry

.. py:module:: xpcsjax.config.parameter_space

.. py:module:: xpcsjax.config.physics_validators

.. py:module:: xpcsjax.config.types

.. py:module:: xpcsjax.core

.. py:module:: xpcsjax.core.diagonal_correction

.. py:module:: xpcsjax.core.fitting

.. py:module:: xpcsjax.core.heterodyne_jax_backend

.. py:module:: xpcsjax.core.heterodyne_model

.. py:module:: xpcsjax.core.heterodyne_models

.. py:module:: xpcsjax.core.heterodyne_model_stateful

.. py:module:: xpcsjax.core.heterodyne_physics_factors

.. py:module:: xpcsjax.core.heterodyne_physics_kernel

.. py:module:: xpcsjax.core.heterodyne_physics_utils

.. py:module:: xpcsjax.core.heterodyne_scaling_utils

.. py:module:: xpcsjax.core.homodyne_model

.. py:module:: xpcsjax.core.jax_backend

.. py:module:: xpcsjax.core.math_primitives

.. py:module:: xpcsjax.core.model_mixins

.. py:module:: xpcsjax.core.models

.. py:module:: xpcsjax.core.physics

.. py:module:: xpcsjax.core.physics_factors

.. py:module:: xpcsjax.core.physics_nlsq

.. py:module:: xpcsjax.core.physics_utils

.. py:module:: xpcsjax.data

.. py:module:: xpcsjax.data.angle_filtering

.. py:module:: xpcsjax.data.config

.. py:module:: xpcsjax.data.filtering_utils

.. py:module:: xpcsjax.data.memory_manager

.. py:module:: xpcsjax.data.optimization

.. py:module:: xpcsjax.data.performance_engine

.. py:module:: xpcsjax.data.phi_filtering

.. py:module:: xpcsjax.data.preprocessing

.. py:module:: xpcsjax.data.quality_controller

.. py:module:: xpcsjax.data.types

.. py:module:: xpcsjax.data.validation

.. py:module:: xpcsjax.data.validators

.. py:module:: xpcsjax.data.xpcs_loader

.. py:module:: xpcsjax.device

.. py:module:: xpcsjax.device.config

.. py:module:: xpcsjax.device.cpu

.. py:module:: xpcsjax.io

.. py:module:: xpcsjax.io.json_utils

.. py:module:: xpcsjax.io.nlsq_writers

.. py:module:: xpcsjax.optimization

.. py:module:: xpcsjax.optimization.batch_statistics

.. py:module:: xpcsjax.optimization.exceptions

.. py:module:: xpcsjax.optimization.nlsq

.. py:module:: xpcsjax.optimization.nlsq.adapter

.. py:module:: xpcsjax.optimization.nlsq.adapter_base

.. py:module:: xpcsjax.optimization.nlsq.adaptive_regularization

.. py:module:: xpcsjax.optimization.nlsq.anti_degeneracy_controller

.. py:module:: xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics

.. py:module:: xpcsjax.optimization.nlsq.cmaes_wrapper

.. py:module:: xpcsjax.optimization.nlsq.config

.. py:module:: xpcsjax.optimization.nlsq.core

.. py:module:: xpcsjax.optimization.nlsq.data_prep

.. py:module:: xpcsjax.optimization.nlsq.fallback_chain

.. py:module:: xpcsjax.optimization.nlsq.fit_computation

.. py:module:: xpcsjax.optimization.nlsq.fourier_reparam

.. py:module:: xpcsjax.optimization.nlsq.gradient_diagnostics

.. py:module:: xpcsjax.optimization.nlsq.gradient_monitor

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_adapter

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_adapter_base

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_config

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_constant_mode

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_core

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_data_prep

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_logging

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_memory

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_multistart

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_result_builder

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_results

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_stratified_data

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_stratified_ls

.. py:module:: xpcsjax.optimization.nlsq.heterodyne_views

.. py:module:: xpcsjax.optimization.nlsq.hierarchical

.. py:module:: xpcsjax.optimization.nlsq.jacobian

.. py:module:: xpcsjax.optimization.nlsq.memory

.. py:module:: xpcsjax.optimization.nlsq.multistart

.. py:module:: xpcsjax.optimization.nlsq.parallel_accumulator

.. py:module:: xpcsjax.optimization.nlsq.parameter_index_mapper

.. py:module:: xpcsjax.optimization.nlsq.parameter_utils

.. py:module:: xpcsjax.optimization.nlsq.progress

.. py:module:: xpcsjax.optimization.nlsq.recovery

.. py:module:: xpcsjax.optimization.nlsq.result_builder

.. py:module:: xpcsjax.optimization.nlsq.results

.. py:module:: xpcsjax.optimization.nlsq.shear_weighting

.. py:module:: xpcsjax.optimization.nlsq.strategies

.. py:module:: xpcsjax.optimization.nlsq.strategies.chunking

.. py:module:: xpcsjax.optimization.nlsq.strategies.executors

.. py:module:: xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming

.. py:module:: xpcsjax.optimization.nlsq.strategies.hybrid_streaming

.. py:module:: xpcsjax.optimization.nlsq.strategies.out_of_core

.. py:module:: xpcsjax.optimization.nlsq.strategies.residual

.. py:module:: xpcsjax.optimization.nlsq.strategies.residual_jit

.. py:module:: xpcsjax.optimization.nlsq.strategies.sequential

.. py:module:: xpcsjax.optimization.nlsq.strategies.stratified_ls

.. py:module:: xpcsjax.optimization.nlsq.transforms

.. py:module:: xpcsjax.optimization.nlsq.validation

.. py:module:: xpcsjax.optimization.nlsq.wrapper

.. py:module:: xpcsjax.optimization.numerical_validation

.. py:module:: xpcsjax.optimization.recovery_strategies

.. py:module:: xpcsjax.post_install

.. py:module:: xpcsjax.runtime.shell

.. py:module:: xpcsjax.runtime.shell.activation

.. py:module:: xpcsjax.runtime.utils

.. py:module:: xpcsjax.runtime.utils.system_validator

.. py:module:: xpcsjax.uninstall_scripts

.. py:module:: xpcsjax.utils

.. py:module:: xpcsjax.utils.async_io

.. py:module:: xpcsjax.utils.logging

.. py:module:: xpcsjax.utils.path_validation

.. py:module:: xpcsjax.viz

.. py:module:: xpcsjax.viz.datashader_backend

.. py:module:: xpcsjax.viz.diagnostics

.. py:module:: xpcsjax.viz.nlsq_plots

