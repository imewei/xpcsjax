# Graph Report - xpcsjax  (2026-09-16)

## Corpus Check
- 566 files · ~668,437 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: .npz 2, (none) 1, .bat 1)

## Summary
- 10042 nodes · 21533 edges · 452 communities (366 shown, 70 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1604 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e4c2bffb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- make_synthetic_two_component
- generate_nlsq_plots
- HeterodyneModel
- fit_heterodyne_stratified_least_squares
- HeterodynePhysicsAdapter
- test_low_level_plots.py
- _evaluate_c2_per_angle
- test_logging.py
- MainWindow
- OptimizationResult
- AnalysisMode
- DataQualityController
- test_heterodyne_cmaes_warmstart_auto_skip.py
- test_heterodyne_fixed_parameters.py
- data/validation.py
- ParameterIndexMapper
- heterodyne_core.py
- fit_nlsq
- HeterodyneParameterSpace
- SingleStartResult
- NLSQWrapper
- StratifiedResidualFunction
- test_uninstall_scripts.py
- optimization/test_debug_audit_2026_07_22.py
- NLSQConfig
- core.py
- main.py
- jax_backend.py
- NLSQConfig
- test_transforms.py
- test_anti_degeneracy_layers.py
- ValueError
- model_fn
- parameter_registry.py
- Project
- Two-Component Config Template
- test_joint_ssr_floor.py
- FitJob
- XPCSDataFilter
- classify_option
- diagonal_correction.py
- plots_view.py
- heterodyne_logging.py
- AnalysisMode
- test_output_resolution.py
- PreprocessingPipeline
- GradientCollapseMonitor
- filtering_utils.py
- .fit
- test_plot_dispatch_logging.py
- rasterize
- XPCSDataLoader
- test_heterodyne_result_builder.py
- test_nlsq_support_modules.py
- test_gui_redesign.py
- nlsq/__init__.py
- AntiDegeneracyController
- test_multistart.py
- cpu.py
- ParameterSpace
- fit_nlsq_jax
- heterodyne_config.py
- adapter.py
- parameter_utils.py
- sequential.py
- StratificationConfig
- AdaptiveRegularizer
- heterodyne_physics_kernel.py
- compute_c2_heterodyne
- run_worker
- test_gui_jax_free.py
- make_cfgmgr_and_data
- test_heterodyne_cmaes_warmstart_success_gate.py
- nlsq_plots.py
- ProjectSidebar
- DatasetOptimizer
- .from_config
- build_workbench
- test_strategy_chunking.py
- quality_controller.py
- _NullLogger
- test_heterodyne_memory_adapter.py
- _make_synthetic_c2
- StratifiedResidualFunctionJIT
- test_review_regressions.py
- HierarchicalOptimizer
- normalize_angle_to_symmetric_range
- .phi
- heterodyne_model_stateful.py
- AntiDegeneracyController
- test_persist.py
- ResultBuilder
- test_jacobian.py
- test_homodyne_engine_preservation.py
- test_cache_source_file_validation.py
- test_heterodyne_tied_parameters.py
- test_post_install.py
- read_c2_preview
- ndarray
- test_shear_weighting.py
- ResultSummary
- DataInspectDialog
- multistart.py
- test_heterodyne_tied_residuals.py
- test_heterodyne_data_prep.py
- PerAngleScalingPlan
- log_exception
- test_engine_route_bugfixes.py
- json_safe
- result_presenter.py
- test_jax_backend.py
- _write_npz_compressed
- post_install.py
- test_quality_gate_fixes.py
- present_failure
- RunController
- test_static_individual_invariant.py
- service/persist.py
- _save_fig
- test_fit_quality.py
- test_result_presenter.py
- load_dataset
- HeterodyneParameterManager
- CreateConfigDialog
- service/config.py
- test_memory_manager_cache_gating.py
- test_cmaes_trigger.py
- test_cache_safety.py
- .fit
- test_gradient_monitor.py
- physics_nlsq.py
- test_strategy_executors.py
- .load_config
- test_aps_u_selective_read_d7.py
- ProjectDialogHandler
- ConfigTextEditorDialog
- test_plots_view.py
- ShearSensitivityWeighting
- heterodyne_memory.py
- validate_xpcs_data
- test_runtime_shell.py
- test_system_validator.py
- get_adaptive_memory_threshold
- SequentialResult
- heterodyne_result_builder.py
- test_load_degradation_signal.py
- test_laminar_mode_banners.py
- format_comparison
- LargeDatasetExecutor
- save_nlsq_json_files
- log_heterodyne_completion
- test_homodyne_covariance_contract.py
- test_memory_manager_logging.py
- test_fit_queue.py
- test_logging_primitives.py
- test_adapter_xdata_cache.py
- build_heterodyne_stratified_data
- optimization/test_validation.py
- ndarray
- References and Citations Doc
- MemoryStats
- residual_jit.py
- fit_with_stratified_hybrid_streaming
- PhysicsFactors
- QualityControlConfig
- xpcsjax/data/__init__.py
- test_main_window.py
- xla_config.py
- FitQueueController
- test_heterodyne_physics_validators.py
- events.py
- PhiAngleFilter
- memory_manager.py
- wrap_stratified_function_with_transforms
- test_out_of_core_route.py
- fit_nlsq entry point
- stratified_ls.py
- configure_logging
- ndarray
- Banner
- ._configure_impl
- compute_jacobian_stats
- PerAngleScaling
- test_covariance_placeholder_contract.py
- BatchStatistics
- test_perf_regression.py
- ExecutionResult
- test_cache_q_validation.py
- test_config_jax_free.py
- test_validation_branches.py
- log_performance
- ValidationResult
- ndarray
- NLSQStrategy
- NumericalValidator
- LHS multistart
- Memory-Aware Strategy Routing
- test_no_pickle_loads.py
- _unpack_result_params
- npz_cache.py
- test_hybrid_streaming_retry.py
- test_layer5_gating.py
- parallel_accumulator.py
- compute_diagonal_overlay_stats
- AdvancedMemoryManager
- ConfigManager
- NLSQ CurveFit (trust-region least squares)
- test_stratified_max_iter_grading.py
- FakeHandle
- .from_config
- HomodyneModel
- fit_nlsq_multistart_heterodyne
- test_escape_disabled_hint.py
- fit_with_out_of_core_accumulation
- test_parallel_accumulator.py
- .__init__
- test_layer_gate_wiring.py
- fit_two_component_via_engine
- compute_chi_squared
- test_heterodyne_config_bounds_override.py
- xpcs_loader.py
- heterodyne_parameter_names.py
- _FakeConfigManager
- test_frame_dimension_guard.py
- test_config_unwrap.py
- test_phase5_model_function_modes.py
- run_fit
- test_adaptive_regularization.py
- _logger_that_raises_on_log
- test_debug_audit_2026_07_23_diagonal_skip.py
- NLSQOptimizationError
- MemoryPressureMonitor
- VizBundle
- test_heterodyne_parameter_manager_tied.py
- build_gradient_collapse_callback
- nlsq/validation.py
- test_io.py
- json_serializer
- completion.sh
- test_docs_structure.py
- TestExecuteLayersNLSQConfigHomodyne
- filter_phi_angles_jax
- TestExecuteLayersNLSQConfigHeterodyne
- hybrid_streaming.py
- test_heterodyne_results.py
- test_post_install_fish.py
- get_safe_output_dir
- test_validation_crash_coverage.py
- maps.py
- interactive_setup
- test_diagonal_correction.py
- TestTypeBoundary
- system_validator.py
- run_validation
- Anti-degeneracy controller (5/4-layer defense)
- All-modules registry page
- test_l4_callback_layout.py
- heterodyne_parameter_space.py
- core/test_debug_audit_2026_07_22.py
- test_reader_final_drain_recovers_terminal_after_grace
- test_unweighted_stratified_data_does_not_materialize_dense_sigma
- InputValidator
- ResultValidator
- TestNoScipyLeastSquares
- load_and_merge_config
- test_loader_performance_engine_cleanup.py
- fit_nlsq
- _fixture
- test_run_controller.py
- .create_nlsq_callbacks
- compute_g2_scaled
- validate_no_nan_inf
- test_lazy_imports.py
- _resolve_color_limits
- test_iteration_callback_seam.py
- StreamingExecutor
- model_adapter.PointEvaluator
- heterodyne_physics_validators.py
- XPCSDataFormatError
- .__init__
- test_config_relative_data_paths.py
- validate_cross_parameter_constraints
- heterodyne_jax_backend.py
- test_stratified_ls_averaged_covariance_transform.py
- load_config
- test_logging_quality_gate.py
- plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2)
- NLSQ Residual Diagnostics Baseline (phi=45deg)
- xpcsjax.cli.plot_dispatch
- xpcsjax.optimization.nlsq.fit_nlsq
- test_freeze_safety.py
- TimedContext
- test_debug_audit_2026_07_23_negative_correlation_repair.py
- _safe_log_memory_strategy
- FitQualityConfig
- TestJSONFormatterCircularRef
- TestLogPerformanceNeverRaise
- TestContextFilterOnLogger
- CMAESWrapper
- TestSafeSincContinuity
- xla_config.bash
- Public API Reference
- test_codex_review_fixes.py
- OptimizationStrategy
- logging.py
- XpcsDataset
- test_docs_no_fourier.py
- _version_at_least
- DatashaderRenderer
- validate_single_parameter
- test_anti_degeneracy_transforms.py
- ._run_sequential_optimization
- plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg)
- ConfigManager
- xpcsjax.core.HeterodyneModel
- AntiDegeneracyController
- 5-layer anti-degeneracy controller
- Second-order (intensity) correlation function c2
- _FakeOpt
- set_log_context
- test_ci_heavy_nodes_parity.py
- TestPropagateInversion
- _build_inputs
- generate_nlsq_plots
- Joint global escape keep-better contract
- main
- ndarray
- test_data_package_features.py
- _ColorFormatter
- validate_single_parameter
- test_no_removed_per_angle_tokens_in_tests_or_package
- fit_with_stratified_least_squares
- Architecture Overview
- xpcsjax.cli.config_generator.main
- xpcsjax Project Logo
- test_aps_u_bin_skip_misalignment.py
- install_xla_activation
- tests/conftest.py
- test_fourier_reparam_removed.py
- test_heterodyne_per_angle_cmaes_fits_without_signature_drift
- test_debug_audit_regressions.py
- test_l1_rename.py
- configure_cpu_hpc
- DatashaderRenderer
- Decision Record: CPU-only Execution
- make test-smoke
- HomodyneModel
- User Guide Overview
- log_phase
- test_simulated_data_grid.py
- compute_c2_heterodyne_pointwise
- create_time_integral_matrix
- compute_c2_heterodyne
- compute_g2_scaled
- xpcsjax.post_install.main
- xpcsjax.cli.xla_config.configure_xla
- xpcsjax.core.models.CombinedModel
- NLSQConfig
- parameter_utils.resolve_optimized_physical_parameters
- Runtime Package API
- configure_logging
- PathValidationError
- compute_diagonal_overlay_stats
- conf.py
- test_heterodyne_smoke_fit_recovers_truth
- per_angle_mode.PerAngleScalingPlan
- gui/conftest.py
- plot_families/__init__.py
- OOCSharedArrays
- controllers/__init__.py
- xpcsjax/gui/__init__.py
- ipc/__init__.py
- project/__init__.py
- views/__init__.py
- main_window_support/__init__.py
- plots/__init__.py
- ContextFilter
- validate_parameters
- activation/__init__.py
- xpcsjax/service/__init__.py
- Domain Documentation Guide
- GitHub Issue Tracker Conventions
- Triage Label Mapping
- service.events
- service.persist
- Anti-degeneracy Controller Theory
- Lazy Public API Implementation
- xpcsjax.cli.main.main
- xpcsjax.cli.main.main_xjexp
- xpcsjax.cli.main.main_xjsim
- xpcsjax.core.models.DiffusionModel
- heterodyne_physics_kernel.compute_c2_unified
- xpcsjax.core.physics_factors.PhysicsFactors
- xpcsjax.core.models.PhysicsModelBase
- xpcsjax.core.physics.ValidationResult
- Data API Reference
- heterodyne_stratified_ls.fit_heterodyne_stratified_least_squares
- Contributing Guide
- restore_by_mask_jax
- tests/benchmarks shard
- tests/core shard
- tests/heterodyne shard
- tests/property shard
- Examples Index
- Autosummary Module Template
- Theory and Physics Index
- Command-Line Interface
- ConfigManager
- test_pointwise_joint_parity.py
- validate_optimized_params
- _build_homodyne_l4_callback
- test_preprocessing_diagonal_dtype.py
- _FakeModel
- test_templates_no_fourier_keys.py
- NLSQ Integration Contract
- test_loader_integration.py
- get_executor
- ConstraintSeverity
- test_reduce_noise_dtype.py
- CLI Module Guide
- ._run_nlsq_refinement
- Config Module Guide
- get_constraint_summary
- test_high_scale_ratio_triggers_cmaes
- Core Module Guide
- test_low_scale_ratio_does_not_trigger
- test_compute_scale_ratio_increases_with_spread
- Data Module Guide
- Device Module Guide
- GUI Module Guide
- IO Module Guide
- Optimization Module Guide
- Runtime Module Guide
- Service Module Guide
- Utils Module Guide
- Viz Module Guide

## God Nodes (most connected - your core abstractions)
1. `NLSQConfig` - 263 edges
2. `ConfigManager` - 243 edges
3. `OptimizationResult` - 170 edges
4. `AnalysisMode` - 159 edges
5. `make_synthetic_two_component()` - 156 edges
6. `get_logger()` - 146 edges
7. `fit_nlsq_multi_phi()` - 109 edges
8. `HeterodyneModel` - 104 edges
9. `ParameterManager` - 96 edges
10. `MainWindow` - 94 edges

## Surprising Connections (you probably didn't know these)
- `test_repair_scaling_issues_method_removed()` --uses--> `DataQualityController`  [INFERRED]
  tests/data/test_repair_defaults_a3.py → xpcsjax/data/quality_controller.py
- `test_residual_map_accepts_array()` --calls--> `ResidualMapView`  [INFERRED]
  tests/gui/test_plots_view.py → xpcsjax/gui/views/plots/maps.py
- `test_phi_grid_pins_scrollbars_to_keep_square_tiles()` --calls--> `PhiResultsGrid`  [INFERRED]
  tests/gui/test_plots_view.py → xpcsjax/gui/views/plots/grid.py
- `test_residual_histogram_renders_and_tolerates_all_nan()` --calls--> `ResidualHistogramView`  [INFERRED]
  tests/gui/test_plots_view.py → xpcsjax/gui/views/plots/residuals.py
- `test_diagonal_residual_traces_diag_against_t1()` --calls--> `DiagonalResidualView`  [INFERRED]
  tests/gui/test_plots_view.py → xpcsjax/gui/views/plots/residuals.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Memory routing strategy tiers** — docs_source_advanced_memory_routing_strategy, docs_source_advanced_memory_routing_standard, docs_source_advanced_memory_routing_out_of_core, docs_source_advanced_memory_routing_hybrid_streaming [EXTRACTED 0.90]
- **GUI-Worker Process Isolation** — docs_source_user_guide_gui, xpcsjax_service_events [EXTRACTED 0.95]
- **Four Canonical Analysis Modes** — docs_source_changelog_analysis_mode_deprecation, xpcsjax_config_templates_xpcsjax_laminar_flow_doc, xpcsjax_config_templates_xpcsjax_static_anisotropic_doc, xpcsjax_config_templates_xpcsjax_static_isotropic_doc, xpcsjax_config_templates_xpcsjax_two_component_doc [EXTRACTED 1.00]
- **Anti-Degeneracy 5-Layer Defense System** — concept_anti_degeneracy_controller, concept_per_angle_reparameterization, concept_hierarchical_optimizer_l2, concept_adaptive_regularization_l3, concept_gradient_collapse_monitor_l4, concept_shear_weighting_l5 [EXTRACTED 1.00]
- **Homodyne 5-layer anti-degeneracy defense system** — docs_source_theory_anti_degeneracy_controller, docs_source_theory_anti_degeneracy_perangleplan, docs_source_theory_anti_degeneracy_hierarchicaloptimizer, docs_source_theory_anti_degeneracy_adaptiveregularizer, docs_source_theory_anti_degeneracy_gradientcollapsemonitor, docs_source_theory_anti_degeneracy_shearsensitivityweighting [EXTRACTED 1.00]
- **5-layer Anti-Degeneracy Defense feeding NLSQ CurveFit** — docs_diagrams_fitting_workflow_l1_per_angle_reparam, docs_diagrams_fitting_workflow_l2_hierarchical_opt, docs_diagrams_fitting_workflow_l3_adaptive_cv_reg, docs_diagrams_fitting_workflow_l5_shear_sensitivity_wt, docs_diagrams_fitting_workflow_nlsq_curvefit [EXTRACTED 1.00]
- **Documentation Coverage Enforcement** — docs_adr_0001_automated_structural_doc_coverage_check, tests_test_docs_structure, docs_source_api_index [EXTRACTED 1.00]
- **Five-Layer Anti-Degeneracy Stack** — docs_diagrams_fitting_workflow_l1_per_angle_reparameterization, docs_diagrams_fitting_workflow_l2_hierarchical_optimization, docs_diagrams_fitting_workflow_l3_adaptive_cv_regularization [EXTRACTED 1.00]
- **Global Search Escape Mechanisms Orchestrated by fit_nlsq** — concept_fit_nlsq, concept_cma_es_escape, concept_multistart [EXTRACTED 1.00]
- **Heterodyne joint-fit global escape flow (CMA-ES + multistart)** — docs_source_theory_heterodyne_anti_degeneracy_fit_joint_cmaes_multi_phi, docs_source_theory_heterodyne_anti_degeneracy_fit_joint_multistart, docs_source_theory_heterodyne_anti_degeneracy_fit_with_cmaes, docs_source_theory_heterodyne_anti_degeneracy_run_multistart_nlsq [EXTRACTED 1.00]
- **Memory Routing Strategy Tiers** — docs_diagrams_fitting_workflow_cmaes_escape, docs_diagrams_fitting_workflow_hybrid_streaming, docs_diagrams_fitting_workflow_stratified_ls, docs_diagrams_fitting_workflow_inmemory_engine_route [EXTRACTED 1.00]
- **xpcsjax NLSQ fit pipeline (config to result)** — docs_source_quickstart_configmanager, docs_source_quickstart_select_nlsq_strategy, docs_source_quickstart_antidegeneracycontroller, docs_source_quickstart_curvefit, docs_source_quickstart_optimizationresult [EXTRACTED 1.00]
- **Memory-budget strategy routing (CMA-ES / multistart / hybrid-streaming / stratified-LS / in-memory)** — docs_diagrams_fitting_workflow_select_nlsq_strategy, docs_diagrams_fitting_workflow_cmaes_escape, docs_diagrams_fitting_workflow_hybrid_streaming, docs_diagrams_fitting_workflow_stratified_ls [EXTRACTED 1.00]
- **Lazy-loaded public API surface** — docs_source_api_public_lazy_exports, docs_source_api_public_fit_nlsq, docs_source_api_public_load_xpcs_data, docs_source_api_public_configmanager, docs_source_api_public_homodynemodel, docs_source_api_public_heterodynemodel [EXTRACTED 1.00]
- **v0.1.6 Version Consistency Across Docs and Templates** — docs_source_development_releasing_version_consistency, docs_source_changelog_v016, docs_source_installation_doc, xpcsjax_config_templates_xpcsjax_laminar_flow_doc [EXTRACTED 1.00]
- **Anti-Degeneracy / Per-Angle-Mode Mechanism Family** — docs_source_changelog_anti_degeneracy_controller, xpcsjax_config_templates_xpcsjax_laminar_flow_anti_degeneracy, xpcsjax_config_templates_xpcsjax_static_anisotropic_per_angle_pinned, xpcsjax_config_templates_xpcsjax_two_component_per_angle_mode [INFERRED 0.85]
- **XPCS Theory Chapter Chain** — docs_source_theory_xpcs_basics, docs_source_theory_transport_coefficient, docs_source_theory_homodyne_model, docs_source_theory_heterodyne_model [INFERRED 0.85]

## Communities (452 total, 70 thin omitted)

### Community 0 - "make_synthetic_two_component"
Cohesion: 0.03
Nodes (117): _diag(), Golden test for heterodyne anti-degeneracy diagnostics emission. Both modes now…, Only the 3 activation flags are unconditional; per-layer DETAIL keys…, test_constant_path_always_emits_activation_keys(), test_detail_keys_preserved_when_enabled(), test_disabled_path_now_emits_activation_keys_false(), test_disabled_path_omits_layer_detail_keys(), test_enabled_path_emits_activation_keys_true() (+109 more)

### Community 1 - "generate_nlsq_plots"
Cohesion: 0.10
Nodes (30): _phi_filename(), _png_sha256(), Path, End-to-end tests for generate_nlsq_plots., Heterodyne now produces plots + NPZ + JSON via per-angle scaling reconstruction., One angle's compute failure leaves NaN; others render; NPZ still written., If datashader is missing, orchestrator warns and uses matplotlib. The…, plot_simulated_data and _generate_heatmap_plots were removed in favor of… (+22 more)

### Community 2 - "HeterodyneModel"
Cohesion: 0.06
Nodes (62): C045 RCA (2026-08-17): a config-supplied ``initial_parameters`` value outside…, test_stratified_ls_clips_out_of_bounds_physics_init(), Proves the tied-parameter mechanism is REAL coupling (gradient sums both usages…, End-to-end discriminator: real during-solve tying vs report-time-only…, Within THIS hand-rolled closure, d(loss)/d(D0_sample_free_var) must equal the…, test_tied_fit_ssr_matches_recompute_from_reported_parameters(), test_tied_gradient_sums_both_usages(), _tied_param_manager() (+54 more)

### Community 3 - "fit_heterodyne_stratified_least_squares"
Cohesion: 0.04
Nodes (77): ResolvedPerAngleMode, test_stratified_ls_scaling_names_match_dedup_phi_count(), Without anti_degeneracy dict, the result must not contain…, test_stratified_ls_result_no_controller_diagnostics_without_ad_config(), The stratified-LS path (the >=1M solver the C044 two_component run took)…, Test averaged: the 2 head scalars broadcast to n_phi., Test _run_hierarchical_layers feeds the scaling-first vector unpermuted. With…, The shared completion chokepoint must emit an anti-degeneracy DEFENSE summary… (+69 more)

### Community 4 - "HeterodynePhysicsAdapter"
Cohesion: 0.02
Nodes (96): Config-driven dispatch returns the right physics model class. Task 28:…, ConfigManager should normalize 'heterodyne' / 'Heterodyne' → 'two_component'., Minimal YAML config setting analysis_mode., analysis_mode: two_component must produce a HeterodynePhysicsAdapter instance., analysis_mode: heterodyne (synonym) must also produce HeterodynePhysicsAdapter., analysis_mode: static_anisotropic must NOT produce a HeterodynePhysicsAdapter…, analysis_mode: laminar_flow must NOT produce a HeterodynePhysicsAdapter., make_model should also work on a raw config dict (no ConfigManager). (+88 more)

### Community 5 - "test_low_level_plots.py"
Cohesion: 0.09
Nodes (39): Figure, filterwarnings, mpl_image_compare, Path, Unit tests for low-level plot functions and helpers., test_plot_nlsq_fit_accepts_t_none(), test_plot_nlsq_fit_residual_cmap_is_rdbu(), test_plot_nlsq_fit_save_path_writes_png() (+31 more)

### Community 6 - "_evaluate_c2_per_angle"
Cohesion: 0.09
Nodes (30): The heterodyne branch of ``_evaluate_c2_per_angle`` must evaluate the FIT-TIME…, test_heterodyne_evaluator_matches_fit_time_model_on_short_grid(), Heterodyne path returns a real c2 surface in the expected [1.0, 1.5] range., Duplicate-valued phi entries must resolve by loop index, not first match., Two angles sharing a phi value each render with THEIR OWN contrast/offset.…, A null (present-but-None) config section degrades to the intended ValueError.…, The homodyne branch's null-config-section guard (mirrors the heterodyne guard…, test_evaluate_heterodyne_duplicate_phi_uses_own_scaling() (+22 more)

### Community 7 - "test_logging.py"
Cohesion: 0.08
Nodes (33): MonkeyPatch, parametrize, Path, Tests for xpcsjax.utils.logging. This module mutates process-global logging…, Present-but-null path/max_size_mb/backup_count must fall back to defaults, not…, test_configure_creates_rotating_file_handler(), test_configure_disables_propagation_when_handler_installed(), test_configure_from_dict_filename_auto_generated() (+25 more)

### Community 8 - "MainWindow"
Cohesion: 0.04
Nodes (31): QMainWindow, _expand_path(), MainWindow, Any, Path, QWidget, Build the File menu: project-lifecycle actions only. Order: Create Project →…, Refresh the persistent "Run will act on: <dataset>" status-bar label. (+23 more)

### Community 9 - "OptimizationResult"
Cohesion: 0.02
Nodes (152): _make_result(), ndarray, `_warn_nlsq_bound_saturation` must not misreport NaN uncertainties. NaN/inf…, A global-escape result (all-NaN uncertainties) emits no saturation warning., A real near-zero uncertainty must still surface a saturation warning., test_genuinely_zero_uncertainty_still_warns(), test_nan_uncertainties_do_not_warn_bound_saturation(), test_bool_coercion_and_determinism() (+144 more)

### Community 10 - "AnalysisMode"
Cohesion: 0.48
Nodes (6): Typed accessors on OptimizationResult (quality-gate type-design fixes). F8:…, _result(), test_accessors_raise_clearly_when_n_physics_unknown(), test_global_escape_typed_read(), test_physics_and_scaling_split_when_n_physics_known(), test_physics_parameters_empty_when_n_physics_zero()

### Community 11 - "DataQualityController"
Cohesion: 0.06
Nodes (43): A dataset that fails the FINAL_DATA quality gate must set ``_degraded`` even…, test_final_qc_gate_failure_sets_degraded(), validate_data_stage(), DataQualityController, Any, log_performance, QualityControlResult, QualityMetrics (+35 more)

### Community 12 - "test_heterodyne_cmaes_warmstart_auto_skip.py"
Cohesion: 0.17
Nodes (13): _cfg(), Parity: heterodyne JOINT CMA-ES escapes honor ``cmaes_warmstart_auto_skip``.…, The auto-skip knob is CMA-ES-specific (matches laminar + the knob name)., dof ≤ 0 (more params than data) never skips — guards a meaningless χ²/dof., SSR/dof below threshold ⇒ skip: tag the auto-skip and never call CMA-ES., SSR/dof above threshold ⇒ no skip: the global search still runs., A good warm start does NOT skip when ``cmaes_warmstart_auto_skip=False``., test_apply_global_escape_auto_skips_cmaes_on_good_warmstart() (+5 more)

### Community 13 - "test_heterodyne_fixed_parameters.py"
Cohesion: 0.16
Nodes (10): _config(), Heterodyne fixed_parameters: vary=False + value write, wins over EVERY overlay…, v2-review-identified gap: fixed_parameters must win even against overlays that…, The parent-side mirror of the child check -- antigravity round-4 finding.…, test_fixed_contrast_is_honored(), test_fixed_offset_is_honored(), test_fixed_parameter_value_is_honored_not_flat_list_value(), test_fixed_parameters_alias_canonical_collision_rejected() (+2 more)

### Community 14 - "data/validation.py"
Cohesion: 0.08
Nodes (38): Quality-gate finding #7: the validation helpers narrow their except clause to…, _report(), test_validate_array_shapes_records_issue_on_arithmeticerror(), test_validate_array_shapes_records_issue_on_attributeerror(), _raise(), _raise_boom(), Data-integrity regression: unexpected validator-body crashes must be loud.…, A crash inside ``_validate_array_shapes`` must log ERROR and fail the report. (+30 more)

### Community 15 - "ParameterIndexMapper"
Cohesion: 0.05
Nodes (44): slice, parametrize, Phase-0 unit tests for the scaling-first canonical layout authority (spec §4…, test_averaged_and_individual_are_not_frozen(), test_blocks_partition_the_vector_scaling_first(), test_canonical_rejects_unresolved_mode(), test_constant_mode_is_frozen_with_empty_scaling_head(), test_group_indices() (+36 more)

### Community 16 - "heterodyne_core.py"
Cohesion: 0.02
Nodes (177): Regression: CMA-ES auto-memory sizing must use the FINAL (post-adaptive-…, At a memory budget where the small (pre-scaling) popsize fits without batching…, test_configure_memory_respects_explicit_popsize_override(), Data-integrity guards on the joint global-escape keep-better decision. These…, test_finite_candidate_beats_nonfinite_warm_start(), test_finite_keep_better_semantics_unchanged(), test_nonfinite_candidate_never_kept(), Regression tests for per-angle CMA-ES diagnostics + warm-start auto-skip. Two… (+169 more)

### Community 17 - "fit_nlsq"
Cohesion: 0.08
Nodes (36): _fit(), Phase 5 — averaged uses EXPANDED constrained-model DOF (2*n_phi + n_physics)., test_averaged_dof_basis_is_expanded(), test_constant_dof_basis_is_physics_only(), test_individual_dof_basis_is_optimizer(), Phase 5 — averaged/constant results expand back to the dense per-angle layout., test_averaged_result_is_dense_per_angle(), test_constant_result_is_dense_per_angle() (+28 more)

### Community 18 - "HeterodyneParameterSpace"
Cohesion: 0.09
Nodes (14): HeterodyneParameterSpace, ndarray, Names of parameters that vary (physics + scaling)., Names of varying physics parameters (excludes scaling)., Get contrast and offset values., Get initial values as a numpy array in canonical order. Returns -------…, Serialize this space to a dict compatible with :meth:`from_config`. Produces…, Get bounds as numpy arrays. Returns ------- tuple of numpy.ndarray… (+6 more)

### Community 19 - "SingleStartResult"
Cohesion: 0.08
Nodes (37): _CfgMgr, _install_model_stub(), Tests for heterodyne joint multistart wiring (Phase 1)., Regression test for the multistart no-op bug. ``fit_nlsq_multi_phi`` seeds its…, ``reseed_initial_values`` must handle the ndarray input form too, not just dict., End-to-end regression test for the multistart no-op bug. Each LHS-sampled…, Minimal OptimizationResult stand-in., _StubModel (+29 more)

### Community 20 - "NLSQWrapper"
Cohesion: 0.07
Nodes (35): curve_fit(), Regression test for NLSQWrapper._prepare_sigma_data's stratified phi indexing.…, test_stratified_sigma_phi_index_matches_own_angle(), Which physical parameters are free vs. fixed for one NLSQ solve. ``free_mask``…, ResolvedPhysicalParameters, create_multistart_warmup_func(), warmup_fit_func(), _extract_n_points() (+27 more)

### Community 21 - "StratifiedResidualFunction"
Cohesion: 0.14
Nodes (30): _full_grid_chunk(), ndarray, SimpleNamespace, Tests for xpcsjax.optimization.nlsq.strategies.residual.…, Regression: t1_unique = np.unique(all_t1) and t2_unique = np.unique(all_t2) are…, An out-of-range t1 query (float drift past t1_unique's max) must clamp to the…, One chunk holding the full phi x t1 x t2 cartesian grid (all angles)., _stratified() (+22 more)

### Community 22 - "test_uninstall_scripts.py"
Cohesion: 0.08
Nodes (60): CaptureFixture, MonkeyPatch, Path, Tests for xpcsjax.uninstall_scripts. Covers venv-path resolution, cleanup-…, Sandbox HOME + VIRTUAL_ENV to tmp and return the fake venv path., test_cleanup_activation_scripts_dry_run(), test_cleanup_activation_scripts_no_venv(), test_cleanup_activation_scripts_scrubs_block() (+52 more)

### Community 23 - "optimization/test_debug_audit_2026_07_22.py"
Cohesion: 0.06
Nodes (44): _linear_residual_fn(), ndarray, Regression tests for the 2026-07-22 debug-audit fixes. Fix 1…, Previously ``_fit_joint_averaged_multi_phi`` never called the floor — this pins…, Previously ``_fit_joint_constant_multi_phi`` never called the floor — this pins…, No-regression: individual mode already called the floor pre-extraction; confirm…, Residual = x - target; SSR is minimized (0) exactly at x == target., No-worse contract: final SSR is always <= the warm-start SSR. (+36 more)

### Community 24 - "NLSQConfig"
Cohesion: 0.05
Nodes (49): F6: NLSQConfig.validate() upper-bound guard for chunk-size fields. A…, test_cmaes_data_chunk_size_above_ceiling_is_rejected(), test_cmaes_data_chunk_size_none_is_accepted(), test_cmaes_data_chunk_size_normal_is_accepted(), test_cmaes_data_chunk_size_zero_still_rejected(), test_hybrid_chunk_size_above_ceiling_is_rejected(), test_hybrid_chunk_size_normal_is_accepted(), test_hybrid_chunk_size_zero_still_rejected() (+41 more)

### Community 25 - "core.py"
Cohesion: 0.05
Nodes (60): Regression: scalar q-extraction sites must reject NaN, not silently poison the…, test_build_model_function_rejects_nan_q(), test_normalize_data_to_object_accepts_finite_q(), test_normalize_data_to_object_rejects_nan_q(), test_restore_by_mask_numpy_round_trip(), ParameterSpace, Parameter space definition with bounds for NLSQ optimization. Implements…, is_adapter_available() (+52 more)

### Community 26 - "main.py"
Cohesion: 0.15
Nodes (16): create_parser(), ArgumentParser, Namespace, Argument parser for the xpcsjax CLI. Parameter override flags map to canonical…, Build the xpcsjax CLI argument parser., Light validation pass. Returns non-fatal warning strings. Raises…, validate_args(), _bootstrap_xla_env() (+8 more)

### Community 27 - "jax_backend.py"
Cohesion: 0.06
Nodes (45): Regression test for gradient_g2 (audit C12). `gradient_g2` was wired as…, batch_chi_squared(), compute_g1_diffusion(), _compute_g1_diffusion_core(), _compute_g1_shear_core(), _sinc2_for_one_phi(), compute_g1_total(), _compute_g1_total_core() (+37 more)

### Community 28 - "NLSQConfig"
Cohesion: 0.03
Nodes (99): _build_model(), _config_dict(), ndarray, Integration smoke tests: real heterodyne NLSQ fits on tiny synthetic data.…, _synthetic_stack(), test_auto_mode_resolves_to_averaged_for_many_angles(), test_cmaes_path_runs(), test_individual_mode_joint_fit() (+91 more)

### Community 29 - "test_transforms.py"
Cohesion: 0.07
Nodes (44): _laminar_index_map(), parametrize, Scientific tests for xpcsjax.optimization.nlsq.transforms. The shear transforms…, test_adjust_covariance_beta_unchanged(), test_adjust_covariance_log_jacobian(), test_adjust_covariance_no_state_or_empty(), test_build_physical_index_map(), test_format_x_scale_for_log() (+36 more)

### Community 30 - "test_anti_degeneracy_layers.py"
Cohesion: 0.08
Nodes (24): Verify the surviving anti-degeneracy layers ported over from homodyne. Task 29…, Homodyne AntiDegeneracyConfig.from_dict must honor config-file values over the…, ``regularization.auto_tune_lambda: False`` must reach the built regularizer.…, ``execute_layers`` is a registered, parseable, INERT config gate. All tests…, ``from_dict({})`` must give ``execute_layers == False``., ``from_dict({"execute_layers": False})`` must give ``False``., ``from_dict({"execute_layers": True})`` must give ``True``., The dataclass default must be ``False`` without calling ``from_dict``. (+16 more)

### Community 31 - "ValueError"
Cohesion: 0.06
Nodes (26): A malformed --phi-angles must not abort the whole fit run — it is only ever…, test_dispatch_fit_is_non_fatal_on_malformed_phi_angles(), test_diagnose_error_convergence_failure_perturbs_zero_param(), test_diagnose_error_convergence_failure_retry_perturbs_zero_param(), test_diagnose_error_matches_nlsq_not_finite_message(), test_diagnose_error_numerical_instability_infinite_bounds_stays_finite(), test_diagnose_error_numerical_instability_resets_to_bounds_center(), test_diagnose_error_unknown_error_perturbs_zero_param() (+18 more)

### Community 32 - "model_fn"
Cohesion: 0.25
Nodes (9): model_fn(), _build_hybrid_streaming_config(), _hier_loss(), _loss(), _loss_jax(), Any, Build a HybridStreamingConfig from a nested override dict. All 24 keys…, Mean-squared residual, optionally weighted by per-point sigma. Mirrors the… (+1 more)

### Community 33 - "parameter_registry.py"
Cohesion: 0.03
Nodes (67): given, settings, parametrize, Heterodyne parameter registry entries — verbatim from heterodyne docs. Source:…, heterodyne' should normalize to 'two_component'., test_heterodyne_mode_lists_14_params(), test_heterodyne_param_specs(), test_heterodyne_synonym_normalize() (+59 more)

### Community 34 - "Project"
Cohesion: 0.05
Nodes (60): QModelIndex, QStandardItem, QStandardItemModel, Tests for the JAX-free Project/Dataset/FitRun model., test_add_dataset_assigns_stable_unique_ids(), test_add_run_is_append_only_and_queued(), test_set_run_status_updates_in_place(), test_unknown_ids_return_none() (+52 more)

### Community 35 - "Two-Component Config Template"
Cohesion: 0.05
Nodes (55): analysis_mode Taxonomy Deprecation, Anti-Degeneracy Controller, Command-Line Interface, D_offset/D0 Negative-Dominant Validator Fix, Changelog Documentation, fixed_parameters/active_parameters Hardening, Desktop Analysis Workbench (GUI), Heterodyne Config Bounds Overrides (+47 more)

### Community 36 - "test_joint_ssr_floor.py"
Cohesion: 0.05
Nodes (36): _degraded_adapter(), _individual_cfg(), Keep-better floor on the heterodyne joint solve + Stage-1 de-duplication. Pins…, The CMA-ES escape builds the joint problem (incl. Stage 1) exactly once. Before…, On revert to x0 the result must NOT carry the rejected solve's covariance. When…, A reverted-to-x0 warm-start must report success=False / 'failed'. Exercises the…, Blast-radius boundary: an assembly backed by a global escape (joint_result is…, The probe-quieting context drops ONLY known noise messages (not a blanket level… (+28 more)

### Community 37 - "FitJob"
Cohesion: 0.07
Nodes (42): codex#3: shutdown() joins+closes the worker process and closes the queue., test_worker_handle_shutdown_reaps_process(), test_fitjob_is_frozen_and_picklable(), _collect(), skipif, pytest-qt tests for WorkerHandle: event forwarding, Died synthesis, cancel., _cancel_blocking() (the atexit/closeEvent teardown path) has no direct coverage…, MagicMock case pinning the exact synchronous escalation order: terminate() ->… (+34 more)

### Community 38 - "XPCSDataFilter"
Cohesion: 0.08
Nodes (32): A clean angle with ONE bad first-lag cell must not lose diagonal quality. Pre-…, test_quality_score_robust_to_single_boundary_artifact(), test_degenerate_matrix_nan_quality_score_is_dropped_not_kept(), _clean_matrix_with_raw_spike(), Regression tests for ``XPCSDataFilter._calculate_matrix_quality_score``. The…, A clean, symmetric correlation matrix carrying the raw tau=0 spike. Off-…, A valid matrix must score full marks despite the raw tau=0 spike., The fix must not mask real problems: lag-1 g2 > 2.0 stays penalized. (+24 more)

### Community 39 - "classify_option"
Cohesion: 0.07
Nodes (39): Action, Hint, _action(), test_choices_become_choices_completion(), test_count_action_is_flag_returns_none(), test_dir_hint_overrides_path(), test_literal_word_hint(), test_path_type_defaults_to_file() (+31 more)

### Community 40 - "diagonal_correction.py"
Cohesion: 0.09
Nodes (44): ArrayLike, Backend, Method, parametrize, Diagonal correction is mandatory for both physics models. Property: after…, Correction must NOT modify off-diagonal entries., ``window_size <= 0`` must not silently leave the diagonal uncorrected.…, A 1x1 matrix has no off-diagonal neighbors; every method must return it… (+36 more)

### Community 41 - "plots_view.py"
Cohesion: 0.08
Nodes (30): _SquareBase, _PhiSection, ndarray, QWidget, Scrollable per-phi results grid. ``_PhiSection`` and ``PhiResultsGrid`` — one…, Build the interactive residual-diagnostics row (3 plots), or a placeholder.…, Rebuild the grid from *bundle* (one interactive section per phi angle).…, Remove all section widgets so a prior run's grid never lingers. (+22 more)

### Community 42 - "heterodyne_logging.py"
Cohesion: 0.08
Nodes (28): test_log_quantile_scaling_never_raises_on_empty(), _layer_state(), log_anti_degeneracy_defense(), log_configured_layers_preamble(), log_effective_mode(), log_fit_start(), log_gradient_sanity_check(), log_optimization_results() (+20 more)

### Community 43 - "AnalysisMode"
Cohesion: 0.03
Nodes (118): Regression: initial_parameters.active_parameters: [] must mean 'fix…, test_empty_active_parameters_list_means_none_active(), test_missing_active_parameters_key_falls_back_to_defaults(), parametrize, Regression tests for the 2026-06-20 adversarial-review (codex) findings.…, test_bounds_reject_nonfinite(), test_finite_bounds_still_load(), test_finite_initial_parameters_still_load() (+110 more)

### Community 44 - "test_output_resolution.py"
Cohesion: 0.19
Nodes (23): _args(), _cfg(), Any, Path, Regression tests for unified CLI output-directory resolution. Adversarial-…, ``run_nlsq``'s Writer 1 (parameters/analysis_results_nlsq/convergence_metrics)…, When generate_nlsq_plots raises, _generate_post_fit_plots must report that…, Minimal ConfigManager stand-in — the resolver only reads ``.config``. (+15 more)

### Community 45 - "PreprocessingPipeline"
Cohesion: 0.05
Nodes (44): NoiseReductionMethod, NormalizationMethod, PreprocessingConfigurationError, PreprocessingError, PreprocessingPipeline, PreprocessingProvenance, PreprocessingResult, PreprocessingStage (+36 more)

### Community 46 - "GradientCollapseMonitor"
Cohesion: 0.05
Nodes (42): test_check_watched_parameter_collapse(), test_compute_reset_params_resets_to_mean(), test_get_diagnostics_with_history_and_watch(), test_config_defaults(), An empty monitor (callback never fired) must produce…, test_l4_fallback_block_when_no_observations(), LogCaptureFixture, Regression tests for the 2026-06-18 whole-codebase debug-audit fixes. Each test… (+34 more)

### Community 47 - "filtering_utils.py"
Cohesion: 0.19
Nodes (11): ndarray, Physical constants and parameter validation for homodyne XPCS analysis.…, # NOTE: These are reference values. The PRIMARY bounds used by NLSQ, Validate parameter values against bounds with detailed error reporting. This is…, Validate parameter values against bounds with tolerance. This is the legacy…, validate_parameters(), validate_parameters_detailed(), FilterCriteria (+3 more)

### Community 48 - ".fit"
Cohesion: 0.15
Nodes (16): _adjust_covariance_for_normalization(), _compute_normalization_factors(), _denormalize_params(), _format_bounds_summary(), _normalize_bounds(), _normalize_params(), ndarray, Normalize parameters from physical space to [0, 1] space. Parameters ----------… (+8 more)

### Community 49 - "test_plot_dispatch_logging.py"
Cohesion: 0.11
Nodes (23): _FakeResult, _make_data(), LogCaptureFixture, MonkeyPatch, Logging-behavior tests for ``xpcsjax.cli.plot_dispatch``. These tests pin the…, A SECOND separate call must emit its own WARNING (no cross-call collapse). With…, ``_save_fit_comparison_only``'s per-angle loop must pass ``phi_index=i`` -- the…, ``log_once`` uses a process-global dedup cache; reset for determinism. (+15 more)

### Community 50 - "rasterize"
Cohesion: 0.09
Nodes (27): P3 (twin-path): the residual heatmap color window must be computed from the…, test_residual_map_levels_computed_from_full_resolution(), test_c2_levels_clamp_to_unit_band(), test_residual_levels_are_symmetric_about_zero(), test_time_rect_none_on_degenerate_axes(), Tests for numeric block-mean display rasterization., test_infinities_are_sanitized_for_display(), test_large_array_is_downsampled_within_max_dim() (+19 more)

### Community 51 - "XPCSDataLoader"
Cohesion: 0.06
Nodes (34): load_xpcs_data(), _maybe_apply_mandatory_diagonal_correction(), Any, log_performance, Load data from NPZ cache file with q-vector validation. See…, Load and process data from HDF5 file., Detect whether an HDF5 file is APS old or APS-U new format. See…, Load data from APS old format HDF5 file. See… (+26 more)

### Community 52 - "test_heterodyne_result_builder.py"
Cohesion: 0.14
Nodes (24): test_build_result_from_nlsq_singular_marker_is_nan_not_inf(), NLSQResult, Tests for the heterodyne NLSQ result layer. Covers the ``NLSQResult`` dataclass…, _result(), test_build_from_nlsq_bad_tuple_length_raises(), test_build_from_nlsq_dict(), test_build_from_nlsq_dict_missing_keys_raises(), test_build_from_nlsq_object() (+16 more)

### Community 53 - "test_nlsq_support_modules.py"
Cohesion: 0.06
Nodes (48): Routing decision for a typical XPCS fit (10k points, 11 params). Expected: well…, Routing decision at chunked-fit scale (10M points, 14 params). Same code path…, test_perf_select_strategy_10k_points(), test_perf_select_strategy_10m_points(), name(), Direct unit tests for memory-aware NLSQ strategy routing. Localizes router…, Normalize the returned value (enum, string, dataclass) to upper-case name., Small datasets fit in memory — STANDARD strategy. (+40 more)

### Community 54 - "test_gui_redesign.py"
Cohesion: 0.07
Nodes (37): _bundle(), parametrize, Tests for the 2026-06-20 GUI redesign. Covers the new surfaces that replaced…, create_config writes the mode's template; the written config's mode matches., Toolbar owns the operational actions; File menu owns project lifecycle plus the…, codex/agy: a degenerate (n_phi,1,1) bundle (no tau=dt lag) renders without…, codex MEDIUM: optional arrays shorter than n_phi must degrade, never IndexError., agy LOW: a finished run with a valid viz bundle switches the central stack to… (+29 more)

### Community 55 - "nlsq/__init__.py"
Cohesion: 0.06
Nodes (44): NLSQAdapterBase, ABC, Any, Abstract base class for NLSQ adapters (FR-012). Shared ABC for NLSQAdapter and…, Abstract base class for NLSQ optimization adapters. Subclasses must implement…, Fit the model to data. Must be implemented by subclasses., NLSQResult, Result container for NLSQ optimization compatible with FitResult. (+36 more)

### Community 56 - "AntiDegeneracyController"
Cohesion: 0.04
Nodes (35): The controller must construct from a minimal (config, n_phi, n_physical,…, test_controller_instantiates_with_minimal_config(), Regression: explicit ``per_angle_mode="averaged"`` is accepted on every laminar…, The laminar hybrid-streaming driver must route per-angle mode resolution…, The shared seam returns ``"averaged"`` verbatim (the contract the inline paths…, The anti-degeneracy controller (standard + CMA-ES laminar paths) must accept an…, The seam refactor must not change the resolved actual mode for the pre-existing…, A truly unknown / removed-legacy token must still raise (via the shared… (+27 more)

### Community 57 - "test_multistart.py"
Cohesion: 0.09
Nodes (40): int64, _fit_factory(), fit(), ndarray, parametrize, Scientific/branch tests for xpcsjax.optimization.nlsq.multistart. Pure…, _single(), test_config_defaults() (+32 more)

### Community 58 - "cpu.py"
Cohesion: 0.08
Nodes (43): _cpu_info_no_extras(), _fake_cpuinfo(), _fake_lscpu(), isolated_configure(), Any, MonkeyPatch, parametrize, Regression tests for the CPU HPC configuration helpers. The thread-reservation… (+35 more)

### Community 59 - "ParameterSpace"
Cohesion: 0.12
Nodes (11): Regression test for Fix 2: homodyne ParameterSpace.from_config's public bounds…, test_from_config_rejects_inf_bound(), test_from_config_rejects_nan_bound(), ParameterSpace, Create ParameterSpace with package defaults (no config file). This method…, Parameter space definition with bounds for NLSQ optimization. This class…, Return a shallow copy safe for localized mutations., Get bounds for a specific parameter. Parameters ---------- param_name : str… (+3 more)

### Community 60 - "fit_nlsq_jax"
Cohesion: 0.08
Nodes (54): _config(), _fixed_value_survives(), _heterodyne_config(), _hybrid_streaming_fake_stratify_and_config(), _force_hybrid_streaming(), _physical_index(), parametrize, Integration tests: fixed_parameters/active_parameters actually constrain the… (+46 more)

### Community 61 - "heterodyne_config.py"
Cohesion: 0.05
Nodes (44): _make_failing_adapter(), fit(), parametrize, Regression: L4 must not report the discarded adapter monitor on fallback. Bug…, Build an NLSQAdapter subclass whose ``fit`` fires the L4 callback once then…, Adapter fires the L4 callback then fails; the wrapper fallback succeeds. The…, test_fallback_does_not_report_discarded_adapter_monitor(), _make_config() (+36 more)

### Community 62 - "adapter.py"
Cohesion: 0.04
Nodes (64): Adapter must not mint a 'good' fit from a missing objective. Quality-gate…, _result(), test_explicit_cost_still_used(), test_missing_objective_yields_nonfinite_chi2_not_good(), _data_unsorted_phi(), Regression test for phi-broadcast ordering in NLSQAdapter._flatten_xpcs_data.…, Duplicate angles in phi must not crash the broadcast (agy review, P2).…, test_flatten_duplicate_phi_does_not_crash() (+56 more)

### Community 63 - "parameter_utils.py"
Cohesion: 0.07
Nodes (48): ndarray, SimpleNamespace, _quantile_flat(), Scientific tests for xpcsjax.optimization.nlsq.parameter_utils. Pure helpers…, Mirror the diffusion-only g1^2 the estimator computes, for self-consistency., Build flat data where small lags sit at the ceiling and large lags at the floor., _static_g1_sq(), _static_stratified() (+40 more)

### Community 64 - "sequential.py"
Cohesion: 0.06
Nodes (48): LeastSquares, A per-angle singular JᵀJ yields a NaN per-angle covariance that the inverse-…, test_sequential_singular_angle_is_excluded_not_weighted_as_sigma_one(), sequential.py's own bounds-equality helpers, relocated but not altered., test_relocated_strip_and_restore_unchanged(), Regression: ``_jax_jacobian`` works under ``jax.jacfwd``. The inner…, Every parameter fixed (issue #58 follow-up coverage gap). Not reachable through…, test_all_physical_fixed_hits_zero_free_vector_fast_path() (+40 more)

### Community 65 - "StratificationConfig"
Cohesion: 0.09
Nodes (25): test_safe_float(), test_safe_int(), Resolve the installed ``xpcsjax_two_component.yaml`` path (import-anchored)., The shipped two_component template's stratification block parses to the…, No stratification block -> homodyne defaults., Explicit YAML nulls mean "unset" -- fall back to defaults, don't crash., `bool(None)` is False -- a null must not turn the safety check off., _shipped_two_component_template_path() (+17 more)

### Community 66 - "AdaptiveRegularizer"
Cohesion: 0.11
Nodes (19): test_controller_passes_true_n_phi_to_regularizer(), Phase 1+2: shared L3 boundary reads the canonical mapper, and…, test_adaptive_regularizer_averaged_groups(), test_adaptive_regularizer_constant_mode_has_empty_groups(), test_adaptive_regularizer_individual_groups(), AdaptiveRegularizationConfig, AdaptiveRegularizer, ndarray (+11 more)

### Community 67 - "heterodyne_physics_kernel.py"
Cohesion: 0.12
Nodes (32): EvalStrategy, _compute_c2_meshgrid(), _compute_c2_pointwise(), compute_c2_unified(), _fraction(), _half_transport_meshgrid(), ndarray, partial (+24 more)

### Community 68 - "compute_c2_heterodyne"
Cohesion: 0.16
Nodes (18): loss(), compute_c2_heterodyne(), single_angle_residual(), Compute the two-time heterodyne correlation (meshgrid path, JIT-compiled). Thin…, _per_phi(), _per_phi(), Any, ndarray (+10 more)

### Community 69 - "run_worker"
Cohesion: 0.10
Nodes (28): Tests for the JAX-free IPC primitives (job / emitter / log_capture)., test_emitter_blocks_for_terminal_but_drops_telemetry_when_full(), test_emitter_stamps_run_id_and_monotonic_seq(), test_log_handler_forwards_records_as_loglines(), EventEmitter, Any, Stamps and enqueues FitEvents from the worker to the parent., Stamp each event with the run id + a monotonic seq, then enqueue it. Terminal… (+20 more)

### Community 70 - "test_gui_jax_free.py"
Cohesion: 0.07
Nodes (41): _probe_import(), The GUI-importable surface must never pull JAX into the process. # Task 3…, xpcsjax.gui.views.plots_view must not import JAX at module level., xpcsjax.gui.views.raster must not import JAX at module level., xpcsjax.gui.project.persist must not import JAX (stdlib + project model only)., xpcsjax.gui.error_presenter must not import JAX (stdlib only)., xpcsjax.service.config must not import JAX (config validator, JAX-free)., xpcsjax.gui.data_inspect must not import JAX (h5py + stdlib only). (+33 more)

### Community 71 - "make_cfgmgr_and_data"
Cohesion: 0.06
Nodes (34): make_cfgmgr_and_data(), Build a real ``ConfigManager`` plus a heterodyne-loader data dict. Returns…, parametrize, Routing tests for the heterodyne standard-tier stratification gate. The gate…, LARGE memory tier + hybrid_streaming.enable=true → hybrid path, not stratified-…, per_angle_mode=individual + >=1M points → stratified-LS solver IS called., >=1M points in ``constant`` per-angle mode → stratified-LS solver IS called.…, Flat enable_cmaes=true (no nested cmaes block) + >=1M → stratified-LS NOT… (+26 more)

### Community 72 - "test_heterodyne_cmaes_warmstart_success_gate.py"
Cohesion: 0.07
Nodes (38): skipif, When CMA-ES actually runs, the per-angle result carries a ``global_escape`` tag…, The ``global_escape`` tag must survive aggregation into…, A broken off-diagonal cost recompute must raise, not silently hand Phase 3's…, A good NLSQ warm-start (reduced χ² < threshold) skips CMA-ES entirely. With a…, test_global_escape_surfaces_in_per_angle_metadata(), test_global_escape_tag_mirrors_winner_when_search_runs(), test_phase3_off_diag_cost_exception_propagates() (+30 more)

### Community 73 - "nlsq_plots.py"
Cohesion: 0.09
Nodes (26): API Reference Index, xpcsjax.io API, Low-level helpers and constants shared across the plots subpackage. Colormaps,…, ndarray, Shared color-limit helper for residual heatmap rendering. Single source of…, Symmetric ``(-v, v)`` color limits for a residual surface. ``v`` is the…, symmetric_residual_limit(), plot_c2_comparison_fast() (+18 more)

### Community 74 - "ProjectSidebar"
Cohesion: 0.09
Nodes (24): pytest-qt tests for the project sidebar + comparison view., A present summary with chi_squared=None (incomplete/older result) must not…, A field where the two runs disagree is prefixed with the diff marker; a field…, A diverged run's non-finite parameter (persisted as None) must render as "NaN"…, _summary(), test_comparison_view_marks_differing_values(), test_comparison_view_renders_nan_parameter_and_flags_diff(), test_comparison_view_shows_two_runs() (+16 more)

### Community 75 - "DatasetOptimizer"
Cohesion: 0.08
Nodes (28): DatasetInfo, Regression: strategy cache key must reflect memory-scaled recommendations.…, Equal .size, sigma crossing the memory limit -> distinct strategies. With…, test_equal_size_different_sigma_get_distinct_cached_strategies(), create_dataset_optimizer(), DatasetOptimizer, optimize_for_method(), Any (+20 more)

### Community 76 - ".from_config"
Cohesion: 0.11
Nodes (29): _ad_config_dict(), Parity tests: heterodyne ≥1M stratified-LS anti-degeneracy controller wiring., The driver must accept an optional anti_degeneracy_dict keyword (default None)., The driver's controller instantiation emits the laminar-style Layer 2/3/4…, No anti_degeneracy dict -> returns None, emits nothing, never raises., hierarchical.enable=False must suppress the Layer 2 banner (L2 IS gated)., gradient_monitoring.enable=False must suppress the Layer 4 banner (L4 IS gated)., L3 is NOT gated by a regularization.enable field — it stays on under master… (+21 more)

### Community 77 - "build_workbench"
Cohesion: 0.08
Nodes (34): QCloseEvent, pytest-qt tests for app wiring + worker cleanup., test_build_workbench_returns_wired_window(), test_close_triggers_shutdown(), Tests for Plan H Task 4: Save/Open project wiring in MainWindow., test_open_tolerates_deleted_result_dir(), test_save_then_open_round_trips_through_window(), Integration tests for the workbench surfaces (post-redesign). Asserts that… (+26 more)

### Community 78 - "test_strategy_chunking.py"
Cohesion: 0.05
Nodes (63): _balanced_dataset(), Any, parametrize, Tests for xpcsjax.optimization.nlsq.strategies.chunking. All pure functions…, _stratify(), test_adaptive_chunk_size_clamped_to_max(), test_adaptive_chunk_size_clamped_to_min(), test_adaptive_chunk_size_docstring_scenario_pinned() (+55 more)

### Community 79 - "quality_controller.py"
Cohesion: 0.10
Nodes (31): _baseline_config(), _minimal_data(), parametrize, Smoke tests for :mod:`xpcsjax.data.quality_controller`. The…, Each pipeline stage must return a QualityControlResult without raising., When the controller is disabled, every stage must return a minimal result…, The module-level convenience function constructs the controller and validates…, QualityLevel maps to the ``validation_level`` config string set in… (+23 more)

### Community 80 - "_NullLogger"
Cohesion: 0.09
Nodes (22): _CapturingOptimizer, _constant_mode_initial_and_bounds(), _FakeStratifiedData, _laminar_dataset(), _NullLogger, Any, ndarray, Shared fakes/fixtures for laminar-flow hybrid-streaming regression tests.… (+14 more)

### Community 81 - "test_heterodyne_memory_adapter.py"
Cohesion: 0.07
Nodes (48): _clear_cache(), _patch_threshold(), MonkeyPatch, Tests for heterodyne memory routing and adapter helper logic. *…, Overcommit prevention: the budget shrinks 1/N with the fit concurrency., test_assess_convergence_no_progress(), test_assess_convergence_nonfinite(), test_assess_convergence_poor_fit() (+40 more)

### Community 82 - "_make_synthetic_c2"
Cohesion: 0.33
Nodes (6): _make_synthetic_c2(), Build a tiny synthetic two-time correlation stack for unit tests., Dual-region quantile estimator recovers per-angle contrast and offset. Small-…, A phi index with no samples in `phi_indices` is a malformed input., test_quantile_estimator_raises_on_empty_phi_cell(), test_quantile_estimator_recovers_synthetic_contrast()

### Community 83 - "StratifiedResidualFunctionJIT"
Cohesion: 0.10
Nodes (28): _full_grid_chunk(), SimpleNamespace, Tests for xpcsjax.optimization.nlsq.strategies.residual_jit.…, _stratified(), test_dt_none_uses_fallback_and_warns(), test_empty_chunks_raises(), test_fixed_scaling_constant_mode(), test_get_diagnostics() (+20 more)

### Community 84 - "test_review_regressions.py"
Cohesion: 0.08
Nodes (34): _check_lazy_viz_import(), _make_heterodyne_config(), _make_heterodyne_data(), _make_heterodyne_result(), parametrize, Regression tests for the bugs surfaced by the Codex+Gemini review of the…, ``generate_nlsq_plots``' ``analysis_mode`` guard (line ~1623) falls back to…, n_total == n_physical (constant mode) → NotImplementedError upfront. (+26 more)

### Community 85 - "HierarchicalOptimizer"
Cohesion: 0.12
Nodes (18): _opt(), Tests for xpcsjax.optimization.nlsq.hierarchical. The two-stage optimizer…, test_config_defaults(), test_create_per_angle_loss_and_grad(), test_create_physical_grad_slices_physical_indices(), grad(), test_create_physical_loss_assembles_full_vector(), loss() (+10 more)

### Community 86 - "normalize_angle_to_symmetric_range"
Cohesion: 0.13
Nodes (24): _config(), Regression test: apply_angle_filtering_for_plot must normalize phi_angles and…, test_plot_and_optimization_paths_select_same_angles_across_wrap(), Regression tests for the 2026-06-17 debug-audit fixes (data/utils layer). Each…, test_frame_range_lower_bound_checked_when_end_is_none(), test_frame_range_valid_start_with_end_none_passes(), test_normalize_angle_preserves_minus_180_array(), test_normalize_angle_preserves_minus_180_scalar() (+16 more)

### Community 87 - ".phi"
Cohesion: 0.20
Nodes (13): _fit(), _revert_records(), Any, NDArray, Return the first present alias key as a NumPy array. Parameters ---------- keys…, Experimental correlation data (``c2_exp`` / ``c2``)., Phi angles (``phi_angles_list`` / ``phi_angles`` / ``phi``)., compute_single_angle() (+5 more)

### Community 88 - "heterodyne_model_stateful.py"
Cohesion: 0.06
Nodes (27): Main heterodyne model wrapper class., HeterodyneModelBase, ABC, ndarray, Model class hierarchy for heterodyne correlation analysis., Set default parameter values., Number of parameters (14)., Parameter names in canonical order. (+19 more)

### Community 89 - "AntiDegeneracyController"
Cohesion: 0.06
Nodes (34): BIPOP restart strategy, CMAESResult, CMAESWrapper, CMAESWrapperConfig, fit_nlsq, fit_nlsq_cmaes, CMA-ES per-mode default rationale, CMAESWrapper.should_use_cmaes (+26 more)

### Community 90 - "test_persist.py"
Cohesion: 0.15
Nodes (20): _probe_import(), Persist service: import-path equivalence + JAX-free guard., Off-shape uncertainties/covariance must be written at the documented shapes.…, A fitted-c2 NPZ older than the primary NPZ is a leftover from a PREVIOUS run in…, Not stale: fitted-c2 written at/after the primary npz (the real-world ordering,…, A 0-D result.parameters (__post_init__ permits it) must not IndexError.…, 0-D numpy arrays (scalar arrays) must coerce, not crash. Regression:…, test_extract_parameters_handles_zero_dim_result() (+12 more)

### Community 91 - "ResultBuilder"
Cohesion: 0.06
Nodes (34): _builder(), Regression tests for ``ResultBuilder`` (homodyne result_builder.py). The…, The iterations field must reflect optimizer iterations (nit), not nfev., With no nit, an explicit 'iterations' key is honored before defaulting., With neither nit nor iterations, default to 0 — never fall back to nfev., test_iterations_defaults_to_zero_without_nfev_fallback(), test_iterations_prefers_explicit_iterations_key_over_default(), test_iterations_uses_nit_not_nfev() (+26 more)

### Community 92 - "test_jacobian.py"
Cohesion: 0.08
Nodes (33): fixture_data(), _polynomial_residual(), ndarray, Smoke tests for the Jacobian-stats utility module.…, Smooth polynomial → low but finite gradient noise across perturbations. The…, 20-point polynomial fixture: xdata + initial params at the truth., Happy path: well-conditioned polynomial returns a (3,3) J^T J and a length-3…, A residual function that raises must yield (None, None) rather than propagating… (+25 more)

### Community 93 - "test_homodyne_engine_preservation.py"
Cohesion: 0.10
Nodes (24): load_or_init_golden(), Any, Path, Golden-snapshot load/init helper for the parity preservation suites. Mechanism:…, Return the golden arrays at ``path``. Args: path: Destination ``.npz`` path…, _build_laminar_fit(), _build_stratified_residual_fn(), ndarray (+16 more)

### Community 94 - "test_cache_source_file_validation.py"
Cohesion: 0.13
Nodes (29): _base_metadata(), _full_loader(), _loader(), LogCaptureFixture, Regression test for review finding A2 (2026-09-15) and its follow-up. The NPZ…, Source identity is checked FIRST: a re-reduced HDF5 that arrives with a re-…, Pointing ``data_file`` at a same-frame-window sibling file (same size,…, A content-preserving touch (``cp`` without ``-p``, ``rsync``, a backup restore)… (+21 more)

### Community 95 - "test_heterodyne_tied_parameters.py"
Cohesion: 0.12
Nodes (28): _base_config(), Tests for heterodyne tied_parameters (equality-constrained physics params). See…, A malformed mapping VALUE (unhashable, e.g. an empty list) must raise the…, The 'also listed as varying' warning must be scoped to an EXPLICIT…, A grouped-format vary: true for a tied child must NOT undo the tie --…, child also listed in active_parameters: tie wins, warning logged, not a hard…, child's configured bounds differing from its tied parent's bounds must log a…, test_tied_parameters_absent_is_noop() (+20 more)

### Community 96 - "test_post_install.py"
Cohesion: 0.12
Nodes (36): fake_venv(), isolated_env(), MonkeyPatch, Path, Tests for xpcsjax.post_install. Covers environment/shell detection, venv-path…, A venv-shaped directory with empty bash + fish activate scripts., Point HOME at a tmp dir and clear venv markers so config paths are sandboxed., test_configure_xla_mode_overwrites_with_force() (+28 more)

### Community 97 - "read_c2_preview"
Cohesion: 0.16
Nodes (21): _make_h5(), test_c2_preview_missing_dataset_returns_none(), test_c2_preview_missing_group_returns_none(), test_c2_preview_nested_group_at_key_returns_none(), test_c2_preview_reconstructs_group_half_matrix(), test_c2_preview_slices_and_downsamples(), test_c2_preview_unknown_type_group_path_returns_none(), test_metadata_lists_datasets_without_loading() (+13 more)

### Community 98 - "ndarray"
Cohesion: 0.12
Nodes (11): log_calls, ndarray, Compute g1 correlation function for this model., Get default parameter values., Compute diffusion contribution to g1. g₁_diff = exp[-q²/2 ∫|t₂-t₁| D(t')dt'], Default values for typical XPCS measurements., Compute shear contribution to g1. g₁_shear = [sinc(Φ)]² where Φ = (qL/2π)…, Default values for typical shear flow. (+3 more)

### Community 99 - "test_shear_weighting.py"
Cohesion: 0.12
Nodes (23): Tests for xpcsjax.optimization.nlsq.shear_weighting. The shear weight ``w(phi)…, test_apply_weights_to_loss_disabled(), test_apply_weights_to_loss_enabled(), test_compute_weighted_mse_disabled(), test_compute_weighted_mse_enabled(), test_create_defaults_phi0_index_when_no_names(), test_create_disabled_returns_none(), test_create_none_config_returns_none() (+15 more)

### Community 100 - "ResultSummary"
Cohesion: 0.05
Nodes (53): QTreeWidgetItem, Regression tests for the 2026-06-21 GUI debug-audit fixes. Each test pins one…, P3: a partial/external nlsq_result.json whose nlsq_diagnostics is null must…, P3: the inspector must not raise when handed a non-dict diagnostics block…, A minimal summary whose convergence_status carries a unique marker so the…, P2 (twin-path): while run A is active, the user clicks an earlier finished run…, Guard against over-correction: with no competing selection, a finishing active…, P3 (stale-state): the append-only Fitting-Process log must be cleared when the… (+45 more)

### Community 101 - "DataInspectDialog"
Cohesion: 0.10
Nodes (21): _make_h5(), Tests for DataInspectDialog — the wiring for the previously-orphaned…, HDF5 names may contain spaces; the label they're rendered into is not a…, A 3-D dataset with a zero-length first axis makes the auto-mode read raise…, A corrupt-but-signature-valid HDF5 file can raise RuntimeError/KeyError/…, test_corrupt_file_error_shown_not_crash(), test_empty_leading_axis_shows_error_not_crash(), test_lists_datasets() (+13 more)

### Community 102 - "multistart.py"
Cohesion: 0.05
Nodes (51): float64, test_check_zero_volume_bounds(), test_generate_lhs_starts_within_bounds_and_reproducible(), test_generate_random_starts_within_bounds_and_reproducible(), test_get_dataset_size(), test_get_phi_from_data(), test_screen_starts_keeps_lowest_cost_sequential(), test_screen_starts_parallel_path() (+43 more)

### Community 103 - "test_heterodyne_tied_residuals.py"
Cohesion: 0.14
Nodes (20): Direct residual-closure tests: every wired heterodyne residual function must…, heterodyne_core.py: _fit_cmaes's model_func (per-angle CMA-ES escape)., heterodyne_core.py: _fit_local's jax_residual_fn., heterodyne_core.py: _make_numpy_residual_fn's residual_fn., heterodyne_stratified_ls.py: residual_fn (scaling-first: scaling head, physics…, strategies/heterodyne_hybrid_streaming.py: model_fn (scaling-first: physics is…, Sanity check exercised again here (redundant with Task 2's test, kept as the…, heterodyne_core.py: _fit_joint_averaged_multi_phi's base_residual_fn. (+12 more)

### Community 104 - "test_heterodyne_data_prep.py"
Cohesion: 0.11
Nodes (29): parametrize, Coverage for heterodyne data-prep pure functions (audit finding #16). Exercises…, test_compute_weights_exclude_diagonal_zeros_diagonal(), test_compute_weights_inverse_variance_requires_sigma(), test_compute_weights_inverse_variance_shape_mismatch(), test_compute_weights_unknown_method(), test_degrees_of_freedom_normal(), test_degrees_of_freedom_underdetermined_floors_at_one() (+21 more)

### Community 105 - "PerAngleScalingPlan"
Cohesion: 0.08
Nodes (36): parametrize, _quantiles(), Phase-0 unit tests for PerAngleScalingPlan (spec §4 Seam 3). quantile_scaling =…, The jnp variant returns the same values as the NumPy one (so the traced…, test_expand_back_averaged_broadcasts_then_splits_physics(), test_expand_back_constant_uses_frozen_quantiles_physics_only_vector(), test_expand_back_individual_reorders_to_full_per_angle(), test_expand_covariance_averaged_replicates_scalar_blocks() (+28 more)

### Community 106 - "log_exception"
Cohesion: 0.03
Nodes (94): Regression: per-angle plot filenames must not collide. Bug history: the three…, test_experimental_missing_phi_list_gets_distinct_files(), test_postfit_close_angles_get_distinct_files(), test_simulated_close_angles_get_distinct_files(), _write_config(), _cm(), Tests for the post-fit plotting service (xpcsjax/service/plots.py)., test_generate_plots_forwards_to_viz_and_forces_agg() (+86 more)

### Community 107 - "test_engine_route_bugfixes.py"
Cohesion: 0.11
Nodes (30): _make_well_posed_case(), Build (model, c2, phi) where ``c2`` is NOISELESS model correlation at a…, _build_single_angle_model(), _make_config(), ndarray, NLSQConfig, Regression tests for two engine-route ``two_component`` defects. Both were…, An explicit per-set ``max_nfev`` is scaled by ``n_phi`` for the combined joint… (+22 more)

### Community 108 - "json_safe"
Cohesion: 0.10
Nodes (6): TestJsonSafeContainers, TestJsonSafeEdgeCases, TestJsonSafeFloatSanitization, TestJsonSafeNumpyTypes, json_safe(), Recursively convert numpy arrays and special types to JSON-safe types.…

### Community 109 - "result_presenter.py"
Cohesion: 0.06
Nodes (39): QApplication, QIcon, QRectF, Unit tests for StatusManager collaborator (set_status / append_log)., StatusManager routes set_status/append_log through to the MainWindow widgets., StatusManager is a QObject child of MainWindow., Calling set_status twice replaces the previous status., Multiple append_log calls accumulate in the log widget. (+31 more)

### Community 110 - "test_jax_backend.py"
Cohesion: 0.07
Nodes (29): Tests for xpcsjax.core.jax_backend identified by Gemini round-2 review. Covers…, The pre-computed-factors hot path must produce bit-identical output to the…, compute_g2_scaled_with_factors must equal compute_g2_scaled to float64., Physics identity: g₂ = offset + 0·g₁² = offset when contrast=0., Sanity check: contrast=1 must not be constant., Batched basic correction must preserve 1x1 matrices like the scalar path., After BUG1 fix the hash key includes interior quartile samples., A concrete dt still produces a finite sinc² surface (guard is None-only). (+21 more)

### Community 111 - "_write_npz_compressed"
Cohesion: 0.14
Nodes (28): floating, integer, Any, MonkeyPatch, Path, Tests for NPZ + JSON artifact serialization., If writing fails mid-stream, no stale .npz or .tmp should remain., _sample_arrays() (+20 more)

### Community 112 - "post_install.py"
Cohesion: 0.12
Nodes (28): test_completion_activation_missing_script(), test_completion_and_xla_source_paths_exist(), test_completion_bash_activation_injection_and_idempotency(), test_fish_completion_is_nonfatal_noop_but_xla_stays(), test_install_completion_activation_skips_outside_venv(), test_install_xla_config_scripts(), detect_shell_type(), get_completion_source_path() (+20 more)

### Community 113 - "test_quality_gate_fixes.py"
Cohesion: 0.10
Nodes (22): parametrize, Regression tests for the quality-gate audit fixes. Locks in the behavioral…, StrEnum result must still compare/serialize as its string value., Construct a minimal OptimizationResult, overriding selected fields., A genuinely failed fit may carry empty parameters — not an error., _result(), test_converged_result_with_empty_parameters_is_rejected(), test_converged_result_with_nonfinite_parameters_is_rejected() (+14 more)

### Community 114 - "present_failure"
Cohesion: 0.33
Nodes (7): Tests for the JAX-free failure->message mapping., test_oom_message_is_friendly(), test_plain_message_passthrough(), test_traceback_is_summarized_with_details_retained(), present_failure(), Map raw failure text to a user-facing (title, message, details) triple., Return ``(title, friendly_message, details)`` for a failure string. Parameters…

### Community 115 - "RunController"
Cohesion: 0.09
Nodes (22): Path, Tests for the JAX-free figure-export helper (export.py). Two required cases:…, export_figures copies *.png/pdf from <result_dir>/plots recursively., Two PNGs with the same filename in different sub-dirs must both survive., export_figures must return [] and NOT raise when there is no plots/ dir., _on_export_figure shim delegates to RunController; export_figures is called., test_export_copies_figures_to_dest(), test_export_disambiguates_same_name_files() (+14 more)

### Community 116 - "test_static_individual_invariant.py"
Cohesion: 0.11
Nodes (26): parametrize, Regression: static modes are pinned to the ``individual`` per-angle layout on…, Wiring: every large-data reduced-chi2 DOF computation in the wrapper must…, Laminar reduction is preserved (byte-identical routing for laminar)., Pin the wiring: the streaming fit must resolve its mode via the helper, never…, _static_cfg(), test_controller_disabled_for_static_so_no_compression(), test_resolve_static_pinned_forces_individual() (+18 more)

### Community 117 - "service/persist.py"
Cohesion: 0.09
Nodes (35): Regression tests for the 2026-06-17 debug-audit fixes (CLI layer)., test_config_summary_reads_real_surface(), test_save_npz_none_covariance_uses_documented_shapes(), test_save_results_rejects_bad_format(), For output_format='both', the durable NPZ must be written before the JSON, so a…, test_extract_parameters_scalar_uncertainties_dropped_no_crash(), test_extract_parameters_scalar_uncertainty_matches_single_param(), test_save_results_both_writes_durable_npz_before_json() (+27 more)

### Community 118 - "_save_fig"
Cohesion: 0.16
Nodes (21): test_save_fig_with_none_is_noop(), _fresh_fig(), Security regression tests: plot save paths must be validated. Quality-gate…, Regression: legitimate saves into a not-yet-existing subdir still work., When an explicit base_dir is given, an absolute path outside it is rejected…, test_datashader_comparison_creates_missing_parent_dir(), test_datashader_heatmap_creates_missing_parent_dir(), test_datashader_heatmap_rejects_traversal() (+13 more)

### Community 119 - "test_fit_quality.py"
Cohesion: 0.13
Nodes (27): _cm(), parametrize, SimpleNamespace, Tests for the mode-agnostic NRMSE metric (xpcsjax/service/fit_quality.py)., Real laminar_flow (homodyne) fit through the service seam: same block, same…, nrmse == sqrt(SSR/n_valid) / std(data) over the t>0, off-diagonal mask,…, _result(), test_angle_count_mismatch_raises_not_truncates() (+19 more)

### Community 120 - "test_result_presenter.py"
Cohesion: 0.06
Nodes (45): QRunnable, Unit tests for ResultPresenter collaborator (show_result / show_error /…, _show_result_with_bundle switches to the grid page (index 1) when a bundle…, _show_result_with_bundle falls back to text (page 0) when result_dir has no…, A slower load for an earlier selection must not clobber a newer one. Regression…, _show_result_with_bundle with result_dir=None forces the text fallback., show_inspector passes the summary to the inspector dock without error., show_inspector(None) clears the inspector without error. (+37 more)

### Community 121 - "load_dataset"
Cohesion: 0.13
Nodes (25): _cm(), SimpleNamespace, Tests for the argparse-free data service (xpcsjax/service/data.py)., An empty phi array must warn-and-skip, not raise from ``np.argmin``.…, End-to-end path the caller-side ``is not None`` guard failed to cover., test_load_dataset_empty_phi_does_not_crash(), test_load_dataset_no_subset_keeps_all(), test_load_dataset_raises_when_no_correlation_matrix() (+17 more)

### Community 122 - "HeterodyneParameterManager"
Cohesion: 0.04
Nodes (34): HeterodyneParameterManager, Any, ndarray, Total number of physics model parameters (14)., Number of physics parameters that vary in optimization., Names of varying physics parameters (excludes scaling)., Indices of varying parameters in the 14-element physics array., ``(child_index, parent_index)`` pairs in the 14-element physics array. Empty… (+26 more)

### Community 123 - "CreateConfigDialog"
Cohesion: 0.07
Nodes (16): workflow LOW/agy NIT: a non-blank malformed numeric surfaces (not silently…, A write failure during the overwrite retry must surface as a warning, not…, test_create_config_dialog_collects_inputs(), test_create_config_dialog_omits_blank_optionals(), test_create_config_dialog_raises_on_invalid_numeric(), test_on_create_config_guards_overwrite_retry_failure(), CreateConfigDialog, _parse_optional() (+8 more)

### Community 124 - "service/config.py"
Cohesion: 0.07
Nodes (45): Regression tests for config_generator scalar substitution (audit C2).…, test_overrides_produce_parseable_yaml(), agy#2 (service side): scalar initial_parameters yields an error, not…, test_validate_config_scalar_initial_parameters_reports_not_crashes(), _ip(), JAX-free config validation + template loading., test_available_modes_are_the_four_known(), test_out_of_bounds_value_is_an_error() (+37 more)

### Community 125 - "test_memory_manager_cache_gating.py"
Cohesion: 0.11
Nodes (26): _make_manager(), skipif, Emergency cleanup must not thrash JAX recompiles under live-array pressure. RCA…, A live-array regime logs pressure warnings at DEBUG, not WARNING. During a…, Critical pressure under live arrays stays visible (WARNING) but not CRITICAL., Outside the live-array regime the original WARNING/CRITICAL are preserved., A first zero-result GC must short-circuit the old 3x collect loop., Repeated warnings must NOT compound gc thresholds toward 0 (agy F2). The old… (+18 more)

### Community 126 - "test_cmaes_trigger.py"
Cohesion: 0.18
Nodes (18): _laminar_cmaes_config(), parametrize, CMA-ES auto-triggers at scale_ratio >= 1000 (homodyne default). XPCS multi-…, All resolved per-angle modes expand back to the dense per-angle 2*n_phi +…, A CMA-ES solve that actually ran must not be discarded into a failed result.…, A converged fixed-constant CMA-ES solve whose covariance is ``None`` must not…, A bug in OptimizationResult construction must crash, not be silently downgraded…, Audit [2026-07-23] (PR #15 review, pr15-test-review): a NaN… (+10 more)

### Community 127 - "test_cache_safety.py"
Cohesion: 0.15
Nodes (21): _bare_loader(), _good_payload(), MonkeyPatch, ndarray, Path, Trust-boundary regression tests for :meth:`XPCSDataLoader._load_from_cache`.…, _peek_npz_array_header must report the same shape/dtype as a full read.…, A cached c2_exp shape that exceeds the allocation budget must be rejected via… (+13 more)

### Community 128 - ".fit"
Cohesion: 0.21
Nodes (9): _wrapped(), Any, ndarray, NLSQResult, Resolve the solver method, fetch/create a cached ``CurveFit``, and run it.…, Run NLSQ optimisation using nlsq.CurveFit. Wraps the residual function into the…, Run NLSQ optimisation using a pure JAX-traceable residual function. This method…, Run NLSQ optimisation with automatic memory-based strategy routing. Parameters… (+1 more)

### Community 129 - "test_gradient_monitor.py"
Cohesion: 0.16
Nodes (15): _monitor(), Tests for xpcsjax.optimization.nlsq.gradient_monitor. The monitor is a pure…, test_check_collapse_after_consecutive_triggers(), test_check_disabled_returns_ok(), test_check_healthy_gradient_is_ok(), test_check_rearms_after_recovery(), test_check_skips_off_interval(), test_check_tracks_best_params() (+7 more)

### Community 130 - "physics_nlsq.py"
Cohesion: 0.17
Nodes (19): _grid_reference_diffusion(), _grid_reference_shear(), parametrize, Regression tests for the degenerate 1x1-grid branch in physics_nlsq.py. The…, The static (<7 params) early return must handle 0-d t1 like the other branches., test_diffusion_degenerate_matches_grid(), test_shear_degenerate_matches_grid(), test_static_shear_accepts_0d_scalar_t1() (+11 more)

### Community 131 - "test_strategy_executors.py"
Cohesion: 0.26
Nodes (17): _logger(), MonkeyPatch, ndarray, Tests for xpcsjax.optimization.nlsq.strategies.executors. The executors wrap…, _resid(), test_large_executor_optimize_result_with_pcov(), test_large_executor_optimize_result_without_pcov(), test_large_executor_reraises() (+9 more)

### Community 132 - ".load_config"
Cohesion: 0.08
Nodes (14): Any, Lightweight configuration validation. Checks for required sections and valid…, T051: Log key configuration values at INFO level. Logs analysis mode, dataset…, Normalize configuration schema for backward compatibility. Handles multiple…, Normalize analysis_mode to canonical lowercase form. Handles case-insensitive…, Validate config_version against package version. Warns if config version…, Normalize experimental_data section. Supports two formats: 1. Template/Legacy:…, Initialize the configuration manager. Loads from ``config_file`` (via… (+6 more)

### Community 133 - "test_aps_u_selective_read_d7.py"
Cohesion: 0.18
Nodes (18): _bare_loader(), _load_aps_u_eager_reference(), Regression test for review finding D7 (2026-09-15). ``_load_aps_u_format`` used…, Replicate the pre-D7 "read every valid bin, then select" algorithm. Uses the…, Default config (no data_filtering): selective and eager reads agree., A phi-range filter (still no quality filtering) also stays selective-safe., Regression test (2026-09-15 PR #79 review): the quality-filtering branch runs…, Quality filtering enabled: candidates are narrowed then loaded, but the final… (+10 more)

### Community 134 - "ProjectDialogHandler"
Cohesion: 0.10
Nodes (12): ProjectDialogHandler, QObject, Project and config dialog slot collaborator for MainWindow., Open a file-chooser and add the selected config as a new dataset. The start…, Open a save-file dialog and persist the current project to disk. Defaults the…, Open a file-chooser and load a previously saved project file. Defaults the…, Prompt for confirmation and close the current project., Open a file-chooser, then a read-only HDF5 dataset/C₂ inspector dialog. The… (+4 more)

### Community 135 - "ConfigTextEditorDialog"
Cohesion: 0.09
Nodes (18): Malformed YAML must not be written to disk — the on-disk config stays intact., Valid YAML still saves and accepts the dialog (validation isn't over-strict)., agy HIGH: a failed load must disable Save so a blank editor can't truncate the…, A write OSError must surface and keep the dialog open — never escape the slot…, test_config_text_editor_accepts_valid_yaml_on_save(), test_config_text_editor_disables_save_on_load_failure(), test_config_text_editor_rejects_invalid_yaml_on_save(), test_config_text_editor_round_trip() (+10 more)

### Community 136 - "test_plots_view.py"
Cohesion: 0.10
Nodes (21): parametrize, Tests for interactive PyQtGraph plot widgets (plots_view)., Colorbar construction must degrade gracefully, like the image item's own…, test_diagonal_residual_traces_diag_against_t1(), test_map_applies_color_lookup_table(), test_map_axes_labelled_t1_t2(), test_map_keeps_equal_t1_t2_range_without_aspect_lock(), test_map_view_survives_unresolvable_colormap() (+13 more)

### Community 137 - "ShearSensitivityWeighting"
Cohesion: 0.11
Nodes (16): Array, _compute_weights_jax(), ndarray, partial, Shear-sensitivity weighted loss for anti-degeneracy defense. This class manages…, Initialize the weighter and precompute weights for the initial phi0. See the…, Compute angle weights for given phi0. Performance Optimization (Spec 001 -…, Update phi0 estimate from current parameters. Parameters ---------- params :… (+8 more)

### Community 138 - "heterodyne_memory.py"
Cohesion: 0.10
Nodes (22): _grep_callers(), parametrize, Path, Chunking / streaming smoke tests for memory-aware NLSQ routing. The Phase 5…, The homodyne (HYBRID_STREAMING) and heterodyne (STREAMING) routers both return…, Return non-test python files under ``xpcsjax/`` that mention ``symbol``., ``select_nlsq_strategy`` must have non-test callers — otherwise the "memory-…, Number of int64 points whose index array exceeds ``factor × threshold_gb``.… (+14 more)

### Community 139 - "validate_xpcs_data"
Cohesion: 0.16
Nodes (25): _clean_data(), _correlation_warnings(), ndarray, parametrize, Tests for xpcsjax.data.validation (M-2: closes the largest coverage gap).…, Warnings emitted by the near-zero-lag correlation-matrix check., Regression: the excluded tau=0 self-correlation spike must not fail QC. The…, A <=1-frame matrix has no lag-1 (first superdiagonal) to check; the fix must… (+17 more)

### Community 140 - "test_runtime_shell.py"
Cohesion: 0.14
Nodes (20): _bash_executable(), parametrize, Path, Tests for xpcsjax.runtime.shell and the runtime package re-exports., Resolve the real Git-Bash executable, not the WSL launcher stub. Windows…, Source xla_config.bash with ``mode`` in an isolated HOME. An empty ``mode``…, _source_xla_config(), test_get_completion_script_returns_absolute_existing_path() (+12 more)

### Community 141 - "test_system_validator.py"
Cohesion: 0.14
Nodes (17): MonkeyPatch, Tests for xpcsjax.runtime.utils.system_validator. Covers the version-parsing…, test_config_templates_probe_finds_all(), test_cpu_info_probe_reports(), test_dependency_versions_probe_buckets_unparseable_version(), test_dependency_versions_probe_passes(), test_jax_installation_probe_x64_enabled(), test_python_version_probe_passes() (+9 more)

### Community 142 - "get_adaptive_memory_threshold"
Cohesion: 0.12
Nodes (27): MonkeyPatch, Concurrency-aware memory-budget routing (OOM-overcommit prevention). Background…, A fit that is STANDARD when alone escalates to OUT_OF_CORE under load., Item-3 regression guard: PYTEST_XDIST_WORKER_COUNT shrinks the budget. This is…, test_detect_fit_concurrency_default_is_one(), test_detect_fit_concurrency_explicit_arg_wins(), test_detect_fit_concurrency_fit_env_beats_xdist(), test_detect_fit_concurrency_floors_at_one() (+19 more)

### Community 143 - "SequentialResult"
Cohesion: 0.22
Nodes (13): fixed_parameters must survive the sequential per-angle tier (Task 7d).…, test_fixed_parameter_wiring_reaches_sequential_solver(), _fake_sequential(), _build_sequential_laminar_fit(), Regression: the Site 4 (sequential per-angle) covariance-rescale call…, Reuse the small synthetic laminar fixture but force the SEQUENTIAL per-angle…, Site 4 (sequential per-angle) result carries the symmetric anti-degeneracy…, test_sequential_laminar_emits_symmetric_activation_keys() (+5 more)

### Community 144 - "heterodyne_result_builder.py"
Cohesion: 0.12
Nodes (23): parametrize, test_build_failed_result_with_initial(), test_build_failed_result_without_initial(), test_build_from_arrays_with_and_without_jacobian(), test_build_from_scipy_with_jacobian(), test_build_from_scipy_without_jacobian(), test_compute_covariance_regularizes_near_singular(), test_compute_covariance_well_conditioned() (+15 more)

### Community 145 - "test_load_degradation_signal.py"
Cohesion: 0.47
Nodes (5): _bare_loader(), DATA-1: degraded-fallback paths must leave a detectable signal. The quality-…, An instance with __init__ bypassed — we only exercise the helper., test_record_degradation_accumulates(), test_record_degradation_appends_and_logs_error()

### Community 146 - "test_laminar_mode_banners.py"
Cohesion: 0.11
Nodes (29): _laminar_controller(), Unit tests for the shared per-angle-mode banner formatter. Covers…, The shared quantile helper is reused by the averaged path, so its banner must…, Pin the wiring: the summary label must be produced by the helper, never re-…, test_averaged_banner_text(), test_broadcast_mode_label_resolves_word(), test_compute_fixed_per_angle_scaling_emits_neutral_banner(), test_constant_banner_has_no_zero_scaling() (+21 more)

### Community 147 - "format_comparison"
Cohesion: 0.14
Nodes (15): Plain (no Qt) unit tests for xpcsjax.gui.comparison.format_comparison.…, chi_squared=None (incomplete/older result) must not crash the render --…, A reported-but-non-finite parameter renders 'NaN', not the '—' sentinel used…, _summary(), test_empty_summaries_render_empty_string(), test_marks_differing_values_with_diff_marker(), test_nan_parameter_renders_distinct_from_absent(), test_shows_two_runs() (+7 more)

### Community 148 - "LargeDatasetExecutor"
Cohesion: 0.11
Nodes (15): test_executor_names_and_progress(), LargeDatasetExecutor, OptimizationExecutor, ABC, Optimization Strategy Executors for NLSQ. This module implements the Strategy…, Strategy name for logging., Whether this strategy supports progress bars., Standard curve_fit optimization for small datasets (<1M points). Uses… (+7 more)

### Community 149 - "save_nlsq_json_files"
Cohesion: 0.20
Nodes (10): A ``Path.stat()`` failure on the post-write size check (the best-effort…, A write that dies mid-file must not leave a truncated artifact. The writes go…, TestSaveNlsqJsonFiles, boom(), _atomic_json_dump(), Any, Path, Write ``obj`` as JSON to ``path`` via a sibling temp file + ``os.replace``.… (+2 more)

### Community 150 - "log_heterodyne_completion"
Cohesion: 0.22
Nodes (16): _chi2_args(), _completion_physics(), _het_result(), Regression tests for the 2026-06-17 debug-audit fixes (optimization/core)., test_averaged_physics_first_marker_reads_physics_from_head(), test_averaged_scaling_first_marker_reads_physics_from_tail(), test_gradient_chi2_does_not_raise_on_static_args(), test_hessian_chi2_does_not_raise_on_static_args() (+8 more)

### Community 151 - "test_homodyne_covariance_contract.py"
Cohesion: 0.11
Nodes (22): _fit_with_pcov(), parametrize, Homodyne covariance contract: no fabricated uncertainties on any wrapper path.…, F1 on the >=1M stratified-LS path (direct strategy call): whenever the strategy…, Same as tests.parity.test_laminar_execute_layers._fit but returns pcov too., F1/F2 on out-of-core: a singular free-block JᵀJ is NaN + flag, not pinv/1e-5.…, test_laminar_stratified_ls_never_ships_identity_sigma(), test_out_of_core_singular_hessian_reports_nan_and_flag() (+14 more)

### Community 152 - "test_memory_manager_logging.py"
Cohesion: 0.11
Nodes (15): _make_manager(), Memory-manager cleanup/monitoring must be observed, not silently swallowed.…, The fallback shim must swallow when policy='suppress' (existing behaviour)., Exercise the fallback shim directly under forced HAS_V2_LOGGING=False.…, The manager<->monitor reference cycle must stay broken. Before the fix,…, A weak-wrapped callback whose target was collected must stay non-fatal.…, Build a manager with the background pressure monitor disabled., A crash inside a best-effort cleanup path logs DEBUG and is swallowed. (+7 more)

### Community 153 - "test_fit_queue.py"
Cohesion: 0.16
Nodes (18): _queue(), pytest-qt tests for the bounded-concurrency fit queue (fake handles)., test_bounded_concurrency_runs_one_at_a_time(), test_cancel_active_frees_slot_and_starts_next(), test_cancel_active_removes_partial_output_dir(), test_cancel_and_shutdown(), test_cancel_does_not_delete_non_per_run_dir(), test_cancel_pending_removes_before_start() (+10 more)

### Community 154 - "test_logging_primitives.py"
Cohesion: 0.14
Nodes (16): Defensive-logging contract tests for the logging primitives. Logging is…, test_json_formatter_handles_empty_exc_info_tuple(), test_json_formatter_never_raises_on_bad_format_args(), test_json_formatter_redaction_does_not_overredact(), test_json_formatter_redacts_secrets(), test_json_formatter_schema_and_jsonsafe(), test_json_safe_recursion_guard_truncates_deep_nesting(), test_log_exception_never_raises_on_bad_context() (+8 more)

### Community 155 - "test_adapter_xdata_cache.py"
Cohesion: 0.19
Nodes (15): _coord_sensitive_model_func(), _expected(), MonkeyPatch, Regression tests for the xdata->JAX conversion cache in get_or_create_model.…, Build an [n,3] xdata array (columns t1, t2, phi_idx=0)., Production model_func with the physics kernel stubbed to g1 = t1 + t2., Two live arrays with different coordinates must get their OWN conversions — no…, Repeated calls with the SAME array object (the cache-hit path) stay correct —… (+7 more)

### Community 156 - "build_heterodyne_stratified_data"
Cohesion: 0.03
Nodes (116): Wiring check: drive the REAL fit_with_stratified_hybrid_streaming_heterodyne…, test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse(), _Cfg, _install_model_stub(), _make_synthetic_heterodyne(), MonkeyPatch, parametrize, Tests for heterodyne stratified hybrid-streaming pipeline (Phase 2). (+108 more)

### Community 157 - "optimization/test_validation.py"
Cohesion: 0.12
Nodes (25): test_validate_covariance_rejects_non_finite(), test_validate_result_consistency_low_and_high_still_pass(), _make_result(), Coverage tests for `xpcsjax.optimization.nlsq.validation`. Closes the /double-…, Build a duck-typed OptimizationResult-like object., The QualityFlag bridge maps the 'acceptable' band to 'marginal' and passes the…, test_classify_quality_flag_maps_band_to_qualityflag(), test_validate_covariance_accepts_symmetric_finite_positive_diag() (+17 more)

### Community 158 - "ndarray"
Cohesion: 0.07
Nodes (26): test_result_dataclass(), callback(), create_gradient_function_with_monitoring(), monitored_grad_fn(), Wrap gradient function to include monitoring. Parameters ---------- grad_fn :…, per_angle_grad(), per_angle_loss(), physical_grad() (+18 more)

### Community 159 - "References and Citations Doc"
Cohesion: 0.19
Nodes (22): Andrade1910 - Viscous flow in metals (creep law), Bradbury2018 - JAX composable transformations, Duri2005 - Time-resolved-correlation measurements, Hansen2016 - CMA evolution strategy tutorial, He2024 - Transport coefficient approach (PNAS), He2025 - Bridging microscopic dynamics and rheology (PNAS), Kubo1966 - Fluctuation-dissipation theorem, Lumma2000 - Area detector based photon correlation (+14 more)

### Community 160 - "MemoryStats"
Cohesion: 0.14
Nodes (10): _as_weak_callable(), MemoryStats, Wrap a bound method so holding it doesn't keep its owner alive. Callers…, Comprehensive memory statistics and monitoring., Refresh the system memory fields from :mod:`psutil` in place., Return a human-readable pressure level. Returns ------- str One of ``"low"``,…, Initialize the memory pressure monitor. Parameters ---------- warning_threshold…, Register a callback invoked when warning pressure is entered. Parameters… (+2 more)

### Community 161 - "residual_jit.py"
Cohesion: 0.06
Nodes (45): Protocol, Tests for the model-agnostic ``PointEvaluator`` adapter. Phase 1.1 introduces a…, The heterodyne adapter satisfies the runtime-checkable Protocol., ``HomodynePointEvaluator.eval_points`` must equal the raw kernel exactly., The adapter satisfies the runtime-checkable Protocol., ``HeterodynePointEvaluator`` must plug into the stratification ENGINE. The…, test_heterodyne_evaluator_is_a_point_evaluator(), test_heterodyne_evaluator_returns_per_angle_meshgrid() (+37 more)

### Community 162 - "fit_with_stratified_hybrid_streaming"
Cohesion: 0.09
Nodes (22): _CapturingAdaptiveOptimizerSpy, _FakeStratifiedDataWithSigma, _HierarchicalOptimizerSpy, Regression: the L2 hierarchical loss must honor per-point sigma weighting,…, Stand-in for HierarchicalOptimizer: captures loss_fn/grad_fn and returns a stub…, _sigma_weighted_mse (the arithmetic building block _hier_loss/_loss_jax call)…, Audit follow-up (2026-07-23): the plan's initial fix only edited the `else`…, Stand-in for NLSQ's AdaptiveHybridStreamingOptimizer: captures the fit() kwargs… (+14 more)

### Community 163 - "PhysicsFactors"
Cohesion: 0.13
Nodes (9): PhysicsFactors, Create PhysicsFactors from experimental configuration. This is the recommended…, Validate physics factors for physical consistency. Checks: 1. All values are…, Convert to tuple for JIT-compatible function calls. Returns the two pre-…, Convert to dictionary for serialization or inspection. Returns ------- dict…, Human-readable string representation., Detailed string representation., Pre-computed physics factors for XPCS correlation calculations. This immutable… (+1 more)

### Community 164 - "QualityControlConfig"
Cohesion: 0.16
Nodes (14): test_quality_config_defaults(), test_quality_config_from_dict_reads_quality_control_section(), test_quality_config_from_empty_dict_uses_defaults(), LogCaptureFixture, Regression test for review finding A3 (2026-09-15). ``_repair_scaling_issues``…, test_repair_nan_values_defaults_to_false(), test_repair_nan_values_disabled_by_default_leaves_data_untouched(), test_repair_nan_values_logs_warning_with_count() (+6 more)

### Community 165 - "xpcsjax/data/__init__.py"
Cohesion: 0.05
Nodes (52): Regression tests for the 2026-08-05 data-module deep-RCA debug-audit fixes.…, test_calculate_matrix_quality_score_never_returns_nan(), test_create_chunked_iterator_rejects_misaligned_phi(), test_create_chunked_iterator_slices_aligned_phi_correctly(), test_phi_angle_filter_matches_raw_360_convention_angle(), test_validate_config_schema_malformed_section_is_flagged_not_absorbed(), test_validate_config_schema_none_section_reports_missing_params(), test_validate_numeric_range_rejects_nan_when_wrapped_and_unbounded() (+44 more)

### Community 166 - "test_main_window.py"
Cohesion: 0.24
Nodes (12): pytest-qt smoke tests for MainWindow (logic-free view, controller-less…, Closing a project must not orphan queued/active fit workers (codex review).…, Two runs for the SAME dataset must write to distinct dirs (no overwrite)., test_close_event_calls_queue_shutdown(), test_close_project_clears_folder_name_from_sidebar(), test_close_project_stops_active_and_pending_runs(), test_create_project_shows_folder_name_in_sidebar(), test_repeated_runs_get_distinct_output_dirs() (+4 more)

### Community 167 - "xla_config.py"
Cohesion: 0.12
Nodes (17): parametrize, test_build_parser_returns_parser(), test_post_install_shell_choices_preserved(), test_xla_config_threads_roundtrip(), build_parser(), configure_xla(), get_cpu_info(), main() (+9 more)

### Community 168 - "FitQueueController"
Cohesion: 0.07
Nodes (24): P3 (signal-wiring): a LogLine/Iteration/Banner for a run no longer in _handles…, test_stale_non_terminal_event_for_freed_run_is_dropped(), Regression tests for the 2026-06-19 GUI debug-audit fixes. Each test pins one…, codex#4: after Open Project, Run works (an active dataset is established)., codex#6/agy#6: the dead-path check expands ${ENV}/~ before testing existence.…, codex#6: clear_map() drops a displayed image so stale data is not retained., agy#4: a Finished arriving after cancel() must keep status 'cancelled', not…, test_cancel_race_finished_stays_cancelled() (+16 more)

### Community 169 - "test_heterodyne_physics_validators.py"
Cohesion: 0.21
Nodes (15): Coverage for the heterodyne physics-constraint validators (audit finding #5).…, test_correlation_inputs_clean_is_valid(), test_correlation_inputs_nan_is_error(), test_correlation_inputs_non_monotonic_time_is_error(), test_correlation_inputs_shape_mismatch_is_error(), test_cross_parameter_fraction_sum_exceeds_unity(), test_cross_parameter_offset_ref_below_threshold_no_violation(), test_cross_parameter_offset_ref_dominance_negative() (+7 more)

### Community 170 - "events.py"
Cohesion: 0.10
Nodes (32): Analysis Workbench (GUI) Guide, emit_started_then_finished(), exit_without_terminal(), Any, Importable spawn-target fakes for the WorkerHandle tests. Run inside spawned…, sleep_forever(), _drain(), _install_fake_services() (+24 more)

### Community 171 - "PhiAngleFilter"
Cohesion: 0.10
Nodes (14): DataFilteringError, Exception, Raised when data filtering encounters an error., PhiAngleFilter, Validate target angle ranges. Wrapped ranges (``min_angle > max_angle``) are…, Phi angle filtering utility class for XPCS analysis. Provides vectorized…, NDArray, Reconstruct full correlation matrix from half matrix (APS storage format).… (+6 more)

### Community 172 - "memory_manager.py"
Cohesion: 0.14
Nodes (11): AllocationError, log_calls(), MemoryManagerError, MemoryPressureError, Any, Exception, Memory pressure monitoring and GC-pressure response for large XPCS fits. This…, Base exception for memory manager errors. (+3 more)

### Community 173 - "wrap_stratified_function_with_transforms"
Cohesion: 0.10
Nodes (16): test_wrap_model_applies_inverse_and_preserves_attrs(), test_wrap_model_no_state_returns_original(), model(), test_wrap_model_noncallable_returns_original(), test_wrap_stratified_applies_inverse_and_delegates_attrs(), test_wrap_stratified_no_state_returns_original(), The JIT-safe jnp inverse must be numerically identical to the numpy inverse…, test_jax_inverse_transform_matches_numpy() (+8 more)

### Community 174 - "test_out_of_core_route.py"
Cohesion: 0.20
Nodes (11): _canned_ooc(), _FakeConfig, _NullLogger, A4: the out-of-core route is one shared function so the initial-decision and…, Static mode + explicit 'constant' must resolve DOF through the pin (->…, laminar_flow honors the requested token unchanged (pin is a no-op)., Wiring guard for the extraction itself: calling the same pure function twice…, _run() (+3 more)

### Community 175 - "fit_nlsq entry point"
Cohesion: 0.33
Nodes (19): analysis_mode (static_isotropic/anisotropic/laminar_flow/two_component), CMA-ES global escape, ConfigManager, fit_nlsq entry point, fixed_parameters / active_parameters, OptimizationResult dataclass, Per-angle reparameterization (L1: constant/averaged/individual/auto), tied_parameters (two_component only) (+11 more)

### Community 176 - "stratified_ls.py"
Cohesion: 0.06
Nodes (41): _nonlinear_residual(), ndarray, E1: ``covariance._chunked_jacfwd_dense`` (moved here from…, test_empty_params(), test_matches_for_various_col_block_sizes(), fn(), test_matches_jax_jacfwd_exactly(), _nonlinear_residual() (+33 more)

### Community 177 - "configure_logging"
Cohesion: 0.25
Nodes (15): _managed_console_handler(), _managed_logger(), Handler, Logger, Phase-1b wiring tests for xpcsjax logging. Confirms env/YAML selection wiring…, Return the managed console handler (StreamHandler, not a FileHandler). Debug-…, test_context_filter_installed_once(), test_debug_precedence_env_over_yaml() (+7 more)

### Community 178 - "ndarray"
Cohesion: 0.16
Nodes (9): ndarray, Trim model time axis to match post-exclusion data length. The data pipeline may…, Return the current full parameter array. Returns ------- np.ndarray Parameter…, Set parameter values. Parameters ---------- params : np.ndarray or dict of str…, Compute the two-time correlation matrix. Parameters ---------- phi_angle :…, Compute residuals between model and data. Parameters ---------- c2_data :…, Compute the reference g1 correlation. Parameters ---------- params :…, Compute the sample g1 correlation. Parameters ---------- params : np.ndarray,… (+1 more)

### Community 179 - "Banner"
Cohesion: 0.20
Nodes (16): Tests for the JAX-free layer-status map + banner classifier., test_classify_banner_ignores_ordinary_log_lines(), test_classify_banner_recognizes_engine_prefixes(), test_layer_status_l5_inactive_sentinels_are_false(), test_layer_status_maps_diagnostics_keys(), classify_banner(), _l5_active(), layer_status_from_diagnostics() (+8 more)

### Community 180 - "._configure_impl"
Cohesion: 0.12
Nodes (15): test_resolve_level(), Any, Path, Apply this configuration to the logging system. Merges default external-library…, Create a configuration from a dictionary. Unknown keys are ignored; missing…, Resolve the log file path from a ``file:`` config block, or None. Builds…, Configure xpcsjax logging. Thread-safe configuration of the logging system.…, Apply the logging configuration (called under the instance lock). (+7 more)

### Community 181 - "compute_jacobian_stats"
Cohesion: 0.13
Nodes (16): test_jacobian_stats_failure_returns_none(), bad_residual(), test_jacobian_stats_plain_residual(), test_jacobian_stats_uses_jax_residual_attr(), C9: the per-angle contrast/offset override parser used to be two byte-identical…, _RecordingLogger, test_malformed_value_is_dropped_with_default_wording(), test_none_input_yields_no_overrides() (+8 more)

### Community 182 - "PerAngleScaling"
Cohesion: 0.07
Nodes (32): _lag_separated_dataset(), Regression tests for xpcsjax.core.heterodyne_scaling_utils.…, C2 that genuinely decays with lag, so lag-separation differs from a global…, The added dt mask must be a no-op when every value is finite., test_all_finite_inputs_unchanged(), test_nan_delta_t_is_dropped_not_poisoning_thresholds(), Characterization tests for PerAngleScaling pack/unpack helpers. Quality-gate…, test_constant_mode_propagates_first_angle_to_all() (+24 more)

### Community 183 - "test_covariance_placeholder_contract.py"
Cohesion: 0.18
Nodes (12): _cfg(), parametrize, One failed-covariance semantics on every heterodyne path. A covariance that is…, The flag judges the SOLVER's reduced covariance, not the expanded one. A tied…, Make every nlsq solve report the singular-Jacobian all-inf pcov., _singular_curve_fit(), test_engine_route_singular_reports_nan_and_flag(), test_fixed_or_tied_slots_do_not_trip_the_placeholder_flag() (+4 more)

### Community 184 - "BatchStatistics"
Cohesion: 0.13
Nodes (11): BatchStatistics, Any, Batch-level statistics tracking for streaming optimization. This module…, Calculate success rate from recent batches in buffer. Returns ------- float…, Calculate average loss from recent successful batches. Returns ------- float…, Calculate average iterations from recent batches. Returns ------- float Average…, Circular buffer for tracking batch-level statistics. Maintains statistics for…, Return comprehensive statistics dictionary. Returns ------- dict Dictionary… (+3 more)

### Community 185 - "test_perf_regression.py"
Cohesion: 0.28
Nodes (8): _build_synthetic_c2(), _het_smoke_config_dict(), ndarray, Path, Wall-clock regression suite for xpcsjax v0.1 hot paths. The /double-check…, End-to-end timing for the heterodyne per-angle local NLSQ fit. Smallest…, Tiny heterodyne config — same shape as test_heterodyne_cmaes.py., test_perf_heterodyne_per_angle_local_fit()

### Community 186 - "ExecutionResult"
Cohesion: 0.17
Nodes (10): test_execution_result_dataclass(), ExecutionResult, Any, ndarray, Execute standard curve_fit optimization., Execute large dataset optimization., Initialize streaming executor. Parameters ---------- checkpoint_config : dict,…, Execute streaming optimization using AdaptiveHybridStreamingOptimizer. (+2 more)

### Community 187 - "test_cache_q_validation.py"
Cohesion: 0.20
Nodes (16): _angle_hash(), _loader_with_q(), LogCaptureFixture, Regression test for cache q-vector validation (audit C1). A q-keyed selective…, Pre-existing caches without the new fingerprint must still load (warn-only)., A cache built for a different dt must not silently reuse its t1/t2 axes., Pre-existing caches without dt fingerprinting must still load (warn-only)., optimization_config.angle_filtering shapes the cached (q, phi) selection. It… (+8 more)

### Community 188 - "test_config_jax_free.py"
Cohesion: 0.29
Nodes (9): _imports_jax(), _probe_import(), Regression guard: importing xpcsjax.config must not load JAX (F1)., Import ``module`` in a fresh interpreter; return 0/1/2 (see contract above)., Return True iff importing ``module`` cleanly loads jax. Raises AssertionError…, test_importing_config_does_not_load_jax(), test_importing_parameter_manager_does_not_load_jax(), test_importing_registry_and_types_does_not_load_jax() (+1 more)

### Community 189 - "test_validation_branches.py"
Cohesion: 0.26
Nodes (17): Any, Branch-coverage complement for xpcsjax.optimization.nlsq.validation.…, _result(), test_fit_quality_acceptable_band_logged(), test_fit_quality_cmaes_max_restarts_warns(), test_fit_quality_cmaes_other_reason_passes(), test_fit_quality_condition_number_ok_via_pcov_fallback(), test_fit_quality_condition_number_too_high() (+9 more)

### Community 190 - "log_performance"
Cohesion: 0.12
Nodes (15): LoggerType, test_log_calls_logs_and_reraises_exception(), test_log_calls_logs_entry_exit_with_args_and_result(), test_log_calls_skips_when_level_disabled(), test_log_performance_logs_above_threshold(), test_log_performance_logs_and_reraises(), boom(), log_calls() (+7 more)

### Community 191 - "ValidationResult"
Cohesion: 0.11
Nodes (13): test_result_to_dict_roundtrip(), test_validation_result_defaults(), Run all validation tests. Each test is isolated: an uncaught exception is…, Check that the running Python is >= 3.12. Returns ------- ValidationResult INFO…, Verify JAX imports, exposes devices, and has x64 precision enabled. Returns…, Import xpcsjax and resolve every public lazy-loaded symbol. Returns -------…, Verify every shipped YAML config template is present on disk. Returns -------…, Report CPU core count and system RAM (informational only). Returns -------… (+5 more)

### Community 192 - "ndarray"
Cohesion: 0.14
Nodes (15): test_validate_bounds_consistency_upper_length_mismatch(), test_validate_initial_params_within_bounds_direct(), test_validate_array_dimensions_accepts_matching_shapes(), test_validate_array_dimensions_rejects_empty_and_mismatched(), test_validate_bounds_consistency_accepts_sorted_bounds(), test_validate_bounds_consistency_rejects_inverted_or_misshaped(), ndarray, Validate all input data. (+7 more)

### Community 193 - "NLSQStrategy"
Cohesion: 0.14
Nodes (19): fixed_parameters must survive the out-of-core (chunk-wise J^T J accumulation)…, test_fixed_parameter_survives_out_of_core_fit(), _force_out_of_core(), _force_strategy(), _force(), _FakeConfig, _FakeStratifiedData, _force_standard() (+11 more)

### Community 194 - "NumericalValidator"
Cohesion: 0.09
Nodes (29): parametrize, Tests for numerical-validation helpers. ``numerical_validation`` raises…, test_set_bounds_disable_enable(), test_validate_gradients_detects_nonfinite(), test_validate_gradients_disabled_skips(), test_validate_gradients_finite_ok(), test_validate_loss_detects_nonfinite(), test_validate_loss_disabled_skips() (+21 more)

### Community 195 - "LHS multistart"
Cohesion: 0.21
Nodes (17): LHS multistart, AntiDegeneracyController (L1-L5), fit_nlsq_multi_phi, NLSQ adapters (CurveFit, trust-region), select_nlsq_strategy, StratifiedResidualFunctionJIT, Anti-Degeneracy Defense (per iteration), CMA-ES escape (seed-pinned, keep-better) (+9 more)

### Community 196 - "Memory-Aware Strategy Routing"
Cohesion: 0.13
Nodes (17): Advanced Topics index, JAX Environment Configuration, JAX_ENABLE_X64 (float64 mandatory), XLA_FLAGS (device count, constant_folding), NLSQStrategy.HYBRID_STREAMING, memory_fraction clamp, NLSQStrategy.OUT_OF_CORE, NLSQStrategy.STANDARD (in-memory) (+9 more)

### Community 197 - "test_no_pickle_loads.py"
Cohesion: 0.19
Nodes (16): expr, _find_violations(), _is_np_load(), _iter_source_files(), Path, Regression guard: no unsafe ``np.load`` calls inside ``xpcsjax/``. The NPZ…, Sanity: variable-smuggled allow_pickle is flagged., Return True if ``node`` is ``np.load``, ``numpy.load``, or bare ``load``. (+8 more)

### Community 198 - "_unpack_result_params"
Cohesion: 0.10
Nodes (20): Regression: homodyne per-angle results are ``[c_0..N-1, o_0..N-1, phys]``. The…, Heterodyne result with wrong param count should raise ValueError., Per-angle layout: [c_0..N-1, o_0..N-1, 14 physical]., Homodyne result with <3 params should raise ValueError., test_unpack_heterodyne_per_angle_layout(), test_unpack_heterodyne_size_mismatch_raises(), test_unpack_homodyne(), test_unpack_homodyne_per_angle_layout_offset_not_misread() (+12 more)

### Community 199 - "npz_cache.py"
Cohesion: 0.17
Nodes (18): dtype, hash_filter_config(), load_from_cache(), migrate_cache_template(), peek_npz_array_header(), Any, log_performance, NPZ cache load/save/validate for… (+10 more)

### Community 200 - "test_hybrid_streaming_retry.py"
Cohesion: 0.12
Nodes (22): _AlwaysFailsOptimizer, _enable(), _fake_hybrid_streaming_config(), _FlakyOptimizer, _logger(), Any, MonkeyPatch, ndarray (+14 more)

### Community 201 - "test_layer5_gating.py"
Cohesion: 0.16
Nodes (16): _make_controller(), parametrize, ShearSensitivityWeighting (anti-degeneracy Layer 5) is gated by analysis mode.…, Build a controller with minimal-but-valid arguments., Layer 5 is active ONLY for laminar_flow (the mode with a shear rate)., Layer 5 is inactive for static modes — no flow direction, no shear peak., Layer 5 is inactive for two_component (heterodyne) mode., The 'heterodyne' synonym must produce the same gating. (+8 more)

### Community 202 - "parallel_accumulator.py"
Cohesion: 0.10
Nodes (23): MonkeyPatch, test_worker_functions_in_process(), _batch_timeout(), create_ooc_kernels(), _ooc_compute_chi2_chunk(), _ooc_compute_chunk(), _ooc_worker_cleanup(), _ooc_worker_init() (+15 more)

### Community 203 - "compute_diagonal_overlay_stats"
Cohesion: 0.16
Nodes (16): Tests for compute_diagonal_overlay_stats., test_diagonal_overlay_out_of_bounds_raises(), test_diagonal_overlay_rmse_matches_manual(), test_diagonal_overlay_shapes_match(), test_diagonal_overlay_variance_ignores_single_inf(), test_compute_diagonal_overlay_stats_2d_input_raises(), test_compute_diagonal_overlay_stats_shape_mismatch_raises(), compute_diagonal_overlay_stats() (+8 more)

### Community 204 - "AdvancedMemoryManager"
Cohesion: 0.08
Nodes (24): _contextmanager, test_advanced_memory_manager_collected_without_gc_sweep(), The atexit monitor cleanup must not emit logging-handler error noise. At…, test_atexit_cleanup_silences_closed_stream_logging_errors(), AdvancedMemoryManager, _cleanup_active_monitors(), logged_errors(), Enter the context manager. Returns ------- AdvancedMemoryManager This instance. (+16 more)

### Community 205 - "ConfigManager"
Cohesion: 0.12
Nodes (16): AdvancedMemoryManager, AnalysisMode (4 modes), ConfigManager, DataQualityController, load_xpcs_data, NLSQConfig, parameter_registry, ParameterManager (+8 more)

### Community 206 - "NLSQ CurveFit (trust-region least squares)"
Cohesion: 0.15
Nodes (16): cli (argparse), fit_nlsq, generate_nlsq_plots, gui.ipc.worker, MainWindow (PySide6 workbench), OptimizationResult, service.fit, service.plots (+8 more)

### Community 207 - "test_stratified_max_iter_grading.py"
Cohesion: 0.21
Nodes (14): _good_chi2_inputs(), NLSQWrapper, R2 (status-grading parity): laminar stratified-LS must grade a max_nfev-limited…, status==0 (SciPy max_nfev code) + finite reduced chi^2 -> max_iter, not failed., When no status code is threaded, the SciPy message string also triggers it., A non-budget failure (e.g. status=-1, no max_nfev reason) is not upgraded., A converged solve stays converged regardless of the threaded reason/status., The relabel must not perturb parameters / chi^2 / covariance. (+6 more)

### Community 208 - "FakeHandle"
Cohesion: 0.13
Nodes (17): FakeHandle, QObject, Shared ``WorkerHandle`` test double: never spawns a real process. Consolidates…, Fake the atexit/closeEvent-only fully-synchronous cancel path., Regression tests for the click-path-audit findings (state-conflict bugs). Each…, test_add_dataset_does_not_retarget_while_run_active(), test_dataset_row_selection_updates_active_dataset(), test_load_config_preserves_prior_sidebar_selection() (+9 more)

### Community 209 - ".from_config"
Cohesion: 0.17
Nodes (17): Regression test for grouped-format value/bounds coercion (audit C10). The…, A config specifying both spellings of the same parameter must raise, not…, min == max is a fixed parameter, not an error (mirrors the registry)., test_equal_bounds_allowed(), test_grouped_format_accepts_public_angle_alias(), test_grouped_format_accepts_public_velocity_alias(), test_grouped_format_coerces_string_scalars_to_float(), test_grouped_format_rejects_alias_and_canonical_collision() (+9 more)

### Community 210 - "HomodyneModel"
Cohesion: 0.07
Nodes (26): HomodyneModel must reject negative end_frame before constructing., end_frame=-1 must raise ValueError before any JAX computation., Error message must reference 'sentinel' so callers understand the fix., A properly resolved end_frame must not raise., TestHomodyneModelInitValidation, _coerce_bool_flag(), HomodyneModel, ndarray (+18 more)

### Community 211 - "fit_nlsq_multistart_heterodyne"
Cohesion: 0.12
Nodes (23): _data_ssr(), ndarray, SimpleNamespace, Multi-seed keep-best for the heterodyne joint CMA-ES escape. RCA (C044…, ``cmaes_n_seeds=3`` ⇒ ``fit_with_cmaes`` is invoked 3× with seeds 42/43/44.…, Fake OptimizationResult: parameters[0] encodes the SSR data_ssr reads., _result(), test_all_failed_returns_first_with_inf_ssr() (+15 more)

### Community 212 - "test_escape_disabled_hint.py"
Cohesion: 0.18
Nodes (15): Actionable hint when a heterodyne joint fit fails with NO global escape…, A failed escape-less fit logs a hint naming cmaes.enable and n_seeds., Missing ``success`` defaults to converged (no hint); the helper never raises., test_log_hint_emits_actionable_cmaes_message(), test_log_hint_robust_to_missing_attributes(), test_log_hint_silent_when_escape_already_enabled(), test_log_hint_silent_when_fit_succeeded(), test_no_hint_when_cmaes_already_enabled() (+7 more)

### Community 213 - "fit_with_out_of_core_accumulation"
Cohesion: 0.13
Nodes (13): Regression: explicit ``per_angle_mode="averaged"`` expands DOF like ``"auto"``.…, test_explicit_averaged_matches_auto_expansion(), Number of fitted parameters., _effective_param_count_for_ooc(), fit_with_out_of_core_accumulation(), _embed_step(), _pcov_from_active_jtj(), Any (+5 more)

### Community 214 - "test_parallel_accumulator.py"
Cohesion: 0.13
Nodes (26): _int_chunks(), _kernels(), parametrize, Scientific tests for xpcsjax.optimization.nlsq.parallel_accumulator. Three…, Integer-valued (JtJ, Jtr, chi2) chunks — exact under float summation., _static_physics_config(), test_accumulate_sequential_empty_raises(), test_accumulate_sequential_sums_correctly() (+18 more)

### Community 215 - ".__init__"
Cohesion: 0.12
Nodes (14): load_xpcs_config(), Exception, log_calls, Path, Raised when required dependencies are not available., Raised when configuration is invalid or missing required parameters., Load an XPCS configuration from a YAML or JSON file. YAML is the primary…, Initialize the loader from a config file path or an in-memory dict. Exactly one… (+6 more)

### Community 216 - "test_layer_gate_wiring.py"
Cohesion: 0.23
Nodes (11): _build(), Task 29 follow-up: verify the model-lineage gate is wired through the…, Construct a controller through the production API., The production constructor must accept ``analysis_mode`` so callers can thread…, Lineage gate must short-circuit ShearSensitivityWeighting for heterodyne…, Homodyne laminar_flow path must keep Layer 5 active — this is the regime the…, Backward-compat: omitting ``analysis_mode`` keeps existing behavior (all layers…, test_from_config_accepts_analysis_mode_kwarg() (+3 more)

### Community 217 - "fit_two_component_via_engine"
Cohesion: 0.03
Nodes (101): _cfg(), NLSQConfig, parametrize, Uncertainty parity between the two heterodyne in-memory joint-fit paths. The…, test_engine_route_uncertainties_match_multi_phi_dof(), test_rescale_covariance_dof_factor(), Locks in the dual-region rationale: diagonal-only input cannot recover offset.…, test_quantile_estimator_diagonal_only_fails_to_recover_offset() (+93 more)

### Community 218 - "compute_chi_squared"
Cohesion: 0.26
Nodes (8): ndarray, All-zero sigma means all pixels excluded → chi-squared = 0., All-positive sigma must give same result before and after the fix path., After BUG2 fix, zero-sigma pixels are excluded (contribute 0), not Inf., A single zero-sigma element must not produce Inf chi-squared., TestChiSquaredZeroSigma, compute_chi_squared(), Compute chi-squared goodness of fit. χ² = Σᵢ [(data_i - theory_i) / σᵢ]²…

### Community 219 - "test_heterodyne_config_bounds_override.py"
Cohesion: 0.17
Nodes (14): Regression: heterodyne ``parameter_space.bounds`` list overrides are honored.…, ``initial_parameters.parameter_names``/``values`` listing both ``v_beta`` and…, ``parameter_space.bounds`` with the template name ``v_beta`` overrides the…, Absent an explicit override, ``beta`` keeps the conservative registry default…, The ``phi0_het`` template name also translates onto canonical ``phi0``., An unrecognised bound name is skipped, not fatal (defensive parity)., ``v_beta`` and its canonical name ``beta`` both listed in…, _resolved_bounds() (+6 more)

### Community 220 - "xpcs_loader.py"
Cohesion: 0.13
Nodes (14): _loader(), Regression test for APS-U empty (q,phi) selection (audit C9). When phi…, test_empty_selection_raises(), test_nonempty_selection_returned_unchanged(), parametrize, Cross-platform safety of the cache-filename guard. The original guard tested…, test_accepts_plain_filename(), test_rejects_unsafe_cache_filenames() (+6 more)

### Community 221 - "heterodyne_parameter_names.py"
Cohesion: 0.24
Nodes (10): _manager(), Tests for ParameterManager.expand_reduced_result., Fully-untied 'constant' mode: n_scaling=0, physics_first irrelevant., A physics param excluded via active_parameters (not tied) must be NaN in the…, test_expand_reduced_result_fixed_physics_param_gets_nan(), test_expand_reduced_result_physics_first_with_scaling(), test_expand_reduced_result_scaling_first_with_scaling(), test_expand_reduced_result_tied_child_mirrors_parent_covariance() (+2 more)

### Community 222 - "_FakeConfigManager"
Cohesion: 0.11
Nodes (11): _FakeConfigManager, nlsq_result.npz must round-trip with allow_pickle=False (no SEC-1 regression).…, Heterodyne fits label from result.nlsq_diagnostics, not the config manager.…, Homodyne fits (no nlsq_diagnostics parameter_names) get names reconstructed…, Multi-angle 'individual' per-angle mode: N (contrast, offset) pairs., End-to-end: a homodyne-shaped result no longer degrades to param_0..N, and the…, test_resolve_parameter_names_prefers_nlsq_diagnostics(), test_resolve_parameter_names_synthesizes_multi_angle_scaling_head() (+3 more)

### Community 223 - "test_frame_dimension_guard.py"
Cohesion: 0.12
Nodes (28): Guard against unbounded allocation from a crafted/corrupt correlation file.…, test_accepts_realistic_allocation(), test_accepts_realistic_frame_count(), test_accepts_square_matrix_shape(), test_rejects_absurd_frame_count(), test_rejects_huge_matrix_count_even_with_legal_frame_count(), test_rejects_negative_matrix_count(), test_rejects_non_2d_matrix_shape() (+20 more)

### Community 224 - "test_config_unwrap.py"
Cohesion: 0.16
Nodes (12): captured_nlsq(), _fake_data(), Regression test for the heterodyne config unwrap in ``_fit_nlsq_heterodyne``.…, Already-flat dicts (legacy/tests) must still parse correctly., ``analysis_mode`` placed only in the nested NLSQ section must reach NLSQConfig.…, Minimal ConfigManager replacement holding only ``self.config``., Patch HeterodyneModel + fit_nlsq_multi_phi to capture the NLSQConfig., Nested ``optimization.nlsq.*`` settings must reach NLSQConfig. (+4 more)

### Community 225 - "test_phase5_model_function_modes.py"
Cohesion: 0.30
Nodes (11): _data_obj(), _make_fn(), _phys(), Phase 5 — the JIT model_function slices per resolved mode (no crash, correct…, # NOTE: plan named ``xpcsjax.core.analysis.AnalysisMode`` / ``NLSQFitter``; the, Averaged with (c,o) must equal individual with all angles = (c,o)., Raw (non-stratified) grid data object the standard path builds., test_averaged_equals_individual_when_uniform() (+3 more)

### Community 226 - "run_fit"
Cohesion: 0.14
Nodes (28): _cm(), parametrize, SimpleNamespace, Equivalence tests for the typed NLSQ override application (F8)., test_max_iterations_is_coerced_to_int(), test_multistart_sets_nested_enable_and_n_starts(), test_non_dict_config_is_noop(), test_none_fields_write_nothing() (+20 more)

### Community 227 - "test_adaptive_regularization.py"
Cohesion: 0.18
Nodes (15): _make(), Coverage for Layer-3 adaptive regularization (audit finding #15). These…, At mean~=0, compute_regularization_jax's CV must fall back to std (not 0),…, A uniform per-angle group (std == 0) must not NaN the gradient. Every per-angle…, H-4: NaN/inf params (a diverged step) must force trust-region rejection. The…, compute_regularization_jax's CV safe-divide must not poison jax.grad. A group…, test_auto_and_absolute_modes_both_finite(), test_disabled_returns_zero() (+7 more)

### Community 228 - "_logger_that_raises_on_log"
Cohesion: 0.12
Nodes (11): _logger_that_raises_on_log(), Logger, Return a fresh logger whose only handler always raises on emit., log_calls: a raising handler must never abort the decorated function., Logger raises on the *entry* emit; function must still return., Logger raises on the *success* emit; function must still return., Even when the logger raises, a real function exception still propagates., include_args=True path: raising logger must not abort the call. (+3 more)

### Community 229 - "test_debug_audit_2026_07_23_diagonal_skip.py"
Cohesion: 0.20
Nodes (12): Regression: the loader's mandatory diagonal correction must not silently…, End-to-end (design spec Testing item 2): drive the REAL load_experimental_data…, # NOTE: PreprocessingPipeline.process() itself does not consult the, End-to-end: apply_diagonal_correction_batch must NOT be called again when the…, _synthetic_data(), test_correct_diagonal_stage_marks_data_as_corrected(), test_disabled_preprocessing_does_not_set_marker(), test_loader_applies_configured_diagonal_correction_end_to_end() (+4 more)

### Community 230 - "NLSQOptimizationError"
Cohesion: 0.28
Nodes (6): NLSQOptimizationError, Exception, Initialize numerical error. Parameters ---------- message : str Detailed error…, Base exception for all NLSQ optimization errors. This is the base class for all…, Initialize base optimization error. Parameters ---------- message : str…, Return formatted error message with context.

### Community 231 - "MemoryPressureMonitor"
Cohesion: 0.10
Nodes (17): Regression: ``get_pressure_trend`` must not iterate ``_pressure_history`` bare.…, test_get_pressure_trend_survives_concurrent_appends(), Regression: pressure-state hysteresis dead-zone must preserve prior state.…, 0.85 -> 0.75 (dead zone) -> 0.5: recovery must fire exactly once. With…, test_deadzone_preserves_state_and_recovery_still_fires(), log_once(), MemoryPressureMonitor, Real-time memory pressure monitoring with adaptive responses. Monitors system… (+9 more)

### Community 232 - "VizBundle"
Cohesion: 0.15
Nodes (17): NpzFile, A shape-mismatched (2-D) phi_angles must degrade to placeholders, not crash at…, test_phi_grid_degrades_on_non_1d_phi_angles(), Tests for the JAX-free viz bundle (writer-shape + loader)., test_exp_only_when_no_fitted_surface(), test_load_full_bundle_from_fitted_artifact(), test_load_missing_returns_none(), _write_fitted() (+9 more)

### Community 233 - "test_heterodyne_parameter_manager_tied.py"
Cohesion: 0.31
Nodes (8): _manager_with_tie(), Tests for ParameterManager tied-parameter support., Changing the free variable that backs the parent must change the reported child…, test_expand_varying_to_full_mirrors_tied_child(), test_expand_varying_to_full_tied_value_updates_with_parent(), test_expand_varying_to_full_untied_matches_previous_behavior(), test_tied_idx_pairs_empty_when_untied(), test_tied_idx_pairs_maps_names_to_physics_indices()

### Community 234 - "build_gradient_collapse_callback"
Cohesion: 0.24
Nodes (12): _monitor(), test_callback_feeds_monitor_per_iteration(), grad_fn(), test_callback_swallows_grad_fn_errors(), test_diagnostics_block_shape_and_mechanism(), test_diagnostics_empty_history_is_fallback(), test_update_frequency_throttles_grad_evals(), test_zero_or_nan_denominator_yields_inf_ratio() (+4 more)

### Community 235 - "nlsq/validation.py"
Cohesion: 0.25
Nodes (7): test_classify_parameter_status_all_three_states(), test_is_physical_param(), _classify_parameter_status(), _is_physical_param(), Validation utilities for NLSQ optimization. Consolidates three modules from the…, Classify each parameter's status relative to bounds., Return True if the label is for a physical (non per-angle-scaling) parameter.

### Community 236 - "test_io.py"
Cohesion: 0.19
Nodes (10): _make_npz_arrays(), Tests for xpcsjax/io module: json_utils and nlsq_writers., Verify json_safe output is always valid JSON (no NaN/Inf tokens)., Build minimal valid arrays for save_nlsq_npz_file., JAX arrays must be coerced without error., TestJsonSafeRoundTrip, TestSaveNlsqNpzFile, ndarray (+2 more)

### Community 237 - "json_serializer"
Cohesion: 0.16
Nodes (8): TestJsonSerializer, I/O operations for xpcsjax XPCS analysis. This module provides functions for…, json_serializer(), Any, JSON utility functions for xpcsjax I/O operations. This module provides helper…, JSON serializer for numpy arrays and other objects. Use as the `default`…, Convert non-finite floats to JSON-safe representations. JSON spec does not…, _sanitize_float()

### Community 238 - "completion.sh"
Cohesion: 0.34
Nodes (13): _filedir(), _init_completion(), mapfile(), completion.sh script, _xpcsjax(), _xpcsjax_cleanup(), _xpcsjax_config(), _xpcsjax_config_xla() (+5 more)

### Community 239 - "test_docs_structure.py"
Cohesion: 0.21
Nodes (12): ADR 0001: Automated structural doc-coverage check, _missing_api_pages(), Every top-level xpcsjax submodule needs a Sphinx API page (ADR-0001).…, Every importable top-level xpcsjax submodule (has __init__.py), sorted., Submodules (minus `excluded`) absent from `existing_page_names` — pure, no I/O., Each top-level xpcsjax submodule needs docs/source/api/{name}.rst (ADR-0001)., Regression proof the check actually fires — synthetic inputs, no filesystem.…, A stale EXCLUDED_PACKAGES entry (e.g. a renamed/removed package) should be… (+4 more)

### Community 240 - "TestExecuteLayersNLSQConfigHomodyne"
Cohesion: 0.20
Nodes (6): ``execute_layers`` round-trips through the homodyne solver ``NLSQConfig``. The…, ``from_dict({})`` must give ``execute_layers is False``., A nested ``anti_degeneracy.execute_layers`` value must be parsed., ``to_dict()["anti_degeneracy"]["execute_layers"]`` echoes the field., ``from_dict(to_dict())`` preserves ``execute_layers`` both ways., TestExecuteLayersNLSQConfigHomodyne

### Community 241 - "filter_phi_angles_jax"
Cohesion: 0.14
Nodes (13): filter_phi_angles(), filter_phi_angles_jax(), _normalize_target_ranges(), Any, ndarray, Filter phi angles based on target ranges for optimization. This method…, Compute statistics about angle distribution vs. target ranges. Parameters…, Filter phi angles via a one-shot convenience wrapper. This is the main entry… (+5 more)

### Community 242 - "TestExecuteLayersNLSQConfigHeterodyne"
Cohesion: 0.20
Nodes (6): ``execute_layers`` round-trips through the heterodyne solver ``NLSQConfig``.…, ``from_dict({})`` must give ``execute_layers is False``., A nested ``anti_degeneracy.execute_layers`` value must be parsed., ``to_dict()["execute_layers"]`` echoes the field (flat key)., ``from_dict(to_dict())`` preserves ``execute_layers`` both ways., TestExecuteLayersNLSQConfigHeterodyne

### Community 243 - "hybrid_streaming.py"
Cohesion: 0.17
Nodes (14): HybridRecoveryConfig, Configuration for hybrid streaming optimizer recovery strategy. T029:…, Get settings for a specific retry attempt. Parameters ---------- attempt : int…, Hybrid streaming optimization strategy for NLSQ optimization. Extracted from…, # NOTE: Both t1 and t2 index into t1_unique because XPCS correlation, forward_transform_per_angle_params(), inverse_transform_per_angle_params(), Any (+6 more)

### Community 244 - "test_heterodyne_results.py"
Cohesion: 0.21
Nodes (12): _minimal_result(), NLSQResult, Smoke tests for the heterodyne NLSQResult dataclass + result helpers.…, The smallest legal NLSQResult — just the four required fields., parameters, parameter_names, success, message are required positional args.…, Two results must not share the same metadata dict (field defaults to a fresh…, The optional uncertainty / covariance / residuals fields accept arrays and…, When success=False the message must carry the diagnostic — a regression to… (+4 more)

### Community 245 - "test_post_install_fish.py"
Cohesion: 0.36
Nodes (11): _appended_block(), _make_fake_venv(), Path, Regression tests for fish-shell XLA activation under conda. Adversarial-review…, If fish is installed, the generated activate.fish must parse cleanly. Guards…, test_fish_activation_is_idempotent(), test_fish_activation_missing_script_returns_false(), test_fish_activation_resolves_conda_prefix() (+3 more)

### Community 246 - "get_safe_output_dir"
Cohesion: 0.21
Nodes (14): NLSQ result saving functions for xpcsjax XPCS analysis. This module provides…, Minimal utilities for the xpcsjax package. Essential utility functions with…, get_safe_output_dir(), _has_traversal_component(), Path, Path validation utilities for secure file operations. This module provides path…, Validate a save path for plot files. Convenience wrapper for validate_save_path…, Sanitize path for logging to prevent log injection. Parameters ---------- path… (+6 more)

### Community 247 - "test_validation_crash_coverage.py"
Cohesion: 0.29
Nodes (11): _fresh_report(), _raise_boom(), F8 TEST-1 GAP-3: crash-logging regression tests for the four missing validator…, A crash inside ``_validate_statistical_properties`` must log ERROR and fail the…, A crash inside ``_compute_data_statistics`` must log ERROR and fail the report., A crash inside ``_validate_physics_parameters`` must log ERROR and fail the…, A crash inside ``_validate_correlation_matrices`` must log ERROR and fail the…, test_compute_data_statistics_crash_logs_error_and_invalidates_report() (+3 more)

### Community 248 - "maps.py"
Cohesion: 0.20
Nodes (12): ColorMap, ImageItem, PlotItem, _apply_colormap(), Resolve a named matplotlib colormap (best-effort); ``None`` if unavailable.…, Apply a named matplotlib colormap to *image_item* (best-effort). Falls back to…, _resolve_colormap(), QWidget (+4 more)

### Community 249 - "interactive_setup"
Cohesion: 0.29
Nodes (8): CaptureFixture, test_interactive_setup_aborts_outside_venv(), test_interactive_setup_full_flow(), test_is_conda_environment(), interactive_setup(), is_conda_environment(), Run interactive post-installation setup., Check whether the interpreter is running inside a conda/mamba environment.…

### Community 250 - "test_diagonal_correction.py"
Cohesion: 0.33
Nodes (5): ndarray, partial, Canonical numerically-safe math primitives shared across physics backends. This…, Overflow-protected exponential, canonical for all physics paths. Clips the…, safe_exp()

### Community 251 - "TestTypeBoundary"
Cohesion: 0.17
Nodes (7): set_log_context and log_context must only accept the 4 known fields;…, All four known keys must be accepted without error., log_context() context manager accepts all 4 known keys., The module must define a literal __all__ listing public symbols., Key public symbols must appear in __all__., log_once must accept a proper logging.Logger (not just Any)., TestTypeBoundary

### Community 252 - "system_validator.py"
Cohesion: 0.17
Nodes (13): test_main_returns_one_on_error(), test_main_returns_zero_when_no_errors(), Runtime utilities for the xpcsjax package. Provides: *…, Runtime utilities for xpcsjax., build_parser(), main(), ArgumentParser, Enum (+5 more)

### Community 253 - "run_validation"
Cohesion: 0.28
Nodes (9): CaptureFixture, test_print_report_renders_all_tags(), test_run_validation_human_report(), test_run_validation_json(), test_validate_verbose_prints(), _print_report(), Print a human-readable report (boxed header, per-test lines, summary).…, Run all validation tests and emit a report. Parameters ---------- verbose :… (+1 more)

### Community 254 - "Anti-degeneracy controller (5/4-layer defense)"
Cohesion: 0.25
Nodes (11): L3 Adaptive CV-based Regularization, Anti-degeneracy controller (5/4-layer defense), L4 Gradient Collapse Monitoring, L2 Hierarchical Optimization, select_nlsq_strategy (STANDARD/OUT_OF_CORE/HYBRID_STREAMING), L5 Shear-Sensitivity Weighting (laminar_flow only), Angle-stratified least squares (>=1M points), L1 Per-Angle Reparameterization (all modes) (+3 more)

### Community 255 - "All-modules registry page"
Cohesion: 0.18
Nodes (11): Doc-coverage completeness rationale, All-modules registry page, xpcsjax.cli, xpcsjax.config, xpcsjax.core, xpcsjax.data, xpcsjax.device, xpcsjax.optimization.nlsq (+3 more)

### Community 256 - "test_l4_callback_layout.py"
Cohesion: 0.29
Nodes (8): _make(), Regression test for the L4 gradient-collapse monitor index layout.…, _StubConfig, _StubModel, _StubParamManager, test_disabled_monitoring_returns_none(), test_physics_first_layout_partitions_physics_as_head(), test_scaling_first_layout_partitions_physics_as_tail()

### Community 257 - "heterodyne_parameter_space.py"
Cohesion: 0.11
Nodes (29): Coverage for xpcsjax/config/types.py's coerce_finite_float bool guard. bool is…, test_coerce_finite_float_accepts_real_numbers(), test_coerce_finite_float_rejects_bool_false(), test_coerce_finite_float_rejects_bool_true(), test_coerce_finite_float_rejects_nan_and_inf(), Build default bounds lookup from the registry, then merge config overrides., Merge config-overridden bounds from ParameterSpace into _default_bounds.…, _apply_fixed_parameters() (+21 more)

### Community 258 - "core/test_debug_audit_2026_07_22.py"
Cohesion: 0.16
Nodes (17): _het_params(), Regression tests for the 2026-07-22 debug-audit fixes (Fix 1-3). Fix 1:…, compute_chi_squared's masked support must equal sum(compute_residuals**2) --…, Perturbing ONLY the t=0 row/col or the diagonal must not change chi2, since…, Static (<7-param) mode with a 0-d scalar t1 must not raise., With N > 10001 unique lag times, passing time_grid must change the (previously…, test_compute_g1_shear_static_mode_accepts_0d_scalar_t1(), test_compute_g1_total_elementwise_time_grid_avoids_truncation() (+9 more)

### Community 259 - "test_reader_final_drain_recovers_terminal_after_grace"
Cohesion: 0.13
Nodes (8): QThread, A terminal event still queued when the grace deadline expires must be recovered…, test_reader_final_drain_recovers_terminal_after_grace(), Any, Emit any still-queued events; return True if a terminal was among them.…, Spawn the worker process and begin draining its events., Drains the event queue onto a Qt signal; synthesizes Died on abnormal exit., _ReaderThread

### Community 260 - "test_unweighted_stratified_data_does_not_materialize_dense_sigma"
Cohesion: 0.40
Nodes (3): build_heterodyne_stratified_data(weights=None) must not allocate a dense…, test_unweighted_stratified_data_does_not_materialize_dense_sigma(), __init__()

### Community 261 - "InputValidator"
Cohesion: 0.17
Nodes (10): test_input_validator_bounds_inconsistent_and_out_of_range(), test_input_validator_dimension_mismatch_recorded(), test_input_validator_errors_property_is_copy(), test_input_validator_non_strict_returns_false_and_records_errors(), test_input_validator_passes_on_clean_input(), test_input_validator_strict_raises_on_bad_input(), InputValidator, Validator for NLSQ optimization input data. (+2 more)

### Community 262 - "ResultValidator"
Cohesion: 0.18
Nodes (9): test_result_validator_consistency_failure_recorded(), test_result_validator_happy_path_true(), test_result_validator_strict_raises_out_of_bounds(), test_result_validator_warnings_property_is_copy(), test_result_validator_records_warnings_for_bad_covariance(), Validator for NLSQ optimization results., Initialize ResultValidator. Parameters ---------- strict_mode : bool, optional…, Get list of validation warnings from last validate_all() call. (+1 more)

### Community 263 - "TestNoScipyLeastSquares"
Cohesion: 0.18
Nodes (7): parametrize, Guard test: no direct ``scipy.optimize.least_squares`` in the NLSQ path.…, ScipyNLSQAdapter (the retired fallback) must not reappear in adapter.py., Verify scipy.optimize.least_squares is absent from the NLSQ path., No NLSQ-path file imports ``scipy.optimize.least_squares``., No NLSQ-path file calls scipy's ``least_squares(...)``. Distinguishes…, TestNoScipyLeastSquares

### Community 264 - "load_and_merge_config"
Cohesion: 0.19
Nodes (13): Regression tests for config_handling.py error-path hardening. * L108:…, test_initial_override_lands_in_initial_parameters_values(), test_initial_override_resolver_failure_raises_value_error(), test_load_failure_names_the_file(), _write_static_isotropic_config(), apply_cli_overrides(), _apply_parameter_overrides(), load_and_merge_config() (+5 more)

### Community 265 - "test_loader_performance_engine_cleanup.py"
Cohesion: 0.33
Nodes (8): _bare_loader(), Regression test: XPCSDataLoader.close() must shut down its memory_manager.…, Bypass __init__ to avoid needing a YAML config / real dataset on disk., B1: the attribute (always None) was removed along with the module., test_close_calls_memory_manager_shutdown(), test_close_is_idempotent_and_safe_with_no_components(), test_context_manager_closes_on_exit(), test_performance_engine_attribute_no_longer_exists()

### Community 266 - "fit_nlsq"
Cohesion: 0.20
Nodes (10): fit_nlsq, load_xpcs_data, OptimizationResult, sigma_is_default interpretation caveat, fit_nlsq, OptimizationResult, heterodyne_views.per_angle_chi2, heterodyne_views.reconstruct_per_angle_scaling (+2 more)

### Community 267 - "_fixture"
Cohesion: 0.22
Nodes (14): _fixture(), _isolate_logging(), Restore the xpcsjax logger's handlers/level and manager state after each test., _agg_backend(), converged_heterodyne_result(), converged_homodyne_result(), heterodyne_model(), homodyne_model() (+6 more)

### Community 268 - "test_run_controller.py"
Cohesion: 0.29
Nodes (9): Path, Tests for RunController.on_cancel's confirm-before-cancel guard. Regression…, Answering Yes to the confirmation dialog actually cancels the run., Answering No (or dismissing) the confirmation dialog must NOT cancel the run., With no run selected, on_cancel must not reach the confirmation/cancel path., test_cancel_confirmed_yes_proceeds(), test_cancel_declined_no_does_not_cancel(), test_cancel_no_selection_shows_message_not_queue_cancel() (+1 more)

### Community 269 - ".create_nlsq_callbacks"
Cohesion: 0.14
Nodes (7): Any, Create kwargs for NLSQ's HybridStreamingConfig. Returns kwargs that can be used…, Get group variance indices for NLSQ regularization. T024: Delegates to…, Get comprehensive diagnostics from all components. Returns ------- dict Nested…, Compute and store fixed per-angle contrast/offset from quantiles. This method…, Check if fixed per-angle scaling has been computed. Returns ------- bool True…, Create callbacks for NLSQ's CurveFit integration. This method creates callbacks…

### Community 270 - "compute_g2_scaled"
Cohesion: 0.12
Nodes (14): A NaN g1 must survive the 1e-10 floor, not be laundered into a finite value.…, test_g1_total_propagates_nan_instead_of_flooring_it(), compute_g2_scaled(), Compute g2 for NLSQ using meshgrid 2D matrices. NLSQ-specific function —…, compute_chunk_accumulators(), r_fn(), compute_chunk_chi2(), evaluate_total_chi2() (+6 more)

### Community 271 - "validate_no_nan_inf"
Cohesion: 0.25
Nodes (7): test_validate_no_nan_inf_with_iteration_and_context(), test_validate_no_nan_inf_accepts_finite(), test_validate_no_nan_inf_rejects_nan_and_inf(), Any, Convert to dictionary for saving in results., Validate that array contains no NaN or Inf values., validate_no_nan_inf()

### Community 272 - "test_lazy_imports.py"
Cohesion: 0.18
Nodes (10): tests/characterization shard, Verify top-level imports are lazy and that homodyne's env setup is mirrored., v0.1 public API symbols importable as of Phase 4 (Task 20). `HeterodyneModel`…, HeterodyneModel is a public lazy export as of Phase 6 (Task 27 + Task 28)., `import xpcsjax` must set the env vars homodyne sets at import time., Importing xpcsjax must not eagerly load jax — CLI arg parsing stays instant.…, test_env_setup_mirrors_homodyne(), test_heterodyne_model_exported() (+2 more)

### Community 273 - "_resolve_color_limits"
Cohesion: 0.33
Nodes (9): Unit tests for _resolve_color_limits., test_all_nan_returns_fallback(), test_empty_matrix_returns_fallback(), test_flat_constant_matrix_returns_widened_range(), test_normal_data_returns_percentile_limits(), test_percentile_clamp_excludes_outliers(), test_returns_floats_not_numpy_scalars(), Percentile-based color limits with NaN/empty/flat fallbacks. Returns ``(1.0,… (+1 more)

### Community 274 - "test_iteration_callback_seam.py"
Cohesion: 0.18
Nodes (11): Task 2 gate: ``on_iteration`` observer threads through the NLSQ engine. The…, fit_nlsq accepts on_iteration and completes without error. Any callback firings…, fit_nlsq() with on_iteration=None must be bit-identical to the default. This is…, A raising observer must be silently swallowed; the fit must complete., fit_nlsq on two_component config: on_iteration accepted, never called. The…, Return (config, data) for the smallest laminar_flow fit. Reuses the exact…, test_default_none_does_not_change_result(), test_on_iteration_accepted_and_wellformed() (+3 more)

### Community 275 - "StreamingExecutor"
Cohesion: 0.16
Nodes (9): _enable_streaming(), Executor must call fit(data_source=(x,y), func=model_fn, p0=...) — the real API., test_streaming_executor_calls_real_fit_api(), test_streaming_executor_missing_pcov_is_nan_placeholder(), test_streaming_executor_reraises(), Streaming optimization for unlimited dataset sizes. Uses NLSQ's…, Strategy name for logging., Whether this strategy supports progress bars (it does). (+1 more)

### Community 276 - "model_adapter.PointEvaluator"
Cohesion: 0.25
Nodes (9): heterodyne_engine_route.fit_two_component_via_engine, model_adapter.HeterodynePointEvaluator, model_adapter.HeterodynePointwiseEvaluator, model_adapter.HomodynePointEvaluator, model_adapter.PointEvaluator, Pointwise evaluator not wired rationale, StratifiedResidualFunctionJIT, coverage.run omit list (engine orchestration files) (+1 more)

### Community 277 - "heterodyne_physics_validators.py"
Cohesion: 0.21
Nodes (10): Physics constraint validators for heterodyne (``two_component``) parameters.…, ConstraintRule, is_non_finite(), PhysicsViolation, Shared physics-validation primitives for homodyne and heterodyne constraint…, Return ``True`` for ``NaN`` / ``±inf``; ``False`` for finite or non-numeric.…, A single triggered physics constraint violation. Attributes ---------- param :…, Render the violation as a single human-readable line. (+2 more)

### Community 278 - "XPCSDataFormatError"
Cohesion: 0.16
Nodes (25): _FakeDS, Quality-gate finding #5: the APS-U loader builds an unbounded intermediate list…, Minimal h5py-dataset stand-in: exposes shape + dtype, no data read., test_guard_noop_when_no_valid_bins_in_range(), test_guard_passes_for_legitimate_small_input(), test_guard_rejects_budget_exceeded(), test_guard_rejects_non_square(), test_guard_rejects_oversized_frame_count() (+17 more)

### Community 279 - ".__init__"
Cohesion: 0.14
Nodes (8): Any, Logger, Pre-compute GLOBAL unique values from ALL chunks to avoid jnp.unique() in JIT.…, Compute flat indices for mapping chunk points to global grid positions. This…, Pre-compile JAX functions for performance. This method sets up JIT-compiled…, Initialize the stratified residual function. Parameters ----------…, Get diagnostic information about the residual function. Returns -------…, Log diagnostic information for monitoring.

### Community 280 - "test_config_relative_data_paths.py"
Cohesion: 0.32
Nodes (12): MonkeyPatch, Path, Relative data paths in a config resolve against the config file's directory.…, A ``..`` relative path must NOT be anchored to an absolute path.…, test_absolute_path_is_left_unchanged(), test_env_var_path_expands_to_absolute(), test_override_path_is_not_resolved(), test_parent_traversal_path_is_left_literal_for_downstream_guard() (+4 more)

### Community 281 - "validate_cross_parameter_constraints"
Cohesion: 0.26
Nodes (12): Coverage for the homodyne physics-constraint validators (audit finding).…, test_cross_parameter_min_severity_filters_info(), test_cross_parameter_missing_keys_no_violation(), test_d_offset_overfitting_flagged_above_half_d0(), test_d_offset_overfitting_flagged_when_negative_and_dominant(), test_d_offset_within_bound_no_violation(), test_validate_all_parameters_aggregates_single_and_cross(), test_validate_all_parameters_no_violations_for_sane_values() (+4 more)

### Community 282 - "heterodyne_jax_backend.py"
Cohesion: 0.24
Nodes (11): Characterization tests for the heterodyne velocity/transport integral kernels.…, test_transport_integral_constant_rate_is_abs_gap_and_symmetric(), test_velocity_integral_constant_velocity_is_linear_signed_gap(), test_velocity_integral_is_antisymmetric_with_zero_diagonal(), compute_transport_integral_matrix(), compute_velocity_integral_matrix(), JAX-accelerated computational backend for heterodyne correlation. This module…, Compute the velocity integral matrix (NLSQ meshgrid path, JIT-compiled).… (+3 more)

### Community 283 - "test_stratified_ls_averaged_covariance_transform.py"
Cohesion: 0.25
Nodes (8): _broadcast_jacobian_transform(), ndarray, Regression: the averaged-mode inverse covariance transform in…, Exact transform now used in ``fit_with_stratified_least_squares``'s averaged-…, n_phi=2, n_physical=1: pcov is 3x3 over [contrast, offset, D0]., Wiring: the averaged-mode inverse block must use the J_full broadcast…, test_hand_computed_n_phi2_n_physical1(), test_source_no_longer_uses_identity_diagonal_shortcut()

### Community 284 - "load_config"
Cohesion: 0.18
Nodes (8): _probe_import(), Tests for the argparse-free config loader (xpcsjax/service/config.py)., test_config_service_is_jax_free(), test_load_config_applies_mode_and_output(), test_load_config_no_overrides_leaves_config(), load_config(), Path, Load a YAML/JSON config and apply the mode + output-dir overrides. Parameters…

### Community 285 - "test_logging_quality_gate.py"
Cohesion: 0.20
Nodes (6): LogRecord, _RaisingHandler, Quality-gate regression tests for xpcsjax/utils/logging.py. Each section…, A handler whose emit() always raises — simulates a broken log backend., log_once with two DISTINCT keys must each emit exactly once., TestLogOnceTwoDistinctKeys

### Community 286 - "plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2)"
Cohesion: 0.43
Nodes (8): Reduced chi-squared goodness-of-fit (chi2_red = 0.906), Experimental Data panel (C2 heatmap), plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2), Fitted Model panel (NLSQ C2 heatmap), plot_nlsq_fit visualization function, NLSQ curve fit (XPCS correlation fit), Residuals panel (data minus model, diverging colormap), Two-time correlation C2(t1,t2) at phi=45 deg

### Community 287 - "NLSQ Residual Diagnostics Baseline (phi=45deg)"
Cohesion: 0.32
Nodes (8): Diagonal Residuals vs Time, NLSQ Residual Diagnostics Baseline (phi=45deg), NLSQ Fit (residual source), plot_residual_map (viz function), Residual Distribution with Normal Overlay, Two-Time Residual Map (t1 vs t2 heatmap), Residuals vs Fitted Value, Matplotlib Baseline Image Validation

### Community 288 - "xpcsjax.cli.plot_dispatch"
Cohesion: 0.25
Nodes (8): xpcsjax.cli.config_handling, xpcsjax.cli.data_pipeline, xpcsjax.cli.optimization_runner, plot_backend.resolve_plots_dir, plot_backend.should_use_datashader, xpcsjax.cli.plot_dispatch, plot_families.simulated.resolve_phi_angles_for_sim, xpcsjax.cli.result_saving

### Community 289 - "xpcsjax.optimization.nlsq.fit_nlsq"
Cohesion: 0.25
Nodes (8): CMAESWrapper, xpcsjax.optimization.nlsq.fit_nlsq, core.fit_nlsq_cmaes, core.fit_nlsq_jax, core.fit_nlsq_multistart, NLSQStrategy, select_nlsq_strategy, StrategyDecision

### Community 290 - "test_freeze_safety.py"
Cohesion: 0.28
Nodes (5): _extract_collect_all_names(), The GUI entry must be freeze-safe and resolvable as a console script., Lowercase package names from the spec's ``collect_all()`` for-loop tuple.…, test_collect_all_extraction_ignores_comment_text(), test_pyinstaller_spec_covers_runtime_deps()

### Community 291 - "TimedContext"
Cohesion: 0.25
Nodes (5): test_timed_context_measures_elapsed(), Context manager for timing optimizer calls. Usage:: timer = TimedContext() with…, Start the timer and return the context manager., Stop the timer, recording the elapsed seconds on :attr:`elapsed`., TimedContext

### Community 292 - "test_debug_audit_2026_07_23_negative_correlation_repair.py"
Cohesion: 0.22
Nodes (12): _data_with_one_skipped_matrix(), _data_with_one_skipped_matrix_robust(), Regression: negative-correlation repair must not clamp matrices that were…, End-to-end (design spec Testing item 6): thread the REAL _normalize_data output…, Integer-dtype c2_exp must not truncate the 1e-6 floor to exactly 0. ``arr[mask]…, ROBUST method's zero-IQR skip branch must also be tracked per-matrix, mirroring…, test_normalize_data_tracks_per_matrix_mask(), test_normalize_data_tracks_per_matrix_mask_robust() (+4 more)

### Community 293 - "_safe_log_memory_strategy"
Cohesion: 0.21
Nodes (11): Logging-parity tests: two_component must emit the same setup-log narrative as…, Mirrors laminar's ``xpcsjax.device.cpu`` configuration banner., Mirrors laminar's ``memory_strategy_selection`` phase + threshold line., Logging must never break a fit even when the optional deps are absent., test_helpers_never_raise(), test_safe_configure_cpu_threading_emits_device_cpu_block(), test_safe_log_memory_strategy_emits_phase_and_threshold(), Best-effort CPU/HPC threading configuration for the heterodyne path. Mirrors… (+3 more)

### Community 294 - "FitQualityConfig"
Cohesion: 0.21
Nodes (11): parametrize, test_classify_fit_quality_bands(), test_classify_fit_quality_respects_custom_thresholds(), test_fit_quality_config_defaults_match_homodyne(), test_fit_quality_config_from_dict_falls_back_on_missing_keys(), test_fit_quality_config_from_none_returns_defaults(), classify_fit_quality(), FitQualityConfig (+3 more)

### Community 295 - "TestJSONFormatterCircularRef"
Cohesion: 0.40
Nodes (3): JSONFormatter must handle a circular-reference context dict without raising and…, Circular ref injected via a record attribute must also be handled., TestJSONFormatterCircularRef

### Community 296 - "TestLogPerformanceNeverRaise"
Cohesion: 0.18
Nodes (5): log_performance: a raising handler must never abort the decorated function., Performance log raises on success; function must still return., Below-threshold path (no log emit) still works with a bad logger., Real function exception still propagates when logger also raises., TestLogPerformanceNeverRaise

### Community 297 - "TestContextFilterOnLogger"
Cohesion: 0.20
Nodes (6): ContextFilter must be attached to the named xpcsjax logger so context fields…, A record emitted before configure() carries the context fields. The…, After configure_logging, the named xpcsjax logger has a ContextFilter., Multiple configure_logging calls must not double-install ContextFilter on the…, ContextFilter.filter() sets all fields to None when context is empty., TestContextFilterOnLogger

### Community 298 - "CMAESWrapper"
Cohesion: 0.15
Nodes (10): Wiring: ``_configure_memory`` must be called AFTER the adaptive scale_ratio >…, test_fit_calls_configure_memory_after_scale_ratio_block_with_final_popsize(), The documented default scale_threshold is 1000.0., test_default_threshold_is_1000(), wrapper(), CMAESWrapper, Wrapper around NLSQ's CMAESOptimizer for homodyne integration. This wrapper…, Initialize CMA-ES wrapper. Parameters ---------- config : CMAESWrapperConfig |… (+2 more)

### Community 299 - "TestSafeSincContinuity"
Cohesion: 0.17
Nodes (7): safe_sinc must be continuous and well-valued at the Taylor threshold., Values just inside and just outside the 1e-4 threshold must agree to better…, sinc(0) = 1 by the Taylor expansion., sin(π)/π ≈ 0; sanity check for the far branch., safe_sinc must return finite values for all x in [-10, 10]., Gradient at x just below threshold must match gradient just above., TestSafeSincContinuity

### Community 300 - "xla_config.bash"
Cohesion: 0.50
Nodes (7): xla_config.bash script, _xpcsjax_configure_xla(), _xpcsjax_get_cpu_count(), _xpcsjax_load_xla_mode(), _xpcsjax_resolve_mode_file(), _xpcsjax_save_xla_mode(), _xpcsjax_xla_setup()

### Community 301 - "Public API Reference"
Cohesion: 0.33
Nodes (7): Public API Reference, ConfigManager, fit_nlsq (single-entry NLSQ wrapper), HeterodyneModel, HomodyneModel, Lazy __getattr__ public-export mechanism, load_xpcs_data

### Community 302 - "test_codex_review_fixes.py"
Cohesion: 0.21
Nodes (11): _laminar_fit(), parametrize, Regression tests for the two confirmed Codex adversarial-review findings. Both…, End-to-end guard for BOTH transform fixes: a laminar fit with config-enabled…, The single DOF authority the out-of-core / hybrid-streaming / stratified-LS…, With ``scaling_head_size`` given, physics indices start at the head end and…, The exact crash from the review: a forward shear transform on a compressed…, test_effective_constrained_dof_rule() (+3 more)

### Community 303 - "OptimizationStrategy"
Cohesion: 0.06
Nodes (38): test_recovery_helper_has_no_floor(), Audit [2026-07-22], updated [2026-07-23] (PR #15 review, Finding #3 scope…, Audit [2026-07-23] (PR #15 review): the plain (enable_recovery=False, non-…, Audit [2026-07-23]: a STREAMING soft-failure (success=False, no exception) must…, Audit [2026-07-23]: enable_recovery=True's execute_with_recovery returning…, test_fallback_no_recovery_reports_failed_on_stagnation(), fake_curve_fit(), test_plain_soft_failure_escalates_to_next_strategy() (+30 more)

### Community 304 - "logging.py"
Cohesion: 0.03
Nodes (77): The CLI data_pipeline adapters must map argparse attrs to service kwargs., test_load_and_validate_data_forwards_phi_as_phi_subset(), test_resolve_phi_angles_forwards_cli_phi_and_phi_angles_str(), _isolate_logging(), Any, MonkeyPatch, Path, Regression test for config-driven file logging on the CLI path. Root cause this… (+69 more)

### Community 305 - "XpcsDataset"
Cohesion: 0.27
Nodes (9): dict, Typed XpcsDataset at the load/fit boundary. Quality-gate type-design finding:…, _raw(), test_is_dict_subclass_backward_compatible(), test_missing_correlation_raises_clear_error(), test_typed_accessors_resolve_canonical_and_alias_keys(), Typed container for loaded XPCS experimental data. ``XpcsDataset`` is the named…, A loaded XPCS dataset: a ``dict`` with typed, alias-resolving accessors. (+1 more)

### Community 306 - "test_docs_no_fourier.py"
Cohesion: 0.29
Nodes (6): parametrize, Path, Phase 7: the Sphinx docs carry no references to the REMOVED per-angle Fourier…, No rst under docs/source references the deleted Fourier-scaling feature., test_all_rst_have_no_deleted_feature_tokens(), test_rst_has_no_removed_mode()

### Community 307 - "_version_at_least"
Cohesion: 0.29
Nodes (6): parametrize, test_version_at_least(), test_version_at_least_returns_none_for_unparseable_version(), Return whether ``actual`` is at least ``minimum`` per PEP 440 ordering.…, Verify every required runtime dependency meets its minimum version. Returns…, _version_at_least()

### Community 308 - "DatashaderRenderer"
Cohesion: 0.20
Nodes (8): Image, The fast path must handle rectangular (n_t1 != n_t2) grids correctly., test_datashader_renderer_handles_rectangular_grid(), DatashaderRenderer, ndarray, Resolve a matplotlib colormap name to a Datashader hex-color list., Fast heatmap rendering using Datashader. Uses Datashader's CPU rasterization…, Rasterize 2D gridded data to a PIL Image using Datashader. Parameters…

### Community 309 - "validate_single_parameter"
Cohesion: 0.20
Nodes (10): test_single_parameter_flags_negative_diffusion(), test_single_parameter_min_severity_filters_warnings(), test_single_parameter_valid_value_no_violations(), PhysicsViolation, Validate a single parameter against its physics constraints. Parameters…, Validate all parameters against single- and cross-parameter constraints.…, A single triggered physics constraint violation. Same fields as…, Alias for :attr:`param` (pre-unification field name). (+2 more)

### Community 310 - "test_anti_degeneracy_transforms.py"
Cohesion: 0.31
Nodes (10): _build(), _make_controller(), _per_angle_params(), ndarray, Round-trip coverage for anti-degeneracy parameter transforms (audit finding…, test_constant_collapse_uses_nanmean_and_preserves_physical(), test_constant_round_trip_is_exact_for_constant_scaling(), test_controller_rejects_removed_tokens_after_phase7() (+2 more)

### Community 311 - "._run_sequential_optimization"
Cohesion: 0.22
Nodes (9): Without ``scaling_head_size`` the legacy dense ``2*n_phi`` head is preserved…, test_physical_index_map_default_is_dense_backward_compatible(), build_physical_index_map(), Build mapping from parameter names to indices. Parameters ----------…, _compute_g2_grid_for_phi(), _expand_compact_layout(), _maybe_expand_x_scale(), residual_func() (+1 more)

### Community 312 - "plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg)"
Cohesion: 0.40
Nodes (6): C2 Value Colorbar / Range Annotation, Correlation Diagonal Ridge (t1=t2), plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg), Scattering Angle phi = 45 deg, C2(t1,t2) Two-Time Correlation Map, plot_simulated_data Visualization Function

### Community 313 - "ConfigManager"
Cohesion: 0.33
Nodes (6): AnalysisMode (StrEnum), ConfigManager, ParameterManager, ParameterRegistry, ParameterSpace, PhysicsViolation / physics validators

### Community 314 - "xpcsjax.core.HeterodyneModel"
Cohesion: 0.40
Nodes (6): xpcsjax.optimization.nlsq.fit_nlsq, heterodyne_jax_backend.compute_c2_heterodyne, xpcsjax.core.heterodyne_model_stateful, xpcsjax.core.HeterodyneModel, xpcsjax.core.jax_backend, HeterodyneModel

### Community 315 - "AntiDegeneracyController"
Cohesion: 0.33
Nodes (6): adaptive_regularization.AdaptiveRegularizer, AntiDegeneracyController, gradient_monitor.GradientCollapseMonitor, hierarchical.HierarchicalOptimizer, per_angle_mode.PerAngleScalingPlan, shear_weighting.ShearSensitivityWeighting

### Community 316 - "5-layer anti-degeneracy controller"
Cohesion: 0.33
Nodes (6): 5-layer anti-degeneracy controller, CMA-ES escape, ConfigManager, NLSQ CurveFit (trust-region solve), OptimizationResult, select_nlsq_strategy

### Community 317 - "Second-order (intensity) correlation function c2"
Cohesion: 0.33
Nodes (6): First-order correlation function c1, Second-order (intensity) correlation function c2, g2(q, tau) equilibrium projection, load_xpcs_data, Siegert relation, Sutton2008 (Siegert relation citation)

### Community 319 - "set_log_context"
Cohesion: 0.24
Nodes (11): test_set_and_log_context_inject_run_id(), Token, log_context(), _LogContext, TypedDict, Set context-local log fields, returning a token for restoration. Only the four…, Restore the log context to the state captured by ``token``., Context manager that sets log context fields for the enclosed scope. Only the… (+3 more)

### Community 320 - "test_ci_heavy_nodes_parity.py"
Cohesion: 0.60
Nodes (5): _ci_yml_node_occurrences(), _makefile_heavy_nodes(), Guards HEAVY_NODES parity between the Makefile and .github/workflows/ci.yml.…, test_ci_yml_deselects_exactly_the_makefile_heavy_nodes(), test_ci_yml_references_each_heavy_node_symmetrically()

### Community 321 - "TestPropagateInversion"
Cohesion: 0.33
Nodes (4): Default no-arg configure() must not force propagate=True when there is no…, With no managed handler, the production branch must NOT set propagate=True., With a managed handler, the production branch sets propagate=False., TestPropagateInversion

### Community 322 - "_build_inputs"
Cohesion: 0.33
Nodes (6): _build_inputs(), Any, ndarray, Heterodyne residual-layout parity gate (corpus-loading, not generating).…, Reconstruct the deterministic inputs — must match the generator exactly., test_xpcsjax_matches_upstream_residual_layout()

### Community 323 - "generate_nlsq_plots"
Cohesion: 0.40
Nodes (5): generate_nlsq_plots, Heterodyne individual-layout requirement rationale, plot_nlsq_fit, plot_residual_map, plot_simulated_data

### Community 324 - "Joint global escape keep-better contract"
Cohesion: 0.40
Nodes (5): Joint global escape keep-better contract, _fit_joint_cmaes_multi_phi, _fit_joint_multistart, cmaes_wrapper.fit_with_cmaes, run_multistart_nlsq

### Community 325 - "main"
Cohesion: 0.22
Nodes (9): parametrize, test_detect_shell_type_from_env(), test_main_dispatches_to_interactive(), test_main_skip_both_returns_zero(), test_validate_xla_mode(), main(), Validate an XLA mode string. Accepts 'auto', 'nlsq', or an integer. Returns the…, CLI entry point for xpcsjax-post-install. (+1 more)

### Community 326 - "ndarray"
Cohesion: 0.24
Nodes (6): ndarray, JAX-native residuals for use in JIT/Jacobian contexts. Performance Optimization…, Vectorized residual computation using concatenated arrays. Performance…, Original chunk-based residual computation — REMOVED. Per-chunk data was freed…, Return JAX-native residuals (for JIT / Jacobian contexts). Parameters…, Compute residuals, returning a NumPy array (NLSQ interface). Parameters…

### Community 327 - "test_data_package_features.py"
Cohesion: 0.33
Nodes (3): Regression guard for xpcsjax.data's import contract. A dangling name in one of…, The per-feature HAS_* shims (and the loader-error escape hatch) are gone., test_has_flags_removed()

### Community 328 - "_ColorFormatter"
Cohesion: 0.22
Nodes (7): Formatter, LogRecord, _record(), test_color_formatter_applies_and_restores_color(), test_color_formatter_no_color(), _ColorFormatter, Optional ANSI color formatter for console logging.

### Community 329 - "validate_single_parameter"
Cohesion: 0.29
Nodes (10): PhysicsViolation, _has_error(), parametrize, test_heterodyne_finite_value_still_passes(), test_heterodyne_nan_is_error(), test_homodyne_finite_value_still_passes(), test_homodyne_inf_is_error(), test_homodyne_nan_is_error() (+2 more)

### Community 330 - "test_no_removed_per_angle_tokens_in_tests_or_package"
Cohesion: 0.40
Nodes (4): skipif, Phase 7 exit gate: the removed per-angle tokens must not survive anywhere. This…, ``rg -n -w <tokens> tests/ xpcsjax/`` returns zero lines after teardown., test_no_removed_per_angle_tokens_in_tests_or_package()

### Community 331 - "fit_with_stratified_least_squares"
Cohesion: 0.13
Nodes (17): _homodyne_config(), parametrize, Phase 6 no-worse-SSR gate: deleting the truncated-basis mode on the laminar…, Self-consistent g2-at-truth drives residuals -> ~0; the individual solve must…, _ssr(), _stratified_info(), test_fourier_rejected_all_laminar_paths(), test_stratified_individual_ssr_tripwire() (+9 more)

### Community 333 - "xpcsjax.cli.config_generator.main"
Cohesion: 0.50
Nodes (4): xpcsjax.cli.config_generator.main, xpcsjax.cli.config_generator.generate_config, xpcsjax.cli.config_generator.show_template, xpcsjax.cli.config_generator.validate_config

### Community 334 - "xpcsjax Project Logo"
Cohesion: 0.67
Nodes (4): xpcsjax Project Logo, Time Axes t1 and t2, Two-Time Correlation Surface c2(t1, t2), xpcsjax Wordmark (xpcs + jax)

### Community 335 - "test_aps_u_bin_skip_misalignment.py"
Cohesion: 0.29
Nodes (9): _loader(), Regression test for review finding A1 (2026-09-15). ``_load_aps_u_format`` used…, Minimal APS-U HDF5 file: n_phi=1 so bin_idx == q_idx directly., Bare loader with just enough state for the (q,phi)-selection code. D7…, Fewer correlation matrices than processed_bins entries must raise. D7…, Sanity check: a consistent file loads all the way through cleanly., test_bin_skip_raises_instead_of_misaligning(), test_matched_bins_and_matrices_load_cleanly() (+1 more)

### Community 336 - "install_xla_activation"
Cohesion: 0.17
Nodes (12): test_install_xla_activation_skips_outside_venv(), test_install_xla_bash_activation_injection_and_idempotency(), test_install_xla_bash_activation_missing_script(), test_is_virtual_environment_false(), test_is_virtual_environment_via_base_prefix(), test_is_virtual_environment_via_conda(), install_xla_activation(), _install_xla_bash_activation() (+4 more)

### Community 337 - "tests/conftest.py"
Cohesion: 0.50
Nodes (3): pytest_addoption(), Root pytest configuration., Block the pytest-qt plugin when PySide6 is absent (bare [dev] / CLI envs).…

### Community 339 - "test_heterodyne_per_angle_cmaes_fits_without_signature_drift"
Cohesion: 0.22
Nodes (9): _cmaes_available(), _cmaes_smoke_config_dict(), Path, skipif, Heterodyne + CMA-ES end-to-end smoke test. Closes the /double-check Phase 5…, End-to-end: the per-angle ``_fit_cmaes`` path completes without raising. The…, Self-contained heterodyne config with CMA-ES enabled and tight budget. The…, Skip-gate for hosts without evosax (CPU-only or barebones installs). (+1 more)

### Community 340 - "test_debug_audit_regressions.py"
Cohesion: 0.13
Nodes (18): _make_aps_old_hdf5(), parametrize, Regression tests for the 2026-06-10 whole-codebase debug audit. Each test pins…, Audit [8]: the active-parameter fallback for two_component must return the…, Write a minimal APS-old-format HDF5 file the loader can parse., Audit [6] + Codex follow-up: an empty (q,phi) selection must fail loudly on…, Audit [26]: get_group_indices('scaling') must resolve, not KeyError. The…, 2026-07-22 audit Fix 1: the APS-old quality-filtering branch must probe-then-… (+10 more)

### Community 342 - "configure_cpu_hpc"
Cohesion: 0.67
Nodes (3): configure_cpu_hpc, xpcsjax.device.cpu.configure_cpu_threading, detect_cpu_info

### Community 343 - "DatashaderRenderer"
Cohesion: 0.67
Nodes (3): DatashaderRenderer, plot_c2_comparison_fast, plot_c2_heatmap_fast

### Community 344 - "Decision Record: CPU-only Execution"
Cohesion: 0.67
Nodes (3): Decision Record: CPU-only Execution, Compilation is the bottleneck, not compute, Float64 erases consumer-GPU advantage

### Community 345 - "make test-smoke"
Cohesion: 0.67
Nodes (3): HEAVY_NODES deselected tests, make test-smoke, make verify

### Community 346 - "HomodyneModel"
Cohesion: 0.67
Nodes (3): ConfigManager, HomodyneModel, static_isotropic analysis mode

### Community 347 - "User Guide Overview"
Cohesion: 0.67
Nodes (3): Lazy Loading Public API, NLSQ-Only Capability Boundary, User Guide Overview

### Community 348 - "log_phase"
Cohesion: 0.20
Nodes (10): test_log_phase_never_raises_when_memory_probe_fails(), test_get_memory_gb_returns_float_on_linux(), test_log_phase_logs_start_and_completion(), test_log_phase_threshold_suppresses_logs(), _get_memory_gb(), log_phase(), PhaseContext, Context object returned by log_phase() with timing and memory info. (+2 more)

### Community 349 - "test_simulated_data_grid.py"
Cohesion: 0.25
Nodes (7): Regression: the standalone ``--plot-simulated-data`` path must evaluate the…, A trailing comma must drop the empty token, not crash float('')., All-empty input (e.g. a bare comma) must warn and fall back to the data's own…, test_resolve_phi_angles_for_sim_all_empty_falls_back_to_data(), test_resolve_phi_angles_for_sim_trailing_comma(), test_simulated_grid_uses_elapsed_time(), _write_config()

### Community 350 - "compute_c2_heterodyne_pointwise"
Cohesion: 0.33
Nodes (8): _params(), parametrize, Pointwise heterodyne kernel must exactly match the meshgrid path., Two phi angles, scattered points; each must match its meshgrid value., test_pointwise_matches_meshgrid(), test_pointwise_multi_phi_gather(), compute_c2_heterodyne_pointwise(), Pointwise heterodyne correlation at scattered ``(phi, t1, t2)`` triples. Thin…

### Community 351 - "create_time_integral_matrix"
Cohesion: 0.28
Nodes (6): N=1 single-point time array must not crash and must return finite (1,1)., The integral from t[0] to t[0] is zero; smooth_abs gives sqrt(eps)., Sanity check: N=2 also works., TestTimeIntegralMatrixN1, create_time_integral_matrix(), r"""Create time integral matrix using trapezoidal numerical integration.…

### Community 364 - "test_heterodyne_smoke_fit_recovers_truth"
Cohesion: 0.22
Nodes (9): _build_synthetic_c2(), ndarray, Path, Self-contained heterodyne config sufficient for HeterodyneModel.from_config.…, Forward-evaluate the model at each phi to build the c2 stack., End-to-end heterodyne fit on synthetic data must converge and recover…, _smoke_config_dict(), test_heterodyne_smoke_fit_recovers_truth() (+1 more)

### Community 368 - "OOCSharedArrays"
Cohesion: 0.14
Nodes (12): shm_required, test_shared_arrays_rejects_non_finite_sigma(), test_shared_arrays_rejects_non_positive_sigma(), test_shared_arrays_roundtrip_and_cleanup(), test_shared_arrays_without_sigma(), OOCSharedArrays, ndarray, Shared memory manager for OOC flat data arrays. Parameters ---------- phi_flat,… (+4 more)

### Community 376 - "ContextFilter"
Cohesion: 0.25
Nodes (6): test_log_context_restores_on_exit(), ContextFilter, LogRecord, Logging filter that injects context-local fields onto each record. Fields named…, Attach context-local fields to ``record``; always returns ``True``., Render ``record`` as a single JSON line, never raising. Falls back to a minimal…

### Community 377 - "validate_parameters"
Cohesion: 0.25
Nodes (8): test_time_integral_negative_alpha_requires_positive_tmin(), test_time_integral_well_posed_is_valid(), ndarray, ValidationResult, Validate heterodyne model parameters against physical constraints. Convenience…, r"""Check the ``D0 * t**alpha`` time integral for numerical hazards. For…, validate_parameters(), validate_time_integral_safety()

### Community 398 - "restore_by_mask_jax"
Cohesion: 0.29
Nodes (8): test_restore_by_mask_jax_matches_numpy_and_is_traceable(), traced(), _stripped_model_func(), model_for_cmaes(), JAX-traceable equivalent of :func:`restore_by_mask_numpy`. Uses immutable…, restore_by_mask_jax(), _solver_model_fn(), _stripped_wrapped_residual_fn()

### Community 407 - "ConfigManager"
Cohesion: 0.05
Nodes (51): _fake_loader_data(), Config-driven phi_filtering must subset the data arrays. Regression: the HDF5…, Minimal two_component config with phi_filtering enabled., 23-angle synthetic dataset mirroring the C044 azimuthal sweep., test_load_and_validate_data_subsets_to_filtered_angles(), test_phi_filtering_disabled_keeps_all_angles(), _write_config(), test_update_config_null_intermediate_creates_mapping() (+43 more)

### Community 408 - "test_pointwise_joint_parity.py"
Cohesion: 0.32
Nodes (7): _assert_pointwise_matches_batched(), _effective_scaling(), parametrize, Phase-0 gate: the flat point-wise heterodyne joint residual must reproduce the…, (contrast_per_angle, offset_per_angle) in SORTED phi_unique order, matching…, Run the full point-wise vs batched SSR comparison for one mode at p0. Steps…, test_pointwise_joint_ssr_matches_batched()

### Community 414 - "validate_optimized_params"
Cohesion: 0.33
Nodes (6): test_validate_optimized_params_below_lower(), test_validate_optimized_params_accepts_in_bounds(), test_validate_optimized_params_rejects_non_finite(), test_validate_optimized_params_rejects_out_of_bounds(), Validate that optimized parameters are finite and within bounds., validate_optimized_params()

### Community 416 - "_build_homodyne_l4_callback"
Cohesion: 0.29
Nodes (8): _build_homodyne_l4_callback(), _l4_plus_observer(), _loss(), _observer_only(), _homodyne_l4_monitoring_enabled(), Read the homodyne L4 gradient-monitoring gate from the config. The flag lives…, Build the L4 per-iteration gradient-collapse monitor + curve_fit callback.…, on_iteration()

### Community 422 - "test_preprocessing_diagonal_dtype.py"
Cohesion: 0.33
Nodes (6): _integer_c2_with_fractional_correction(), Regression test for integer-dtype truncation in enhanced diagonal correction.…, One 8x8 integer matrix whose basic diagonal correction is fractional. The basic…, ``window_size <= 0`` must not silently leave the diagonal uncorrected. This…, test_diagonal_correction_upcasts_integer_c2_no_truncation(), test_statistical_diagonal_correction_warns_and_clamps_on_nonpositive_window_size()

### Community 423 - "_FakeModel"
Cohesion: 0.33
Nodes (5): _FakeModel, Minimal stand-in model for the fit-comparison render-logging tests., Regression test for the pre-existing bug: a fixed (non-tied) physics param must…, test_build_hybrid_streaming_result_expands_fixed_physics_param(), test_build_hybrid_streaming_result_mirrors_tied_child()

### Community 424 - "test_templates_no_fourier_keys.py"
Cohesion: 0.47
Nodes (5): parametrize, Phase 7 (static-config resolution): the removed reparam-order YAML keys are…, test_template_has_no_live_removed_keys(), _walk(), test_template_still_loads()

### Community 429 - "test_loader_integration.py"
Cohesion: 0.40
Nodes (5): Round-trip load test using a self-contained synthetic NPZ cache. Historically…, Write a minimal, valid 1-D-time-axis NPZ cache the loader can read directly.…, Load a synthetic NPZ cache end-to-end and assert the XPCS data invariants., test_load_synthetic_npz_cache_roundtrip(), _write_synthetic_npz_cache()

### Community 430 - "get_executor"
Cohesion: 0.33
Nodes (6): parametrize, test_get_executor_dispatch(), test_get_executor_streaming_passes_checkpoint_config(), test_get_executor_unknown_raises(), get_executor(), Get the executor instance for a named optimization strategy. Parameters…

### Community 432 - "ConstraintSeverity"
Cohesion: 0.33
Nodes (5): ValidationResult, Validate physics-based parameter constraints beyond simple bounds. Checks for…, ConstraintSeverity, StrEnum, Severity levels for physics constraint violations. ``StrEnum`` (a ``str``…

### Community 433 - "test_reduce_noise_dtype.py"
Cohesion: 0.50
Nodes (4): _integer_c2(), parametrize, Regression test for integer-dtype truncation in noise reduction. Bug:…, test_reduce_noise_upcasts_integer_c2_no_truncation()

### Community 436 - "._run_nlsq_refinement"
Cohesion: 0.40
Nodes (3): Any, Convert to NLSQ CMAESConfig. Parameters ---------- n_params : int Number of…, Run NLSQ TRF refinement on CMA-ES solution. Uses NLSQ's curve_fit with…

### Community 438 - "get_constraint_summary"
Cohesion: 0.67
Nodes (3): get_constraint_summary(), Any, Summarize the defined constraint registry. Returns ------- dict Keys…

## Knowledge Gaps
- **188 isolated node(s):** `completion.sh script`, `g1/g2/c2 correlation functions`, `Bradbury2018 - JAX composable transformations`, `Duri2005 - Time-resolved-correlation measurements`, `Hansen2016 - CMA evolution strategy tutorial` (+183 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 4200 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **70 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `logging.py` to `heterodyne_parameter_space.py`, `physics_nlsq.py`, `fit_heterodyne_stratified_least_squares`, `HeterodynePhysicsAdapter`, `test_logging.py`, `OptimizationResult`, `heterodyne_memory.py`, `_fixture`, `data/validation.py`, `heterodyne_core.py`, `heterodyne_result_builder.py`, `LargeDatasetExecutor`, `heterodyne_physics_validators.py`, `StratifiedResidualFunction`, `.__init__`, `test_memory_manager_logging.py`, `core.py`, `main.py`, `jax_backend.py`, `test_logging_primitives.py`, `NLSQWrapper`, `parameter_registry.py`, `residual_jit.py`, `xpcsjax/data/__init__.py`, `FitJob`, `diagonal_correction.py`, `heterodyne_logging.py`, `AnalysisMode`, `memory_manager.py`, `PreprocessingPipeline`, `filtering_utils.py`, `OptimizationStrategy`, `test_plot_dispatch_logging.py`, `configure_logging`, `stratified_ls.py`, `._configure_impl`, `test_nlsq_support_modules.py`, `compute_jacobian_stats`, `nlsq/__init__.py`, `cpu.py`, `heterodyne_config.py`, `adapter.py`, `parameter_utils.py`, `sequential.py`, `log_performance`, `npz_cache.py`, `nlsq_plots.py`, `parallel_accumulator.py`, `DatasetOptimizer`, `AdvancedMemoryManager`, `.from_config`, `test_strategy_chunking.py`, `quality_controller.py`, `StratifiedResidualFunctionJIT`, `normalize_angle_to_symmetric_range`, `heterodyne_model_stateful.py`, `fit_two_component_via_engine`, `ResultBuilder`, `xpcs_loader.py`, `log_phase`, `test_frame_dimension_guard.py`, `run_fit`, `_logger_that_raises_on_log`, `multistart.py`, `test_heterodyne_data_prep.py`, `log_exception`, `nlsq/validation.py`, `result_presenter.py`, `hybrid_streaming.py`, `service/persist.py`, `get_safe_output_dir`, `test_fit_quality.py`, `service/config.py`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `ConfigManager` connect `ConfigManager` to `make_synthetic_two_component`, `HeterodyneModel`, `fit_heterodyne_stratified_least_squares`, `HeterodynePhysicsAdapter`, `.load_config`, `load_and_merge_config`, `OptimizationResult`, `SequentialResult`, `heterodyne_core.py`, `fit_nlsq`, `test_iteration_callback_seam.py`, `optimization/test_debug_audit_2026_07_22.py`, `test_config_relative_data_paths.py`, `test_homodyne_covariance_contract.py`, `core.py`, `build_heterodyne_stratified_data`, `NLSQConfig`, `load_config`, `ValueError`, `test_templates_no_fourier_keys.py`, `AnalysisMode`, `test_output_resolution.py`, `test_codex_review_fixes.py`, `GradientCollapseMonitor`, `logging.py`, `nlsq/__init__.py`, `test_perf_regression.py`, `fit_nlsq_jax`, `adapter.py`, `NLSQStrategy`, `StratificationConfig`, `fit_with_stratified_least_squares`, `HomodyneModel`, `test_heterodyne_per_angle_cmaes_fits_without_signature_drift`, `test_debug_audit_regressions.py`, `fit_two_component_via_engine`, `test_simulated_data_grid.py`, `test_homodyne_engine_preservation.py`, `run_fit`, `log_exception`, `test_engine_route_bugfixes.py`, `test_heterodyne_smoke_fit_recovers_truth`, `test_quality_gate_fixes.py`, `test_static_individual_invariant.py`, `service/persist.py`, `test_fit_quality.py`, `load_dataset`, `service/config.py`, `test_cmaes_trigger.py`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `OptimizationResult` connect `OptimizationResult` to `make_synthetic_two_component`, `generate_nlsq_plots`, `fit_heterodyne_stratified_least_squares`, `test_low_level_plots.py`, `_evaluate_c2_per_angle`, `AnalysisMode`, `_fixture`, `heterodyne_core.py`, `heterodyne_result_builder.py`, `SingleStartResult`, `NLSQWrapper`, `log_heterodyne_completion`, `optimization/test_debug_audit_2026_07_22.py`, `core.py`, `NLSQConfig`, `build_heterodyne_stratified_data`, `ValueError`, `test_output_resolution.py`, `GradientCollapseMonitor`, `stratified_ls.py`, `nlsq/__init__.py`, `._run_sequential_optimization`, `test_perf_regression.py`, `fit_nlsq_jax`, `heterodyne_config.py`, `adapter.py`, `_unpack_result_params`, `HomodyneModel`, `test_heterodyne_per_angle_cmaes_fits_without_signature_drift`, `test_review_regressions.py`, `fit_nlsq_multistart_heterodyne`, `fit_two_component_via_engine`, `test_persist.py`, `_FakeConfigManager`, `run_fit`, `multistart.py`, `log_exception`, `test_heterodyne_smoke_fit_recovers_truth`, `test_quality_gate_fixes.py`, `service/persist.py`, `test_fit_quality.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 139 inferred relationships involving `NLSQConfig` (e.g. with `_cfg()` and `_diag()`) actually correct?**
  _`NLSQConfig` has 139 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `ConfigManager` (e.g. with `test_config_summary_reads_real_surface()` and `test_initial_parameter_value_rejects_nonfinite()`) actually correct?**
  _`ConfigManager` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 195 inferred relationships involving `ValueError` (e.g. with `test_dispatch_fit_is_non_fatal_on_malformed_phi_angles()` and `_boom()`) actually correct?**
  _`ValueError` has 195 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `OptimizationResult` (e.g. with `test_perf_heterodyne_per_angle_local_fit()` and `test_heterodyne_smoke_fit_recovers_truth()`) actually correct?**
  _`OptimizationResult` has 44 INFERRED edges - model-reasoned connections that need verification._