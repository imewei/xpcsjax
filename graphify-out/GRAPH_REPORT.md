# Graph Report - xpcsjax  (2026-11-27)

## Corpus Check
- 623 files · ~711,214 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: .npz 2, (none) 1, .bat 1)

## Summary
- 12291 nodes · 25752 edges · 584 communities (439 shown, 145 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1905 edges (avg confidence: 0.9)
- Token cost: 2,221,470 input · 226,881 output

## Graph Freshness
- Built from commit: `e35451e9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- make_synthetic_two_component
- generate_nlsq_plots
- .from_config
- fit_heterodyne_stratified_least_squares
- HeterodynePhysicsAdapter
- test_low_level_plots.py
- nlsq_plots.py
- test_logging.py
- MainWindow
- OptimizationResult
- fit_two_component_via_engine
- DataQualityController
- ndarray
- test_heterodyne_fixed_parameters.py
- data/validation.py
- ParameterIndexMapper
- HeterodyneModel
- execute_with_recovery
- ProjectSidebar
- CMAESResult
- NLSQWrapper
- StratifiedResidualFunction
- Path
- make_cfgmgr_and_data
- NLSQConfig
- core.py
- main.py
- jax_backend.py
- NLSQConfig
- test_transforms.py
- test_anti_degeneracy_layers.py
- ValueError
- apply_angle_filtering_for_optimization
- parameter_registry.py
- Project
- Two-Component Config Template
- test_joint_ssr_floor.py
- FitJob
- XPCSDataFilter
- generate_completion.py
- diagonal_correction.py
- test_plots_view.py
- .fit
- AnalysisMode
- test_output_resolution.py
- PreprocessingPipeline
- GradientCollapseMonitor
- commands.py
- cmaes_wrapper.py
- test_plot_dispatch_logging.py
- nlsq/__init__.py
- XPCSDataLoader
- manager.py
- test_perf_regression.py
- test_gui_redesign.py
- wrapper.py
- n_optimized
- test_multistart.py
- cpu.py
- ParameterSpace
- fit_nlsq_jax
- test_uninstall_scripts.py
- NLSQAdapter
- test_parameter_utils.py
- sequential.py
- test_heterodyne_stratified_execute_layers.py
- optimization_runner.py
- heterodyne_jax_backend.py
- classify_option
- run_worker
- test_gui_jax_free.py
- test_heterodyne_stratification_config.py
- test_heterodyne_cmaes_warmstart_success_gate.py
- test_codex_ioviz.py
- DatasetOptimizer
- test_persist.py
- build_workbench
- _FakeConfigManager
- quality_controller.py
- _NullLogger
- test_heterodyne_memory_adapter.py
- _ooc_probe
- StratifiedResidualFunctionJIT
- test_review_regressions.py
- SingleStartResult
- QualityControlConfig
- .phi
- heterodyne_model_stateful.py
- AntiDegeneracyController
- chunking.py
- ResultBuilder
- test_heterodyne_data_prep.py
- test_homodyne_engine_preservation.py
- test_cache_source_file_validation.py
- test_heterodyne_tied_parameters.py
- test_post_install.py
- read_c2_preview
- models.py
- format_comparison
- ResultSummary
- DataInspectDialog
- NDArray
- test_heterodyne_tied_residuals.py
- heterodyne_core.py
- PerAngleScalingPlan
- NLSQStrategy
- resolve_per_angle_mode
- json_safe
- theme.py
- HomodyneModel
- _write_npz_compressed
- post_install.py
- test_quality_gate_fixes.py
- HeterodyneParameterSpace
- RunController
- test_static_individual_invariant.py
- service/persist.py
- test_save_path_validation.py
- test_fit_quality.py
- test_result_presenter.py
- load_dataset
- HeterodyneParameterManager
- ConfigTextEditorDialog
- test_codex_cli.py
- test_memory_manager_cache_gating.py
- test_max_imbalance_ratio_can_loosen_above_5
- test_cache_safety.py
- _make_synthetic_heterodyne
- get_or_create_fitter
- jit
- test_strategy_executors.py
- test_heterodyne_constant_mode.py
- test_aps_u_selective_read_d7.py
- ProjectDialogHandler
- test_jacobian.py
- ShearSensitivityWeighting
- test_homodyne_covariance_contract.py
- validate_xpcs_data
- test_runtime_shell.py
- test_system_validator.py
- get_adaptive_memory_threshold
- assemble_anti_degeneracy_diagnostics
- compute_g2_scaled
- test_load_degradation_signal.py
- test_laminar_mode_banners.py
- multistart.py
- OptimizationExecutor
- save_nlsq_json_files
- Bugs
- test_codex_gui.py
- test_memory_manager_logging.py
- test_heterodyne_cmaes_warmstart_auto_skip.py
- .__init__
- test_adapter_xdata_cache.py
- fit_with_stratified_hybrid_streaming_heterodyne
- optimization/test_validation.py
- HierarchicalOptimizer
- References and Citations Doc
- MemoryStats
- detect_degeneracy
- fit_with_stratified_hybrid_streaming
- PhysicsFactors
- main_window.py
- xpcsjax/data/__init__.py
- FitQueueController
- xla_config.py
- NLSQResult
- test_heterodyne_physics_validators.py
- TestMeshgridCacheCollision
- Bugs
- simulated.py
- assemble_anti_degeneracy_diagnostics
- json_serializer
- fit_nlsq entry point
- gauss_newton_covariance
- PhiResultsGrid
- test_codex_optstrat.py
- test_fit_queue.py
- test_get_model.py
- Bugs
- heterodyne_scaling_utils.py
- hybrid_streaming
- BatchStatistics
- ExecutionResult
- test_cache_q_validation.py
- test_config_jax_free.py
- test_validation_branches.py
- hybrid_streaming
- ValidationResult
- nlsq/validation.py
- test_completion_parity.py
- NumericalValidator
- LHS multistart
- Memory-Aware Strategy Routing
- test_no_pickle_loads.py
- hybrid_streaming
- npz_cache.py
- test_hybrid_streaming_retry.py
- test_layer5_gating.py
- hybrid_streaming
- compute_diagonal_overlay_stats
- AdvancedMemoryManager
- load_xpcs_data
- NLSQ CurveFit (trust-region least squares)
- test_stratified_max_iter_grading.py
- C. New findings
- create_time_integral_matrix
- numpy
- fit_nlsq_multistart_heterodyne
- _fit_nlsq_heterodyne
- test_chunked_jacfwd_dense.py
- test_compute_pool_matches_direct_kernel
- logging.py
- test_layer_gate_wiring.py
- _FlakyOptimizer
- compute_chi_squared
- AntiDegeneracyController
- physics.py
- ParameterRegistry
- test_pr80_review.py
- XPCSDataFormatError
- test_config_unwrap.py
- test_phase5_model_function_modes.py
- run_fit
- AdaptiveRegularizer
- _logger_that_raises_on_log
- test_debug_audit_2026_07_23_diagonal_skip.py
- NLSQOptimizationError
- MemoryPressureMonitor
- cmaes
- metadata
- build_gradient_collapse_callback
- cmaes
- test_io.py
- cmaes
- completion.sh
- test_docs_structure.py
- TestExecuteLayersNLSQConfigHomodyne
- phi_filtering.py
- TestExecuteLayersNLSQConfigHeterodyne
- stratification
- fit_with_out_of_core_accumulation
- test_post_install_fish.py
- cmaes
- test_validation_crash_coverage.py
- plots_view.py
- interactive_setup
- anti_degeneracy_controller.py
- TestTypeBoundary
- run_validation
- _normalize_data_to_object
- select_nlsq_strategy
- All-modules registry page
- uninstall_scripts.py
- heterodyne_parameter_space.py
- core/test_debug_audit_2026_07_22.py
- Finished
- service/config.py
- InputValidator
- ResultValidator
- TestNoScipyLeastSquares
- generate_plots
- test_loader_performance_engine_cleanup.py
- fit_nlsq
- _build_laminar_stratified_data
- test_run_controller.py
- test_phase5_vector_build.py
- file
- FitQualityReport
- test_lazy_imports.py
- _resolve_color_limits
- fit_nlsq
- parse_per_angle_scaling_overrides
- model_adapter.PointEvaluator
- plotting
- xpcs_loader.py
- nlsq
- test_pr81_review.py
- build_heterodyne_pointwise_model
- test_optstrat_b4
- test_stratified_ls_averaged_covariance_transform.py
- color_scale
- test_logging_quality_gate.py
- plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2)
- NLSQ Residual Diagnostics Baseline (phi=45deg)
- xpcsjax.cli.plot_dispatch
- xpcsjax.optimization.nlsq.fit_nlsq
- test_freeze_safety.py
- test_debug_audit_2026_07_23_negative_correlation_repair.py
- file
- FitQualityConfig
- TestJSONFormatterCircularRef
- TestLogPerformanceNeverRaise
- TestContextFilterOnLogger
- file
- file
- xla_config.bash
- Public API Reference
- _PhiSection
- OptimizationStrategy
- _FakeModel
- parallel_accumulator.py
- test_docs_no_fourier.py
- _version_at_least
- Bugs
- A. Fix review
- test_anti_degeneracy_transforms.py
- test_two_component_smoke.py
- plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg)
- ConfigManager
- xpcsjax.core.HeterodyneModel
- AntiDegeneracyController
- 5-layer anti-degeneracy controller
- Second-order (intensity) correlation function c2
- _FakeOpt
- ContextFilter
- test_ci_heavy_nodes_parity.py
- TestPropagateInversion
- hierarchical
- generate_nlsq_plots
- Joint global escape keep-better contract
- test_config_debug_null_nonfinite.py
- test_data_package_features.py
- test_config_relative_data_paths.py
- Bugs
- test_no_removed_per_angle_tokens_in_tests_or_package
- test_laminar_no_worse_phase6.py
- Architecture Overview
- xpcsjax.cli.config_generator.main
- xpcsjax Project Logo
- test_aps_u_bin_skip_misalignment.py
- post_install.py
- tests/conftest.py
- _build_stratified_residual_fn
- _install_completion_bash_activation
- configure_cpu_hpc
- DatashaderRenderer
- Decision Record: CPU-only Execution
- make test-smoke
- HomodyneModel
- User Guide Overview
- accumulate_chunks_sequential
- anti_degeneracy
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
- regularization
- Bugs
- per_angle_mode.PerAngleScalingPlan
- nlsq
- nlsq
- nlsq
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
- tests/benchmarks shard
- tests/core shard
- tests/heterodyne shard
- tests/property shard
- Examples Index
- Autosummary Module Template
- Theory and Physics Index
- Command-Line Interface
- ConfigManager
- XpcsDataset
- physics_validators.py
- OOCSharedArrays
- test_worker_functions_in_process
- xpcsjax/__init__.py
- NLSQ Integration Contract
- test_codex_data.py
- Round 1 ledger (Codex claims -> Claude verification)
- OOCComputePool
- CLI Module Guide
- Maintainer-decision items from the 2026-11-24 review — dispositions
- Config Module Guide
- A. Fix review
- plotting
- Core Module Guide
- anti_degeneracy
- plotting
- anti_degeneracy
- plotting
- Changelog Documentation
- test_heterodyne_parameter_manager_tied.py
- test_laminar_execute_layers.py
- optimization
- Data Module Guide
- optimization
- optimization
- log_heterodyne_completion
- A. Fix review
- test_out_of_core_route.py
- Device Module Guide
- GUI Module Guide
- IO Module Guide
- Optimization Module Guide
- performance
- output
- phi_filtering
- performance
- output
- phi_filtering
- performance
- output
- phi_filtering
- performance
- output
- Runtime Module Guide
- Service Module Guide
- phi_filtering
- test_heterodyne_config_bounds_override.py
- metadata
- Utils Module Guide
- Viz Module Guide
- Decision round: close the 3 remaining open items
- Findings
- Installation Guide
- experimental_data
- experimental_data
- experimental_data
- experimental_data
- ValueError
- v0.1.0 Initial Release
- _get_dataset_size
- xpcsjax_laminar_flow.yaml
- multi_start
- streaming
- xpcsjax_static_anisotropic.yaml
- multi_start
- streaming
- xpcsjax_static_isotropic.yaml
- multi_start
- streaming
- xpcsjax_two_component.yaml
- multi_start
- streaming
- Findings
- Closing check: fixes for your PR #81 review
- Review request: xpcsjax PR #81 (draft)
- FakeHandle
- analyzer_parameters
- quality_control
- analyzer_parameters
- quality_control
- analyzer_parameters
- quality_control
- analyzer_parameters
- quality_control
- initial_parameters
- rasterize
- test_chunked_jacfwd.py
- heterodyne_logging.py
- agy position — round 2
- test_aps_u_alloc_guard.py
- validation
- initial_parameters
- test_streaming_resolves_canonical_mode
- stratification
- validation
- initial_parameters
- types
- stratification
- validation
- initial_parameters
- stratification
- validation
- metadata
- StreamingExecutor
- color_scale
- metadata
- regularization
- regularization
- codex position — round 2
- test_streaming_l2_real_hessian_covariance_via_inv_on_success
- bounds
- quality_validation
- bounds
- quality_validation
- bounds
- quality_validation
- bounds
- color_scale
- quality_validation
- StandardExecutor
- ConfigManager
- Codex closing report
- codex position — round 1
- codex closing check of PR #81
- detect_shell_type
- color_scale
- system_validator.py
- hierarchical
- shear_weighting
- matplotlib
- matplotlib
- matplotlib
- _effective_param_count_for_ooc
- test_templates_no_fourier_keys.py
- diagnostics
- diagnostics
- diagnostics
- test_streaming_l2_singular_hessian_falls_back_to_placeholder
- test_streaming_averaged_optimizes_scaling_and_sets_group_variance

## God Nodes (most connected - your core abstractions)
1. `NLSQConfig` - 280 edges
2. `ConfigManager` - 249 edges
3. `make_synthetic_two_component()` - 197 edges
4. `OptimizationResult` - 171 edges
5. `get_logger()` - 147 edges
6. `AnalysisMode` - 142 edges
7. `fit_nlsq_multi_phi()` - 129 edges
8. `HeterodyneModel` - 103 edges
9. `ParameterManager` - 94 edges
10. `XPCSDataLoader` - 93 edges

## Surprising Connections (you probably didn't know these)
- `opthet-B9: Zero-valued regularization rows shrink the reported direct-fit covariance` --references--> `rescale_covariance_dof()`  [INFERRED]
  docs/reviews/2026-11-24-evidence/round1-codex-shards/opthet.md → xpcsjax/optimization/nlsq/covariance.py
- `R3-C1: Stateful homodyne plots — The measured-q rebuild adds mandatory frame fields to plotting config` --references--> `HomodyneModel`  [INFERRED]
  docs/reviews/2026-11-24-evidence/round3-codex-report.md → xpcsjax/core/homodyne_model.py
- `data-B2: Selective HDF5 loading silently broadcasts a short matrix into a full correlation surface` --references--> `XPCSDataFormatError`  [INFERRED]
  docs/reviews/2026-11-24-evidence/round1-codex-shards/data.md → xpcsjax/data/xpcs_loader.py
- `R2-C2: HDF5 allocation — Guards budget the source dtype before allocating the promoted buffer` --references--> `XPCSDataFormatError`  [INFERRED]
  docs/reviews/2026-11-24-evidence/round2-codex-report.md → xpcsjax/data/xpcs_loader.py
- `data-B5: The HDF5 readers and the quality filter read different configuration keys` --references--> `XPCSDataFilter`  [INFERRED]
  docs/reviews/2026-11-24-evidence/round1-codex-shards/data.md → xpcsjax/data/filtering_utils.py

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

## Communities (584 total, 145 thin omitted)

### Community 0 - "make_synthetic_two_component"
Cohesion: 0.03
Nodes (84): opthet-B13: Sequential reduced chi-squared subtracts the size of the reporting vector, F1: Zero-DOF masked heterodyne joint fits crash in `OptimizationResult.__post_init__` across all three modes, _diag(), test_constant_path_always_emits_activation_keys(), test_detail_keys_preserved_when_enabled(), test_disabled_path_now_emits_activation_keys_false(), test_disabled_path_omits_layer_detail_keys(), test_enabled_path_emits_activation_keys_true() (+76 more)

### Community 1 - "generate_nlsq_plots"
Cohesion: 0.12
Nodes (19): _phi_filename(), _png_sha256(), test_c2_exp_shape_mismatch_raises(), test_homodyne_model_no_legacy_plot_methods(), test_invalid_compression_raises(), test_missing_c2_exp_raises(), test_missing_q_raises(), test_orchestrator_closes_all_figures() (+11 more)

### Community 2 - ".from_config"
Cohesion: 0.08
Nodes (25): test_tied_fit_ssr_matches_recompute_from_reported_parameters(), test_tied_gradient_sums_both_usages(), _tied_param_manager(), _assert_tied_result_shape(), _build_synthetic_c2(), _run_tied_fit(), test_averaged_mode_tied_fit_reports_full_physics(), test_constant_mode_tied_fit_reports_full_physics() (+17 more)

### Community 3 - "fit_heterodyne_stratified_least_squares"
Cohesion: 0.03
Nodes (52): _params(), test_pointwise_matches_meshgrid(), test_pointwise_multi_phi_gather(), test_stratified_ls_scaling_names_match_dedup_phi_count(), test_averaged_scaling_expander_broadcasts(), test_bounds_clip_enforces_bounds_on_marginally_out_of_bounds_result(), fit(), test_bounds_clip_is_noop_for_in_bounds_result() (+44 more)

### Community 4 - "HeterodynePhysicsAdapter"
Cohesion: 0.06
Nodes (10): TestHeterodyneComputeG1MultiPhi, TestHeterodyneComputeResidual, test_analysis_mode_is_two_component(), test_default_params_length_14(), test_is_physics_model(), test_model_constructible_via_from_config(), test_param_bounds_match_docs(), test_param_names_match_registry() (+2 more)

### Community 5 - "test_low_level_plots.py"
Cohesion: 0.08
Nodes (33): ioviz-B8: Per-angle simulated heatmaps display the mean contrast and offset, ioviz-B9: Residual traces use unequal-time points when the axes have different origins, test_evaluate_homodyne_2d_finite(), test_evaluate_unsupported_raises(), test_plot_nlsq_fit_accepts_t_none(), test_plot_nlsq_fit_residual_cmap_is_rdbu(), test_plot_nlsq_fit_save_path_writes_png(), test_plot_nlsq_fit_shared_color_scale() (+25 more)

### Community 6 - "nlsq_plots.py"
Cohesion: 0.05
Nodes (30): R2-A11: gui-B4/ioviz-B2 — The documented stateful plotting path still evaluates the wrong q, test_ioviz_b2(), test_heterodyne_evaluator_matches_fit_time_model_on_short_grid(), test_evaluate_heterodyne_returns_2d_finite(), test_evaluate_null_scattering_section_raises_valueerror(), test_resolve_phi_index_prefers_explicit_index(), test_unpack_heterodyne_per_angle_layout(), test_unpack_heterodyne_size_mismatch_raises() (+22 more)

### Community 7 - "test_logging.py"
Cohesion: 0.03
Nodes (39): _record(), test_color_formatter_applies_and_restores_color(), test_color_formatter_no_color(), test_configure_creates_rotating_file_handler(), test_configure_disables_propagation_when_handler_installed(), test_configure_from_dict_filename_auto_generated(), test_configure_from_dict_filename_timestamp_placeholder(), test_configure_from_dict_filename_without_run_id_placeholder() (+31 more)

### Community 8 - "MainWindow"
Cohesion: 0.03
Nodes (9): gui-B5: Reloading a project saved during a fit can permanently block Run for that dataset, test_expand_path_resolves_env_and_user(), test_append_log_multiple_lines(), test_status_manager_is_qobject_parented_to_window(), test_status_manager_overwrite(), test_status_manager_sets_status_and_appends_log(), _expand_path(), MainWindow (+1 more)

### Community 9 - "OptimizationResult"
Cohesion: 0.05
Nodes (30): ioviz-B4: A legal missing-uncertainty placeholder aborts plot artifact export, test_config_summary_reads_real_surface(), test_save_npz_none_covariance_uses_documented_shapes(), test_per_angle_chi2_raises_when_missing(), test_per_angle_chi2_reads_from_diagnostics(), test_reconstruct_per_angle_scaling_auto_resolves_to_averaged(), test_reconstruct_per_angle_scaling_averaged_mode_falls_back_to_head(), test_reconstruct_per_angle_scaling_averaged_mode_from_diagnostics() (+22 more)

### Community 10 - "fit_two_component_via_engine"
Cohesion: 0.03
Nodes (56): opthet-B1: The in-memory engine swaps angle-specific scaling on unsorted input, _cfg(), test_engine_route_uncertainties_match_multi_phi_dof(), _cfg(), _singular_curve_fit(), test_engine_route_singular_reports_nan_and_flag(), test_fixed_or_tied_slots_do_not_trip_the_placeholder_flag(), test_in_memory_multi_phi_singular_reports_nan_and_flag() (+48 more)

### Community 12 - "ndarray"
Cohesion: 0.06
Nodes (12): Bugs, Codex review — shard core, core-B10: Parameter validation succeeds when it cannot convert a concrete parameter, core-B1: Heterodyne fits shift the entire time axis when the loader clamps the end frame, core-B2: The scattered homodyne kernel changes the modeled correlation on the same frame grid, core-B3: The opt-in homodyne adapter flattens correlations beyond 10,000 frame intervals, core-B5: A masked homodyne observation can make chi-squared NaN, core-B8: JAX diagonal correction truncates fractional interpolants for integer arrays (+4 more)

### Community 13 - "test_heterodyne_fixed_parameters.py"
Cohesion: 0.09
Nodes (13): _manager(), test_expand_reduced_result_fixed_physics_param_gets_nan(), test_expand_reduced_result_physics_first_with_scaling(), test_expand_reduced_result_scaling_first_with_scaling(), test_expand_reduced_result_tied_child_mirrors_parent_covariance(), test_expand_reduced_result_untied_physics_first_no_scaling(), _config(), test_fixed_contrast_is_honored() (+5 more)

### Community 14 - "data/validation.py"
Cohesion: 0.08
Nodes (19): _report(), test_validate_array_shapes_records_issue_on_arithmeticerror(), test_validate_array_shapes_records_issue_on_attributeerror(), _raise(), _raise_boom(), test_array_shapes_crash_logs_error_and_invalidates_report(), test_report_add_issue_flips_is_valid_on_error(), test_report_warning_does_not_invalidate() (+11 more)

### Community 15 - "ParameterIndexMapper"
Cohesion: 0.07
Nodes (13): _linear_residual(), test_estimate_gradient_noise(), test_jacobian_condition_number(), test_jacobian_stats_ill_conditioned_qr_branch(), test_jacobian_stats_jax_residual_attr(), test_jacobian_stats_well_conditioned(), test_mapper_constant_mode(), test_mapper_diagnostics() (+5 more)

### Community 16 - "HeterodyneModel"
Cohesion: 0.03
Nodes (47): core-B7: Stateful heterodyne residuals silently broadcast incomplete correlation matrices, R4-A3, Item 1 — R4-A3: zero-weight rows in the heterodyne joint covariance DOF, Report format (your file is named in the message that sent you here), Review request: xpcsjax PR #81 (draft), Rules, What to look for, Where everything is (exact paths; do NOT search the filesystem) (+39 more)

### Community 17 - "execute_with_recovery"
Cohesion: 0.10
Nodes (16): _result(), _run(), solver(), _solve(), test_a_later_exception_keeps_the_earlier_usable_solve(), test_an_exception_without_a_usable_solve_still_raises(), test_lowest_cost_attempt_is_the_one_accepted(), test_solver_cost_ranks_the_attempts() (+8 more)

### Community 18 - "ProjectSidebar"
Cohesion: 0.05
Nodes (21): test_add_dataset_assigns_stable_unique_ids(), test_add_run_is_append_only_and_queued(), test_set_run_status_updates_in_place(), test_unknown_ids_return_none(), _summary(), test_comparison_view_marks_differing_values(), test_comparison_view_renders_nan_parameter_and_flags_diff(), test_comparison_view_shows_two_runs() (+13 more)

### Community 19 - "CMAESResult"
Cohesion: 0.10
Nodes (18): _laminar_cmaes_config(), test_cmaes_fixed_constant_survives_none_covariance(), _fake_fit(), test_cmaes_laminar_does_not_silently_fail(), test_cmaes_laminar_result_param_count(), test_cmaes_nan_q_is_rejected_not_silently_fit(), test_cmaes_result_construction_error_propagates(), test_cmaes_typo_active_parameters_propagates_not_silently_failed() (+10 more)

### Community 20 - "NLSQWrapper"
Cohesion: 0.06
Nodes (13): test_stratified_sigma_phi_index_matches_own_angle(), test_optwrap_b1(), _build_homodyne_l4_callback(), _l4_plus_observer(), _loss(), _observer_only(), on_iteration(), create_multistart_warmup_func() (+5 more)

### Community 21 - "StratifiedResidualFunction"
Cohesion: 0.07
Nodes (19): _full_grid_chunk(), _stratified(), test_compute_flat_indices_clamps_out_of_range_axis_independently(), test_deprecated_paths_raise(), test_diag_mask_uses_values_not_independent_unique_indices(), test_empty_chunks_raises(), test_factory_with_validation(), test_factory_without_validation() (+11 more)

### Community 22 - "Path"
Cohesion: 0.12
Nodes (21): test_cleanup_completion_files_dry_run(), test_cleanup_completion_files_removes(), test_cleanup_xla_config_dry_run(), test_cleanup_xla_config_removes_and_prunes_parent(), test_find_cleanup_targets_detects_legacy_xla_mode(), test_find_cleanup_targets_flags_empty_parent_dir(), test_find_cleanup_targets_marks_existing(), test_interactive_cleanup_cancelled() (+13 more)

### Community 23 - "make_cfgmgr_and_data"
Cohesion: 0.10
Nodes (12): make_cfgmgr_and_data(), test_flat_enable_cmaes_skips_stratified_ls(), test_ge_1M_constant_mode_uses_stratified_ls(), test_ge_1M_stratifies(), _fake(), test_hybrid_streaming_takes_precedence_over_stratified_ls(), _fake_select(), test_individual_mode_uses_stratified_ls() (+4 more)

### Community 24 - "NLSQConfig"
Cohesion: 0.07
Nodes (27): test_cmaes_data_chunk_size_above_ceiling_is_rejected(), test_cmaes_data_chunk_size_none_is_accepted(), test_cmaes_data_chunk_size_normal_is_accepted(), test_cmaes_data_chunk_size_zero_still_rejected(), test_hybrid_chunk_size_above_ceiling_is_rejected(), test_hybrid_chunk_size_normal_is_accepted(), test_hybrid_chunk_size_zero_still_rejected(), test_streaming_chunk_size_above_ceiling_is_rejected() (+19 more)

### Community 25 - "core.py"
Cohesion: 0.05
Nodes (26): optescape-B6: Multistart drops fixed parameters from the reported initial vectors, ParameterSpace, is_adapter_available(), _bounds_to_arrays(), _cmaes_failed_result(), _estimate_contrast_offset_from_data(), _extract_shear_transform_config(), fit_nlsq_cmaes() (+18 more)

### Community 26 - "main.py"
Cohesion: 0.14
Nodes (8): cli-B3: `--threads 2` aborts JAX backend initialization, gui-B3: `--phi nan` silently selects the first angle, create_parser(), validate_args(), _bootstrap_xla_env(), main(), main_xjexp(), main_xjsim()

### Community 27 - "jax_backend.py"
Cohesion: 0.05
Nodes (23): core-B3, test_validate_backend_gradient_support_true_on_working_jax(), TestDiagonalCorrectionBatchN1, TestG2ScaledContrastZero, TestG2ScaledFactorsParity, batch_chi_squared(), compute_g1_diffusion(), _compute_g1_diffusion_core() (+15 more)

### Community 28 - "NLSQConfig"
Cohesion: 0.03
Nodes (66): _build_model(), _config_dict(), _synthetic_stack(), test_auto_mode_resolves_to_averaged_for_many_angles(), test_cmaes_path_runs(), test_individual_mode_joint_fit(), test_multi_phi_ssr_conservation(), test_multistart_path_runs() (+58 more)

### Community 29 - "test_transforms.py"
Cohesion: 0.04
Nodes (54): _laminar_index_map(), test_adjust_covariance_beta_unchanged(), test_adjust_covariance_log_jacobian(), test_adjust_covariance_no_state_or_empty(), test_build_physical_index_map(), test_format_x_scale_for_log(), test_forward_bounds_none_or_empty_state_passthrough(), test_forward_bounds_rejects_nonpositive_gamma() (+46 more)

### Community 30 - "test_anti_degeneracy_layers.py"
Cohesion: 0.08
Nodes (10): test_anti_degeneracy_config_overrides_defaults(), test_controller_instantiates_with_minimal_config(), test_controller_source_references_all_layers(), test_module_exports_all_layer_classes(), test_regularization_auto_tune_lambda_wired_through_controller(), TestExecuteLayersFlag, test_anti_degeneracy_config_null_hierarchical_section_falls_back_to_defaults(), test_anti_degeneracy_config_null_hierarchical_tolerance_falls_back_to_default() (+2 more)

### Community 31 - "ValueError"
Cohesion: 0.13
Nodes (12): _linear_residual_fn(), test_averaged_mode_now_wires_the_keep_better_floor(), _spy(), test_build_joint_problem_individual_bounds_match_registry(), test_constant_mode_now_wires_the_keep_better_floor(), test_individual_mode_still_wires_the_floor_unchanged(), test_keep_better_floor_keeps_a_better_candidate(), test_keep_better_floor_never_regresses_ssr() (+4 more)

### Community 32 - "apply_angle_filtering_for_optimization"
Cohesion: 0.13
Nodes (13): _config(), test_plot_and_optimization_paths_select_same_angles_across_wrap(), test_frame_range_lower_bound_checked_when_end_is_none(), test_frame_range_valid_start_with_end_none_passes(), test_normalize_angle_preserves_minus_180_array(), test_normalize_angle_preserves_minus_180_scalar(), test_apply_angle_filtering_explicit_none_values_do_not_crash(), angle_in_range() (+5 more)

### Community 33 - "parameter_registry.py"
Cohesion: 0.07
Nodes (16): test_heterodyne_mode_lists_14_params(), test_heterodyne_param_specs(), test_heterodyne_synonym_normalize(), test_manager_bounds_match_registry_for_all_modes(), test_no_inline_bound_constants_in_manager(), test_clip_value_lands_inside_bounds(), test_get_param_names_returns_valid_registry_keys(), test_heterodyne_synonyms_all_normalize() (+8 more)

### Community 34 - "Project"
Cohesion: 0.06
Nodes (21): test_load_accepts_absolute_path_by_design(), test_load_rejects_backslash_style_traversal(), test_load_rejects_null_byte_in_path(), test_load_rejects_relative_path_that_escapes_project_directory(), test_load_rejects_relative_result_dir_that_escapes_project_directory(), test_load_rejects_unknown_schema(), test_round_trip_preserves_datasets_and_runs(), test_save_is_atomic_failed_write_keeps_original_intact() (+13 more)

### Community 35 - "Two-Component Config Template"
Cohesion: 0.22
Nodes (14): analysis_mode Taxonomy Deprecation, Config-Schema Version, Laminar Flow 5-Layer Anti-Degeneracy Stack, Laminar Flow CMA-ES Config, Laminar Flow Config Template, Laminar Flow Physics Model (7-param), Static Anisotropic Config Template, Static Anisotropic Physics Model (3-param) (+6 more)

### Community 36 - "test_joint_ssr_floor.py"
Cohesion: 0.05
Nodes (22): _degraded_adapter(), _individual_cfg(), test_build_floor_fallback_is_x0_for_non_individual_modes(), test_build_floor_fallback_is_x0_for_uniform_scaling(), test_build_floor_fallback_preserves_distinct_per_angle_scaling(), test_build_joint_problem_exposes_floor_fallback(), test_cmaes_escape_builds_joint_problem_once(), test_escape_kept_assembly_still_reports_success() (+14 more)

### Community 37 - "FitJob"
Cohesion: 0.06
Nodes (22): emit_started_then_finished(), exit_without_terminal(), sleep_forever(), test_worker_handle_shutdown_reaps_process(), test_emitter_stamps_run_id_and_monotonic_seq(), test_fitjob_is_frozen_and_picklable(), _collect(), test_cancel_and_shutdown_join_reader_thread() (+14 more)

### Community 38 - "XPCSDataFilter"
Cohesion: 0.05
Nodes (17): test_bounds_reject_nonfinite(), test_finite_bounds_still_load(), test_initial_parameter_value_rejects_nonfinite(), test_per_angle_scaling_rejects_nonfinite(), test_quality_score_robust_to_single_boundary_artifact(), test_quality_score_still_flags_genuinely_overnormalized(), test_degenerate_matrix_nan_quality_score_is_dropped_not_kept(), _clean_matrix_with_raw_spike() (+9 more)

### Community 39 - "generate_completion.py"
Cohesion: 0.13
Nodes (7): CommandSpec, _all_option_strings(), _emit_footer(), _emit_function(), generate(), main(), _value_completion_bash()

### Community 40 - "diagonal_correction.py"
Cohesion: 0.08
Nodes (18): test_degenerate_1x1_does_not_crash(), test_diagonal_is_replaced(), test_off_diagonal_preserved(), test_statistical_correction_warns_and_clamps_on_nonpositive_window_size(), apply_diagonal_correction(), apply_diagonal_correction_batch(), _basic_correction_numpy(), _diagonal_correction_batch_jax() (+10 more)

### Community 41 - "test_plots_view.py"
Cohesion: 0.14
Nodes (9): test_map_applies_color_lookup_table(), test_map_axes_labelled_t1_t2(), test_map_keeps_equal_t1_t2_range_without_aspect_lock(), test_map_view_survives_unresolvable_colormap(), test_plot_widgets_render_square(), test_two_time_map_accepts_array(), test_two_time_map_falls_back_to_frame_indices(), test_two_time_map_places_image_on_physical_t1_t2_grid() (+1 more)

### Community 42 - ".fit"
Cohesion: 0.12
Nodes (8): opthet-B6: The single-angle JAX adapter reports convergence after the solver exhausts its budget, test_assess_convergence_no_progress(), test_assess_convergence_nonfinite(), test_assess_convergence_poor_fit(), test_assess_convergence_success(), _assess_convergence(), _wrapped(), _run_curve_fit()

### Community 43 - "AnalysisMode"
Cohesion: 0.05
Nodes (34): cli-B5: Standalone simulation drops fixed parameters before assembling the model vector, config-B7: Public homodyne `ParameterSpace` rejects a null bound that its manager accepts, test_empty_active_parameters_list_means_none_active(), test_missing_active_parameters_key_falls_back_to_defaults(), test_analysis_mode_property_returns_enum(), test_null_bound_falls_back_to_default(), test_empty_tied_parameters_dict_is_noop_for_homodyne(), test_malformed_tied_parameters_still_rejected_for_homodyne() (+26 more)

### Community 44 - "test_output_resolution.py"
Cohesion: 0.08
Nodes (19): _args(), _cfg(), test_cli_override_wins_over_config(), test_legacy_output_settings_fallback(), test_none_when_unconfigured(), test_plots_dir_agrees_with_result_dir_template_schema(), test_plots_dir_honors_cli_override(), test_post_fit_plots_return_none_on_failure() (+11 more)

### Community 45 - "PreprocessingPipeline"
Cohesion: 0.05
Nodes (14): _integer_c2_with_fractional_correction(), test_diagonal_correction_upcasts_integer_c2_no_truncation(), test_statistical_diagonal_correction_warns_and_clamps_on_nonpositive_window_size(), _integer_c2(), test_reduce_noise_upcasts_integer_c2_no_truncation(), NoiseReductionMethod, NormalizationMethod, PreprocessingConfigurationError (+6 more)

### Community 46 - "GradientCollapseMonitor"
Cohesion: 0.06
Nodes (26): _monitor(), test_check_collapse_after_consecutive_triggers(), test_check_disabled_returns_ok(), test_check_healthy_gradient_is_ok(), test_check_rearms_after_recovery(), test_check_skips_off_interval(), test_check_tracks_best_params(), test_check_watched_parameter_collapse() (+18 more)

### Community 47 - "commands.py"
Cohesion: 0.07
Nodes (21): cli-B7: A plot-directory error changes a converged fit into a failed command, test_initial_override_lands_in_initial_parameters_values(), test_initial_override_resolver_failure_raises_value_error(), _write_static_isotropic_config(), test_dispatch_fit_is_non_fatal_on_malformed_phi_angles(), test_load_and_validate_data_forwards_phi_as_phi_subset(), test_resolve_phi_angles_forwards_cli_phi_and_phi_angles_str(), _stub_fit() (+13 more)

### Community 48 - "cmaes_wrapper.py"
Cohesion: 0.05
Nodes (25): R2-A6: optwrap-B5 — Sequential, adapter, stratified and CMA-ES diagnostics still charge fixed slots, test_configure_memory_respects_explicit_popsize_override(), test_fit_calls_configure_memory_after_scale_ratio_block_with_final_popsize(), test_default_threshold_is_1000(), test_cmaes_seed_defaults_to_none_when_unset(), test_cmaes_seed_field_exists_with_none_default(), test_cmaes_seed_parsed_from_dict(), test_cmaes_seed_reaches_wrapper_config() (+17 more)

### Community 49 - "test_plot_dispatch_logging.py"
Cohesion: 0.13
Nodes (11): _FakeResult, _make_data(), _reset_log_once(), _stub_c2_eval(), test_per_phi_render_failure_logs_once_not_per_angle(), _fit_raises(), _resid_raises(), test_phi_index_threaded_matches_loop_position() (+3 more)

### Community 50 - "nlsq/__init__.py"
Cohesion: 0.04
Nodes (25): test_estimate_peak_memory_gb_formula(), test_build_parameter_labels_no_per_angle(), test_build_parameter_labels_per_angle(), AdapterConfig, NLSQAdapterBase, get_adapter(), HybridRecoveryConfig, NLSQResult (+17 more)

### Community 51 - "XPCSDataLoader"
Cohesion: 0.04
Nodes (6): data-B18: Memory-manager shutdown overwrites process GC thresholds it never changed, _make_aps_old_hdf5(), test_aps_old_quality_filter_guards_allocation_before_accumulation(), test_aps_old_zero_selection_raises(), _maybe_apply_mandatory_diagonal_correction(), XPCSDataLoader

### Community 52 - "manager.py"
Cohesion: 0.08
Nodes (12): load_xpcs_config(), _NoYAMLError, BoundDict, ExperimentalDataConfig, HomodyneConfig, InitialParametersConfig, NLSQValidationConfig, OptimizationConfig (+4 more)

### Community 53 - "test_perf_regression.py"
Cohesion: 0.18
Nodes (5): _build_synthetic_c2(), _het_smoke_config_dict(), test_perf_heterodyne_per_angle_local_fit(), test_perf_select_strategy_10k_points(), test_perf_select_strategy_10m_points()

### Community 54 - "test_gui_redesign.py"
Cohesion: 0.05
Nodes (15): test_close_project_resets_to_empty_state(), test_close_project_tears_down_active_worker(), test_create_config_dialog_collects_inputs(), test_create_config_dialog_omits_blank_optionals(), test_create_config_dialog_raises_on_invalid_numeric(), test_create_config_writes_template_for_mode(), test_create_project_sets_working_dir(), test_file_menu_has_workflow_actions() (+7 more)

### Community 55 - "wrapper.py"
Cohesion: 0.04
Nodes (26): test_restore_by_mask_jax_matches_numpy_and_is_traceable(), traced(), test_optrest_b1(), AllocationError, get_logger(), MemoryManagerError, MemoryPressureError, _chunked_jacfwd_dense() (+18 more)

### Community 56 - "n_optimized"
Cohesion: 0.14
Nodes (12): test_averaged_and_individual_are_not_frozen(), test_blocks_partition_the_vector_scaling_first(), test_canonical_rejects_unresolved_mode(), test_constant_mode_is_frozen_with_empty_scaling_head(), test_group_indices(), test_legacy_fourier_constructor_is_untouched(), test_vector_length_property(), test_mapper_drives_streaming_L3_L4() (+4 more)

### Community 57 - "test_multistart.py"
Cohesion: 0.10
Nodes (24): _fit_factory(), fit(), test_config_defaults(), test_config_to_nlsq_global_config(), test_config_to_nlsq_global_config_screening_off(), test_get_n_workers(), test_include_custom_starts_all_out_of_bounds_returns_generated(), test_include_custom_starts_filters_out_of_bounds() (+16 more)

### Community 58 - "cpu.py"
Cohesion: 0.06
Nodes (22): _cpu_info_no_extras(), _fake_cpuinfo(), _fake_lscpu(), isolated_configure(), test_configure_jax_cpu_skips_warning_when_flag_already_applied(), test_configure_jax_cpu_warns_when_flag_entirely_missing(), test_configure_jax_cpu_warns_when_flag_value_differs(), test_explicit_thread_count_is_not_divided() (+14 more)

### Community 59 - "ParameterSpace"
Cohesion: 0.12
Nodes (3): test_from_config_rejects_inf_bound(), test_from_config_rejects_nan_bound(), ParameterSpace

### Community 60 - "fit_nlsq_jax"
Cohesion: 0.06
Nodes (45): _config(), _hybrid_streaming_fake_stratify_and_config(), _force_hybrid_streaming(), _physical_index(), _synthetic_data(), test_fixed_parameter_survives_cmaes_fit(), test_fixed_parameter_survives_hybrid_streaming_fit(), test_fixed_parameter_survives_hybrid_streaming_fit_individual_mode() (+37 more)

### Community 61 - "test_uninstall_scripts.py"
Cohesion: 0.14
Nodes (17): test_cleanup_activation_scripts_dry_run(), test_cleanup_activation_scripts_no_venv(), test_cleanup_activation_scripts_scrubs_block(), test_cleanup_activation_scripts_skips_clean_files(), test_get_venv_path_conda(), test_get_venv_path_none(), test_get_venv_path_via_sys_prefix(), test_get_venv_path_virtual_env() (+9 more)

### Community 62 - "NLSQAdapter"
Cohesion: 0.04
Nodes (28): core-B3 (B), _result(), test_explicit_cost_still_used(), test_missing_objective_yields_nonfinite_chi2_not_good(), _data_unsorted_phi(), test_flatten_duplicate_phi_does_not_crash(), test_flatten_non_rectangular_raises(), test_flatten_preserves_incoming_phi_block_order() (+20 more)

### Community 63 - "test_parameter_utils.py"
Cohesion: 0.05
Nodes (31): TestSafeSincContinuity, _quantile_flat(), _static_g1_sq(), _static_stratified(), test_classify_status_detects_bounds(), test_classify_status_no_bounds_all_active(), test_consistent_init_chunks_format_smoke(), test_consistent_init_falls_back_to_defaults_on_flat_data() (+23 more)

### Community 64 - "sequential.py"
Cohesion: 0.07
Nodes (25): optstrat-B15: Sequential fitting drops valid parameter scaling when a parameter is fixed, optstrat-B7: Sequential fits report finite variance for collinear parameters, test_sequential_singular_angle_is_excluded_not_weighted_as_sigma_one(), test_relocated_strip_and_restore_unchanged(), test_all_physical_fixed_hits_zero_free_vector_fast_path(), test_jax_jacobian_returns_real_jacobian_under_jacfwd(), residual(), restore_fixed_parameters() (+17 more)

### Community 65 - "test_heterodyne_stratified_execute_layers.py"
Cohesion: 0.07
Nodes (16): _fit(), test_execute_layers_averaged_l3_is_flag_only(), test_execute_layers_covariance_recompute_skipped_over_memory_budget(), test_execute_layers_exception_falls_back_to_baseline(), test_execute_layers_honors_hierarchical_budget(), test_execute_layers_keep_better_never_worse(), test_execute_layers_l2_reports_real_covariance(), test_execute_layers_l2_singular_jacobian_reports_nan_not_pinv() (+8 more)

### Community 66 - "optimization_runner.py"
Cohesion: 0.11
Nodes (11): _make_result(), test_genuinely_zero_uncertainty_still_warns(), test_nan_uncertainties_do_not_warn_bound_saturation(), _build_analysis_dict(), _build_convergence_dict(), _build_param_dict(), format_nlsq_summary(), _overrides_from_args() (+3 more)

### Community 67 - "heterodyne_jax_backend.py"
Cohesion: 0.08
Nodes (20): test_transport_integral_constant_rate_is_abs_gap_and_symmetric(), test_velocity_integral_constant_velocity_is_linear_signed_gap(), test_velocity_integral_is_antisymmetric_with_zero_diagonal(), compute_transport_integral_matrix(), compute_velocity_integral_matrix(), _compute_c2_meshgrid(), _compute_c2_pointwise(), compute_c2_unified() (+12 more)

### Community 68 - "classify_option"
Cohesion: 0.25
Nodes (14): _action(), test_choices_become_choices_completion(), test_count_action_is_flag_returns_none(), test_dir_hint_overrides_path(), test_literal_word_hint(), test_path_type_defaults_to_file(), test_plain_value_no_hint_no_choices_is_none_kind(), test_store_false_is_flag_returns_none() (+6 more)

### Community 69 - "run_worker"
Cohesion: 0.08
Nodes (16): Analysis Workbench (GUI) Guide, test_emitter_blocks_for_terminal_but_drops_telemetry_when_full(), test_log_handler_forwards_records_as_loglines(), _drain(), _install_fake_services(), _run_fit(), test_run_worker_emits_failed_on_exception(), test_run_worker_happy_path_emits_started_then_finished() (+8 more)

### Community 70 - "test_gui_jax_free.py"
Cohesion: 0.07
Nodes (22): _probe_import(), test_app_module_is_jax_free_at_import(), test_config_dialogs_is_jax_free(), test_data_inspect_is_jax_free(), test_diagnostics_module_is_jax_free(), test_error_presenter_is_jax_free(), test_export_is_jax_free(), test_fit_queue_controller_is_jax_free() (+14 more)

### Community 71 - "test_heterodyne_stratification_config.py"
Cohesion: 0.09
Nodes (12): test_safe_float(), test_safe_int(), _shipped_two_component_template_path(), test_shipped_template_stratification_defaults(), test_stratification_defaults(), test_stratification_null_bool_does_not_silently_disable_memory_safety(), test_stratification_null_scalars_fall_back_to_defaults(), test_use_index_based_flows_into_diagnostics() (+4 more)

### Community 72 - "test_heterodyne_cmaes_warmstart_success_gate.py"
Cohesion: 0.08
Nodes (17): _cfg(), test_apply_global_escape_no_skip_when_warmstart_failed(), _spy(), test_apply_global_escape_still_skips_on_converged_warmstart(), test_averaged_auto_skip_preserves_covariance(), test_averaged_callsite_threads_warm_success(), test_constant_callsite_threads_warm_success(), test_decision_does_not_skip_bad_basin_with_real_c2_data() (+9 more)

### Community 74 - "test_codex_ioviz.py"
Cohesion: 0.22
Nodes (8): homodyne_config(), homodyne_model(), homodyne_result(), multi_angle_data(), _restore_xpcsjax_logger(), test_ioviz_b3(), test_ioviz_b4(), test_ioviz_b5()

### Community 75 - "DatasetOptimizer"
Cohesion: 0.08
Nodes (8): test_create_chunked_iterator_rejects_misaligned_phi(), test_create_chunked_iterator_slices_aligned_phi_correctly(), test_equal_size_different_sigma_get_distinct_cached_strategies(), create_dataset_optimizer(), DatasetOptimizer, optimize_for_method(), DatasetInfo, ProcessingStrategy

### Community 76 - "test_persist.py"
Cohesion: 0.18
Nodes (13): gui-B10: Fitted-surface enrichment can report an already-saved fit as failed, _probe_import(), test_extract_parameters_handles_zero_dim_result(), test_merge_fitted_c2_accepts_same_or_newer_fitted_npz(), test_merge_fitted_c2_happy_path(), test_merge_fitted_c2_missing_fitted_npz_is_noop(), test_merge_fitted_c2_missing_primary_npz_is_noop(), test_merge_fitted_c2_rejects_stale_fitted_npz() (+5 more)

### Community 77 - "build_workbench"
Cohesion: 0.07
Nodes (15): test_build_workbench_returns_wired_window(), test_close_triggers_shutdown(), test_open_tolerates_deleted_result_dir(), test_save_then_open_round_trips_through_window(), _summary(), test_bottom_dock_is_log_only(), test_central_area_is_results_stack(), test_inspector_dock_exists() (+7 more)

### Community 78 - "_FakeConfigManager"
Cohesion: 0.11
Nodes (6): _FakeConfigManager, test_resolve_parameter_names_prefers_nlsq_diagnostics(), test_resolve_parameter_names_synthesizes_multi_angle_scaling_head(), test_resolve_parameter_names_synthesizes_scaling_head_for_homodyne(), test_save_results_json_labels_homodyne_scaling_params(), test_save_results_npz_readable_without_allow_pickle()

### Community 79 - "quality_controller.py"
Cohesion: 0.09
Nodes (14): _baseline_config(), _minimal_data(), test_controller_exposes_stage_enum_via_attribute(), test_controller_instantiates_from_config_dict(), test_create_quality_controller_factory_returns_controller(), test_quality_config_defaults(), test_validate_data_stage_disabled_short_circuits(), test_validate_data_stage_returns_result() (+6 more)

### Community 80 - "_NullLogger"
Cohesion: 0.09
Nodes (11): _CapturingOptimizer, _constant_mode_initial_and_bounds(), _FakeStratifiedData, _laminar_dataset(), _NullLogger, test_streaming_auto_tune_lambda_reaches_adaptive_regularization_config(), test_constant_mode_quantile_failure_resolves_l3_l4_as_averaged(), test_explicit_constant_mode_quantile_failure_uses_reduced_layout() (+3 more)

### Community 81 - "test_heterodyne_memory_adapter.py"
Cohesion: 0.09
Nodes (25): _patch_threshold(), test_build_tier_list_drops_large_when_disabled(), test_build_tier_list_from_each_start(), test_detect_total_system_memory_gb(), test_estimate_peak_memory_gb(), test_get_memory_threshold_clamps(), test_get_memory_threshold_concurrency_divides(), test_get_memory_threshold_default() (+17 more)

### Community 82 - "_ooc_probe"
Cohesion: 0.21
Nodes (5): _enable_generic_streaming(), _ooc_probe(), _fake_kernels(), test_optstrat_b14(), test_optstrat_b5()

### Community 83 - "StratifiedResidualFunctionJIT"
Cohesion: 0.31
Nodes (14): _full_grid_chunk(), _stratified(), test_dt_none_uses_fallback_and_warns(), test_empty_chunks_raises(), test_fixed_scaling_constant_mode(), test_get_diagnostics(), test_inconsistent_q_raises(), test_log_diagnostics_runs() (+6 more)

### Community 84 - "test_review_regressions.py"
Cohesion: 0.08
Nodes (22): _check_lazy_viz_import(), _make_heterodyne_config(), _make_heterodyne_data(), _make_heterodyne_result(), test_averaged_physics_first_uncertainties_use_head_slice(), test_datashader_module_load_does_not_block_when_missing(), test_generate_plots_missing_analysis_mode_falls_back_to_laminar_flow(), test_generate_plots_missing_analyzer_parameters_raises_valueerror() (+14 more)

### Community 85 - "SingleStartResult"
Cohesion: 0.08
Nodes (24): _CfgMgr, _install_model_stub(), _StubModel, _StubParamManager, _StubResult, test_build_multistart_config_defaults_on_empty(), test_build_multistart_config_reads_nested_keys(), test_dispatch_cmaes_takes_precedence_over_multistart() (+16 more)

### Community 86 - "QualityControlConfig"
Cohesion: 0.15
Nodes (9): test_quality_config_from_dict_reads_quality_control_section(), test_quality_config_from_empty_dict_uses_defaults(), test_quality_levels_are_valid_validation_levels(), test_repair_nan_values_defaults_to_false(), test_repair_nan_values_disabled_by_default_leaves_data_untouched(), test_repair_nan_values_logs_warning_with_count(), test_repair_scaling_issues_field_removed(), test_repair_scaling_issues_method_removed() (+1 more)

### Community 87 - ".phi"
Cohesion: 0.15
Nodes (10): _fit(), _revert_records(), test_optwrap_b8(), truth(), compute_single_angle(), compute_single_angle_scaled(), _single_fit(), model_function() (+2 more)

### Community 88 - "heterodyne_model_stateful.py"
Cohesion: 0.07
Nodes (3): test_core_b9_reported_fraction_is_the_one_the_kernel_uses(), HeterodyneModelBase, TwoComponentModel

### Community 89 - "AntiDegeneracyController"
Cohesion: 0.06
Nodes (25): BIPOP restart strategy, CMAESResult, CMAESWrapper, CMAESWrapperConfig, fit_nlsq, fit_nlsq_cmaes, CMAESWrapper.should_use_cmaes, laminar_flow analysis mode (+17 more)

### Community 90 - "chunking.py"
Cohesion: 0.05
Nodes (38): _balanced_dataset(), _stratify(), test_adaptive_chunk_size_clamped_to_max(), test_adaptive_chunk_size_clamped_to_min(), test_adaptive_chunk_size_docstring_scenario_pinned(), test_adaptive_chunk_size_memory_constrained(), test_adaptive_chunk_size_zero_angles(), test_analyze_angle_distribution_balanced() (+30 more)

### Community 91 - "ResultBuilder"
Cohesion: 0.06
Nodes (10): _builder(), test_iterations_defaults_to_zero_without_nfev_fallback(), test_iterations_prefers_explicit_iterations_key_over_default(), test_iterations_uses_nit_not_nfev(), compute_quality_metrics(), compute_uncertainties(), determine_convergence_status(), normalize_nlsq_result() (+2 more)

### Community 92 - "test_heterodyne_data_prep.py"
Cohesion: 0.05
Nodes (42): opthet-B3: Stratified fitting turns zero-weight observations back on, Anything the plan missed, <codex|agy> position — round 2, core-B3 open point (check inside model_func): DROP | INSIST — <one or two sentences>, core-B3: RATIFY | OBJECT — <reason if OBJECT>, data-B9 (B), data-B9: RATIFY | OBJECT — <reason if OBJECT>, R4-A3 (A) (+34 more)

### Community 93 - "test_homodyne_engine_preservation.py"
Cohesion: 0.10
Nodes (9): load_or_init_golden(), _assert_covariance_matches_flag(), _build_laminar_fit(), _recorded_versions(), test_explicit_individual_pins_layout_and_stays_bit_identical(), test_golden_files_are_committed(), test_laminar_flow_end_to_end_golden(), test_laminar_flow_unknown_covariance_is_flagged() (+1 more)

### Community 94 - "test_cache_source_file_validation.py"
Cohesion: 0.08
Nodes (28): Behaviour changes users can see, Claude + Codex review/debug loop — 2026-11-24, Fixes that change reported uncertainties or DOF, How it was run, Maintainer decisions, Not done, Recommendation, Summary (+20 more)

### Community 95 - "test_heterodyne_tied_parameters.py"
Cohesion: 0.09
Nodes (33): test_grouped_format_accepts_public_angle_alias(), test_grouped_format_accepts_public_velocity_alias(), test_grouped_format_coerces_string_scalars_to_float(), test_grouped_format_rejects_alias_and_canonical_collision(), test_grouped_format_rejects_inf_bounds(), test_grouped_format_rejects_inverted_bounds(), test_grouped_format_rejects_nan_value(), test_initial_parameters_flat_format_rejects_nan() (+25 more)

### Community 96 - "test_post_install.py"
Cohesion: 0.14
Nodes (24): fake_venv(), isolated_env(), test_configure_xla_mode_overwrites_with_force(), test_configure_xla_mode_preserves_existing_without_force(), test_configure_xla_mode_writes(), test_get_xla_mode_path_home_fallback(), test_get_xla_mode_path_venv(), test_get_xla_mode_path_xdg() (+16 more)

### Community 97 - "read_c2_preview"
Cohesion: 0.15
Nodes (14): _make_h5(), test_c2_preview_missing_dataset_returns_none(), test_c2_preview_missing_group_returns_none(), test_c2_preview_nested_group_at_key_returns_none(), test_c2_preview_reconstructs_group_half_matrix(), test_c2_preview_slices_and_downsamples(), test_c2_preview_unknown_type_group_path_returns_none(), test_metadata_lists_datasets_without_loading() (+6 more)

### Community 98 - "models.py"
Cohesion: 0.05
Nodes (14): core-B6: The public homodyne wrapper builds a laminar model from the shipped static templates, TestMakeModelDispatch, test_combined_model_param_count(), test_combined_model_static_mode_param_count(), test_diffusion_model_g1_runs(), test_diffusion_model_param_count(), test_compute_chi_squared_accepts_and_forwards_dt(), CombinedModel (+6 more)

### Community 99 - "format_comparison"
Cohesion: 0.19
Nodes (8): _summary(), test_empty_summaries_render_empty_string(), test_marks_differing_values_with_diff_marker(), test_nan_parameter_renders_distinct_from_absent(), test_shows_two_runs(), test_tolerates_missing_summary(), test_tolerates_none_chi_squared(), format_comparison()

### Community 100 - "ResultSummary"
Cohesion: 0.06
Nodes (22): _summary(), test_finishing_active_run_does_not_clobber_selected_run(), test_finishing_run_still_shown_when_nothing_else_selected(), test_fit_log_cleared_between_runs_and_on_close(), test_inspector_tolerates_non_dict_diagnostics(), test_loader_coerces_null_diagnostics_to_dict(), _window(), _summary() (+14 more)

### Community 101 - "DataInspectDialog"
Cohesion: 0.09
Nodes (9): gui-B13: A valid HDF5 null dataset aborts the data inspector, _make_h5(), test_corrupt_file_error_shown_not_crash(), test_empty_leading_axis_shows_error_not_crash(), test_lists_datasets(), test_missing_file_shows_error_not_crash(), test_preview_renders_selected_dataset(), test_preview_uses_stored_name_not_display_text() (+1 more)

### Community 102 - "NDArray"
Cohesion: 0.15
Nodes (8): test_check_zero_volume_bounds(), test_generate_lhs_starts_within_bounds_and_reproducible(), test_generate_random_starts_within_bounds_and_reproducible(), check_zero_volume_bounds(), generate_lhs_starts(), generate_random_starts(), _OptimizeWorker, _run_sequential()

### Community 103 - "test_heterodyne_tied_residuals.py"
Cohesion: 0.14
Nodes (10): test_averaged_mode_residual_enforces_tie(), test_constant_mode_residual_pattern_enforces_tie(), test_expand_varying_to_full_used_by_all_residual_sites_is_tied(), test_hybrid_streaming_model_fn_pattern_enforces_tie(), test_individual_mode_residual_pattern_enforces_tie(), test_make_numpy_residual_fn_pattern_enforces_tie(), test_per_angle_cmaes_residual_pattern_enforces_tie(), test_per_angle_local_residual_pattern_enforces_tie() (+2 more)

### Community 104 - "heterodyne_core.py"
Cohesion: 0.03
Nodes (74): test_finite_candidate_beats_nonfinite_warm_start(), test_finite_keep_better_semantics_unchanged(), test_nonfinite_candidate_never_kept(), _CaptureAndStopError, _install_capturing_spy(), spy(), test_all_cmaes_config_sites_pin_seed(), test_all_cmaes_config_sites_pin_sigma() (+66 more)

### Community 105 - "PerAngleScalingPlan"
Cohesion: 0.13
Nodes (19): _quantiles(), test_expand_back_averaged_broadcasts_then_splits_physics(), test_expand_back_constant_uses_frozen_quantiles_physics_only_vector(), test_expand_back_individual_reorders_to_full_per_angle(), test_expand_covariance_averaged_replicates_scalar_blocks(), test_expand_covariance_constant_has_zero_scaling_block(), test_expand_covariance_individual_is_identity(), test_expand_covariance_none_passes_through() (+11 more)

### Community 106 - "NLSQStrategy"
Cohesion: 0.14
Nodes (6): _grep_callers(), _streaming_size_for_threshold(), test_homodyne_and_heterodyne_routers_both_escalate_to_streaming(), test_hybrid_streaming_triggers_when_index_exceeds_threshold(), test_select_strategy_is_reached_from_production_code(), NLSQStrategy

### Community 107 - "resolve_per_angle_mode"
Cohesion: 0.17
Nodes (9): test_resolver_passes_through_explicit_averaged(), test_stratified_ls_modes_resolve_via_canonical_resolver(), test_auto_honors_custom_threshold(), test_auto_resolves_by_default_threshold(), test_explicit_modes_pass_through_identity(), test_n_optimized_truth_table(), test_unknown_and_removed_tokens_raise_valueerror(), test_explicit_averaged_resolves_like_auto_for_dof() (+1 more)

### Community 108 - "json_safe"
Cohesion: 0.09
Nodes (6): TestJsonSafeContainers, TestJsonSafeEdgeCases, TestJsonSafeFloatSanitization, TestJsonSafeNumpyTypes, TestJsonSafeRoundTrip, json_safe()

### Community 109 - "theme.py"
Cohesion: 0.14
Nodes (8): main(), app_icon(), apply_theme(), current_palette(), detect_scheme(), Palette, _qpalette(), stylesheet()

### Community 110 - "HomodyneModel"
Cohesion: 0.05
Nodes (34): TestHomodyneModelInitValidation, _fit(), test_averaged_dof_basis_is_expanded(), test_constant_dof_basis_is_physics_only(), test_individual_dof_basis_is_optimizer(), test_averaged_result_is_dense_per_angle(), test_constant_result_is_dense_per_angle(), _run() (+26 more)

### Community 111 - "_write_npz_compressed"
Cohesion: 0.12
Nodes (19): API Reference Index, xpcsjax.io API, _sample_arrays(), _sample_artifact_inputs(), test_save_artifacts_creates_output_dir(), test_save_artifacts_json_schema(), test_save_artifacts_lzma_fallback_on_memoryerror(), test_save_artifacts_npz_schema() (+11 more)

### Community 112 - "post_install.py"
Cohesion: 0.17
Nodes (10): test_completion_and_xla_source_paths_exist(), test_fish_completion_is_nonfatal_noop_but_xla_stays(), test_install_bash_completion(), test_install_shell_completion_skips_outside_venv(), test_install_zsh_completion(), get_completion_source_path(), get_xla_config_source_path(), install_bash_completion() (+2 more)

### Community 113 - "test_quality_gate_fixes.py"
Cohesion: 0.09
Nodes (13): _result(), test_converged_result_with_empty_parameters_is_rejected(), test_converged_result_with_nonfinite_parameters_is_rejected(), test_covariance_shape_mismatch_is_rejected(), test_failed_result_with_empty_parameters_is_allowed(), test_parse_allows_bare_static_when_opted_in(), test_parse_canonicalizes_synonyms(), test_parse_rejects_bare_static_by_default() (+5 more)

### Community 115 - "RunController"
Cohesion: 0.08
Nodes (6): test_export_copies_figures_to_dest(), test_export_disambiguates_same_name_files(), test_export_returns_empty_when_plots_dir_absent(), test_on_export_figure_routes_through_run_controller(), export_figures(), RunController

### Community 116 - "test_static_individual_invariant.py"
Cohesion: 0.07
Nodes (20): _static_cfg(), test_resolve_static_pinned_forces_individual(), test_resolve_static_pinned_laminar_honors_resolver(), test_routing_effective_n_params_laminar_reduction_unchanged(), _r(), test_routing_effective_n_params_static_is_dense_for_any_mode(), test_standard_inmemory_static_is_individual_despite_auto(), test_static_dof_uses_individual_param_count() (+12 more)

### Community 117 - "service/persist.py"
Cohesion: 0.12
Nodes (12): gui-B9: Concurrent JSON result saves share the same staging file, test_json_safe_handles_zero_dim_ndarray(), _canonical_physics_names(), _config_summary(), _extract_metadata(), _resolve_parameter_names(), save_results(), save_results_json() (+4 more)

### Community 118 - "test_save_path_validation.py"
Cohesion: 0.06
Nodes (21): test_r2_ioviz_b10_singleton_axis_is_a_clear_error(), _fresh_fig(), test_datashader_comparison_creates_missing_parent_dir(), test_datashader_heatmap_creates_missing_parent_dir(), test_datashader_heatmap_rejects_traversal(), test_save_fig_accepts_valid_path_and_creates_parent(), test_save_fig_rejects_non_image_extension(), test_save_fig_rejects_null_byte() (+13 more)

### Community 119 - "test_fit_quality.py"
Cohesion: 0.11
Nodes (18): gui-B4: Homodyne NRMSE can measure a different model from the one fitted, ioviz-B2: Homodyne reconstruction uses the requested wavevector instead of the fitted wavevector, _cm(), _result(), test_angle_count_mismatch_raises_not_truncates(), test_attach_never_raises_on_broken_config_manager(), test_failed_angle_is_skipped_not_fatal(), _eval() (+10 more)

### Community 120 - "test_result_presenter.py"
Cohesion: 0.10
Nodes (17): _summary(), test_result_presenter_is_qobject_parented_to_window(), test_show_error_does_not_switch_to_grid(), test_show_error_invalidates_in_flight_bundle_load(), test_show_error_renders_fit_failed(), test_show_inspector_none_clears(), test_show_inspector_populates_inspector(), test_show_result_contains_summary_fields() (+9 more)

### Community 121 - "load_dataset"
Cohesion: 0.12
Nodes (14): _cm(), test_load_dataset_empty_phi_does_not_crash(), test_load_dataset_no_subset_keeps_all(), test_load_dataset_raises_when_no_correlation_matrix(), test_load_dataset_subsets_by_phi(), test_resolve_phi_angles_cli_wins_and_normalizes(), test_resolve_phi_angles_none_when_no_source(), test_resolve_phi_angles_parses_comma_string() (+6 more)

### Community 122 - "HeterodyneParameterManager"
Cohesion: 0.05
Nodes (4): config-B10: A rejected heterodyne update mutates values and leaves the cached array stale, test_config_b10(), HeterodyneParameterManager, HeterodyneValidationResult

### Community 123 - "ConfigTextEditorDialog"
Cohesion: 0.08
Nodes (9): gui-B8: A failed config save can destroy the previous YAML, test_config_text_editor_accepts_valid_yaml_on_save(), test_config_text_editor_disables_save_on_load_failure(), test_config_text_editor_rejects_invalid_yaml_on_save(), test_config_text_editor_round_trip(), test_config_text_editor_surfaces_write_failure(), test_gui_b8(), test_r2_c3_config_save_preserves_symlink() (+1 more)

### Community 124 - "test_codex_cli.py"
Cohesion: 0.05
Nodes (33): Bugs, cli-B10: The public `xpcsjax.cli.main` export changes type with import order, cli-B1: Generated configs can direct an analysis to the wrong dataset, cli-B2: A JSON-only rerun can write current C₂ arrays into a previous fit’s NPZ, cli-B6: Config validation reports empty and scalar YAML documents as valid, cli-B8: Config generation emits broken YAML for a data path containing a newline, cli-B9: A previous shell activation overrides the newly activated environment’s XLA mode, Codex review — shard cli (+25 more)

### Community 125 - "test_memory_manager_cache_gating.py"
Cohesion: 0.11
Nodes (12): _make_manager(), test_critical_demoted_to_warning_in_live_array_regime(), test_emergency_gc_does_not_loop_when_unproductive(), test_first_cache_clear_allowed_on_low_uptime(), test_gc_thresholds_divide_from_baseline_not_compounding(), test_jax_cache_clear_respects_cooldown(), test_live_array_pressure_skips_jax_cache_clear(), test_maybe_clear_jax_caches_direct_call_is_safe() (+4 more)

### Community 126 - "test_max_imbalance_ratio_can_loosen_above_5"
Cohesion: 0.16
Nodes (8): test_max_imbalance_ratio_can_loosen_above_5(), _fake_dist(), test_max_imbalance_ratio_disables_stratification(), _fake_dist(), test_stratification_enabled_false_disables(), _sentinel(), test_stratification_target_chunk_size_flows(), AngleDistributionStats

### Community 127 - "test_cache_safety.py"
Cohesion: 0.15
Nodes (9): _bare_loader(), _good_payload(), test_corrupt_metadata_json_is_rejected_clearly(), test_legacy_object_metadata_is_refused(), test_non_object_metadata_json_is_rejected(), test_object_dtype_under_data_key_is_refused(), test_oversized_cached_shape_rejected_before_materialization(), test_peek_npz_array_header_matches_full_read() (+1 more)

### Community 128 - "_make_synthetic_heterodyne"
Cohesion: 0.07
Nodes (27): _Cfg, _install_model_stub(), _make_synthetic_heterodyne(), test_build_hybrid_streaming_result(), test_build_hybrid_streaming_result_dof_fallback_on_missing_varying_names(), test_build_hybrid_streaming_result_genuine_failure_stays_failed_poor(), test_build_hybrid_streaming_result_honours_nested_covariance_placeholder(), test_build_hybrid_streaming_result_max_nfev_is_max_iter_not_failed() (+19 more)

### Community 129 - "get_or_create_fitter"
Cohesion: 0.15
Nodes (10): _clear_cache(), test_cache_distinct_keys(), test_cache_key_equality(), test_cache_miss_then_hit(), test_clear_model_cache_resets(), CachedModel, clear_model_cache(), get_cache_stats() (+2 more)

### Community 130 - "jit"
Cohesion: 0.12
Nodes (14): _grid_reference_diffusion(), _grid_reference_shear(), test_diffusion_degenerate_matches_grid(), test_g1_total_propagates_nan_instead_of_flooring_it(), test_shear_degenerate_matches_grid(), test_static_shear_accepts_0d_scalar_t1(), _compute_g1_diffusion_meshgrid(), _compute_g1_shear_meshgrid() (+6 more)

### Community 131 - "test_strategy_executors.py"
Cohesion: 0.19
Nodes (18): _enable_streaming(), _logger(), _resid(), test_large_executor_optimize_result_with_pcov(), test_large_executor_optimize_result_without_pcov(), test_large_executor_reraises(), test_large_executor_three_tuple_with_info(), test_large_executor_two_tuple() (+10 more)

### Community 132 - "test_heterodyne_constant_mode.py"
Cohesion: 0.11
Nodes (12): _b2_config_dict(), _build_minimal_heterodyne_model(), _build_synthetic_c2_stack(), _expected_synthetic_scaling(), _make_synthetic_c2(), test_constant_mode_fit_optimizes_only_physics(), test_constant_mode_recovers_synthetic_truth(), test_individual_aggregate_converts_final_cost_to_ssr() (+4 more)

### Community 133 - "test_aps_u_selective_read_d7.py"
Cohesion: 0.18
Nodes (9): _bare_loader(), _load_aps_u_eager_reference(), test_aps_old_quality_filtering_empty_phi_match_records_one_degradation(), test_quality_filtering_empty_phi_match_records_one_degradation(), test_selective_read_matches_eager_read_default_config(), test_selective_read_matches_eager_read_with_phi_filtering(), test_selective_read_matches_eager_read_with_quality_filtering(), _write_multi_bin_aps_old_file() (+1 more)

### Community 135 - "test_jacobian.py"
Cohesion: 0.11
Nodes (13): fixture_data(), _polynomial_residual(), test_analyze_parameter_sensitivity_normalized_to_unit(), test_compute_jacobian_condition_number_is_finite(), test_compute_jacobian_stats_handles_broken_residual(), test_compute_jacobian_stats_returns_jtj_and_norms(), test_estimate_gradient_noise_finite_for_smooth_residual(), test_analyze_parameter_sensitivity() (+5 more)

### Community 137 - "ShearSensitivityWeighting"
Cohesion: 0.06
Nodes (24): test_apply_weights_to_loss_disabled(), test_apply_weights_to_loss_enabled(), test_compute_weighted_mse_disabled(), test_compute_weighted_mse_enabled(), test_config_defaults(), test_create_defaults_phi0_index_when_no_names(), test_create_disabled_returns_none(), test_create_none_config_returns_none() (+16 more)

### Community 138 - "test_homodyne_covariance_contract.py"
Cohesion: 0.08
Nodes (14): _fit_with_pcov(), test_laminar_stratified_ls_never_ships_identity_sigma(), test_recovery_helper_has_no_floor(), test_wrapper_create_fit_result_flags_placeholder(), test_wrapper_helper_honours_placeholder_and_structural_zero(), _FakeConfig, _FakeStratifiedData, _force_standard() (+6 more)

### Community 139 - "validate_xpcs_data"
Cohesion: 0.16
Nodes (15): _clean_data(), _correlation_warnings(), test_clean_data_is_valid(), test_levels_run_without_crashing_on_clean_data(), test_missing_required_key_is_error(), test_non_positive_q_is_error(), test_nonfinite_in_correlation_is_error(), test_overnormalized_near_zero_lag_is_still_flagged() (+7 more)

### Community 140 - "test_runtime_shell.py"
Cohesion: 0.13
Nodes (12): _bash_executable(), _source_xla_config(), test_get_completion_script_returns_absolute_existing_path(), test_get_xla_config_script_defaults_to_bash(), test_get_xla_config_script_per_shell(), test_get_xla_config_script_rejects_unknown_shell(), test_xla_config_bash_accepts_integer_mode(), test_xla_config_bash_no_arg_recovers_from_poisoned_mode_file() (+4 more)

### Community 141 - "test_system_validator.py"
Cohesion: 0.13
Nodes (14): test_config_templates_probe_finds_all(), test_cpu_info_probe_reports(), test_dependency_versions_probe_buckets_unparseable_version(), test_dependency_versions_probe_passes(), test_jax_installation_probe_x64_enabled(), test_main_returns_zero_when_no_errors(), test_python_version_probe_passes(), test_severity_values() (+6 more)

### Community 142 - "get_adaptive_memory_threshold"
Cohesion: 0.14
Nodes (17): test_detect_fit_concurrency_default_is_one(), test_detect_fit_concurrency_explicit_arg_wins(), test_detect_fit_concurrency_fit_env_beats_xdist(), test_detect_fit_concurrency_floors_at_one(), test_detect_fit_concurrency_ignores_garbage(), test_detect_fit_concurrency_reads_xdist(), test_heterodyne_select_strategy_escalates_under_concurrency(), test_heterodyne_threshold_shrinks_with_concurrency() (+9 more)

### Community 143 - "assemble_anti_degeneracy_diagnostics"
Cohesion: 0.15
Nodes (11): _fake_sequential(), _build_sequential_laminar_fit(), test_block_honest_active_from_streaming_info(), test_block_inactive_for_stratified_controller_only_info(), test_block_markers_when_no_info(), test_sequential_laminar_emits_symmetric_activation_keys(), _fake_sequential(), test_sequential_laminar_rescales_covariance_for_active_shear_transform() (+3 more)

### Community 144 - "compute_g2_scaled"
Cohesion: 0.14
Nodes (7): compute_g2_batch(), compute_g2_batch_with_per_angle_scaling(), compute_theoretical_fits(), extract_parameters_from_result(), get_physical_param_count(), normalize_analysis_mode(), solve_lstsq_batch()

### Community 145 - "test_load_degradation_signal.py"
Cohesion: 0.17
Nodes (6): _bare_loader(), test_final_qc_gate_failure_sets_degraded(), validate_data_stage(), test_record_degradation_accumulates(), test_record_degradation_appends_and_logs_error(), QualityMetrics

### Community 146 - "test_laminar_mode_banners.py"
Cohesion: 0.11
Nodes (18): _laminar_controller(), test_averaged_banner_text(), test_broadcast_mode_label_resolves_word(), test_compute_fixed_per_angle_scaling_emits_neutral_banner(), test_constant_banner_has_no_zero_scaling(), test_fit_nlsq_cmaes_summary_routes_through_helper(), test_from_controller_constant_has_no_reason_line(), test_from_controller_disabled_is_noop() (+10 more)

### Community 147 - "multistart.py"
Cohesion: 0.09
Nodes (6): _is_pickle_error(), _pool_worker_init_concurrency(), _run_full_strategy(), _run_parallel_with_progress(), _run_start_safely(), MultiStartProgressTracker

### Community 148 - "OptimizationExecutor"
Cohesion: 0.14
Nodes (6): test_get_executor_dispatch(), test_get_executor_streaming_passes_checkpoint_config(), test_get_executor_unknown_raises(), test_r2_optstrat_b16_standard_executor_accepts_unbounded(), get_executor(), OptimizationExecutor

### Community 149 - "save_nlsq_json_files"
Cohesion: 0.20
Nodes (4): TestSaveNlsqJsonFiles, boom(), _atomic_json_dump(), save_nlsq_json_files()

### Community 150 - "Bugs"
Cohesion: 0.14
Nodes (12): Bugs, Codex review — shard config, config-B11: Heterodyne correlation validation accepts an infinite time coordinate, config-B2: Equal heterodyne bounds leave a fixed coordinate in the optimizer’s free vector, config-B3: Misspelled heterodyne constraints change the fitted model instead of stopping the fit, config-B4: Quoted heterodyne `vary: "false"` frees the parameter, config-B6: Blank optional heterodyne parameter groups crash model construction, config-B9: Midpoint initialization drops explicitly configured per-angle scaling (+4 more)

### Community 151 - "test_codex_gui.py"
Cohesion: 0.08
Nodes (9): _opt_result(), _partial_frame_then_die(), test_gui_b10(), test_gui_b3(), test_gui_b4(), test_gui_b6(), test_gui_b9(), _replace() (+1 more)

### Community 152 - "test_memory_manager_logging.py"
Cohesion: 0.09
Nodes (9): _make_manager(), _reset_log_once_cache(), test_atexit_cleanup_silences_closed_stream_logging_errors(), test_cleanup_failure_is_logged_at_debug_and_does_not_escape(), test_dead_weak_callback_does_not_crash_pressure_trigger(), test_logged_errors_fallback_reraise_policy_propagates(), test_logged_errors_fallback_shim_reraise_direct(), test_logged_errors_fallback_suppress_policy_swallows() (+1 more)

### Community 153 - "test_heterodyne_cmaes_warmstart_auto_skip.py"
Cohesion: 0.17
Nodes (7): _cfg(), test_apply_global_escape_auto_skips_cmaes_on_good_warmstart(), _spy(), test_apply_global_escape_never_auto_skips_multistart(), test_apply_global_escape_no_skip_when_dof_nonpositive(), test_apply_global_escape_respects_disabled_autoskip(), test_apply_global_escape_runs_cmaes_on_poor_warmstart()

### Community 154 - ".__init__"
Cohesion: 0.24
Nodes (3): _apply_colormap(), _resolve_colormap(), _fit_square_view()

### Community 155 - "test_adapter_xdata_cache.py"
Cohesion: 0.19
Nodes (7): _coord_sensitive_model_func(), _expected(), test_distinct_xdata_objects_do_not_cross_contaminate(), test_model_cache_evicts_least_recently_used_entry(), test_repeated_same_object_is_cached_and_correct(), _xdata(), clear_model_cache()

### Community 156 - "fit_with_stratified_hybrid_streaming_heterodyne"
Cohesion: 0.05
Nodes (27): test_streaming_plain_branch_keeps_real_pcov(), test_fit_with_stratified_hybrid_streaming_heterodyne_dispatch(), test_initial_params_override_is_honored(), test_l2_individual_runs_and_beats_frozen_baseline(), test_streaming_above_threshold_ssr_not_worse_after_unification(), test_streaming_auto_threshold_routing(), test_streaming_averaged_real_run_ssr_not_worse(), test_streaming_constant_matches_legacy_frozen_scaling() (+19 more)

### Community 157 - "optimization/test_validation.py"
Cohesion: 0.09
Nodes (25): test_validate_bounds_consistency_upper_length_mismatch(), test_validate_covariance_rejects_non_finite(), test_validate_optimized_params_below_lower(), test_validate_result_consistency_low_and_high_still_pass(), _make_result(), test_validate_bounds_consistency_accepts_sorted_bounds(), test_validate_bounds_consistency_rejects_inverted_or_misshaped(), test_validate_covariance_accepts_symmetric_finite_positive_diag() (+17 more)

### Community 158 - "HierarchicalOptimizer"
Cohesion: 0.06
Nodes (24): optescape-B1: Hierarchical fitting can declare convergence immediately after scaling invalidates the physical solution, _opt(), test_config_defaults(), test_create_per_angle_loss_and_grad(), test_create_physical_grad_slices_physical_indices(), grad(), test_create_physical_loss_assembles_full_vector(), loss() (+16 more)

### Community 159 - "References and Citations Doc"
Cohesion: 0.19
Nodes (22): Andrade1910 - Viscous flow in metals (creep law), Bradbury2018 - JAX composable transformations, Duri2005 - Time-resolved-correlation measurements, Hansen2016 - CMA evolution strategy tutorial, He2024 - Transport coefficient approach (PNAS), He2025 - Bridging microscopic dynamics and rheology (PNAS), Kubo1966 - Fluctuation-dissipation theorem, Lumma2000 - Area detector based photon correlation (+14 more)

### Community 160 - "MemoryStats"
Cohesion: 0.11
Nodes (3): _as_weak_callable(), log_calls(), MemoryStats

### Community 161 - "detect_degeneracy"
Cohesion: 0.14
Nodes (10): _single(), test_detect_degeneracy_multiple_basins(), test_detect_degeneracy_single_basin(), test_detect_degeneracy_too_few_successful(), test_run_sequential_collects_and_handles_failure(), opt(), test_to_optimization_result_no_covariance_failed(), test_to_optimization_result_quality_flags() (+2 more)

### Community 162 - "fit_with_stratified_hybrid_streaming"
Cohesion: 0.10
Nodes (8): _CapturingAdaptiveOptimizerSpy, _FakeStratifiedDataWithSigma, _HierarchicalOptimizerSpy, test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse(), test_heterodyne_hier_loss_honors_nonuniform_sigma(), test_laminar_loss_fn_combines_sigma_with_shear_weighting(), test_laminar_loss_fn_honors_nonuniform_sigma(), test_laminar_plain_path_threads_sigma_into_optimizer_fit()

### Community 164 - "main_window.py"
Cohesion: 0.05
Nodes (9): test_oom_message_is_friendly(), test_plain_message_passthrough(), test_traceback_is_summarized_with_details_retained(), present_failure(), repolish(), show_failure(), _BundleLoadSignals, _BundleLoadTask (+1 more)

### Community 165 - "xpcsjax/data/__init__.py"
Cohesion: 0.05
Nodes (29): data-B17: The schema validator raises on a field type it has already identified as invalid, test_advanced_memory_manager_collected_without_gc_sweep(), test_calculate_matrix_quality_score_never_returns_nan(), test_validate_config_schema_malformed_section_is_flagged_not_absorbed(), test_validate_config_schema_none_section_reports_missing_params(), test_validate_numeric_range_rejects_nan_when_wrapped_and_unbounded(), test_validate_numeric_range_rejects_nan_with_require_positive(), test_validate_positive_value_rejects_nan() (+21 more)

### Community 166 - "FitQueueController"
Cohesion: 0.08
Nodes (15): test_add_dataset_does_not_retarget_while_run_active(), test_dataset_row_selection_updates_active_dataset(), test_load_config_preserves_prior_sidebar_selection(), test_on_open_project_surfaces_malformed_file_as_warning_not_crash(), test_on_run_blocks_duplicate_run_for_same_dataset(), test_open_project_raises_on_malformed_file_leaves_state_untouched(), test_open_project_resets_stale_run_state(), test_run_failed_does_not_clobber_a_different_pinned_view() (+7 more)

### Community 167 - "xla_config.py"
Cohesion: 0.11
Nodes (8): test_build_parser_returns_parser(), test_post_install_shell_choices_preserved(), test_xla_config_threads_roundtrip(), build_parser(), configure_xla(), get_cpu_info(), main(), build_parser()

### Community 168 - "NLSQResult"
Cohesion: 0.02
Nodes (56): R3-A1: R2-A1 — Duplicate-angle diagnostics still mix raw rows and shared slots, _make_failing_adapter(), fit(), test_fallback_does_not_report_discarded_adapter_monitor(), _make_config(), _simple_bounds(), _simple_residual(), TestNLSQAdapterErrorRecovery (+48 more)

### Community 169 - "test_heterodyne_physics_validators.py"
Cohesion: 0.10
Nodes (22): test_correlation_inputs_clean_is_valid(), test_correlation_inputs_nan_is_error(), test_correlation_inputs_non_monotonic_time_is_error(), test_correlation_inputs_shape_mismatch_is_error(), test_cross_parameter_fraction_sum_exceeds_unity(), test_cross_parameter_offset_ref_below_threshold_no_violation(), test_cross_parameter_offset_ref_dominance_negative(), test_cross_parameter_offset_ref_dominance_positive() (+14 more)

### Community 170 - "TestMeshgridCacheCollision"
Cohesion: 0.21
Nodes (3): core-B4: Distinct time axes can share a cache key and return stale meshgrids, TestMeshgridCacheCollision, clear_meshgrid_cache()

### Community 171 - "Bugs"
Cohesion: 0.17
Nodes (11): Bugs, Codex review — shard gui, Files read, gui-B12: Reopened runs with a missing or unreadable summary still appear as normal completed runs, gui-B14: Malformed parameter names make config validation raise, gui-B1: Persistence can label fitted homodyne parameters with the wrong names, gui-B2: Explicit angle selection leaves q values attached to discarded angles, gui-B6: A partial worker event can leave the IPC reader blocked after the worker dies (+3 more)

### Community 172 - "simulated.py"
Cohesion: 0.15
Nodes (7): test_resolve_phi_angles_for_sim_all_empty_falls_back_to_data(), test_resolve_phi_angles_for_sim_trailing_comma(), test_simulated_grid_uses_elapsed_time(), _write_config(), _evaluate_model_c2(), _plot_simulated_from_config(), resolve_phi_angles_for_sim()

### Community 173 - "assemble_anti_degeneracy_diagnostics"
Cohesion: 0.26
Nodes (6): test_bool_coercion_and_determinism(), test_core_activation_keys_always_present(), test_empty_gradient_monitor_is_still_included(), test_gradient_monitor_included_when_provided(), test_layer_detail_merged_verbatim(), assemble_anti_degeneracy_diagnostics()

### Community 174 - "json_serializer"
Cohesion: 0.15
Nodes (3): TestJsonSerializer, json_serializer(), _sanitize_float()

### Community 175 - "fit_nlsq entry point"
Cohesion: 0.25
Nodes (25): L3 Adaptive CV-based Regularization, analysis_mode (static_isotropic/anisotropic/laminar_flow/two_component), Anti-degeneracy controller (5/4-layer defense), CMA-ES global escape, ConfigManager, fit_nlsq entry point, fixed_parameters / active_parameters, L2 Hierarchical Optimization (+17 more)

### Community 176 - "gauss_newton_covariance"
Cohesion: 0.14
Nodes (9): test_rescale_covariance_dof_factor(), test_finalize_rejects_inf_zero_negative_and_missing(), test_gauss_newton_covariance_is_strict(), test_gauss_newton_covariance_matches_nlsq_curve_fit_estimator(), test_rescale_covariance_dof(), test_robust_scaling_matches_nlsq(), gauss_newton_covariance(), rescale_covariance_dof() (+1 more)

### Community 177 - "PhiResultsGrid"
Cohesion: 0.06
Nodes (18): _bundle(), test_phi_grid_builds_interactive_diagnostics_with_fit(), test_phi_grid_degenerate_bundle_is_finite_safe(), test_phi_grid_exp_only_has_no_diagnostics(), test_phi_grid_handles_missing_fit_surfaces(), test_phi_grid_section_count_matches_n_phi(), test_phi_grid_set_bundle_none_clears(), test_phi_grid_tolerates_mismatched_optional_lengths() (+10 more)

### Community 178 - "test_codex_optstrat.py"
Cohesion: 0.03
Nodes (37): test_heterodyne_evaluator_is_a_point_evaluator(), test_homodyne_evaluator_is_a_point_evaluator(), test_homodyne_evaluator_matches_compute_g2_scaled(), _make_single_chunk_chunked(), _params14(), test_engine_takes_scattered_branch_and_matches_grid(), eval_scattered(), test_meshgrid_evaluator_does_not_advertise_scattered() (+29 more)

### Community 179 - "test_fit_queue.py"
Cohesion: 0.09
Nodes (24): test_classify_banner_ignores_ordinary_log_lines(), test_classify_banner_recognizes_engine_prefixes(), test_layer_status_l5_inactive_sentinels_are_false(), test_layer_status_maps_diagnostics_keys(), _queue(), test_bounded_concurrency_runs_one_at_a_time(), test_cancel_active_frees_slot_and_starts_next(), test_cancel_active_removes_partial_output_dir() (+16 more)

### Community 180 - "test_get_model.py"
Cohesion: 0.17
Nodes (7): test_config_manager_normalizes_heterodyne_synonym(), test_heterodyne_synonym_dispatches_to_heterodyne(), test_laminar_flow_does_not_dispatch_to_heterodyne(), test_make_model_accepts_dict(), test_static_does_not_dispatch_to_heterodyne(), test_two_component_dispatches_to_heterodyne(), _write_config()

### Community 181 - "Bugs"
Cohesion: 0.12
Nodes (11): Bugs, Codex review — shard optescape, Files read, Non-blocking notes, optescape-B2: CMA-ES refinement reports convergence when the local solver exhausts its evaluation budget, optescape-B3: Selecting streaming CMA-ES refinement prevents refinement from running, optescape-B4: Adaptive population sizing overrides an explicit CMA-ES generation cap, optescape-B5: A timed-out multistart worker survives fallback and can block process exit (+3 more)

### Community 182 - "heterodyne_scaling_utils.py"
Cohesion: 0.06
Nodes (13): _lag_separated_dataset(), test_all_finite_inputs_unchanged(), test_nan_delta_t_is_dropped_not_poisoning_thresholds(), test_constant_mode_propagates_first_angle_to_all(), test_individual_mode_varying_roundtrip_is_identity(), _BoundsAdapter, compute_averaged_scaling(), estimate_contrast_offset_from_quantiles() (+5 more)

### Community 183 - "hybrid_streaming"
Cohesion: 0.08
Nodes (26): checkpoint_frequency, chunk_size, cost_increase_tolerance, enable, enable_adaptive_warmup_lr, enable_checkpoints, enable_cost_guard, enable_step_clipping (+18 more)

### Community 186 - "ExecutionResult"
Cohesion: 0.14
Nodes (4): optstrat-B16: The public standard executor crashes for unbounded fits, curve_fit(), test_execution_result_dataclass(), ExecutionResult

### Community 187 - "test_cache_q_validation.py"
Cohesion: 0.20
Nodes (10): _angle_hash(), _loader_with_q(), test_angle_filtering_mismatch_is_rejected(), test_dt_mismatch_is_rejected(), test_matching_angle_filtering_is_accepted(), test_matching_dt_is_accepted(), test_matching_q_is_accepted(), test_missing_angle_filtering_metadata_warns_not_raises() (+2 more)

### Community 188 - "test_config_jax_free.py"
Cohesion: 0.29
Nodes (6): _imports_jax(), _probe_import(), test_importing_config_does_not_load_jax(), test_importing_parameter_manager_does_not_load_jax(), test_importing_registry_and_types_does_not_load_jax(), test_parameter_validation_still_works_after_lazy_import()

### Community 189 - "test_validation_branches.py"
Cohesion: 0.26
Nodes (14): _result(), test_fit_quality_acceptable_band_logged(), test_fit_quality_cmaes_max_restarts_warns(), test_fit_quality_cmaes_other_reason_passes(), test_fit_quality_condition_number_ok_via_pcov_fallback(), test_fit_quality_condition_number_too_high(), test_fit_quality_convergence_failed(), test_fit_quality_convergence_ok() (+6 more)

### Community 190 - "hybrid_streaming"
Cohesion: 0.08
Nodes (26): checkpoint_frequency, chunk_size, cost_increase_tolerance, enable, enable_adaptive_warmup_lr, enable_checkpoints, enable_cost_guard, enable_step_clipping (+18 more)

### Community 191 - "ValidationResult"
Cohesion: 0.11
Nodes (4): test_result_to_dict_roundtrip(), test_validation_result_defaults(), _result_to_dict(), ValidationResult

### Community 192 - "nlsq/validation.py"
Cohesion: 0.11
Nodes (13): test_classify_parameter_status_all_three_states(), test_is_physical_param(), test_validate_initial_params_within_bounds_direct(), test_validate_no_nan_inf_with_iteration_and_context(), test_validate_array_dimensions_accepts_matching_shapes(), test_validate_array_dimensions_rejects_empty_and_mismatched(), test_validate_no_nan_inf_accepts_finite(), test_validate_no_nan_inf_rejects_nan_and_inf() (+5 more)

### Community 193 - "test_completion_parity.py"
Cohesion: 0.22
Nodes (3): _generate_via_subprocess(), generated(), test_generated_script_is_valid_bash()

### Community 194 - "NumericalValidator"
Cohesion: 0.14
Nodes (14): test_set_bounds_disable_enable(), test_validate_gradients_detects_nonfinite(), test_validate_gradients_disabled_skips(), test_validate_gradients_finite_ok(), test_validate_loss_detects_nonfinite(), test_validate_loss_disabled_skips(), test_validate_loss_finite_ok(), test_validate_parameters_below_lower_bound() (+6 more)

### Community 195 - "LHS multistart"
Cohesion: 0.18
Nodes (18): LHS multistart, AntiDegeneracyController (L1-L5), fit_nlsq, fit_nlsq_multi_phi, NLSQ adapters (CurveFit, trust-region), select_nlsq_strategy, StratifiedResidualFunctionJIT, Anti-Degeneracy Defense (per iteration) (+10 more)

### Community 196 - "Memory-Aware Strategy Routing"
Cohesion: 0.13
Nodes (13): Advanced Topics index, JAX Environment Configuration, NLSQStrategy.HYBRID_STREAMING, NLSQStrategy.OUT_OF_CORE, NLSQStrategy.STANDARD (in-memory), Memory-Aware Strategy Routing, OptimizationResult, Development index (+5 more)

### Community 197 - "test_no_pickle_loads.py"
Cohesion: 0.19
Nodes (7): _find_violations(), _is_np_load(), _iter_source_files(), test_guard_can_detect_a_violation(), test_guard_flags_nonliteral_allow_pickle(), test_guard_passes_on_safe_call(), test_no_unsafe_np_load_calls()

### Community 198 - "hybrid_streaming"
Cohesion: 0.08
Nodes (26): checkpoint_frequency, chunk_size, cost_increase_tolerance, enable, enable_adaptive_warmup_lr, enable_checkpoints, enable_cost_guard, enable_step_clipping (+18 more)

### Community 199 - "npz_cache.py"
Cohesion: 0.18
Nodes (5): hash_filter_config(), _is_raw_source(), save_to_cache(), source_hdf_stat(), validate_cache_q_vector()

### Community 200 - "test_hybrid_streaming_retry.py"
Cohesion: 0.17
Nodes (11): optstrat-B10: Generic hybrid streaming rejects the wrapper’s normal config type, _enable(), _fake_hybrid_streaming_config(), _logger(), _residual_fn(), test_config_multipliers_apply_from_original_each_attempt(), test_non_recoverable_exception_propagates_without_retry(), test_raises_after_exhausting_all_retries() (+3 more)

### Community 201 - "test_layer5_gating.py"
Cohesion: 0.16
Nodes (7): _make_controller(), test_default_no_mode_keeps_all_layers_active(), test_heterodyne_synonym_also_inactive(), test_other_four_layers_active_for_heterodyne(), test_shear_layer_active_for_laminar_flow(), test_shear_layer_inactive_for_heterodyne(), test_shear_layer_inactive_for_static()

### Community 202 - "hybrid_streaming"
Cohesion: 0.08
Nodes (26): checkpoint_frequency, chunk_size, cost_increase_tolerance, enable, enable_adaptive_warmup_lr, enable_checkpoints, enable_cost_guard, enable_step_clipping (+18 more)

### Community 203 - "compute_diagonal_overlay_stats"
Cohesion: 0.16
Nodes (9): test_diagonal_overlay_out_of_bounds_raises(), test_diagonal_overlay_rmse_matches_manual(), test_diagonal_overlay_shapes_match(), test_diagonal_overlay_variance_ignores_single_inf(), test_compute_diagonal_overlay_stats_2d_input_raises(), test_compute_diagonal_overlay_stats_shape_mismatch_raises(), compute_diagonal_overlay_stats(), DiagonalOverlayResult (+1 more)

### Community 204 - "AdvancedMemoryManager"
Cohesion: 0.09
Nodes (3): test_manager_collected_by_refcounting_without_cyclic_gc(), AdvancedMemoryManager, logged_errors()

### Community 205 - "load_xpcs_data"
Cohesion: 0.25
Nodes (8): AdvancedMemoryManager, DataQualityController, load_xpcs_data, service.data, Fitting Workflow Diagram, load_xpcs_data, Raw C2 data + mode config, Raw C2 data + mode config

### Community 206 - "NLSQ CurveFit (trust-region least squares)"
Cohesion: 0.13
Nodes (18): L4 Gradient Collapse Monitoring, L5 Shear-Sensitivity Weighting (laminar_flow only), cli (argparse), generate_nlsq_plots, gui.ipc.worker, MainWindow (PySide6 workbench), OptimizationResult, service.fit (+10 more)

### Community 207 - "test_stratified_max_iter_grading.py"
Cohesion: 0.21
Nodes (7): _good_chi2_inputs(), test_converged_status_is_never_downgraded(), test_genuine_failure_stays_failed(), test_grading_is_numerics_safe(), test_max_nfev_with_good_chi2_grades_max_iter_via_reason_string(), test_max_nfev_with_good_chi2_grades_max_iter_via_status(), _wrapper()

### Community 208 - "C. New findings"
Cohesion: 0.22
Nodes (8): B. Disputes, C. New findings, Codex round 2 report, R2-C1: CLI standalone simulation — Reordered complete active lists scramble physics, R2-C2: HDF5 allocation — Guards budget the source dtype before allocating the promoted buffer, R2-C3: GUI config save — Atomic replacement breaks shared YAML symlinks, R2-C4: NPZ cache replay — A reused loader accumulates the same degradation on every warm hit, Verdict

### Community 210 - "numpy"
Cohesion: 0.03
Nodes (71): test_build_result_from_nlsq_singular_marker_is_nan_not_inf(), _result(), test_build_failed_result_with_initial(), test_build_failed_result_without_initial(), test_build_from_arrays_with_and_without_jacobian(), test_build_from_nlsq_bad_tuple_length_raises(), test_build_from_nlsq_dict(), test_build_from_nlsq_dict_missing_keys_raises() (+63 more)

### Community 211 - "fit_nlsq_multistart_heterodyne"
Cohesion: 0.22
Nodes (11): _data_ssr(), _result(), test_all_failed_returns_first_with_inf_ssr(), test_failed_runs_are_deprioritized_but_never_lose_a_result(), run(), test_joint_escape_runs_n_seeds_distinct_draws(), _fake_cmaes(), test_keeps_lowest_ssr_across_seeds() (+3 more)

### Community 212 - "_fit_nlsq_heterodyne"
Cohesion: 0.05
Nodes (37): Bugs, Codex review — shard opthet, Files read, Non-blocking notes, opthet-B10: Nested configuration enables regularization even when `enable` is false, opthet-B11: Uniform non-unit weights change the robust objective but the engine discards them, opthet-B12: Stratified budget exhaustion is graded as a hard failure, opthet-B2: Stratified-LS results label sorted scaling and per-angle chi-squared with input-order angles (+29 more)

### Community 213 - "test_chunked_jacfwd_dense.py"
Cohesion: 0.28
Nodes (5): _nonlinear_residual(), test_empty_params(), test_matches_for_various_col_block_sizes(), fn(), test_matches_jax_jacfwd_exactly()

### Community 214 - "test_compute_pool_matches_direct_kernel"
Cohesion: 0.20
Nodes (10): _kernels(), _static_physics_config(), test_accumulate_sequential_empty_raises(), test_compute_pool_matches_direct_kernel(), test_kernel_accumulator_chi2_matches_chi2_kernel(), test_kernel_diagonal_chunk_has_zero_chi2(), test_should_use_parallel_accumulation(), test_should_use_parallel_compute() (+2 more)

### Community 215 - "logging.py"
Cohesion: 0.04
Nodes (30): _isolate_logging(), test_dispatch_command_writes_log_file_when_config_enables_file(), _fake_standalone(), test_dispatch_command_writes_no_file_when_config_disables_file(), test_experimental_missing_phi_list_gets_distinct_files(), test_log_once_emits_once_per_key(), test_logged_errors_reraise_propagates_original_and_logs(), test_logged_errors_suppress_swallows() (+22 more)

### Community 216 - "test_layer_gate_wiring.py"
Cohesion: 0.23
Nodes (5): _build(), test_from_config_accepts_analysis_mode_kwarg(), test_layer5_active_for_laminar_flow(), test_layer5_active_when_lineage_omitted(), test_layer5_gated_off_for_two_component()

### Community 217 - "_FlakyOptimizer"
Cohesion: 0.28
Nodes (3): _AlwaysFailsOptimizer, _FlakyOptimizer, factory()

### Community 218 - "compute_chi_squared"
Cohesion: 0.20
Nodes (4): TestChiSquaredZeroSigma, test_core_b5(), chi2(), compute_chi_squared()

### Community 219 - "AntiDegeneracyController"
Cohesion: 0.04
Nodes (10): _source_of(), test_controller_accepts_explicit_averaged(), test_controller_existing_tokens_unchanged(), resolved(), test_controller_rejects_unknown_token(), test_hybrid_streaming_routes_through_shared_resolver(), test_controller_disabled_for_static_so_no_compression(), test_controller_callbacks_carry_no_l3_groups_when_l3_is_off() (+2 more)

### Community 220 - "physics.py"
Cohesion: 0.11
Nodes (3): validate_parameters(), validate_parameters_detailed(), ValidationResult

### Community 222 - "test_pr80_review.py"
Cohesion: 0.12
Nodes (16): test_config_b1(), test_config_b3(), test_config_b4(), test_config_b6(), test_config_b7(), test_config_b8(), _het_model(), test_core_b7() (+8 more)

### Community 223 - "XPCSDataFormatError"
Cohesion: 0.10
Nodes (9): _c2_out_dtype(), detect_format(), guard_aps_u_intermediate_allocation(), load_aps_old_format(), load_aps_u_format(), _read_matrices(), _or_combines_criteria(), _quality_filtering_requested() (+1 more)

### Community 224 - "test_config_unwrap.py"
Cohesion: 0.16
Nodes (6): captured_nlsq(), _fake_data(), _StubConfigManager, test_analysis_mode_only_in_nested_section(), test_flat_legacy_config_still_works(), test_nested_optimization_nlsq_is_applied()

### Community 225 - "test_phase5_model_function_modes.py"
Cohesion: 0.30
Nodes (7): _data_obj(), _make_fn(), _phys(), test_averaged_equals_individual_when_uniform(), test_averaged_vector_runs_and_broadcasts(), test_constant_vector_runs_physics_only(), test_individual_vector_runs()

### Community 226 - "run_fit"
Cohesion: 0.12
Nodes (19): _cm(), test_max_iterations_is_coerced_to_int(), test_multistart_sets_nested_enable_and_n_starts(), test_non_dict_config_is_noop(), test_none_fields_write_nothing(), test_tolerance_relaxes_ftol_xtol_and_gtol_together(), test_verbose_quiet_mapping(), test_run_fit_attaches_fit_quality_two_component_end_to_end() (+11 more)

### Community 227 - "AdaptiveRegularizer"
Cohesion: 0.08
Nodes (16): _make(), test_auto_and_absolute_modes_both_finite(), test_disabled_returns_zero(), test_jax_path_finite_gradient_at_zero_mean_group(), test_jax_path_matches_numpy_fallback_at_zero_mean_group(), test_nonfinite_params_return_inf(), test_out_of_range_group_is_skipped_not_crash(), test_relative_mode_gradient_is_finite_for_a_uniform_group() (+8 more)

### Community 228 - "_logger_that_raises_on_log"
Cohesion: 0.12
Nodes (3): _logger_that_raises_on_log(), TestLogCallsNeverRaise, boom()

### Community 229 - "test_debug_audit_2026_07_23_diagonal_skip.py"
Cohesion: 0.20
Nodes (8): _synthetic_data(), test_correct_diagonal_stage_marks_data_as_corrected(), test_disabled_preprocessing_does_not_set_marker(), test_loader_applies_configured_diagonal_correction_end_to_end(), fake_load_from_hdf(), test_loader_applies_mandatory_correction_when_not_preprocessed(), test_loader_skips_mandatory_correction_when_already_corrected(), spy()

### Community 231 - "MemoryPressureMonitor"
Cohesion: 0.11
Nodes (4): test_get_pressure_trend_survives_concurrent_appends(), test_deadzone_preserves_state_and_recovery_still_fires(), log_once(), MemoryPressureMonitor

### Community 232 - "cmaes"
Cohesion: 0.08
Nodes (24): auto_select, enable, max_generations, max_restarts, memory_limit_gb, normalization_epsilon, normalize, popsize (+16 more)

### Community 233 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, complexity, config_version, description, generated_at, generated_by, parameter_count, physics_model (+1 more)

### Community 234 - "build_gradient_collapse_callback"
Cohesion: 0.24
Nodes (10): _monitor(), test_callback_feeds_monitor_per_iteration(), grad_fn(), test_callback_swallows_grad_fn_errors(), test_diagnostics_block_shape_and_mechanism(), test_diagnostics_empty_history_is_fallback(), test_update_frequency_throttles_grad_evals(), test_zero_or_nan_denominator_yields_inf_ratio() (+2 more)

### Community 235 - "cmaes"
Cohesion: 0.08
Nodes (24): auto_select, enable, max_generations, max_restarts, memory_limit_gb, normalization_epsilon, normalize, popsize (+16 more)

### Community 236 - "test_io.py"
Cohesion: 0.27
Nodes (3): _make_npz_arrays(), TestSaveNlsqNpzFile, save_nlsq_npz_file()

### Community 237 - "cmaes"
Cohesion: 0.08
Nodes (24): auto_select, enable, max_generations, max_restarts, memory_limit_gb, normalization_epsilon, normalize, popsize (+16 more)

### Community 238 - "completion.sh"
Cohesion: 0.34
Nodes (13): _filedir(), _init_completion(), mapfile(), completion.sh script, _xpcsjax(), _xpcsjax_cleanup(), _xpcsjax_config(), _xpcsjax_config_xla() (+5 more)

### Community 239 - "test_docs_structure.py"
Cohesion: 0.21
Nodes (6): ADR 0001: Automated structural doc-coverage check, _missing_api_pages(), test_every_submodule_has_an_api_page(), test_excluded_packages_are_real_submodules(), test_missing_api_pages_detects_a_gap(), _top_level_submodules()

### Community 241 - "phi_filtering.py"
Cohesion: 0.10
Nodes (7): test_phi_angle_filter_matches_raw_360_convention_angle(), create_anisotropic_ranges(), create_isotropic_ranges(), filter_phi_angles(), filter_phi_angles_jax(), _normalize_target_ranges(), PhiAngleFilter

### Community 243 - "stratification"
Cohesion: 0.08
Nodes (24): enabled, target_angle_ranges, optimization, angle_filtering, method, recovery, sequential, stratification (+16 more)

### Community 244 - "fit_with_out_of_core_accumulation"
Cohesion: 0.12
Nodes (10): optstrat-B13: Out-of-core ignores the documented iteration limit, optstrat-B5: Out-of-core covariance counts masked rows and fixed parameters, create_ooc_kernels(), compute_chunk_accumulators(), r_fn(), compute_chunk_chi2(), fit_with_out_of_core_accumulation(), _accumulate_at() (+2 more)

### Community 245 - "test_post_install_fish.py"
Cohesion: 0.36
Nodes (7): _appended_block(), _make_fake_venv(), test_fish_activation_is_idempotent(), test_fish_activation_missing_script_returns_false(), test_fish_activation_resolves_conda_prefix(), test_generated_fish_block_parses(), _install_xla_fish_activation()

### Community 246 - "cmaes"
Cohesion: 0.08
Nodes (24): auto_select, enable, max_generations, max_restarts, memory_limit_gb, normalization_epsilon, normalize, popsize (+16 more)

### Community 247 - "test_validation_crash_coverage.py"
Cohesion: 0.29
Nodes (6): _fresh_report(), _raise_boom(), test_compute_data_statistics_crash_logs_error_and_invalidates_report(), test_correlation_matrices_crash_logs_error_and_invalidates_report(), test_physics_parameters_crash_logs_error_and_invalidates_report(), test_statistical_properties_crash_logs_error_and_invalidates_report()

### Community 248 - "plots_view.py"
Cohesion: 0.06
Nodes (18): test_residual_map_levels_computed_from_full_resolution(), test_c2_levels_clamp_to_unit_band(), test_diagonal_residual_traces_diag_against_t1(), test_residual_histogram_renders_and_tolerates_all_nan(), test_residual_levels_are_symmetric_about_zero(), test_residual_map_accepts_array(), test_residuals_vs_fitted_decimates_large_cloud(), test_residuals_vs_fitted_tolerates_all_nan() (+10 more)

### Community 249 - "interactive_setup"
Cohesion: 0.15
Nodes (11): test_get_venv_path_conda_fallback(), test_get_venv_path_prefers_virtual_env(), test_get_venv_path_sys_prefix_fallback(), test_install_completion_activation_skips_outside_venv(), test_is_virtual_environment_false(), test_is_virtual_environment_via_base_prefix(), test_is_virtual_environment_via_conda(), get_venv_path() (+3 more)

### Community 250 - "anti_degeneracy_controller.py"
Cohesion: 0.11
Nodes (6): test_result_dataclass(), test_optstrat_b8(), _indefinite_hessian(), safe_float(), safe_int(), HierarchicalResult

### Community 252 - "run_validation"
Cohesion: 0.28
Nodes (6): test_print_report_renders_all_tags(), test_run_validation_human_report(), test_run_validation_json(), test_validate_verbose_prints(), _print_report(), run_validation()

### Community 253 - "_normalize_data_to_object"
Cohesion: 0.09
Nodes (20): data-B1: Optimization angle filtering leaves the fitted q attached to an excluded angle, Bugs, Codex review — shard optwrap, Files read, Non-blocking notes, optwrap-B10: Homodyne routing ignores the configured memory budget, optwrap-B1: The standard wrapper fits unsorted observations against the wrong angles, optwrap-B2: Unsorted angle input swaps per-angle uncertainty weights (+12 more)

### Community 254 - "select_nlsq_strategy"
Cohesion: 0.11
Nodes (21): _strategy_name(), test_large_data_with_tight_threshold_escalates(), test_memory_fraction_clamped_to_valid_range(), test_n_params_zero_yields_zero_peak_and_standard(), test_out_of_core_isolated_from_hybrid_streaming(), test_router_executes_without_exception_for_typical_inputs(), test_small_data_routes_to_standard(), _patch_threshold() (+13 more)

### Community 255 - "All-modules registry page"
Cohesion: 0.18
Nodes (10): All-modules registry page, xpcsjax.cli, xpcsjax.config, xpcsjax.core, xpcsjax.data, xpcsjax.device, xpcsjax.optimization.nlsq, xpcsjax.runtime.shell (+2 more)

### Community 257 - "heterodyne_parameter_space.py"
Cohesion: 0.10
Nodes (13): test_coerce_finite_float_accepts_real_numbers(), test_coerce_finite_float_rejects_bool_false(), test_coerce_finite_float_rejects_bool_true(), test_coerce_finite_float_rejects_nan_and_inf(), _apply_fixed_parameters(), _apply_initial_parameters(), _apply_parameter_space_bounds(), _apply_tied_parameters() (+5 more)

### Community 258 - "core/test_debug_audit_2026_07_22.py"
Cohesion: 0.11
Nodes (10): _het_params(), test_compute_g1_shear_static_mode_accepts_0d_scalar_t1(), test_compute_g1_total_elementwise_time_grid_avoids_truncation(), test_heterodyne_chi_squared_ignores_diagonal_and_frame0_perturbation(), test_heterodyne_chi_squared_matches_residuals_masking(), TestShearDtNoneContract, compute_chi_squared(), _compute_residuals_jit() (+2 more)

### Community 259 - "Finished"
Cohesion: 0.09
Nodes (13): test_cancel_race_finished_stays_cancelled(), test_map_views_clear_removes_image(), test_open_project_sets_active_dataset(), _window(), test_reader_final_drain_recovers_terminal_after_grace(), _imports_jax(), _probe_import(), test_event_subclasses_carry_run_id_and_seq() (+5 more)

### Community 260 - "service/config.py"
Cohesion: 0.10
Nodes (18): test_validate_config_scalar_initial_parameters_reports_not_crashes(), _probe_import(), test_config_service_is_jax_free(), test_load_config_applies_mode_and_output(), test_load_config_no_overrides_leaves_config(), _ip(), test_available_modes_are_the_four_known(), test_out_of_bounds_value_is_an_error() (+10 more)

### Community 261 - "InputValidator"
Cohesion: 0.17
Nodes (7): test_input_validator_bounds_inconsistent_and_out_of_range(), test_input_validator_dimension_mismatch_recorded(), test_input_validator_errors_property_is_copy(), test_input_validator_non_strict_returns_false_and_records_errors(), test_input_validator_passes_on_clean_input(), test_input_validator_strict_raises_on_bad_input(), InputValidator

### Community 262 - "ResultValidator"
Cohesion: 0.18
Nodes (6): test_result_validator_consistency_failure_recorded(), test_result_validator_happy_path_true(), test_result_validator_strict_raises_out_of_bounds(), test_result_validator_warnings_property_is_copy(), test_result_validator_records_warnings_for_bad_covariance(), ResultValidator

### Community 264 - "generate_plots"
Cohesion: 0.20
Nodes (6): _cm(), test_generate_plots_forwards_to_viz_and_forces_agg(), test_generate_plots_returns_none_when_get_model_fails(), test_generate_plots_returns_none_when_render_fails(), _generate_nlsq_plots(), generate_plots()

### Community 265 - "test_loader_performance_engine_cleanup.py"
Cohesion: 0.33
Nodes (5): _bare_loader(), test_close_calls_memory_manager_shutdown(), test_close_is_idempotent_and_safe_with_no_components(), test_context_manager_closes_on_exit(), test_performance_engine_attribute_no_longer_exists()

### Community 266 - "fit_nlsq"
Cohesion: 0.20
Nodes (9): fit_nlsq, load_xpcs_data, OptimizationResult, fit_nlsq, OptimizationResult, heterodyne_views.per_angle_chi2, heterodyne_views.reconstruct_per_angle_scaling, ConfigManager (+1 more)

### Community 267 - "_build_laminar_stratified_data"
Cohesion: 0.29
Nodes (3): R2-A7: optstrat-B4 — Stratified covariance still counts rows masked by sigma, _build_laminar_stratified_data(), test_optstrat_b3()

### Community 268 - "test_run_controller.py"
Cohesion: 0.29
Nodes (4): test_cancel_confirmed_yes_proceeds(), test_cancel_declined_no_does_not_cancel(), test_cancel_no_selection_shows_message_not_queue_cancel(), _window_with_queued_run()

### Community 269 - "test_phase5_vector_build.py"
Cohesion: 0.43
Nodes (5): _run(), test_fit_vector_length_observed_averaged_expands_to_dense(), test_fit_vector_length_observed_individual(), test_vector_length_per_mode(), _vlen()

### Community 270 - "file"
Cohesion: 0.09
Nodes (22): colors, enabled, format, level, show_progress, backup_count, enabled, filename (+14 more)

### Community 272 - "test_lazy_imports.py"
Cohesion: 0.18
Nodes (5): tests/characterization shard, test_env_setup_mirrors_homodyne(), test_heterodyne_model_exported(), test_public_exports_phase4(), test_top_level_import_does_not_load_jax()

### Community 273 - "_resolve_color_limits"
Cohesion: 0.33
Nodes (7): test_all_nan_returns_fallback(), test_empty_matrix_returns_fallback(), test_flat_constant_matrix_returns_widened_range(), test_normal_data_returns_percentile_limits(), test_percentile_clamp_excludes_outliers(), test_returns_floats_not_numpy_scalars(), _resolve_color_limits()

### Community 274 - "fit_nlsq"
Cohesion: 0.06
Nodes (16): _cmaes_available(), _cmaes_smoke_config_dict(), test_heterodyne_per_angle_cmaes_fits_without_signature_drift(), test_default_none_does_not_change_result(), test_on_iteration_accepted_and_wellformed(), test_raising_observer_does_not_abort_fit(), test_two_component_accepts_on_iteration_and_never_calls_it(), _tiny_static_config_and_data() (+8 more)

### Community 275 - "parse_per_angle_scaling_overrides"
Cohesion: 0.32
Nodes (7): _RecordingLogger, test_malformed_value_is_dropped_with_default_wording(), test_none_input_yields_no_overrides(), test_sequential_wording_matches_original_second_call_site(), test_valid_overrides_pass_through(), test_wrong_length_is_dropped_with_default_wording(), parse_per_angle_scaling_overrides()

### Community 276 - "model_adapter.PointEvaluator"
Cohesion: 0.25
Nodes (8): heterodyne_engine_route.fit_two_component_via_engine, model_adapter.HeterodynePointEvaluator, model_adapter.HeterodynePointwiseEvaluator, model_adapter.HomodynePointEvaluator, model_adapter.PointEvaluator, StratifiedResidualFunctionJIT, coverage.run omit list (engine orchestration files), tests/optimization shard

### Community 277 - "plotting"
Cohesion: 0.09
Nodes (22): canvas_height, canvas_width, interpolation, savefig_kwargs, use_tight_layout, plotting, angle_correlation, angle_coverage (+14 more)

### Community 278 - "xpcs_loader.py"
Cohesion: 0.07
Nodes (34): data-B11: The NPZ boundary accepts angle metadata with no corresponding matrix, _loader(), test_empty_selection_raises(), test_nonempty_selection_returned_unchanged(), test_accepts_plain_filename(), test_rejects_unsafe_cache_filenames(), test_accepts_realistic_allocation(), test_accepts_realistic_frame_count() (+26 more)

### Community 279 - "nlsq"
Cohesion: 0.09
Nodes (22): check_gradients, enabled, log_jacobian_norms, sample_size, diagnostics, goal, gtol, loss (+14 more)

### Community 280 - "test_pr81_review.py"
Cohesion: 0.11
Nodes (16): agy closing check of PR #81, New issues introduced by the fixes, Per-finding status, Verdict, test_clamp_metadata_accepts_numpy_integer_frame_counts(), test_duplicate_phi_rows_keep_their_own_mask_on_the_pointwise_paths(), test_engine_route_declines_non_positive_weights_under_linear_loss(), test_heterodyne_streaming_keeps_treating_a_null_l3_enable_as_off() (+8 more)

### Community 281 - "build_heterodyne_pointwise_model"
Cohesion: 0.09
Nodes (12): optstrat-B2: Weighted heterodyne streaming swaps sigma slabs for unsorted angles, test_frozen_per_angle_scaling_invariant_to_input_phi_order(), _frozen(), test_model_fn_pointwise_matches_kernel(), test_pointwise_data_excludes_t0_and_diagonal(), test_pointwise_model_averaged_param_layout(), test_pointwise_model_rejects_removed_token(), _assert_pointwise_matches_batched() (+4 more)

### Community 282 - "test_optstrat_b4"
Cohesion: 0.29
Nodes (3): test_optstrat_b15(), test_optstrat_b4(), _spy()

### Community 283 - "test_stratified_ls_averaged_covariance_transform.py"
Cohesion: 0.25
Nodes (3): _broadcast_jacobian_transform(), test_hand_computed_n_phi2_n_physical1(), test_source_no_longer_uses_identity_diagonal_shortcut()

### Community 284 - "color_scale"
Cohesion: 0.29
Nodes (7): fixed_max, fixed_min, mode, percentile_max, percentile_min, pin_legacy_range, color_scale

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
Nodes (3): _extract_collect_all_names(), test_collect_all_extraction_ignores_comment_text(), test_pyinstaller_spec_covers_runtime_deps()

### Community 292 - "test_debug_audit_2026_07_23_negative_correlation_repair.py"
Cohesion: 0.22
Nodes (8): _data_with_one_skipped_matrix(), _data_with_one_skipped_matrix_robust(), test_normalize_data_tracks_per_matrix_mask(), test_normalize_data_tracks_per_matrix_mask_robust(), test_repair_negative_correlations_clamps_everything_when_unmarked(), test_repair_negative_correlations_respects_per_matrix_mask(), test_repair_negative_correlations_respects_real_pipeline_mask(), test_repair_negative_correlations_upcasts_integer_c2()

### Community 293 - "file"
Cohesion: 0.10
Nodes (21): colors, enabled, format, level, show_progress, backup_count, enabled, filename (+13 more)

### Community 294 - "FitQualityConfig"
Cohesion: 0.21
Nodes (7): test_classify_fit_quality_bands(), test_classify_fit_quality_respects_custom_thresholds(), test_fit_quality_config_defaults_match_homodyne(), test_fit_quality_config_from_dict_falls_back_on_missing_keys(), test_fit_quality_config_from_none_returns_defaults(), classify_fit_quality(), FitQualityConfig

### Community 298 - "file"
Cohesion: 0.10
Nodes (21): colors, enabled, format, level, show_progress, backup_count, enabled, filename (+13 more)

### Community 299 - "file"
Cohesion: 0.10
Nodes (21): colors, enabled, format, level, show_progress, backup_count, enabled, filename (+13 more)

### Community 300 - "xla_config.bash"
Cohesion: 0.50
Nodes (7): xla_config.bash script, _xpcsjax_configure_xla(), _xpcsjax_get_cpu_count(), _xpcsjax_load_xla_mode(), _xpcsjax_resolve_mode_file(), _xpcsjax_save_xla_mode(), _xpcsjax_xla_setup()

### Community 301 - "Public API Reference"
Cohesion: 0.33
Nodes (6): Public API Reference, ConfigManager, fit_nlsq (single-entry NLSQ wrapper), HeterodyneModel, HomodyneModel, load_xpcs_data

### Community 303 - "OptimizationStrategy"
Cohesion: 0.06
Nodes (18): test_combine_angle_results_excludes_nonfinite_covariance(), test_combine_angle_results_per_parameter_precision_dominates(), test_fallback_no_recovery_reports_failed_on_stagnation(), fake_curve_fit(), test_get_group_indices_scaling_resolves(), test_plain_soft_failure_escalates_to_next_strategy(), test_recovery_soft_failure_escalates_to_next_strategy(), test_sequential_reduced_chi2_no_zerodiv_when_underdetermined() (+10 more)

### Community 304 - "_FakeModel"
Cohesion: 0.33
Nodes (3): _FakeModel, test_build_hybrid_streaming_result_expands_fixed_physics_param(), test_build_hybrid_streaming_result_mirrors_tied_child()

### Community 305 - "parallel_accumulator.py"
Cohesion: 0.21
Nodes (3): _batch_timeout(), _ooc_compute_chi2_chunk(), _ooc_compute_chunk()

### Community 307 - "_version_at_least"
Cohesion: 0.29
Nodes (3): test_version_at_least(), test_version_at_least_returns_none_for_unparseable_version(), _version_at_least()

### Community 308 - "Bugs"
Cohesion: 0.15
Nodes (11): Bugs, Codex review — shard ioviz, Files read, ioviz-B10: Datashader crashes when a heatmap axis has one coordinate, ioviz-B1: Homodyne reconstruction assigns fitted scaling to the wrong angles, ioviz-B3: The JSON size guard blocks saving fits with a modest diagnostic covariance, ioviz-B5: Failure to open the optional log file can abort the CLI before fitting, ioviz-B7: Removing managed log handlers leaves host logging disconnected (+3 more)

### Community 309 - "A. Fix review"
Cohesion: 0.18
Nodes (11): A. Fix review, R2-A10: data-B11 — NPZ validation accepts multiple phi coordinates per matrix, R2-A12: cli-B4 — Partial nested scattering hides the top-level q, R2-A13: cli-B8 — Newline paths containing a non-BMP character do not round-trip, R2-A1: opthet-B2 — Restoring duplicate phi rows permutes physics parameters, R2-A2: opthet-B1/B2 — Sorting an individual fit indexes through a shared weight slab, R2-A3: opthet-B9 — The direct individual covariance still counts L3 penalty rows, R2-A4: optescape-B2 — CMA-ES still reports convergence after accepting an unfinished refinement (+3 more)

### Community 310 - "test_anti_degeneracy_transforms.py"
Cohesion: 0.31
Nodes (8): _build(), _make_controller(), _per_angle_params(), test_constant_collapse_uses_nanmean_and_preserves_physical(), test_constant_round_trip_is_exact_for_constant_scaling(), test_controller_rejects_removed_tokens_after_phase7(), test_controller_resolves_individual_no_legacy_scaling_attr(), test_get_diagnostics_returns_mapping()

### Community 311 - "test_two_component_smoke.py"
Cohesion: 0.24
Nodes (4): _build_synthetic_c2(), _smoke_config_dict(), test_heterodyne_smoke_fit_recovers_truth(), _write_config()

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

### Community 319 - "ContextFilter"
Cohesion: 0.06
Nodes (28): ioviz-B6: Nested log contexts overwrite inherited fields with `Ellipsis`, test_ioviz_b6(), test_json_formatter_handles_empty_exc_info_tuple(), test_json_formatter_never_raises_on_bad_format_args(), test_json_formatter_redaction_does_not_overredact(), test_json_formatter_redacts_secrets(), test_json_formatter_schema_and_jsonsafe(), test_json_safe_recursion_guard_truncates_deep_nesting() (+20 more)

### Community 320 - "test_ci_heavy_nodes_parity.py"
Cohesion: 0.60
Nodes (4): _ci_yml_node_occurrences(), _makefile_heavy_nodes(), test_ci_yml_deselects_exactly_the_makefile_heavy_nodes(), test_ci_yml_references_each_heavy_node_symmetrically()

### Community 322 - "hierarchical"
Cohesion: 0.33
Nodes (6): hierarchical, enable, max_outer_iterations, outer_tolerance, per_angle_max_iterations, physical_max_iterations

### Community 323 - "generate_nlsq_plots"
Cohesion: 0.40
Nodes (4): generate_nlsq_plots, plot_nlsq_fit, plot_residual_map, plot_simulated_data

### Community 324 - "Joint global escape keep-better contract"
Cohesion: 0.40
Nodes (4): _fit_joint_cmaes_multi_phi, _fit_joint_multistart, cmaes_wrapper.fit_with_cmaes, run_multistart_nlsq

### Community 325 - "test_config_debug_null_nonfinite.py"
Cohesion: 0.22
Nodes (10): _has_error(), test_heterodyne_finite_value_still_passes(), test_heterodyne_nan_is_error(), test_homodyne_finite_value_still_passes(), test_homodyne_inf_is_error(), test_homodyne_nan_is_error(), test_inverted_bounds_rejected(), test_one_sided_override_inverting_against_default_rejected() (+2 more)

### Community 328 - "test_config_relative_data_paths.py"
Cohesion: 0.32
Nodes (8): test_absolute_path_is_left_unchanged(), test_env_var_path_expands_to_absolute(), test_override_path_is_not_resolved(), test_parent_traversal_path_is_left_literal_for_downstream_guard(), test_relative_data_folder_anchored_to_config_dir(), test_relative_subdir_is_joined(), test_unset_env_var_is_not_mis_anchored(), _write_config()

### Community 329 - "Bugs"
Cohesion: 0.14
Nodes (13): Bugs, Codex review — shard optstrat, Files read, Non-blocking notes, optstrat-B11: Generic hybrid streaming supplies an incorrect prediction callback, optstrat-B14: Out-of-core reports an exact optimum as a line-search stall, optstrat-B1: Homodyne stratified fits assign sigma to the wrong angles, optstrat-B3: Stratified-LS rejects a legal zero-contrast starting point before solving (+5 more)

### Community 331 - "test_laminar_no_worse_phase6.py"
Cohesion: 0.33
Nodes (6): _homodyne_config(), _ssr(), _stratified_info(), test_fourier_rejected_all_laminar_paths(), test_stratified_individual_ssr_tripwire(), test_stratified_ssr_finite()

### Community 333 - "xpcsjax.cli.config_generator.main"
Cohesion: 0.50
Nodes (4): xpcsjax.cli.config_generator.main, xpcsjax.cli.config_generator.generate_config, xpcsjax.cli.config_generator.show_template, xpcsjax.cli.config_generator.validate_config

### Community 334 - "xpcsjax Project Logo"
Cohesion: 0.67
Nodes (4): xpcsjax Project Logo, Time Axes t1 and t2, Two-Time Correlation Surface c2(t1, t2), xpcsjax Wordmark (xpcs + jax)

### Community 335 - "test_aps_u_bin_skip_misalignment.py"
Cohesion: 0.29
Nodes (4): _loader(), test_bin_skip_raises_instead_of_misaligning(), test_matched_bins_and_matrices_load_cleanly(), _write_aps_u_file()

### Community 336 - "post_install.py"
Cohesion: 0.15
Nodes (9): test_install_xla_activation_skips_outside_venv(), test_install_xla_bash_activation_injection_and_idempotency(), test_install_xla_bash_activation_missing_script(), test_install_xla_config_scripts(), test_is_conda_environment(), install_xla_activation(), _install_xla_bash_activation(), install_xla_config_scripts() (+1 more)

### Community 340 - "_install_completion_bash_activation"
Cohesion: 0.50
Nodes (3): test_completion_activation_missing_script(), test_completion_bash_activation_injection_and_idempotency(), _install_completion_bash_activation()

### Community 342 - "configure_cpu_hpc"
Cohesion: 0.67
Nodes (3): configure_cpu_hpc, xpcsjax.device.cpu.configure_cpu_threading, detect_cpu_info

### Community 343 - "DatashaderRenderer"
Cohesion: 0.67
Nodes (3): DatashaderRenderer, plot_c2_comparison_fast, plot_c2_heatmap_fast

### Community 345 - "make test-smoke"
Cohesion: 0.67
Nodes (3): HEAVY_NODES deselected tests, make test-smoke, make verify

### Community 346 - "HomodyneModel"
Cohesion: 0.67
Nodes (3): ConfigManager, HomodyneModel, static_isotropic analysis mode

### Community 348 - "accumulate_chunks_sequential"
Cohesion: 0.29
Nodes (8): _int_chunks(), test_accumulate_sequential_sums_correctly(), test_parallel_below_threshold_falls_back(), test_parallel_fallback_on_worker_failure(), test_parallel_matches_sequential_bit_exact(), test_parallel_zero_workers_falls_back(), accumulate_chunks_parallel(), accumulate_chunks_sequential()

### Community 350 - "anti_degeneracy"
Cohesion: 0.11
Nodes (18): constant_scaling_threshold, enable, execute_layers, gradient_monitoring, per_angle_mode, regularization, consecutive_triggers, enable (+10 more)

### Community 363 - "regularization"
Cohesion: 0.11
Nodes (18): constant_scaling_threshold, enable, execute_layers, gradient_monitoring, per_angle_mode, regularization, consecutive_triggers, enable (+10 more)

### Community 364 - "Bugs"
Cohesion: 0.11
Nodes (18): Bugs, Codex review — shard data, data-B12: A full-turn phi range silently collapses to a single direction, data-B13: Default JAX output skips final QC symmetry and decay calculations, data-B14: QC retains a hard validation error after successfully repairing the reported values, data-B15: Incremental QC can reuse a passing result after the time axis becomes invalid, data-B16: A skipped preprocessing stage can leave the loaded dataset marked clean, data-B2: Selective HDF5 loading silently broadcasts a short matrix into a full correlation surface (+10 more)

### Community 368 - "nlsq"
Cohesion: 0.12
Nodes (17): goal, gtol, loss, max_iterations, memory_fraction, progress, tolerance, trust_region_scale (+9 more)

### Community 376 - "nlsq"
Cohesion: 0.12
Nodes (17): goal, gtol, loss, max_iterations, memory_fraction, progress, tolerance, trust_region_scale (+9 more)

### Community 377 - "nlsq"
Cohesion: 0.12
Nodes (17): goal, gtol, loss, max_iterations, memory_fraction, progress, tolerance, trust_region_scale (+9 more)

### Community 407 - "ConfigManager"
Cohesion: 0.02
Nodes (54): cli-B4: Standalone heterodyne simulation ignores the grouped wavevector, config-B1: Template legacy fields override the explicitly selected dataset, config-B5: Grouped heterodyne initial values are lost in config-driven previews, config-B8: `ConfigManager.load_config()` leaves cached parameters from the previous file, data-B10: The public loader rejects the modern `experimental_data.file_path` configuration, agy position — round 1, data-B9, Disagreements with Claude's recommendation (+46 more)

### Community 408 - "XpcsDataset"
Cohesion: 0.36
Nodes (5): _raw(), test_is_dict_subclass_backward_compatible(), test_missing_correlation_raises_clear_error(), test_typed_accessors_resolve_canonical_and_alias_keys(), XpcsDataset

### Community 414 - "physics_validators.py"
Cohesion: 0.09
Nodes (15): test_cross_parameter_min_severity_filters_info(), test_cross_parameter_missing_keys_no_violation(), test_d_offset_overfitting_flagged_above_half_d0(), test_d_offset_overfitting_flagged_when_negative_and_dominant(), test_d_offset_within_bound_no_violation(), test_validate_all_parameters_aggregates_single_and_cross(), test_validate_all_parameters_no_violations_for_sane_values(), ConstraintRule (+7 more)

### Community 416 - "OOCSharedArrays"
Cohesion: 0.16
Nodes (6): test_shared_arrays_rejects_non_finite_sigma(), test_shared_arrays_rejects_non_positive_sigma(), test_shared_arrays_roundtrip_and_cleanup(), test_shared_arrays_without_sigma(), test_worker_functions_in_process(), OOCSharedArrays

### Community 424 - "xpcsjax/__init__.py"
Cohesion: 0.14
Nodes (6): test_core_b10(), test_core_b6(), test_core_b8(), _detect_worker_count(), __getattr__(), _xla_host_device_count()

### Community 429 - "test_codex_data.py"
Cohesion: 0.07
Nodes (44): R3-A4: R2-A9 — Legacy cache validation misidentifies direct NPZ sources and misses the all-frames request, R3-A5: R2-A10 — Size-only phi validation admits row vectors that downstream treats as a single angle, R4-A1: R3-A3 — Stratified covariance counts index diagonals instead of intersecting the masks, R4-A2: R3-A5 — Accepted NPZ phi columns still break plotting, test_load_synthetic_npz_cache_roundtrip(), _write_synthetic_npz_cache(), _bare_loader(), _half() (+36 more)

### Community 432 - "Round 1 ledger (Codex claims -> Claude verification)"
Cohesion: 0.12
Nodes (15): cli (cli + runtime + device + __init__; 10 claims: 9 confirmed, 1 design), config (11 claims: 7 confirmed, 3 design, 1 rejected), core (10 claims: 7 confirmed, 2 design, 1 rejected), data (18 claims: 15 confirmed, 3 rejected) — verifier rule: CONFIRMED needs an in-package caller, gui (gui + service; 14 claims: 12 confirmed, 1 design, 1 rejected), ioviz (io + utils + viz; 10 claims: 6 confirmed, 2 design, 2 rejected), LEAD ADJUDICATIONS (one rule applied to all shards), optescape (8 claims: 6 confirmed, 1 design, 1 rejected) (+7 more)

### Community 436 - "Maintainer-decision items from the 2026-11-24 review — dispositions"
Cohesion: 0.20
Nodes (9): Closed as intended, Decided by the three-way round, Fixed, Maintainer-decision items from the 2026-11-24 review — dispositions, Recommendation, Review of PR #81 (Codex + Antigravity), Six-pass review of PR #81, Summary (+1 more)

### Community 438 - "A. Fix review"
Cohesion: 0.11
Nodes (17): A. Fix review, B. Disputes, C. New findings, Codex round 3 report, R3-A10: R2-C2 — Mixed-width HDF quality candidates can still exceed preflight allocation limits, R3-A11: R2-A8 — Mixed floating-point bins no longer use their source arithmetic on selective reads, R3-A2: R2-A2 — Shared weight slabs still fail during engine scoring, R3-A3: R2-A7 — The covariance count diverges from the actual residual mask (+9 more)

### Community 439 - "plotting"
Cohesion: 0.12
Nodes (16): canvas_height, canvas_width, plotting, angle_correlation, angle_coverage, correlation_function, datashader, dpi (+8 more)

### Community 441 - "anti_degeneracy"
Cohesion: 0.12
Nodes (16): constant_scaling_threshold, enable, execute_layers, gradient_monitoring, hierarchical, per_angle_mode, consecutive_triggers, enable (+8 more)

### Community 442 - "plotting"
Cohesion: 0.12
Nodes (16): canvas_height, canvas_width, plotting, angle_correlation, angle_coverage, correlation_function, datashader, dpi (+8 more)

### Community 443 - "anti_degeneracy"
Cohesion: 0.12
Nodes (16): constant_scaling_threshold, enable, execute_layers, gradient_monitoring, hierarchical, per_angle_mode, consecutive_triggers, enable (+8 more)

### Community 444 - "plotting"
Cohesion: 0.12
Nodes (16): canvas_height, canvas_width, plotting, angle_correlation, angle_coverage, correlation_function, datashader, dpi (+8 more)

### Community 445 - "Changelog Documentation"
Cohesion: 0.14
Nodes (12): Changelog Documentation, fixed_parameters/active_parameters Hardening, Tied Parameters Feature, v0.1.1 Release, v0.1.2 Release, v0.1.3 Release, v0.1.4 Release, v0.1.5 Release (+4 more)

### Community 446 - "test_heterodyne_parameter_manager_tied.py"
Cohesion: 0.31
Nodes (6): _manager_with_tie(), test_expand_varying_to_full_mirrors_tied_child(), test_expand_varying_to_full_tied_value_updates_with_parent(), test_expand_varying_to_full_untied_matches_previous_behavior(), test_tied_idx_pairs_empty_when_untied(), test_tied_idx_pairs_maps_names_to_physics_indices()

### Community 447 - "test_laminar_execute_layers.py"
Cohesion: 0.25
Nodes (5): _fit(), test_flag_off_is_single_solve_no_l2(), test_flag_on_executes_l2_and_l3_keep_better(), test_flag_on_l2_without_l3(), test_info_threads_convergence_reason_and_status()

### Community 448 - "optimization"
Cohesion: 0.13
Nodes (15): enabled, target_angle_ranges, optimization, angle_filtering, method, recovery, sequential, enable (+7 more)

### Community 450 - "optimization"
Cohesion: 0.13
Nodes (15): enabled, target_angle_ranges, optimization, angle_filtering, method, recovery, sequential, enable (+7 more)

### Community 451 - "optimization"
Cohesion: 0.13
Nodes (15): enabled, target_angle_ranges, optimization, angle_filtering, method, recovery, sequential, enable (+7 more)

### Community 453 - "log_heterodyne_completion"
Cohesion: 0.22
Nodes (13): _chi2_args(), _completion_physics(), _het_result(), test_averaged_physics_first_marker_reads_physics_from_head(), test_averaged_scaling_first_marker_reads_physics_from_tail(), test_gradient_chi2_does_not_raise_on_static_args(), test_hessian_chi2_does_not_raise_on_static_args(), _logged_physics() (+5 more)

### Community 454 - "A. Fix review"
Cohesion: 0.14
Nodes (12): A. Fix review, B. Disputes, C. New findings, Codex round 4 report, R4-A3: R3-A8 — Zero-weight observations still make a zero-DOF fit report covariance, R4-A4: R3-C2 — The standard executor still declares a failed fit converged via its top-level status, R4-A5: R3-A2 — Shared non-unit weights now reach an engine that fits the wrong robust objective, R4-C1: Stateful homodyne plotting — Each measured-q replacement keeps the previous cached model alive (+4 more)

### Community 455 - "test_out_of_core_route.py"
Cohesion: 0.10
Nodes (17): Bugs, Codex review — shard optrest, Files read, Non-blocking notes, optrest-B2: A successful retry produces a failed public result, optrest-B3: Out-of-core reduced chi-squared counts configured frozen scaling even when scaling was fitted, optrest-B4: Standard recovery discards the configured parameter scales before the first solve, optrest-B5: Homodyne strategy routing ignores `NLSQ_MEMORY_FRACTION` (+9 more)

### Community 461 - "performance"
Cohesion: 0.14
Nodes (14): cpu_threads, enable_jit, vectorization_level, cache_strategy, chunk_size, enable_caching, enabled, max_memory_usage_gb (+6 more)

### Community 462 - "output"
Cohesion: 0.14
Nodes (14): csv, hdf5, json, npz, yaml, output, base_directory, compress_hdf5 (+6 more)

### Community 463 - "phi_filtering"
Cohesion: 0.14
Nodes (14): phi_filtering, algorithm, enabled, fallback_to_all_angles, quality_control, target_ranges, tolerance, max_angle_spread (+6 more)

### Community 464 - "performance"
Cohesion: 0.14
Nodes (14): cpu_threads, enable_jit, vectorization_level, cache_strategy, chunk_size, enable_caching, enabled, max_memory_usage_gb (+6 more)

### Community 465 - "output"
Cohesion: 0.14
Nodes (14): csv, hdf5, json, npz, yaml, output, base_directory, compress_hdf5 (+6 more)

### Community 466 - "phi_filtering"
Cohesion: 0.14
Nodes (14): phi_filtering, algorithm, enabled, fallback_to_all_angles, quality_control, target_ranges, tolerance, max_angle_spread (+6 more)

### Community 467 - "performance"
Cohesion: 0.14
Nodes (14): cpu_threads, enable_jit, vectorization_level, cache_strategy, chunk_size, enable_caching, enabled, max_memory_usage_gb (+6 more)

### Community 468 - "output"
Cohesion: 0.14
Nodes (14): csv, hdf5, json, npz, yaml, output, base_directory, compress_hdf5 (+6 more)

### Community 469 - "phi_filtering"
Cohesion: 0.14
Nodes (14): phi_filtering, algorithm, enabled, fallback_to_all_angles, quality_control, target_ranges, tolerance, max_angle_spread (+6 more)

### Community 470 - "performance"
Cohesion: 0.14
Nodes (14): cpu_threads, enable_jit, vectorization_level, cache_strategy, chunk_size, enable_caching, enabled, max_memory_usage_gb (+6 more)

### Community 471 - "output"
Cohesion: 0.14
Nodes (14): csv, hdf5, json, npz, yaml, output, base_directory, compress_hdf5 (+6 more)

### Community 474 - "phi_filtering"
Cohesion: 0.14
Nodes (14): phi_filtering, algorithm, enabled, fallback_to_all_angles, quality_control, target_ranges, tolerance, max_angle_spread (+6 more)

### Community 475 - "test_heterodyne_config_bounds_override.py"
Cohesion: 0.17
Nodes (7): _resolved_bounds(), test_initial_parameters_flat_values_alias_canonical_collision_rejected(), test_parameter_space_bounds_alias_canonical_collision_rejected(), test_parameter_space_bounds_default_unchanged_without_override(), test_parameter_space_bounds_override_phi0_het_translation(), test_parameter_space_bounds_override_v_beta_negative(), test_parameter_space_bounds_unknown_name_is_skipped()

### Community 476 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, complexity, config_version, description, generated_at, generated_by, parameter_count, physics_model (+1 more)

### Community 480 - "Decision round: close the 3 remaining open items"
Cohesion: 0.17
Nodes (11): <codex|agy> position — round <n>, Context, core-B3, data-B9, Decision round: close the 3 remaining open items, Disagreements with Claude's recommendation, Item 2 — data-B9: `experimental_data.cache_directory` is ignored, Item 3 — core-B3: the opt-in adapter path clips times beyond `10000 * dt` (+3 more)

### Community 481 - "Findings"
Cohesion: 0.17
Nodes (11): Closed-as-intended items you dispute, codex review of PR #81, F1: The in-memory engine treats a fully masked linear fit as unweighted, F2: Large masked fits report finite reduced chi-squared at zero data DOF, F3: Duplicate phi rows pick up another measurement's mask on both large-data paths, F4: Hybrid-streaming reduced chi-squared includes the error of masked pixels, F5: Legacy regularization settings can override the Layer 3 off switch, F6: Programmatic NumPy frame counts lose the tail-clamp metadata (+3 more)

### Community 482 - "Installation Guide"
Cohesion: 0.18
Nodes (12): Data-Layer Provenance Version, Releasing Guide, Package Version, PyPI Trusted Publishing (OIDC), Release Steps, release.yml Publishing Workflow, Version Consistency Table, From-Source Development Install (+4 more)

### Community 485 - "experimental_data"
Cohesion: 0.17
Nodes (12): experimental_data, cache_compression, cache_filename_template, data_file_name, data_folder_path, data_type, exchange_key, file_format (+4 more)

### Community 486 - "experimental_data"
Cohesion: 0.17
Nodes (12): experimental_data, cache_compression, cache_filename_template, data_file_name, data_folder_path, data_type, exchange_key, file_format (+4 more)

### Community 487 - "experimental_data"
Cohesion: 0.17
Nodes (12): experimental_data, cache_compression, cache_filename_template, data_file_name, data_folder_path, data_type, exchange_key, file_format (+4 more)

### Community 488 - "experimental_data"
Cohesion: 0.17
Nodes (12): experimental_data, cache_compression, cache_filename_template, data_file_name, data_folder_path, data_type, exchange_key, file_format (+4 more)

### Community 490 - "ValueError"
Cohesion: 0.06
Nodes (20): _broken_result(), test_diagnose_error_convergence_failure_perturbs_zero_param(), test_diagnose_error_convergence_failure_retry_perturbs_zero_param(), test_diagnose_error_matches_nlsq_not_finite_message(), test_diagnose_error_numerical_instability_infinite_bounds_stays_finite(), test_diagnose_error_numerical_instability_resets_to_bounds_center(), test_diagnose_error_unknown_error_perturbs_zero_param(), test_analyze_parameter_sensitivity_failure_returns_empty() (+12 more)

### Community 491 - "v0.1.0 Initial Release"
Cohesion: 0.18
Nodes (11): Anti-Degeneracy Controller, Command-Line Interface, Desktop Analysis Workbench (GUI), Heterodyne Config Bounds Overrides, Heterodyne Joint Global Escapes, Heterodyne Streaming Anti-Degeneracy, NLSQ Engine Split, Unified Public API (7 Lazy Exports) (+3 more)

### Community 493 - "_get_dataset_size"
Cohesion: 0.20
Nodes (4): test_get_dataset_size(), test_get_phi_from_data(), _get_dataset_size(), _get_phi_from_data()

### Community 494 - "xpcsjax_laminar_flow.yaml"
Cohesion: 0.18
Nodes (10): analysis_mode, analysis_settings, model_description, parameters, per_angle_scaling, physics, type, noise_estimation (+2 more)

### Community 495 - "multi_start"
Cohesion: 0.18
Nodes (11): degeneracy_threshold, enable, n_starts, n_workers, refine_top_k, refinement_ftol, sampling_strategy, screen_keep_fraction (+3 more)

### Community 496 - "streaming"
Cohesion: 0.18
Nodes (11): streaming, adaptive_batching, batch_size, checkpoint_dir, checkpoint_frequency, enable_checkpoints, enable_fault_tolerance, keep_last_checkpoints (+3 more)

### Community 497 - "xpcsjax_static_anisotropic.yaml"
Cohesion: 0.18
Nodes (10): analysis_mode, analysis_settings, model_description, parameters, per_angle_scaling, physics, type, noise_estimation (+2 more)

### Community 498 - "multi_start"
Cohesion: 0.18
Nodes (11): degeneracy_threshold, enable, n_starts, n_workers, refine_top_k, refinement_ftol, sampling_strategy, screen_keep_fraction (+3 more)

### Community 499 - "streaming"
Cohesion: 0.18
Nodes (11): streaming, adaptive_batching, batch_size, checkpoint_dir, checkpoint_frequency, enable_checkpoints, enable_fault_tolerance, keep_last_checkpoints (+3 more)

### Community 500 - "xpcsjax_static_isotropic.yaml"
Cohesion: 0.18
Nodes (10): analysis_mode, analysis_settings, model_description, parameters, per_angle_scaling, physics, type, noise_estimation (+2 more)

### Community 501 - "multi_start"
Cohesion: 0.18
Nodes (11): degeneracy_threshold, enable, n_starts, n_workers, refine_top_k, refinement_ftol, sampling_strategy, screen_keep_fraction (+3 more)

### Community 502 - "streaming"
Cohesion: 0.18
Nodes (11): streaming, adaptive_batching, batch_size, checkpoint_dir, checkpoint_frequency, enable_checkpoints, enable_fault_tolerance, keep_last_checkpoints (+3 more)

### Community 503 - "xpcsjax_two_component.yaml"
Cohesion: 0.18
Nodes (10): analysis_mode, analysis_settings, model_description, parameters, per_angle_scaling, physics, type, noise_estimation (+2 more)

### Community 504 - "multi_start"
Cohesion: 0.18
Nodes (11): degeneracy_threshold, enable, n_starts, n_workers, refine_top_k, refinement_ftol, sampling_strategy, screen_keep_fraction (+3 more)

### Community 505 - "streaming"
Cohesion: 0.18
Nodes (11): streaming, adaptive_batching, batch_size, checkpoint_dir, checkpoint_frequency, enable_checkpoints, enable_fault_tolerance, keep_last_checkpoints (+3 more)

### Community 509 - "Findings"
Cohesion: 0.18
Nodes (10): agy review of PR #81, Closed-as-intended items you dispute, F2: `heterodyne_stratified_ls` and `heterodyne_hybrid_streaming` return finite `reduced_chi_squared` at zero data DOF, and stratified-LS omits `covariance_is_placeholder=True` when adapter covariance becomes NaN, F3: `heterodyne_hybrid_streaming` excludes zero-weight rows from `info["n_data_points"]` while `info["ssr"]` still sums masked rows unweighted, F4: `fit_two_component_via_engine` silently accepts all-zero (and non-positive) uniform weights under `loss="linear"`, solving unweighted, F5: Layer 3 off-switch half-fixes: `create_nlsq_callbacks` still emits `group_variance_indices` when `enable: false`, and streaming layer builders ignore `mode: "none"`, F6: Pre-binding `_applied_start_frame`/`_applied_end_frame = None` breaks `save_to_cache`'s `getattr` fallback, F7: `test_fallback_chain_names_the_unsupported_streaming_tier` inspects source text instead of testing runtime behavior, and `fallback_chain.py:328` skips the guard for `config=None` (+2 more)

### Community 510 - "Closing check: fixes for your PR #81 review"
Cohesion: 0.20
Nodes (9): Closing check: fixes for your PR #81 review, <codex|agy> closing check of PR #81, Exact paths (do NOT search the filesystem), New issues introduced by the fixes, Per-finding status, Questions, Rules, Verdict (+1 more)

### Community 511 - "Review request: xpcsjax PR #81 (draft)"
Cohesion: 0.33
Nodes (5): Closed-as-intended items you dispute, <codex|agy> review of PR #81, F1: <one-line claim>, Findings, Verdict

### Community 512 - "FakeHandle"
Cohesion: 0.12
Nodes (10): FakeHandle, test_close_event_calls_queue_shutdown(), test_close_project_clears_folder_name_from_sidebar(), test_close_project_stops_active_and_pending_runs(), test_create_project_shows_folder_name_in_sidebar(), test_repeated_runs_get_distinct_output_dirs(), test_show_result_renders_summary(), test_status_and_log_slots_update_widgets() (+2 more)

### Community 513 - "analyzer_parameters"
Cohesion: 0.20
Nodes (10): analyzer_parameters, dt, end_frame, frame_step, geometry, scattering, start_frame, time_unit (+2 more)

### Community 514 - "quality_control"
Cohesion: 0.20
Nodes (10): check_angle_distribution, min_angles_per_range, validate_angle_coverage, check_correlation_consistency, detect_anomalous_angles, validate_angle_dependencies, quality_control, angle_quality (+2 more)

### Community 515 - "analyzer_parameters"
Cohesion: 0.20
Nodes (10): analyzer_parameters, dt, end_frame, frame_step, geometry, scattering, start_frame, time_unit (+2 more)

### Community 516 - "quality_control"
Cohesion: 0.20
Nodes (10): check_angle_distribution, min_angles_per_range, validate_angle_coverage, check_correlation_consistency, detect_anomalous_angles, validate_angle_dependencies, quality_control, angle_quality (+2 more)

### Community 517 - "analyzer_parameters"
Cohesion: 0.20
Nodes (10): analyzer_parameters, dt, end_frame, frame_step, geometry, scattering, start_frame, time_unit (+2 more)

### Community 518 - "quality_control"
Cohesion: 0.20
Nodes (10): check_angle_distribution, min_angles_per_range, validate_angle_coverage, check_correlation_consistency, detect_anomalous_angles, validate_angle_dependencies, quality_control, angle_quality (+2 more)

### Community 519 - "analyzer_parameters"
Cohesion: 0.20
Nodes (10): analyzer_parameters, dt, end_frame, frame_step, geometry, scattering, start_frame, time_unit (+2 more)

### Community 520 - "quality_control"
Cohesion: 0.20
Nodes (10): check_angle_distribution, min_angles_per_range, validate_angle_coverage, check_correlation_consistency, detect_anomalous_angles, validate_angle_dependencies, quality_control, angle_quality (+2 more)

### Community 521 - "initial_parameters"
Cohesion: 0.20
Nodes (10): initial_parameters, active_parameters, fixed_parameters, parameter_names, per_angle_scaling, tied_parameters, units, values (+2 more)

### Community 525 - "rasterize"
Cohesion: 0.26
Nodes (6): test_infinities_are_sanitized_for_display(), test_large_array_is_downsampled_within_max_dim(), test_nan_is_preserved_for_masked_regions(), test_small_array_is_returned_unchanged(), test_thin_array_does_not_collapse_to_zero_dim(), rasterize()

### Community 526 - "test_chunked_jacfwd.py"
Cohesion: 0.21
Nodes (4): _nonlinear_residual(), test_chunked_jacfwd_default_block_is_identical(), test_chunked_jacfwd_matches_jacfwd(), test_chunked_jacfwd_works_through_jit()

### Community 528 - "heterodyne_logging.py"
Cohesion: 0.09
Nodes (10): _layer_state(), log_anti_degeneracy_defense(), log_configured_layers_preamble(), log_fit_start(), log_optimization_results(), log_physical_parameters(), log_strategy_selection(), log_stratification_diagnostics() (+2 more)

### Community 529 - "agy position — round 2"
Cohesion: 0.17
Nodes (8): agy position — round 2, Anything the plan missed, core-B3 open point (check inside model_func): DROP — All in-package adapter calls already pass through `_build_model_function` outside any JAX trace, so checking there closes the bug with minimal risk and avoids putting eager array checks inside a potentially traced closure. Direct callers of `get_or_create_model` can rely on a brief docstring note pointing to the `[0, (GRID_SIZE - 1) * dt]` fallback bound., core-B3: RATIFY, data-B9: RATIFY, R4-A3: RATIFY, test_classify_quality_flag_maps_band_to_qualityflag(), classify_quality_flag()

### Community 530 - "test_aps_u_alloc_guard.py"
Cohesion: 0.31
Nodes (6): _FakeDS, test_guard_noop_when_no_valid_bins_in_range(), test_guard_passes_for_legitimate_small_input(), test_guard_rejects_budget_exceeded(), test_guard_rejects_non_square(), test_guard_rejects_oversized_frame_count()

### Community 531 - "validation"
Cohesion: 0.22
Nodes (9): min_angle_count, require_multiple_angles, validate_angle_ranges, validation, angle_validation, check_file_existence, check_mode_compatibility, strict_mode (+1 more)

### Community 532 - "initial_parameters"
Cohesion: 0.22
Nodes (9): initial_parameters, active_parameters, fixed_parameters, parameter_names, per_angle_scaling, units, values, contrast (+1 more)

### Community 533 - "test_streaming_resolves_canonical_mode"
Cohesion: 0.20
Nodes (4): test_diagnostics_stamps_canonical_token_and_n_optimized(), test_meta_has_no_reparam_keys(), test_pointwise_model_individual_layout(), test_streaming_resolves_canonical_mode()

### Community 534 - "stratification"
Cohesion: 0.22
Nodes (9): stratification, check_memory_safety, collect_diagnostics, enabled, force_sequential_fallback, log_diagnostics, max_imbalance_ratio, target_chunk_size (+1 more)

### Community 535 - "validation"
Cohesion: 0.22
Nodes (9): min_angle_count, require_multiple_angles, validate_angle_ranges, validation, angle_validation, check_file_existence, check_mode_compatibility, strict_mode (+1 more)

### Community 536 - "initial_parameters"
Cohesion: 0.22
Nodes (9): initial_parameters, active_parameters, fixed_parameters, parameter_names, per_angle_scaling, units, values, contrast (+1 more)

### Community 538 - "stratification"
Cohesion: 0.22
Nodes (9): stratification, check_memory_safety, collect_diagnostics, enabled, force_sequential_fallback, log_diagnostics, max_imbalance_ratio, target_chunk_size (+1 more)

### Community 539 - "validation"
Cohesion: 0.22
Nodes (9): min_angle_count, require_multiple_angles, validate_angle_ranges, validation, angle_validation, check_file_existence, check_mode_compatibility, strict_mode (+1 more)

### Community 540 - "initial_parameters"
Cohesion: 0.22
Nodes (9): initial_parameters, active_parameters, fixed_parameters, parameter_names, per_angle_scaling, units, values, contrast (+1 more)

### Community 542 - "stratification"
Cohesion: 0.22
Nodes (9): stratification, check_memory_safety, collect_diagnostics, enabled, force_sequential_fallback, log_diagnostics, max_imbalance_ratio, target_chunk_size (+1 more)

### Community 543 - "validation"
Cohesion: 0.22
Nodes (9): min_angle_count, require_multiple_angles, validate_angle_ranges, validation, angle_validation, check_file_existence, check_mode_compatibility, strict_mode (+1 more)

### Community 544 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, complexity, config_version, description, generated_at, generated_by, parameter_count, physics_model (+1 more)

### Community 545 - "StreamingExecutor"
Cohesion: 0.25
Nodes (3): optstrat-B12: The public streaming executor fails on partial batches, test_streaming_executor_calls_real_fit_api(), StreamingExecutor

### Community 546 - "color_scale"
Cohesion: 0.29
Nodes (7): fixed_max, fixed_min, mode, percentile_max, percentile_min, pin_legacy_range, color_scale

### Community 547 - "metadata"
Cohesion: 0.22
Nodes (9): metadata, complexity, config_version, description, generated_at, generated_by, parameter_count, physics_model (+1 more)

### Community 551 - "regularization"
Cohesion: 0.25
Nodes (8): regularization, auto_tune_lambda, enable, lambda, max_cv, mode, target_contribution, target_cv

### Community 552 - "regularization"
Cohesion: 0.25
Nodes (8): regularization, auto_tune_lambda, enable, lambda, max_cv, mode, target_contribution, target_cv

### Community 554 - "codex position — round 2"
Cohesion: 0.29
Nodes (6): Anything the plan missed, codex position — round 2, core-B3 open point (check inside model_func): DROP — Validate both time coordinates once at the adapter boundary, where their meaning and effective `dt` are known. A check tied to cache misses and concrete NumPy arrays would leave traced calls uncovered., core-B3: RATIFY, data-B9: RATIFY, R4-A3: RATIFY

### Community 556 - "bounds"
Cohesion: 0.29
Nodes (7): max, min, name, unit, parameter_space, bounds, model

### Community 558 - "quality_validation"
Cohesion: 0.29
Nodes (7): quality_validation, bounds_tolerance, enable, reduced_chi_squared_threshold, warn_on_bounds_hit, warn_on_convergence_failure, warn_on_max_restarts

### Community 559 - "bounds"
Cohesion: 0.29
Nodes (7): max, min, name, unit, parameter_space, bounds, model

### Community 561 - "quality_validation"
Cohesion: 0.29
Nodes (7): quality_validation, bounds_tolerance, enable, reduced_chi_squared_threshold, warn_on_bounds_hit, warn_on_convergence_failure, warn_on_max_restarts

### Community 562 - "bounds"
Cohesion: 0.29
Nodes (7): max, min, name, unit, parameter_space, bounds, model

### Community 564 - "quality_validation"
Cohesion: 0.29
Nodes (7): quality_validation, bounds_tolerance, enable, reduced_chi_squared_threshold, warn_on_bounds_hit, warn_on_convergence_failure, warn_on_max_restarts

### Community 565 - "bounds"
Cohesion: 0.29
Nodes (7): max, min, name, unit, parameter_space, bounds, model

### Community 566 - "color_scale"
Cohesion: 0.29
Nodes (7): fixed_max, fixed_min, mode, percentile_max, percentile_min, pin_legacy_range, color_scale

### Community 567 - "quality_validation"
Cohesion: 0.29
Nodes (7): quality_validation, bounds_tolerance, enable, reduced_chi_squared_threshold, warn_on_bounds_hit, warn_on_convergence_failure, warn_on_max_restarts

### Community 569 - "ConfigManager"
Cohesion: 0.33
Nodes (6): AnalysisMode (4 modes), ConfigManager, NLSQConfig, parameter_registry, ParameterManager, service.config

### Community 570 - "Codex closing report"
Cohesion: 0.33
Nodes (5): Codex closing report, Fix check, Handoffs, Regressions, Verdict

### Community 571 - "codex position — round 1"
Cohesion: 0.33
Nodes (5): codex position — round 1, core-B3, data-B9, Disagreements with Claude's recommendation, R4-A3

### Community 572 - "codex closing check of PR #81"
Cohesion: 0.33
Nodes (5): codex closing check of PR #81, F1: An explicit null now turns on Layer 3 in heterodyne hybrid streaming, New issues introduced by the fixes, Per-finding status, Verdict

### Community 573 - "detect_shell_type"
Cohesion: 0.15
Nodes (8): test_detect_shell_type_from_env(), test_detect_shell_type_unknown_via_psutil_match(), name(), test_detect_shell_type_unknown_when_probe_raises(), test_install_shell_completion_routes(), test_validate_xla_mode(), detect_shell_type(), _validate_xla_mode()

### Community 574 - "color_scale"
Cohesion: 0.29
Nodes (7): fixed_max, fixed_min, mode, percentile_max, percentile_min, pin_legacy_range, color_scale

### Community 577 - "system_validator.py"
Cohesion: 0.21
Nodes (4): test_main_returns_one_on_error(), build_parser(), main(), Severity

### Community 578 - "hierarchical"
Cohesion: 0.33
Nodes (6): hierarchical, enable, max_outer_iterations, outer_tolerance, per_angle_max_iterations, physical_max_iterations

### Community 579 - "shear_weighting"
Cohesion: 0.33
Nodes (6): shear_weighting, alpha, enable, min_weight, normalize, update_frequency

### Community 580 - "matplotlib"
Cohesion: 0.33
Nodes (6): interpolation, savefig_kwargs, use_tight_layout, matplotlib, bbox_inches, pad_inches

### Community 581 - "matplotlib"
Cohesion: 0.33
Nodes (6): interpolation, savefig_kwargs, use_tight_layout, matplotlib, bbox_inches, pad_inches

### Community 582 - "matplotlib"
Cohesion: 0.33
Nodes (6): interpolation, savefig_kwargs, use_tight_layout, matplotlib, bbox_inches, pad_inches

### Community 584 - "test_templates_no_fourier_keys.py"
Cohesion: 0.40
Nodes (3): test_template_has_no_live_removed_keys(), _walk(), test_template_still_loads()

### Community 588 - "diagnostics"
Cohesion: 0.40
Nodes (5): check_gradients, enabled, log_jacobian_norms, sample_size, diagnostics

### Community 589 - "diagnostics"
Cohesion: 0.40
Nodes (5): check_gradients, enabled, log_jacobian_norms, sample_size, diagnostics

### Community 590 - "diagnostics"
Cohesion: 0.40
Nodes (5): check_gradients, enabled, log_jacobian_norms, sample_size, diagnostics

## Knowledge Gaps
- **187 isolated node(s):** `g1/g2/c2 correlation functions`, `Bradbury2018 - JAX composable transformations`, `Duri2005 - Time-resolved-correlation measurements`, `Hansen2016 - CMA evolution strategy tutorial`, `Lumma2000 - Area detector based photon correlation` (+182 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 5646 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **145 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `logging.py` to `fit_heterodyne_stratified_least_squares`, `nlsq_plots.py`, `test_logging.py`, `fit_two_component_via_engine`, `test_heterodyne_fixed_parameters.py`, `data/validation.py`, `HeterodyneModel`, `heterodyne_logging.py`, `NLSQWrapper`, `StratifiedResidualFunction`, `types`, `main.py`, `jax_backend.py`, `core.py`, `test_transforms.py`, `apply_angle_filtering_for_optimization`, `parameter_registry.py`, `XPCSDataFilter`, `diagonal_correction.py`, `test_output_resolution.py`, `PreprocessingPipeline`, `GradientCollapseMonitor`, `commands.py`, `cmaes_wrapper.py`, `nlsq/__init__.py`, `manager.py`, `wrapper.py`, `test_multistart.py`, `cpu.py`, `test_parameter_utils.py`, `sequential.py`, `optimization_runner.py`, `DatasetOptimizer`, `quality_controller.py`, `test_heterodyne_memory_adapter.py`, `chunking.py`, `ResultBuilder`, `test_heterodyne_data_prep.py`, `models.py`, `heterodyne_core.py`, `HomodyneModel`, `service/persist.py`, `test_save_path_validation.py`, `test_fit_quality.py`, `load_dataset`, `test_codex_cli.py`, `jit`, `ShearSensitivityWeighting`, `compute_g2_scaled`, `multistart.py`, `OptimizationExecutor`, `test_memory_manager_logging.py`, `PhysicsFactors`, `main_window.py`, `xpcsjax/data/__init__.py`, `NLSQResult`, `simulated.py`, `json_serializer`, `test_codex_optstrat.py`, `heterodyne_scaling_utils.py`, `nlsq/validation.py`, `numpy`, `_fit_nlsq_heterodyne`, `physics.py`, `XPCSDataFormatError`, `run_fit`, `AdaptiveRegularizer`, `_logger_that_raises_on_log`, `phi_filtering.py`, `anti_degeneracy_controller.py`, `heterodyne_parameter_space.py`, `service/config.py`, `generate_plots`, `xpcs_loader.py`, `test_pr81_review.py`, `OptimizationStrategy`, `ContextFilter`, `physics_validators.py`, `test_worker_functions_in_process`, `.__init__`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 125 inferred relationships involving `NLSQConfig` (e.g. with `_cfg()` and `_diag()`) actually correct?**
  _`NLSQConfig` has 125 INFERRED edges - model-reasoned connections that need verification._
- **What connects `g1/g2/c2 correlation functions`, `Bradbury2018 - JAX composable transformations`, `Duri2005 - Time-resolved-correlation measurements` to the rest of the system?**
  _187 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `make_synthetic_two_component` be split into smaller, more focused modules?**
  _Cohesion score 0.02631578947368421 - nodes in this community are weakly interconnected._
- **Why does `ConfigManager` connect `ConfigManager` to `make_synthetic_two_component`, `_make_synthetic_heterodyne`, `.from_config`, `fit_heterodyne_stratified_least_squares`, `test_heterodyne_constant_mode.py`, `service/config.py`, `generate_plots`, `OptimizationResult`, `fit_two_component_via_engine`, `_build_laminar_stratified_data`, `HeterodyneModel`, `compute_g2_scaled`, `fit_nlsq`, `CMAESResult`, `test_codex_gui.py`, `core.py`, `NLSQConfig`, `test_transforms.py`, `XPCSDataFilter`, `AnalysisMode`, `simulated.py`, `test_codex_data.py`, `test_output_resolution.py`, `commands.py`, `cmaes_wrapper.py`, `OptimizationStrategy`, `test_get_model.py`, `test_perf_regression.py`, `manager.py`, `test_two_component_smoke.py`, `wrapper.py`, `fit_nlsq_jax`, `NLSQAdapter`, `optimization_runner.py`, `test_config_debug_null_nonfinite.py`, `test_heterodyne_stratification_config.py`, `test_hybrid_streaming_retry.py`, `test_config_relative_data_paths.py`, `test_templates_no_fourier_keys.py`, `test_laminar_no_worse_phase6.py`, `test_homodyne_engine_preservation.py`, `test_pr80_review.py`, `Decision round: close the 3 remaining open items`, `run_fit`, `Bugs`, `HomodyneModel`, `fit_with_out_of_core_accumulation`, `test_static_individual_invariant.py`, `service/persist.py`, `test_fit_quality.py`, `load_dataset`, `test_codex_cli.py`, `Findings`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `ConfigManager` (e.g. with `data-B9: The loader ignores the shipped cache enable flag and cache directory` and `optstrat-B10: Generic hybrid streaming rejects the wrapper’s normal config type`) actually correct?**
  _`ConfigManager` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Should `generate_nlsq_plots` be split into smaller, more focused modules?**
  _Cohesion score 0.12315270935960591 - nodes in this community are weakly interconnected._