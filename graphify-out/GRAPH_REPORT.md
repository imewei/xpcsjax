# Graph Report - xpcsjax  (2026-08-06)

## Corpus Check
- 526 files · ~644,239 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 9734 nodes · 19834 edges · 640 communities (357 shown, 283 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 767 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ff26bf55`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- ParameterManager
- heterodyne_core.py
- logging.py
- make_synthetic_two_component
- XPCSDataLoader
- NLSQConfig
- DataQualityController
- xpcsjax/data/__init__.py
- _chunked_jacfwd_dense
- PerformanceEngine
- NLSQResult
- _resolve_effective_mode
- fit_two_component_via_engine
- ConfigManager
- TwoComponentModel
- MainWindow
- NLSQWrapper
- data/config.py
- _build_laminar_fit
- heterodyne_constant_mode.py
- DataQualityReport
- AnalysisMode
- select_nlsq_strategy
- parallel_accumulator.py
- StratifiedResidualFunction
- get_logger
- ParameterManager
- ParameterIndexMapper
- Project
- ResultSummary
- jax_backend.py
- test_shear_weighting.py
- cmaes_wrapper.py
- test_heterodyne_hybrid_streaming.py
- core.py
- AdaptiveRegularizer
- test_heterodyne_tied_result_assembly.py
- ParameterSpace
- diagonal_correction.py
- multistart.py
- test_post_install.py
- log_exception
- FitOverrides
- AntiDegeneracyController
- _write_config
- test_heterodyne_config.py
- HeterodynePointEvaluator
- heterodyne_logging.py
- simulated.py
- test_gui_redesign.py
- test_prepare_sigma_data_phi_index.py
- ResultBuilder
- normalize_angle_to_symmetric_range
- test_config_null_sections.py
- test_result_presenter.py
- DatasetOptimizer
- fit_heterodyne_stratified_least_squares
- PerAngleScalingPlan
- heterodyne_physics_kernel.py
- test_low_level_plots.py
- optimization/test_debug_audit_2026_07_22.py
- nlsq_plots.py
- classify_option
- heterodyne_hybrid_streaming.py
- events.py
- sequential.py
- XPCSDataFilter
- test_gui_jax_free.py
- save_results_npz
- test_heterodyne_physics_validators.py
- executors.py
- build_workbench
- result_presenter.py
- ParameterSpace
- configure_logging
- optimization/test_validation.py
- _save_fig
- _PhysicsModelProtocol
- fit_with_stratified_hybrid_streaming
- ProjectSidebar
- HierarchicalOptimizer
- FitJob
- test_uninstall_scripts.py
- AsyncWriter
- resolve_per_angle_mode
- _StubParamManager
- ParameterRegistry
- test_recovery_and_numerical.py
- test_engine_heterodyne_fit_parity.py
- _PhaseRecord
- NLSQAdapter
- NLSQConfig
- MultiLevelCache
- helpers.py
- _build_joint_problem
- test_heterodyne_stratified_anti_degeneracy_parity.py
- test_parameter_utils.py
- test_review_regressions.py
- test_heterodyne_result_builder.py
- run_worker
- test_heterodyne_memory_adapter.py
- .from_config
- test_validation_branches.py
- .get_active_parameters
- test_gui_debug_2026_06_21.py
- test_output_resolution.py
- MemoryPressureMonitor
- test_transforms.py
- generate_nlsq_plots
- ShearSensitivityWeighting
- test_artifacts.py
- log_heterodyne_completion
- test_quality_controller_smoke.py
- _fit_nlsq_heterodyne
- test_static_individual_invariant.py
- compute_c2_heterodyne
- validate_xpcs_data
- AdvancedMemoryManager
- RunController
- test_heterodyne_data_prep.py
- test_jacobian.py
- json_safe
- MonkeyPatch
- plots_view.py
- config_generator.py
- test_plot_dispatch_logging.py
- test_memory_manager_cache_gating.py
- DataInspectDialog
- test_strategy_chunking.py
- ndarray
- .fit
- test_strategy_executors.py
- ConfigTextEditorDialog
- .n_optimized
- test_heterodyne_tied_parameters.py
- test_cache_safety.py
- load_project
- test_heterodyne_cmaes_warmstart_success_gate.py
- ndarray
- AnalysisSummaryLogger
- test_cpu_config.py
- parameter_names.py
- test_memory_manager_logging.py
- ProjectDialogHandler
- _NullLogger
- test_escape_disabled_hint.py
- test_hybrid_streaming_retry.py
- test_homodyne_engine_preservation.py
- test_runtime_shell.py
- FitQueueController
- _unpack_result_params
- maps.py
- .from_optimization_block
- test_cmaes_trigger.py
- get_adaptive_memory_threshold
- RecoveryStrategyApplicator
- StratifiedResidualFunctionJIT
- ._write_json
- make_cfgmgr_and_data
- test_debug_audit_2026_07_23_sigma_weighting.py
- test_gradient_diagnostics.py
- test_heterodyne_tied_residuals.py
- test_parallel_accumulator.py
- test_phase5_standard_resolver.py
- execute_optimization_with_fallback
- ndarray
- validate_backend
- data_prep.py
- References and Citations
- HomodyneModel
- test_cache_loader_security.py
- _Any
- test_system_validator.py
- NLSQ vs xpcsjax ownership split contract
- XpcsDataset
- physics_nlsq.py
- test_fit_queue.py
- service/config.py
- test_main_window.py
- json_serializer
- fit_computation.py
- ParameterInfo
- HeterodyneModel
- Any
- test_plots_view.py
- test_quality_gate_fixes.py
- ValidationResult
- save_nlsq_npz_file
- compute_diagonal_overlay_stats
- heterodyne_parameter_space.py
- fit_with_out_of_core_accumulation
- BatchStatistics
- load_viz_bundle
- build_failed_result
- test_debug_audit_2026_06_18.py
- MemoryStats
- performance_engine.py
- PhysicsFactors
- test_no_pickle_loads.py
- TestHeterodyneComputeResidual
- test_diagnostics_map.py
- test_anti_degeneracy_layers.py
- test_debug_audit_2026_08_05.py
- test_heterodyne_cmaes_warmstart_auto_skip.py
- _version_at_least
- .__init__
- test_stratified_max_iter_grading.py
- xla_config.py
- test_heterodyne_grouped_coercion.py
- test_cache_no_pickle_exec.py
- result_builder.py
- test_layer5_gating.py
- test_debug_audit_regressions.py
- validators.py
- test_codex_review_fixes.py
- _logger_that_raises_on_log
- test_gradient_monitor.py
- post_install.py
- test_persist.py
- test_memory_routing.py
- test_config_unwrap.py
- test_laminar_streaming_diag.py
- test_laminar_no_worse_phase6.py
- load_dataset
- _validate_loaded_arrays
- logged_errors
- completion.sh
- test_debug_audit_2026_07_23_negative_correlation_repair.py
- ndarray
- test_laminar_execute_layers.py
- NLSQOptimizationError
- Fit pipeline (load -> optimise -> save -> plot)
- test_post_install_fish.py
- ResultPresenter
- test_validation_crash_coverage.py
- fit_nlsq
- OOCSharedArrays
- test_phase5_model_function_modes.py
- test_model_adapter.py
- test_docs_structure.py
- save_nlsq_json_files
- viz/conftest.py
- Laminar Anti-Degeneracy 5-Layer Defense
- .get_cache_stats
- test_cache_q_validation.py
- test_logging_wiring.py
- test_heterodyne_config_bounds_override.py
- heterodyne_parameter_manager.py
- TestChiSquaredZeroSigma
- test_jax_backend.py
- test_debug_audit_2026_07_23_diagonal_skip.py
- test_anti_degeneracy_transforms.py
- test_l4_callback_layout.py
- FitQualityConfig
- validate_no_nan_inf
- run_validation
- InputValidator
- test_config_jax_free.py
- XPCSDataFormatError
- test_run_controller.py
- PhysicsFactors
- heterodyne_scaling_utils.py
- TestNoScipyLeastSquares
- test_lazy_imports.py
- TestContextFilterOnLogger
- TestTypeBoundary
- PerAngleScaling
- .validate_parameters
- fit_nlsq(data, config.yaml)
- AntiDegeneracyController (5-layer composed controller)
- get_or_create_fitter
- main_window.py
- test_stratified_ls_averaged_covariance_transform.py
- ResultValidator
- test_logging_quality_gate.py
- _ReaderThread
- PointEvaluator
- ._apply_strategy
- .wait_all
- plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2)
- NLSQ Residual Diagnostics Baseline (phi=45deg)
- NLSQ owns trust-region solve; xpcsjax owns strategy
- test_optimization_logging.py
- .__init__
- test_homodyne_models.py
- data/test_debug_audit_2026_06_17.py
- quality_controller.py
- test_freeze_safety.py
- LogRecord
- TestLogPerformanceNeverRaise
- estimate_per_angle_scaling_from_quantile
- ScalingConfig
- get_stratified_chunk_iterator
- xla_config.bash
- Issue tracker: GitHub
- xpcsjax-post-install console script
- fit_with_stratified_least_squares
- Public API Reference
- _build_inputs
- test_phase5_vector_build.py
- test_wrapper_finalize_result.py
- test_adversarial_review_nonfinite_coercion.py
- TimedContext
- plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg)
- Automated structural doc-coverage check, content accuracy stays manual
- Domain Docs
- test_aps_u_empty_selection.py
- test_docs_no_fourier.py
- compute_transport_integral_matrix
- TestPropagateInversion
- adapter.py
- system_validator.py
- _LAZY_EXPORTS table + module __getattr__
- estimate_contrast_offset_from_quantiles
- test_io.py
- test_validation_integrity_logging.py
- nlsq/validation.py
- .compute_g1
- safe_exp
- XPCSJAX Static Anisotropic Template
- User Guide: Data Loading
- tests/conftest.py
- test_fourier_reparam_removed.py
- ndarray
- test_no_removed_per_angle_tokens_in_tests_or_package
- test_l1_rename.py
- _FakeModel
- test_result_accessors.py
- test_install_shell_completion_routes
- test_loader_integration.py
- test_heterodyne_tied_gradient_coupling.py
- xpcsjax Architecture
- xpcsjax Fitting Workflow
- Decision Record: CPU-only Execution
- Lumma et al. 2000 — Two-time correlation matrix estimator
- validate_optimized_params
- log_quantile_scaling
- triage-labels.md
- HomodyneModel (static_* / laminar_flow)
- Lazy public API (_LAZY_EXPORTS, 7 symbols)
- CLI lazy-loading via __getattr__
- Runtime Package API
- conf.py
- Kubo 1966 — The fluctuation-dissipation theorem
- Core Physics Tests
- gui/conftest.py
- isolated_env
- plot_families/__init__.py
- .clear_cache
- controllers/__init__.py
- xpcsjax/gui/__init__.py
- ipc/__init__.py
- project/__init__.py
- views/__init__.py
- main_window_support/__init__.py
- plots/__init__.py
- activation/__init__.py
- xpcsjax/service/__init__.py
- Symmetric Anti-Degeneracy Diagnostics Contract
- L4 Observational Gradient-Collapse Callback
- L5 Shear-Sensitivity Mode Gating
- Lazy Public-API Loading (no eager JAX)
- Memory-Aware NLSQ Strategy Routing
- LHS Multistart Escape
- Logging Is Observational Only (never-raise)
- PointEvaluator Model-Adapter Seam
- PerAngleScaling
- APS-U intermediate-allocation pre-guard
- Cross-platform cache-filename safety guard
- Trusted-cache loader defense-in-depth gates
- SEC-1 no-pickle-execution disk-cache guard
- Frame-count unbounded-allocation guard
- Self-contained synthetic NPZ cache round-trip
- Loader imports clean with no homodyne references
- Live-array-aware JAX cache-clear gating (anti-thrash)
- C044 emergency-cleanup recompile-thrash RCA
- AST scan: no unsafe np.load (allow_pickle) calls
- Optional-component init-failure observability
- DataQualityController three-stage pipeline coverage
- Validator crash-site logging-and-invalidation
- validate_xpcs_data integrity contract (no silent data loss)
- tau0 self-correlation spike not over-flagged
- TestG2ScaledContrastZero
- docs/adr Architecture Decision Records
- CONTEXT-MAP.md
- CONTEXT.md
- /domain-modeling skill (lazy CONTEXT.md/ADR creation)
- Domain glossary vocabulary discipline
- gh CLI (GitHub issue/PR operations)
- GitHub Issues as PRD/issue tracker
- GitHub native issue dependencies (blocked_by)
- PRs-as-request-surface flag (default: no)
- Wayfinder map/child-ticket pattern
- Wayfinder map issue (wayfinder:map label)
- needs-info label
- needs-triage label
- ready-for-agent label
- ready-for-human label
- wontfix label
- async wait_all
- Logging Phase 2
- plot_dispatch
- AdvancedMemoryManager
- AnalysisMode (4 modes)
- AntiDegeneracyController (L1-L5)
- compute_c2_heterodyne (reference + sample)
- compute_g2_scaled (homodyne g1/g2)
- ConfigManager
- DataQualityController
- fit_nlsq (single entry)
- fit_nlsq_multi_phi
- generate_nlsq_plots
- load_xpcs_data
- MainWindow (PySide6 workbench)
- NLSQ adapters (CurveFit, trust-region)
- NLSQConfig
- parameter_registry (single source of truth)
- ParameterManager (bounds, per-angle expansion)
- StratifiedResidualFunctionJIT (engine route)
- CMA-ES escape (seed-pinned, keep-better)
- ConfigManager: resolve analysis_mode + bounds
- fit_nlsq_multi_phi: resolve effective per-angle mode
- hybrid-streaming path
- in-memory engine route (StratifiedResidualFunctionJIT)
- L2 Hierarchical Optimization (all modes)
- L3 Adaptive CV Regularization (all modes)
- L4 Gradient-Collapse Monitor (diagnostic)
- LHS multistart
- load_xpcs_data
- NLSQ CurveFit: trust-region least squares
- OptimizationResult (params, chi2, uncertainties, diagnostics)
- select_nlsq_strategy: memory budget decision
- stratified-LS (double-chunking, >=1M points)
- generate_nlsq_plots / GUI
- JAX must be configured before import (env-var constraint)
- xpcsjax-config console script
- compute_c2_heterodyne
- All Modules (Orphan)
- xpcsjax.optimization.nlsq
- Changelog
- Contributing Guide
- Releasing Guide
- Testing Guide
- Examples Index
- Installation
- Autosummary Module Template
- Bradbury et al. 2018 — JAX: composable transformations
- Interpreting Results Guide
- _jax_jacobian
- Heterodyne Residuals Smoke Baseline
- Golden Snapshot: payload only on regen, no leak
- Illegal States Unrepresentable (quality flag requires finite chi2)
- L5 sentinel split is deliberate two-value design
- Shared _build_joint_problem Helper
- Chunk-Size Upper-Bound Guard (MAX_CHUNK_SIZE)
- CMA-ES fit_with_cmaes Signature Drift Guard
- CMA-ES Warm-Start Auto-Skip
- NaN-Aware Keep-Better Decision
- global_escape Diagnostics Symmetry (per-angle vs joint)
- Per-Iteration Gradient Collapse Callback
- Gradient Collapse Monitor (L4)
- L4 Mechanism: per_iteration vs post_solve_fallback
- Why keep-better must be NaN-aware
- HeterodynePointEvaluator (meshgrid)
- HeterodynePointwiseEvaluator
- OptimizationResult Quality Invariant
- ShearWeightingConfig
- Identity pcov fallback safer than zeros (sqrt(diag) info-preserving)
- OptimizationExecutor Strategy pattern (Standard/Large/Streaming)
- Gradient-Based x_scale Recommender
- _install_xla_bash_activation
- Meshgrid-kept Engine Route (pointwise basin fragility)
- Why L4 Callback Is Strictly Observational
- Why L5 Is laminar_flow-Only
- Characterization Tests
- Config-driven phi_filtering Subsets Data Arrays
- phi_filtering Loader-Wiring Regression Rationale
- Config-driven CLI File Logging
- dispatch_command Logging No-op Rationale
- Fish-shell XLA Activation Under Conda
- Conda VIRTUAL_ENV-unset Fish Path Rationale
- Uninstall Activation-block Scrubbing State Machine
- Typed analysis_mode Property (AnalysisMode enum)
- Config-driven Model Dispatch (get_model)
- Heterodyne Physics-constraint Validators
- Heterodyne 14-param Registry Entries
- beta->v_beta / phi0->phi0_het Rename Rationale
- Canonical Parameter-name Single Source of Truth
- ParameterManager Bounds Derive from Registry
- Inline-bound Drift-bug Rationale
- Heterodyne Velocity/Transport Integral Kernels
- HeterodyneModel PhysicsModelBase Contract
- Pointwise-meshgrid heterodyne kernel parity invariant
- Homodyne model attribute/param naming corrections
- Meshgrid cache collision guard (endpoint-only key hazard)
- Safe sinc continuity invariant
- Zero-sigma chi-squared finiteness guard
- test_loader_smoke.py
- .shutdown
- test_config_analysis_mode.py
- Validation Except-Clause Narrowing
- Why AttributeError/ArithmeticError Must Be Caught
- Validator Crash ERROR-Log + Report Invalidation
- Silent Validator Crash Is A Data-Integrity Bug
- XpcsDataset Typed Load/Fit Boundary
- Additive dict-subclass Schema Not A Break
- Heterodyne optimization.nlsq Config Unwrap
- Nested Keys Silently Ignored Without Unwrap
- Heterodyne Setup-Log Parity With Laminar
- NLSQAdapter Pathological-Residual Contract
- Synthetic two_component Fixture Builder
- Missing-Objective Non-Finite Chi-Squared Guard
- Missing Objective Must Not Mint A Good Fit
- L3 Adaptive Regularization Branch Coverage
- NaN Params Force +inf Penalty To Reject Step
- assemble_anti_degeneracy_diagnostics Contract
- Column-Blocked Covariance Jacobian Parity
- Chunked JVP Caps Post-Solve Tangent-Width Memory
- _reset_log_once
- Per-Angle Scaling Pack/Unpack Roundtrip
- Pointwise Scattered-Eval Seam
- Quality-Gate Fixes
- Recovery Strategy Applicator
- Shear-Sensitivity Weighting (L5)
- Strategy Executors / ExecutionResult
- Residual Across Scaling Modes
- Parameter Status Classification
- Fit Quality Validation
- Symmetric Anti-Degeneracy Activation Keys
- Any
- Laminar stratified-LS execute_layers mechanism oracle
- No-scipy.least_squares architectural guard
- Parameter-registry Hypothesis invariants
- Device-probe observational-logging fallback
- Viz NPZ+JSON artifact serialization tests
- Viz color-limit resolution fallback tests
- CLI Lazy-Import Surface
- CLI Exit Code Contract (0/1/2/130)
- NLSQ-Only CLI (No Bayesian Flags) Rationale
- Parameter Override Precedence (CLI>YAML>Registry)
- --multistart store_const default=None Rationale
- generate_config (String Substitution)
- interactive_builder
- Mode→Template Filename Map
- Exact-Placeholder Substitution Rationale
- validate_config (YAML + ConfigManager)
- _CLI_PARAM_MAP (Flag→Canonical Name)
- Canonical initial_parameters Block Write Rationale
- NLSQ Knob Single-Authority Split Rationale
- Shared Output-Dir Resolver Anti-Drift Rationale
- Config phi_filtering Applied Pre-Fit Rationale
- Defensive --phi Subsetting (All-or-Warn) Rationale
- Tolerant Key Spellings (_pick)
- xjexp / xjsim Console Shims
- XLA_FLAGS Read Once Before First Import Rationale
- JSON Summary vs NPZ Full-Fidelity Split
- _json_safe (Non-Finite Coercion)
- NaN/Inf → None for Valid JSON Rationale
- save_results (JSON/NPZ Dispatcher)
- CLI xla-config Informational-Only Rationale
- Config Package (Registry Single Source of Truth)
- Kernel beta/phi0 → Registry v_beta/phi0_het Alias
- Heterodyne 14-Param Name Constants
- Scaling Params Outside 14-Element Physics Array
- gamma_dot_t0 Naming Bug Fix Rationale
- Canonical Parameter Names & Ordering
- Constraints Complement Hard Bounds
- Homodyne Physics Validators
- xpcsjax — LAMINAR FLOW TEMPLATE
- Config TypedDict Definitions
- Shared Heterodyne Physics Kernel
- Physics Factored from Evaluation Strategy
- Pointwise=Meshgrid Gather Identity Parity
- safe_exp clip=700 float64 bound
- math_primitives canonical math
- intentionally NOT consolidated (safe_sinc/trapezoid)
- single source of truth for shared primitives
- intelligent backend selection (JAX vs NumPy fallback)
- physics.py
- validate_parameters
- jnp.where floor preserves gradients (not jnp.maximum)
- smooth abs sqrt(x^2+eps) gradient stability
- Taylor sinc avoids gradient discontinuity at 0
- ._first
- test_l2_branch_uses_identity_layout_no_permute
- shared types break optimization<->performance_engine cycle
- BatchStatistics Circular Buffer
- Empty-Buffer Optimistic Prior (1.0) Rationale
- NLSQ Exception Hierarchy
- Shared Anti-Degeneracy Diagnostics Assembler
- Symmetric Diagnostics Contract Rationale
- dt required for J(t1,t2) integration; raise not silent 0.1s default
- extract_parameters_from_result (per-angle vs scalar layout)
- lstsq scaling re-fit is post-hoc; NLSQ-optimized scaling is authoritative
- compute_g2_batch_with_per_angle_scaling
- test_combine_angle_results_excludes_nonfinite_covariance
- Heterodyne Workflow Guide
- Optimization Tests
- Heterodyne Config Template
- parameters is physics-first [physics|contrast|offset]
- build_heterodyne_stratified_data (flat slab for create_stratified_chunks)
- HeterodyneStratifiedData (homodyne StratifiedData field mirror)
- sigma=None unit sentinel (avoids dense ones array)
- Engine calls evaluator.eval_points instead of hard-coding kernel (Phase 1.1)
- Single-time-axis: t1 authoritative, prepend length-1 phi axis for squeeze parity
- supports_scattered duck-typed, not in Protocol (homodyne unaffected)
- Uniform angle sum cancels cos(phi0-phi) gradient -> gamma_dot collapse
- jnp.where underflow guard (gradient-safe) not jnp.maximum on traced phi0
- Angle-completeness ensures non-zero per-angle gradients
- Free per-chunk arrays after concatenation (M1 memory optimization)
- Concatenated vectorized residual (compute g2 grid once, gather flat indices)
- Precomputed integer diagonal mask (t1==t2 zeroed, removes 370MB float temporaries)
- Host-side flat-index precompute (np.unique/searchsorted, int64 overflow guard)
- ERROR_RECOVERY_STRATEGIES Map
- Activate-Script Block Markers
- Legacy ~/.xpcsjax_xla_mode Migration
- Relocation-Safe $VIRTUAL_ENV/$CONDA_PREFIX Resolution
- Post-Install Shell Completion + XLA Setup
- XLA Mode File Decoupled from Activation Hook
- xpcsjax.runtime package (validation + shell)
- Shell script path accessors (get_completion_script / get_xla_config_script)
- Cleanup Utilities (xpcsjax-cleanup)
- _remove_xpcsjax_blocks State Machine
- Line-by-Line State Machine over Regex Rationale

## God Nodes (most connected - your core abstractions)
1. `ConfigManager` - 200 edges
2. `AnalysisMode` - 154 edges
3. `get_logger()` - 154 edges
4. `OptimizationResult` - 148 edges
5. `make_synthetic_two_component()` - 138 edges
6. `NLSQConfig` - 135 edges
7. `fit_nlsq_multi_phi()` - 100 edges
8. `MainWindow` - 96 edges
9. `XPCSDataLoader` - 69 edges
10. `ParameterManager` - 66 edges

## Surprising Connections (you probably didn't know these)
- `xpcsjax.viz API` --references--> `xpcsjax.viz`  [EXTRACTED]
  docs/source/api/viz.rst → xpcsjax/viz/__init__.py
- `test_config_summary_reads_real_surface()` --calls--> `_config_summary()`  [INFERRED]
  tests/cli/test_debug_audit_2026_06_17.py → xpcsjax/service/persist.py
- `test_phi_grid_pins_scrollbars_to_keep_square_tiles()` --calls--> `PhiResultsGrid`  [INFERRED]
  tests/gui/test_plots_view.py → xpcsjax/gui/views/plots/grid.py
- `xpcsjax.utils API` --references--> `xpcsjax.utils`  [EXTRACTED]
  docs/source/api/utils.rst → xpcsjax/utils/__init__.py
- `Visualization Guide` --references--> `xpcsjax.viz`  [EXTRACTED]
  docs/source/user_guide/visualization.rst → xpcsjax/viz/__init__.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Data Loading and Validation Flow** — xpcsjax_data_xpcs_loader_load_xpcs_data, xpcsjax_data_xpcs_loader_xpcsdataloader, xpcsjax_data_dataset_xpcsdataset, xpcsjax_data_xpcs_loader_xpcsdataformaterror [EXTRACTED 1.00]
- **Test Domain Sharding** — tests_core, tests_optimization, tests_characterization [EXTRACTED 1.00]
- **Documentation Coverage Enforcement** — docs_adr_0001_automated_structural_doc_coverage_check, tests_test_docs_structure, docs_source_api_index [EXTRACTED 1.00]
- **Heterodyne Analysis Pipeline** — docs_source_user_guide_heterodyne_workflow, xpcsjax_config_templates_xpcsjax_two_component, docs_source_user_guide_interpreting_results [INFERRED 0.90]
- **GUI-Worker Process Isolation** — docs_source_user_guide_gui, xpcsjax_service_events, docs_source_development_testing [EXTRACTED 0.95]
- **Five-Layer Anti-Degeneracy Stack (L1-L5)** — docs_source_advanced_anti_degeneracy_l1_per_angle, docs_source_advanced_anti_degeneracy_l2_hierarchical, docs_source_advanced_anti_degeneracy_l3_adaptive_reg, docs_source_advanced_anti_degeneracy_l4_gradient_monitor, docs_source_advanced_anti_degeneracy_l5_shear_weighting, docs_source_advanced_anti_degeneracy_controller [EXTRACTED 1.00]
- **NLSQ vs xpcsjax ownership-split contract** — docs_source_development_nlsq_integration_curvefit, docs_source_development_nlsq_integration_select_nlsq_strategy, docs_source_development_nlsq_integration_anti_degeneracy_controller, docs_source_advanced_architecture_nlsq_ownership_split [EXTRACTED 1.00]

## Communities (640 total, 283 thin omitted)

### Community 0 - "ParameterManager"
Cohesion: 0.03
Nodes (91): _has_error(), parametrize, Regression tests for null / non-finite handling in the config subsystem. Covers…, Only 'min' supplied, inverting against the registry default max., test_heterodyne_finite_value_still_passes(), test_heterodyne_nan_is_error(), test_homodyne_finite_value_still_passes(), test_homodyne_inf_is_error() (+83 more)

### Community 1 - "heterodyne_core.py"
Cohesion: 0.05
Nodes (87): QualityFlag, _data_ssr(), ndarray, SimpleNamespace, Multi-seed keep-best for the heterodyne joint CMA-ES escape. RCA (C044…, ``cmaes_n_seeds=3`` ⇒ ``fit_with_cmaes`` is invoked 3× with seeds 42/43/44.…, Fake OptimizationResult: parameters[0] encodes the SSR data_ssr reads., _result() (+79 more)

### Community 2 - "logging.py"
Cohesion: 0.03
Nodes (94): Phase 5 — quantile scaling accepts a raw (non-stratified, <100k) grid object., The object the <100k unstratified path returns: grid arrays, NO phi_flat., _RawGrid, test_quantile_on_raw_grid_no_attribute_error(), test_estimate_nlsq_optimization_memory_jacobian_dominant(), JAX-First Optimization for xpcsjax. JAX-native NLSQ-only optimization system…, Adaptive Relative Regularization for Anti-Degeneracy Defense. This module…, Anti-Degeneracy Controller - Orchestrator for 5-Layer Defense System. This… (+86 more)

### Community 3 - "make_synthetic_two_component"
Cohesion: 0.02
Nodes (148): _diag(), Golden test for heterodyne anti-degeneracy diagnostics emission. Both modes now…, Only the 3 activation flags are unconditional; per-layer DETAIL keys…, test_constant_path_always_emits_activation_keys(), test_detail_keys_preserved_when_enabled(), test_disabled_path_now_emits_activation_keys_false(), test_disabled_path_omits_layer_detail_keys(), test_enabled_path_emits_activation_keys_true() (+140 more)

### Community 4 - "XPCSDataLoader"
Cohesion: 0.04
Nodes (61): parametrize, Cross-platform safety of the cache-filename guard. The original guard tested…, test_accepts_plain_filename(), test_rejects_unsafe_cache_filenames(), Validate experimental setup parameters for physical reasonableness. Parameters…, validate_experimental_setup(), _assert_safe_cache_filename(), get_logger() (+53 more)

### Community 5 - "NLSQConfig"
Cohesion: 0.04
Nodes (89): _build_model(), _config_dict(), ndarray, Integration smoke tests: real heterodyne NLSQ fits on tiny synthetic data.…, _synthetic_stack(), test_auto_mode_resolves_to_averaged_for_many_angles(), test_cmaes_path_runs(), test_individual_mode_joint_fit() (+81 more)

### Community 6 - "DataQualityController"
Cohesion: 0.06
Nodes (44): DataQualityController, log_performance(), Any, QualityControlResult, Run the combined quality assessment for final data. Aggregates basic and…, Check shape consistency across ``c2_exp``, ``t1``, and ``t2``. Parameters…, Check correlation data for preprocessing artifacts. Parameters ----------…, Compute the fidelity of a data transformation. Parameters ----------… (+36 more)

### Community 7 - "xpcsjax/data/__init__.py"
Cohesion: 0.04
Nodes (88): _integer_c2_with_fractional_correction(), Regression test for integer-dtype truncation in enhanced diagonal correction.…, One 8x8 integer matrix whose basic diagonal correction is fractional. The basic…, ``window_size <= 0`` must not silently leave the diagonal uncorrected. This…, test_diagonal_correction_upcasts_integer_c2_no_truncation(), test_statistical_diagonal_correction_warns_and_clamps_on_nonpositive_window_size(), _integer_c2(), parametrize (+80 more)

### Community 8 - "_chunked_jacfwd_dense"
Cohesion: 0.21
Nodes (13): _nonlinear_residual(), ndarray, parametrize, Parity guard for the column-blocked covariance Jacobian. The heterodyne…, A nonlinear R^n_in -> R^n_out map that exercises mixed partials. Shaped like a…, Chunked Jacobian == jax.jacfwd for every column-block width., The residual is jax.jit-wrapped in production; the helper must handle it., The production default block (no col_block kwarg) is also exact. (+5 more)

### Community 9 - "PerformanceEngine"
Cohesion: 0.08
Nodes (22): Future, _bare_loader(), XPCSDataLoader, Regression test: XPCSDataLoader must not leak its PerformanceEngine's…, Bypass __init__ to avoid needing a YAML config / real dataset on disk., The memory_manager.shutdown() branch — untested when every other test in this…, test_close_calls_memory_manager_shutdown(), test_close_is_idempotent_and_safe_with_no_components() (+14 more)

### Community 10 - "NLSQResult"
Cohesion: 0.04
Nodes (65): _make_config(), ndarray, NLSQConfig, Integration tests for NLSQAdapter error recovery. Ports the focused subset of…, A residual that raises ValueError yields a failed NLSQResult. Verifies the…, A parameter pinned by lower == upper does not crash the adapter., Sanity-check companion: the same adapter converges on benign input. Verifies…, Trivial well-conditioned residual: r_i = p_i - target_i. JAX-traceable: returns… (+57 more)

### Community 11 - "_resolve_effective_mode"
Cohesion: 0.06
Nodes (41): ResolvedPerAngleMode, test_stratified_ls_scaling_names_match_dedup_phi_count(), Tests for heterodyne per-angle mode vocabulary parity with homodyne., `auto` with large n_phi dispatches averaged. Unified rule: at large n_phi (e.g.…, Unified rule: auto ∈ {individual, averaged} for ALL n_phi; explicit…, The averaged-scaling joint solver uses the corrected name., `per_angle_mode='constant'` reaches `_fit_joint_constant_multi_phi`., `auto` with n_phi < constant_scaling_threshold resolves to individual. Unified… (+33 more)

### Community 12 - "fit_two_component_via_engine"
Cohesion: 0.07
Nodes (51): _make_well_posed_case(), Build (model, c2, phi) where ``c2`` is NOISELESS model correlation at a…, _build_single_angle_model(), _make_config(), HeterodyneModel, ndarray, NLSQConfig, Regression tests for two engine-route ``two_component`` defects. Both were… (+43 more)

### Community 13 - "ConfigManager"
Cohesion: 0.02
Nodes (171): Configuration Guide, Regression tests for config_handling.py error-path hardening. Tests the three…, test_load_failure_names_the_file(), test_non_dict_output_block_is_logged(), test_normalize_gate_tolerates_object_without_method(), The CLI data_pipeline adapters must map argparse attrs to service kwargs., A malformed --phi-angles must not abort the whole fit run — it is only ever…, test_dispatch_fit_is_non_fatal_on_malformed_phi_angles() (+163 more)

### Community 14 - "TwoComponentModel"
Cohesion: 0.05
Nodes (31): create_model(), HeterodyneModelBase, ABC, ndarray, Model class hierarchy for heterodyne correlation analysis., Set default parameter values., Number of parameters (14)., Parameter names in canonical order. (+23 more)

### Community 15 - "MainWindow"
Cohesion: 0.03
Nodes (42): QMainWindow, codex#6/agy#6: the dead-path check expands ${ENV}/~ before testing existence.…, test_expand_path_resolves_env_and_user(), Unit tests for StatusManager collaborator (set_status / append_log)., StatusManager routes set_status/append_log through to the MainWindow widgets., StatusManager is a QObject child of MainWindow., Calling set_status twice replaces the previous status., Multiple append_log calls accumulate in the log widget. (+34 more)

### Community 16 - "NLSQWrapper"
Cohesion: 0.06
Nodes (39): Scattering wavevector magnitude., _build_homodyne_l4_callback(), create_multistart_warmup_func(), _extract_n_points(), _homodyne_l4_monitoring_enabled(), NLSQWrapper, Any, Logger (+31 more)

### Community 17 - "data/config.py"
Cohesion: 0.11
Nodes (32): apply_config_defaults(), ConfigValidationResult, create_example_yaml_config(), get_logger(), load_json_config(), load_yaml_config(), migrate_json_to_yaml_config(), Any (+24 more)

### Community 18 - "_build_laminar_fit"
Cohesion: 0.16
Nodes (18): _build_laminar_fit(), Phase-0 gate: NLSQ's curve_fit callback must be observational (cannot perturb a…, Build a small synthetic laminar_flow fit that routes through the live…, _recording_callback(), _result_chi2(), _result_params(), test_heterodyne_curve_fit_callback_is_observational_and_per_iteration(), test_homodyne_curve_fit_callback_is_observational_and_per_iteration() (+10 more)

### Community 19 - "heterodyne_constant_mode.py"
Cohesion: 0.06
Nodes (42): test_bool_coercion_and_determinism(), test_core_activation_keys_always_present(), test_empty_gradient_monitor_is_still_included(), test_gradient_monitor_included_when_provided(), test_layer_detail_merged_verbatim(), _b2_config_dict(), _build_minimal_heterodyne_model(), _build_synthetic_c2_stack() (+34 more)

### Community 20 - "DataQualityReport"
Cohesion: 0.07
Nodes (60): Quality-gate finding #7: the validation helpers narrow their except clause to…, _report(), test_validate_array_shapes_records_issue_on_arithmeticerror(), test_validate_array_shapes_records_issue_on_attributeerror(), _cache_validation_result(), _check_validation_cache(), _compute_data_hash(), _compute_data_statistics() (+52 more)

### Community 21 - "AnalysisMode"
Cohesion: 0.03
Nodes (81): Config-driven dispatch returns the right physics model class. Task 28:…, make_model should also work on a raw config dict (no ConfigManager)., test_make_model_accepts_dict(), Gap-filling tests for xpcsjax.core identified by Codex + Gemini review. Covers…, make_model dispatches correctly for edge-case mode strings., heterodyne' synonym must dispatch to HeterodyneModel., two_component' must dispatch to HeterodyneModel., two-component' hyphenated variant must dispatch to HeterodyneModel. (+73 more)

### Community 22 - "select_nlsq_strategy"
Cohesion: 0.06
Nodes (42): _build_synthetic_c2(), _het_smoke_config_dict(), ndarray, Path, Wall-clock regression suite for xpcsjax v0.1 hot paths. The /double-check…, End-to-end timing for the heterodyne per-angle local NLSQ fit. Smallest…, Routing decision for a typical XPCS fit (10k points, 11 params). Expected: well…, Routing decision at chunked-fit scale (10M points, 14 params). Same code path… (+34 more)

### Community 23 - "parallel_accumulator.py"
Cohesion: 0.12
Nodes (22): test_worker_functions_in_process(), _batch_timeout(), create_ooc_kernels(), _ooc_compute_chi2_chunk(), _ooc_compute_chunk(), _ooc_worker_cleanup(), _ooc_worker_init(), OOCComputePool (+14 more)

### Community 24 - "StratifiedResidualFunction"
Cohesion: 0.06
Nodes (43): _full_grid_chunk(), ndarray, SimpleNamespace, Tests for xpcsjax.optimization.nlsq.strategies.residual.…, One chunk holding the full phi x t1 x t2 cartesian grid (all angles)., _stratified(), test_deprecated_paths_raise(), test_empty_chunks_raises() (+35 more)

### Community 25 - "get_logger"
Cohesion: 0.02
Nodes (116): Formatter, The atexit monitor cleanup must not emit logging-handler error noise. At…, test_atexit_cleanup_silences_closed_stream_logging_errors(), _isolate_logging(), test_logged_errors_reraise_propagates_original_and_logs(), test_logged_errors_suppress_swallows(), test_phaselogger_banner_widths_and_never_raises(), fixture (+108 more)

### Community 26 - "ParameterManager"
Cohesion: 0.04
Nodes (34): _manager_with_tie(), ParameterManager, Tests for ParameterManager tied-parameter support., Changing the free variable that backs the parent must change the reported child…, test_expand_varying_to_full_mirrors_tied_child(), test_expand_varying_to_full_tied_value_updates_with_parent(), test_expand_varying_to_full_untied_matches_previous_behavior(), test_tied_idx_pairs_empty_when_untied() (+26 more)

### Community 27 - "ParameterIndexMapper"
Cohesion: 0.07
Nodes (31): _patch_threshold(), MonkeyPatch, Tests for three NLSQ support modules. * parameter_index_mapper: parameter-group…, test_adaptive_threshold_default_source(), test_adaptive_threshold_env_source(), test_adaptive_threshold_fallback_on_detection_failure(), test_adaptive_threshold_invalid_env_warns(), test_detect_total_system_memory() (+23 more)

### Community 28 - "Project"
Cohesion: 0.06
Nodes (41): QModelIndex, QStandardItem, QStandardItemModel, Tests for the JAX-free Project/Dataset/FitRun model., test_add_dataset_assigns_stable_unique_ids(), test_add_run_is_append_only_and_queued(), test_set_run_status_updates_in_place(), test_unknown_ids_return_none() (+33 more)

### Community 29 - "ResultSummary"
Cohesion: 0.08
Nodes (30): QTreeWidgetItem, P3: the inspector must not raise when handed a non-dict diagnostics block…, test_inspector_tolerates_non_dict_diagnostics(), Tests for InspectorDock (parameters/uncertainties/diagnostics view)., _summary(), test_inspector_clears_on_none(), test_inspector_renders_params_and_diagnostics(), pytest-qt tests for the project sidebar + comparison view. (+22 more)

### Community 30 - "jax_backend.py"
Cohesion: 0.07
Nodes (40): Regression test for gradient_g2 (audit C12). `gradient_g2` was wired as…, compute_g2_scaled_with_factors must equal compute_g2_scaled to float64., batch_chi_squared(), compute_chi_squared(), compute_g1_diffusion(), _compute_g1_diffusion_core(), _compute_g1_shear_core(), compute_g1_total() (+32 more)

### Community 31 - "test_shear_weighting.py"
Cohesion: 0.14
Nodes (22): Tests for xpcsjax.optimization.nlsq.shear_weighting. The shear weight ``w(phi)…, test_apply_weights_to_loss_disabled(), test_apply_weights_to_loss_enabled(), test_compute_weighted_mse_disabled(), test_compute_weighted_mse_enabled(), test_create_defaults_phi0_index_when_no_names(), test_create_disabled_returns_none(), test_create_none_config_returns_none() (+14 more)

### Community 32 - "cmaes_wrapper.py"
Cohesion: 0.04
Nodes (62): Regression: CMA-ES auto-memory sizing must use the FINAL (post-adaptive-…, At a memory budget where the small (pre-scaling) popsize fits without batching…, Wiring: ``_configure_memory`` must be called AFTER the adaptive scale_ratio >…, test_configure_memory_respects_explicit_popsize_override(), test_fit_calls_configure_memory_after_scale_ratio_block_with_final_popsize(), Regression: cmaes.seed in YAML config must actually reach…, test_cmaes_seed_defaults_to_none_when_unset(), test_cmaes_seed_field_exists_with_none_default() (+54 more)

### Community 33 - "test_heterodyne_hybrid_streaming.py"
Cohesion: 0.04
Nodes (101): _Cfg, _install_model_stub(), _make_synthetic_heterodyne(), MonkeyPatch, parametrize, Tests for heterodyne stratified hybrid-streaming pipeline (Phase 2)., Build a tiny synthetic heterodyne dataset from the live kernel. Returns (model,…, Task 3: L4 gradient-collapse monitor diagnostics appear in anti_degeneracy;… (+93 more)

### Community 34 - "core.py"
Cohesion: 0.06
Nodes (63): ParameterSpace, Regression: scalar q-extraction sites must reject NaN, not silently poison the…, test_normalize_data_to_object_accepts_finite_q(), test_normalize_data_to_object_rejects_nan_q(), _bounds_to_arrays(), _cmaes_failed_result(), _estimate_contrast_offset_from_data(), _extract_shear_transform_config() (+55 more)

### Community 35 - "AdaptiveRegularizer"
Cohesion: 0.09
Nodes (27): _make(), Coverage for Layer-3 adaptive regularization (audit finding #15). These…, H-4: NaN/inf params (a diverged step) must force trust-region rejection. The…, test_auto_and_absolute_modes_both_finite(), test_disabled_returns_zero(), test_nonfinite_params_return_inf(), test_out_of_range_group_is_skipped_not_crash(), test_singleton_group_is_skipped() (+19 more)

### Community 36 - "test_heterodyne_tied_result_assembly.py"
Cohesion: 0.10
Nodes (41): End-to-end discriminator: real during-solve tying vs report-time-only…, test_tied_fit_ssr_matches_recompute_from_reported_parameters(), _assert_tied_result_shape(), _build_synthetic_c2(), ndarray, End-to-end result-assembly tests: a tied config's OptimizationResult must…, Single-angle (``len(phi_angles) <= 1``) top-level tied fits fall through…, ``fit_nlsq_jax`` -> ``_fit_local`` (single-angle, per-angle local optimization)… (+33 more)

### Community 37 - "ParameterSpace"
Cohesion: 0.08
Nodes (16): ParameterSpace, ndarray, Names of varying physics parameters (excludes scaling)., Get contrast and offset values., Get initial values as a numpy array in canonical order. Returns -------…, Get bounds as numpy arrays. Returns ------- tuple of numpy.ndarray…, Get a boolean mask for the varying parameters. Returns ------- numpy.ndarray…, Convert a parameter array to a dictionary. Parameters ---------- arr :… (+8 more)

### Community 38 - "diagonal_correction.py"
Cohesion: 0.08
Nodes (48): ArrayLike, Backend, Method, parametrize, Diagonal correction is mandatory for both physics models. Property: after…, Correction must NOT modify off-diagonal entries., ``window_size <= 0`` must not silently leave the diagonal uncorrected.…, A 1x1 matrix has no off-diagonal neighbors; every method must return it… (+40 more)

### Community 39 - "multistart.py"
Cohesion: 0.03
Nodes (116): float64, int64, _CfgMgr, _install_model_stub(), Tests for heterodyne joint multistart wiring (Phase 1)., End-to-end regression test for the multistart no-op bug. Each LHS-sampled…, Minimal OptimizationResult stand-in., _StubModel (+108 more)

### Community 40 - "test_post_install.py"
Cohesion: 0.15
Nodes (28): CaptureFixture, Path, Tests for xpcsjax.post_install. Covers environment/shell detection, venv-path…, test_completion_activation_missing_script(), test_completion_bash_activation_injection_and_idempotency(), test_configure_xla_mode_overwrites_with_force(), test_configure_xla_mode_preserves_existing_without_force(), test_configure_xla_mode_writes() (+20 more)

### Community 41 - "log_exception"
Cohesion: 0.05
Nodes (52): test_experimental_missing_phi_list_gets_distinct_files(), LogCaptureFixture, MonkeyPatch, Phase-2 observational-logging tests for the device probe fallbacks. These…, A raising JAX-backend probe is logged with context; CPU default returned., test_jax_probe_failure_logs_with_context_and_returns_default(), test_log_exception_never_raises_on_bad_context(), test_log_exception_with_traceback_and_context() (+44 more)

### Community 42 - "FitOverrides"
Cohesion: 0.15
Nodes (27): _cm(), parametrize, SimpleNamespace, Equivalence tests for the typed NLSQ override application (F8)., test_max_iterations_is_coerced_to_int(), test_multistart_sets_nested_enable_and_n_starts(), test_non_dict_config_is_noop(), test_none_fields_write_nothing() (+19 more)

### Community 43 - "AntiDegeneracyController"
Cohesion: 0.04
Nodes (37): _build(), Task 29 follow-up: verify the model-lineage gate is wired through the…, Construct a controller through the production API., The production constructor must accept ``analysis_mode`` so callers can thread…, Lineage gate must short-circuit ShearSensitivityWeighting for heterodyne…, Homodyne laminar_flow path must keep Layer 5 active — this is the regime the…, Backward-compat: omitting ``analysis_mode`` keeps existing behavior (all layers…, test_from_config_accepts_analysis_mode_kwarg() (+29 more)

### Community 44 - "_write_config"
Cohesion: 0.17
Nodes (12): ConfigManager should normalize 'heterodyne' / 'Heterodyne' → 'two_component'., Minimal YAML config setting analysis_mode., analysis_mode: two_component must produce a HeterodyneModel instance., analysis_mode: heterodyne (synonym) must also produce HeterodyneModel., analysis_mode: static_anisotropic must NOT produce a HeterodyneModel (sanity)., analysis_mode: laminar_flow must NOT produce a HeterodyneModel., test_config_manager_normalizes_heterodyne_synonym(), test_heterodyne_synonym_dispatches_to_heterodyne() (+4 more)

### Community 45 - "test_heterodyne_config.py"
Cohesion: 0.08
Nodes (21): LogCaptureFixture, parametrize, Tests for xpcsjax.optimization.nlsq.heterodyne_config. Covers the safe type-…, Config-file anti_degeneracy settings must OVERRIDE dataclass defaults. Defaults…, test_advisory_warning_does_not_error(), test_config_overrides_defaults_not_silently_dropped(), test_defaults_construct_and_validate_clean(), test_from_dict_explicit_null_on_numeric_field_uses_default_not_zero() (+13 more)

### Community 46 - "HeterodynePointEvaluator"
Cohesion: 0.12
Nodes (23): _make_single_chunk_chunked(), _params14(), ndarray, Tests for the pointwise scattered-eval seam on the heterodyne evaluator. Also…, The scattered branch (a) is actually exercised and (b) equals the grid path., build_heterodyne_stratified_data(weights=None) must not allocate a dense…, A physically-plausible 14-param heterodyne vector (registry order). Values are…, Build a 1-chunk ``StratifiedChunkedData`` for the engine. Parameters ----------… (+15 more)

### Community 47 - "heterodyne_logging.py"
Cohesion: 0.06
Nodes (33): Defensive-logging contract tests for the logging primitives. Logging is…, If the failure-path ``logger.log`` itself raises, the caller's ORIGINAL…, test_json_formatter_handles_empty_exc_info_tuple(), test_json_formatter_never_raises_on_bad_format_args(), test_json_formatter_redaction_does_not_overredact(), test_json_formatter_redacts_secrets(), test_json_formatter_schema_and_jsonsafe(), test_json_safe_recursion_guard_truncates_deep_nesting() (+25 more)

### Community 48 - "simulated.py"
Cohesion: 0.08
Nodes (31): Visualization Guide, LoggerType, Regression: per-angle plot filenames must not collide. Bug history: the three…, test_postfit_close_angles_get_distinct_files(), test_simulated_close_angles_get_distinct_files(), _write_config(), Regression: the standalone ``--plot-simulated-data`` path must evaluate the…, A trailing comma must drop the empty token, not crash float(''). (+23 more)

### Community 49 - "test_gui_redesign.py"
Cohesion: 0.06
Nodes (45): _bundle(), parametrize, Tests for the 2026-06-20 GUI redesign. Covers the new surfaces that replaced…, create_config writes the mode's template; the written config's mode matches., Toolbar owns the operational actions; File menu owns project lifecycle plus the…, codex/agy: a degenerate (n_phi,1,1) bundle (no tau=dt lag) renders without…, codex MEDIUM: optional arrays shorter than n_phi must degrade, never IndexError., agy LOW: a finished run with a valid viz bundle switches the central stack to… (+37 more)

### Community 51 - "ResultBuilder"
Cohesion: 0.08
Nodes (19): _builder(), Regression tests for ``ResultBuilder`` (homodyne result_builder.py). The…, The iterations field must reflect optimizer iterations (nit), not nfev., With no nit, an explicit 'iterations' key is honored before defaulting., With neither nit nor iterations, default to 0 — never fall back to nfev., test_iterations_defaults_to_zero_without_nfev_fallback(), test_iterations_prefers_explicit_iterations_key_over_default(), test_iterations_uses_nit_not_nfev() (+11 more)

### Community 52 - "normalize_angle_to_symmetric_range"
Cohesion: 0.08
Nodes (34): _config(), Regression test: apply_angle_filtering_for_plot must normalize phi_angles and…, test_plot_and_optimization_paths_select_same_angles_across_wrap(), angle_in_range(), apply_angle_filtering(), apply_angle_filtering_for_optimization(), apply_angle_filtering_for_plot(), normalize_angle_to_symmetric_range() (+26 more)

### Community 53 - "test_config_null_sections.py"
Cohesion: 0.06
Nodes (23): ``execute_layers`` is a registered, parseable, INERT config gate. All tests…, ``from_dict({})`` must give ``execute_layers == False``., ``from_dict({"execute_layers": False})`` must give ``False``., ``from_dict({"execute_layers": True})`` must give ``True``., The dataclass default must be ``False`` without calling ``from_dict``., ``controller.execute_layers`` must proxy ``controller.config.execute_layers``., Enabling ``execute_layers`` must NOT alter any activation marker. With the flag…, All FOUR YAML templates parse and expose the intended ``execute_layers``. The… (+15 more)

### Community 54 - "test_result_presenter.py"
Cohesion: 0.12
Nodes (27): Unit tests for ResultPresenter collaborator (show_result / show_error /…, _show_result_with_bundle switches to the grid page (index 1) when a bundle…, _show_result_with_bundle falls back to text (page 0) when result_dir has no…, _show_result_with_bundle with result_dir=None forces the text fallback., show_inspector passes the summary to the inspector dock without error., show_inspector(None) clears the inspector without error., ResultPresenter is a QObject child of MainWindow., show_result renders text and keeps the stack on page 0 (text-summary page). (+19 more)

### Community 55 - "DatasetOptimizer"
Cohesion: 0.05
Nodes (46): DatasetInfo, Regression: strategy cache key must reflect memory-scaled recommendations.…, Equal .size, sigma crossing the memory limit -> distinct strategies. With…, test_equal_size_different_sigma_get_distinct_cached_strategies(), AdvancedDatasetOptimizer, create_advanced_dataset_optimizer(), create_dataset_optimizer(), DatasetOptimizer (+38 more)

### Community 56 - "fit_heterodyne_stratified_least_squares"
Cohesion: 0.04
Nodes (75): The stratified-LS path (the >=1M solver the C044 two_component run took)…, Seed-42 shuffle is a PRE-shuffle that preserves per-chunk angle balance. A…, Flat pointwise residual is finite and has the off-diagonal/t>0 support length., Individual mode runs successfully on the stratified-LS path. Explicit…, Test constant mode RUNS on stratified-LS (frozen scaling, physics-only solve).…, Fix 4: diagnostics parameter_names must align with the FULL popt length. The…, Heterodyne's joint vector is canonical SCALING-FIRST ([scaling | physics],…, Jacfwd fallback path: when the adapter returns ``covariance=None``, the… (+67 more)

### Community 57 - "PerAngleScalingPlan"
Cohesion: 0.09
Nodes (35): parametrize, _quantiles(), Phase-0 unit tests for PerAngleScalingPlan (spec §4 Seam 3). quantile_scaling =…, The jnp variant returns the same values as the NumPy one (so the traced…, test_expand_back_averaged_broadcasts_then_splits_physics(), test_expand_back_constant_uses_frozen_quantiles_physics_only_vector(), test_expand_back_individual_reorders_to_full_per_angle(), test_expand_covariance_averaged_replicates_scalar_blocks() (+27 more)

### Community 58 - "heterodyne_physics_kernel.py"
Cohesion: 0.11
Nodes (35): EvalStrategy, _compute_c2_meshgrid(), _compute_c2_pointwise(), compute_c2_unified(), _fraction(), _half_transport_meshgrid(), ndarray, partial (+27 more)

### Community 59 - "test_low_level_plots.py"
Cohesion: 0.09
Nodes (37): Figure, filterwarnings, mpl_image_compare, Path, Unit tests for low-level plot functions and helpers., test_plot_nlsq_fit_accepts_t_none(), test_plot_nlsq_fit_residual_cmap_is_rdbu(), test_plot_nlsq_fit_save_path_writes_png() (+29 more)

### Community 60 - "optimization/test_debug_audit_2026_07_22.py"
Cohesion: 0.09
Nodes (34): _linear_residual_fn(), ndarray, Regression tests for the 2026-07-22 debug-audit fixes. Fix 1…, Previously ``_fit_joint_averaged_multi_phi`` never called the floor — this pins…, Previously ``_fit_joint_constant_multi_phi`` never called the floor — this pins…, No-regression: individual mode already called the floor pre-extraction; confirm…, Residual = x - target; SSR is minimized (0) exactly at x == target., No-worse contract: final SSR is always <= the warm-start SSR. (+26 more)

### Community 61 - "nlsq_plots.py"
Cohesion: 0.07
Nodes (42): Unit tests for _resolve_color_limits., test_all_nan_returns_fallback(), test_empty_matrix_returns_fallback(), test_flat_constant_matrix_returns_widened_range(), test_normal_data_returns_percentile_limits(), test_percentile_clamp_excludes_outliers(), test_returns_floats_not_numpy_scalars(), Heterodyne path returns a real c2 surface in the expected [1.0, 1.5] range. (+34 more)

### Community 62 - "classify_option"
Cohesion: 0.07
Nodes (40): Action, Hint, _action(), test_choices_become_choices_completion(), test_count_action_is_flag_returns_none(), test_dir_hint_overrides_path(), test_literal_word_hint(), test_path_type_defaults_to_file() (+32 more)

### Community 63 - "heterodyne_hybrid_streaming.py"
Cohesion: 0.05
Nodes (49): _monitor(), test_callback_feeds_monitor_per_iteration(), test_callback_swallows_grad_fn_errors(), test_diagnostics_block_shape_and_mechanism(), test_diagnostics_empty_history_is_fallback(), test_update_frequency_throttles_grad_evals(), test_zero_or_nan_denominator_yields_inf_ratio(), test_check_watched_parameter_collapse() (+41 more)

### Community 64 - "events.py"
Cohesion: 0.07
Nodes (46): Analysis Workbench (GUI) Guide, test_queue_forwards_per_run_diagnostics(), Tests for the JAX-free IPC primitives (job / emitter / log_capture)., test_emitter_blocks_for_terminal_but_drops_telemetry_when_full(), test_emitter_stamps_run_id_and_monotonic_seq(), test_fitjob_is_frozen_and_picklable(), test_log_handler_forwards_records_as_loglines(), _imports_jax() (+38 more)

### Community 65 - "sequential.py"
Cohesion: 0.14
Nodes (29): AngleSubset, _coerce_mapping_to_array(), _coerce_numeric_array(), combine_angle_results(), _compute_final_jacobian_norms(), _estimate_initial_jacobian_norms(), _normalize_least_squares_kwargs(), optimize_per_angle_sequential() (+21 more)

### Community 66 - "XPCSDataFilter"
Cohesion: 0.07
Nodes (40): A clean angle with ONE bad first-lag cell must not lose diagonal quality. Pre-…, test_quality_score_robust_to_single_boundary_artifact(), test_degenerate_matrix_nan_quality_score_is_dropped_not_kept(), _clean_matrix_with_raw_spike(), Regression tests for ``XPCSDataFilter._calculate_matrix_quality_score``. The…, A clean, symmetric correlation matrix carrying the raw tau=0 spike. Off-…, A valid matrix must score full marks despite the raw tau=0 spike., The fix must not mask real problems: lag-1 g2 > 2.0 stays penalized. (+32 more)

### Community 67 - "test_gui_jax_free.py"
Cohesion: 0.07
Nodes (41): _probe_import(), The GUI-importable surface must never pull JAX into the process. # Task 3…, xpcsjax.gui.views.plots_view must not import JAX at module level., xpcsjax.gui.views.raster must not import JAX at module level., xpcsjax.gui.project.persist must not import JAX (stdlib + project model only)., xpcsjax.gui.error_presenter must not import JAX (stdlib only)., xpcsjax.service.config must not import JAX (config validator, JAX-free)., xpcsjax.gui.data_inspect must not import JAX (h5py + stdlib only). (+33 more)

### Community 68 - "save_results_npz"
Cohesion: 0.16
Nodes (23): test_save_results_rejects_bad_format(), Compatibility shim -- result persistence moved to xpcsjax.service.persist. Kept…, _config_summary(), _extract_metadata(), _json_safe(), Any, Namespace, ndarray (+15 more)

### Community 69 - "test_heterodyne_physics_validators.py"
Cohesion: 0.09
Nodes (38): Coverage for the heterodyne physics-constraint validators (audit finding #5).…, test_correlation_inputs_clean_is_valid(), test_correlation_inputs_nan_is_error(), test_correlation_inputs_non_monotonic_time_is_error(), test_correlation_inputs_shape_mismatch_is_error(), test_cross_parameter_fraction_sum_exceeds_unity(), test_cross_parameter_valid_fractions(), test_single_parameter_flags_negative_diffusion() (+30 more)

### Community 70 - "executors.py"
Cohesion: 0.13
Nodes (16): ExecutionResult, get_executor(), OptimizationExecutor, ABC, Any, ndarray, Optimization Strategy Executors for NLSQ. This module implements the Strategy…, Strategy name for logging. (+8 more)

### Community 71 - "build_workbench"
Cohesion: 0.09
Nodes (32): pytest-qt tests for app wiring + worker cleanup., test_build_workbench_returns_wired_window(), test_close_triggers_shutdown(), Tests for Plan H Task 4: Save/Open project wiring in MainWindow., test_open_tolerates_deleted_result_dir(), test_save_then_open_round_trips_through_window(), Integration tests for the workbench surfaces (post-redesign). Asserts that…, The bottom 'Fitting Process' dock holds only the log (no SSR/chips/banners). (+24 more)

### Community 72 - "result_presenter.py"
Cohesion: 0.09
Nodes (25): QRectF, app_icon(), apply_theme(), current_palette(), detect_scheme(), Palette, _qpalette(), Application theme — a system-aware "precision instrument" palette and global… (+17 more)

### Community 73 - "ParameterSpace"
Cohesion: 0.05
Nodes (23): Regression tests for the 2026-07-22 config-module debug audit. Five…, test_active_parameters_does_not_drop_per_angle_scaling(), test_fixed_per_angle_scaling_name_is_excluded(), test_grouped_format_keys_no_unknown_warning(), test_metadata_none_config_override_constructs(), test_metadata_none_yaml_file_constructs(), test_new_param_bounds_missing_min_max_raises(), Regression test for Fix 2: homodyne ParameterSpace.from_config's public bounds… (+15 more)

### Community 74 - "configure_logging"
Cohesion: 0.09
Nodes (34): Handler, test_configure_from_dict_null_levels_default_to_info(), _managed_console_handler(), _managed_logger(), Logger, Phase-1b wiring tests for xpcsjax logging. Confirms env/YAML selection wiring…, Return the managed console handler (StreamHandler, not a FileHandler). Debug-…, test_context_filter_installed_once() (+26 more)

### Community 75 - "optimization/test_validation.py"
Cohesion: 0.13
Nodes (22): test_validate_covariance_rejects_non_finite(), test_validate_result_consistency_low_and_high_still_pass(), _make_result(), Coverage tests for `xpcsjax.optimization.nlsq.validation`. Closes the /double-…, Build a duck-typed OptimizationResult-like object., The QualityFlag bridge maps the 'acceptable' band to 'marginal' and passes the…, test_classify_quality_flag_maps_band_to_qualityflag(), test_validate_covariance_accepts_symmetric_finite_positive_diag() (+14 more)

### Community 76 - "_save_fig"
Cohesion: 0.09
Nodes (30): Image, test_save_fig_with_none_is_noop(), The fast path must handle rectangular (n_t1 != n_t2) grids correctly., test_datashader_renderer_handles_rectangular_grid(), test_save_fig_closes_on_savefig_exception(), _fresh_fig(), Security regression tests: plot save paths must be validated. Quality-gate…, Regression: legitimate saves into a not-yet-existing subdir still work. (+22 more)

### Community 77 - "_PhysicsModelProtocol"
Cohesion: 0.09
Nodes (15): _PhysicsModelProtocol, Any, ndarray, Protocol, Check if gradient computation is available., Get the best available gradient method for optimization algorithms., Get comprehensive gradient capability information., Generate human-readable backend summary. (+7 more)

### Community 78 - "fit_with_stratified_hybrid_streaming"
Cohesion: 0.08
Nodes (26): N=1 single-point time array must not crash and must return finite (1,1)., The integral from t[0] to t[0] is zero; smooth_abs gives sqrt(eps)., Sanity check: N=2 also works., safe_sinc must be continuous and well-valued at the Taylor threshold., Values just inside and just outside the 1e-4 threshold must agree to better…, sinc(0) = 1 by the Taylor expansion., sin(π)/π ≈ 0; sanity check for the far branch., safe_sinc must return finite values for all x in [-10, 10]. (+18 more)

### Community 79 - "ProjectSidebar"
Cohesion: 0.10
Nodes (14): test_sidebar_reflects_project(), test_sidebar_selected_run_ids_returns_run_rows_only(), ProjectSidebar, Any, QWidget, Return the dataset ids of selected dataset (top-level) rows., Render ``[(label, ResultSummary|None), ...]`` as a real side-by-side table.…, A tree of datasets -> runs with multi-select. (+6 more)

### Community 80 - "HierarchicalOptimizer"
Cohesion: 0.08
Nodes (32): _opt(), Tests for xpcsjax.optimization.nlsq.hierarchical. The two-stage optimizer…, test_config_defaults(), test_create_per_angle_loss_and_grad(), test_create_physical_grad_slices_physical_indices(), test_create_physical_loss_assembles_full_vector(), test_fit_converges_on_separable_quadratic(), test_fit_invokes_outer_callback_and_logs() (+24 more)

### Community 81 - "FitJob"
Cohesion: 0.08
Nodes (34): emit_started_then_finished(), exit_without_terminal(), Any, Importable spawn-target fakes for the WorkerHandle tests. Run inside spawned…, sleep_forever(), Regression tests for the 2026-06-19 GUI debug-audit fixes. Each test pins one…, codex#3: shutdown() joins+closes the worker process and closes the queue., codex#4: after Open Project, Run works (an active dataset is established). (+26 more)

### Community 82 - "test_uninstall_scripts.py"
Cohesion: 0.07
Nodes (61): CaptureFixture, fixture, MonkeyPatch, Path, Tests for xpcsjax.uninstall_scripts. Covers venv-path resolution, cleanup-…, Sandbox HOME + VIRTUAL_ENV to tmp and return the fake venv path., test_cleanup_activation_scripts_dry_run(), test_cleanup_activation_scripts_no_venv() (+53 more)

### Community 83 - "AsyncWriter"
Cohesion: 0.09
Nodes (26): Background-write failures in :mod:`xpcsjax.utils.async_io` must be observed.…, A single wait_all with N>=3 failing futures emits exactly one WARNING., A second separate wait_all() call still logs (no cross-call suppression).…, A write that fails AFTER shutdown's wait_all() timeout must still be logged.…, ``timeout`` bounds the whole wait_all() call, not each pending future.…, StopIteration from load_fn must surface as an error, not silent truncation., A non-StopIteration error from the source must still be re-raised., A failed background write surfaces a WARNING in wait_all (control intact). (+18 more)

### Community 84 - "resolve_per_angle_mode"
Cohesion: 0.08
Nodes (29): Regression: explicit ``per_angle_mode="averaged"`` is accepted on every laminar…, The laminar hybrid-streaming driver must route per-angle mode resolution…, The shared seam returns ``"averaged"`` verbatim (the contract the inline paths…, The anti-degeneracy controller (standard + CMA-ES laminar paths) must accept an…, The seam refactor must not change the resolved actual mode for the pre-existing…, A truly unknown / removed-legacy token must still raise (via the shared…, _source_of(), test_controller_accepts_explicit_averaged() (+21 more)

### Community 85 - "_StubParamManager"
Cohesion: 0.24
Nodes (5): Regression test for the multistart no-op bug. ``fit_nlsq_multi_phi`` seeds its…, ``reseed_initial_values`` must handle the ndarray input form too, not just dict., _StubParamManager, test_reseed_initial_values_accepts_ndarray_form(), test_reseed_initial_values_moves_get_initial_values()

### Community 86 - "ParameterRegistry"
Cohesion: 0.05
Nodes (46): given, settings, parametrize, Heterodyne parameter registry entries — verbatim from heterodyne docs. Source:…, heterodyne' should normalize to 'two_component'., test_heterodyne_mode_lists_14_params(), test_heterodyne_param_specs(), test_heterodyne_synonym_normalize() (+38 more)

### Community 87 - "test_recovery_and_numerical.py"
Cohesion: 0.14
Nodes (19): parametrize, Tests for error-recovery and numerical-validation helpers.…, test_set_bounds_disable_enable(), test_validate_gradients_detects_nonfinite(), test_validate_gradients_disabled_skips(), test_validate_gradients_finite_ok(), test_validate_loss_detects_nonfinite(), test_validate_loss_disabled_skips() (+11 more)

### Community 88 - "test_engine_heterodyne_fit_parity.py"
Cohesion: 0.09
Nodes (30): _builder_head_len(), expand_to_engine_scaling_first(), ndarray, Test-local oracle for the canonical heterodyne scaling-first layout. The…, Length of the scaling head the pointwise builder emits for *mode*. ``constant``…, Expand the builder's canonical scaling-first ``p0`` to the engine's layout.…, _validate(), _assert_engine_reaches_minimum_no_worse() (+22 more)

### Community 89 - "_PhaseRecord"
Cohesion: 0.33
Nodes (4): test_phase_record_duration(), _PhaseRecord, Internal record for phase timing., Mark a phase as started for timing. Parameters ---------- name Phase name (e.g.…

### Community 90 - "NLSQAdapter"
Cohesion: 0.05
Nodes (50): Adapter must not mint a 'good' fit from a missing objective. Quality-gate…, _result(), test_explicit_cost_still_used(), test_missing_objective_yields_nonfinite_chi2_not_good(), _data_unsorted_phi(), Regression test for phi-broadcast ordering in NLSQAdapter._flatten_xpcs_data.…, Duplicate angles in phi must not crash the broadcast (agy review, P2).…, test_flatten_duplicate_phi_does_not_crash() (+42 more)

### Community 91 - "NLSQConfig"
Cohesion: 0.09
Nodes (24): F6: NLSQConfig.validate() upper-bound guard for chunk-size fields. A…, test_cmaes_data_chunk_size_above_ceiling_is_rejected(), test_cmaes_data_chunk_size_none_is_accepted(), test_cmaes_data_chunk_size_normal_is_accepted(), test_cmaes_data_chunk_size_zero_still_rejected(), test_hybrid_chunk_size_above_ceiling_is_rejected(), test_hybrid_chunk_size_normal_is_accepted(), test_hybrid_chunk_size_zero_still_rejected() (+16 more)

### Community 92 - "MultiLevelCache"
Cohesion: 0.10
Nodes (18): Audit [20] (double-check follow-up): cache_hit_rate must be a true…, test_cache_hit_rate_is_a_true_hit_rate(), MultiLevelCache, Path, Estimate the in-memory size of an item in MB. Recurses into lists, tuples, and…, Record an access for intelligent caching and eviction decisions. Parameters…, Return the recent access frequency for a key in accesses per minute. Parameters…, Evict least valuable item from memory cache. (+10 more)

### Community 93 - "helpers.py"
Cohesion: 0.08
Nodes (28): ColorMap, ImageItem, PlotItem, P3 (twin-path): the residual heatmap color window must be computed from the…, test_residual_map_levels_computed_from_full_resolution(), test_c2_levels_clamp_to_unit_band(), test_residual_levels_are_symmetric_about_zero(), test_time_rect_none_on_degenerate_axes() (+20 more)

### Community 94 - "_build_joint_problem"
Cohesion: 0.06
Nodes (53): _degraded_adapter(), _individual_cfg(), NLSQConfig, Keep-better floor on the heterodyne joint solve + Stage-1 de-duplication. Pins…, The CMA-ES escape builds the joint problem (incl. Stage 1) exactly once. Before…, On revert to x0 the result must NOT carry the rejected solve's covariance. When…, A reverted-to-x0 warm-start must report success=False / 'failed'. Exercises the…, Blast-radius boundary: an assembly backed by a global escape (joint_result is… (+45 more)

### Community 95 - "test_heterodyne_stratified_anti_degeneracy_parity.py"
Cohesion: 0.09
Nodes (32): _ad_config_dict(), Parity tests: heterodyne ≥1M stratified-LS anti-degeneracy controller wiring., The driver's controller instantiation emits the laminar-style Layer 2/3/4…, No anti_degeneracy dict -> returns None, emits nothing, never raises., hierarchical.enable=False must suppress the Layer 2 banner (L2 IS gated)., gradient_monitoring.enable=False must suppress the Layer 4 banner (L4 IS gated)., L3 is NOT gated by a regularization.enable field — it stays on under master…, The param-count banner must report heterodyne's real n_physical (14), not… (+24 more)

### Community 96 - "test_parameter_utils.py"
Cohesion: 0.12
Nodes (24): ndarray, SimpleNamespace, _quantile_flat(), Scientific tests for xpcsjax.optimization.nlsq.parameter_utils. Pure helpers…, Mirror the diffusion-only g1^2 the estimator computes, for self-consistency., Build flat data where small lags sit at the ceiling and large lags at the floor., _static_g1_sq(), _static_stratified() (+16 more)

### Community 97 - "test_review_regressions.py"
Cohesion: 0.09
Nodes (32): _check_lazy_viz_import(), _make_heterodyne_config(), _make_heterodyne_data(), _make_heterodyne_result(), parametrize, Regression tests for the bugs surfaced by the Codex+Gemini review of the…, ``generate_nlsq_plots``' ``analysis_mode`` guard (line ~1623) falls back to…, n_total == n_physical (constant mode) → NotImplementedError upfront. (+24 more)

### Community 98 - "test_heterodyne_result_builder.py"
Cohesion: 0.14
Nodes (23): NLSQResult, Tests for the heterodyne NLSQ result layer. Covers the ``NLSQResult`` dataclass…, _result(), test_build_from_nlsq_bad_tuple_length_raises(), test_build_from_nlsq_dict(), test_build_from_nlsq_dict_missing_keys_raises(), test_build_from_nlsq_object(), test_build_from_nlsq_object_missing_popt_raises() (+15 more)

### Community 99 - "run_worker"
Cohesion: 0.22
Nodes (13): _drain(), _install_fake_services(), run_worker emits LayerStatus (post-fit) + Banner (from engine log lines)., test_worker_emits_layerstatus_and_banner(), _drain(), _install_fake_services(), Headless tests for run_worker — fake service modules keep JAX out., Inject JAX-free fake service modules into sys.modules. (+5 more)

### Community 100 - "test_heterodyne_memory_adapter.py"
Cohesion: 0.13
Nodes (28): _patch_threshold(), MonkeyPatch, Tests for heterodyne memory routing and adapter helper logic. *…, Overcommit prevention: the budget shrinks 1/N with the fit concurrency., test_build_tier_list_drops_large_when_disabled(), test_build_tier_list_from_each_start(), test_cache_key_equality(), test_detect_total_system_memory_gb() (+20 more)

### Community 101 - ".from_config"
Cohesion: 0.10
Nodes (30): _laminar_controller(), Unit tests for the shared per-angle-mode banner formatter. Covers…, The shared quantile helper is reused by the averaged path, so its banner must…, Pin the wiring: the summary label must be produced by the helper, never re-…, test_averaged_banner_text(), test_broadcast_mode_label_resolves_word(), test_compute_fixed_per_angle_scaling_emits_neutral_banner(), test_constant_banner_has_no_zero_scaling() (+22 more)

### Community 102 - "test_validation_branches.py"
Cohesion: 0.26
Nodes (17): Any, Branch-coverage complement for xpcsjax.optimization.nlsq.validation.…, _result(), test_fit_quality_acceptable_band_logged(), test_fit_quality_cmaes_max_restarts_warns(), test_fit_quality_cmaes_other_reason_passes(), test_fit_quality_condition_number_ok_via_pcov_fallback(), test_fit_quality_condition_number_too_high() (+9 more)

### Community 103 - ".get_active_parameters"
Cohesion: 0.08
Nodes (16): ndarray, ValidationResult, Extract base parameter name from indexed parameter names. Handles per-angle…, Get parameter bounds configuration (with caching for performance). Parameters…, Get list of active (physical) parameters from configuration (cached). This…, Get default active parameters based on analysis mode., Get all parameter names including scaling parameters. Returns ------- list of…, Get the effective number of physical parameters (excludes scaling). Returns… (+8 more)

### Community 104 - "test_gui_debug_2026_06_21.py"
Cohesion: 0.12
Nodes (24): Regression tests for the 2026-06-21 GUI debug-audit fixes. Each test pins one…, P3: a partial/external nlsq_result.json whose nlsq_diagnostics is null must…, A minimal summary whose convergence_status carries a unique marker so the…, P2 (twin-path): while run A is active, the user clicks an earlier finished run…, Guard against over-correction: with no competing selection, a finishing active…, P3 (stale-state): the append-only Fitting-Process log must be cleared when the…, _summary(), test_finishing_active_run_does_not_clobber_selected_run() (+16 more)

### Community 105 - "test_output_resolution.py"
Cohesion: 0.15
Nodes (29): _args(), _cfg(), Any, Path, Regression tests for unified CLI output-directory resolution. Adversarial-…, ``run_nlsq``'s Writer 1 (parameters/analysis_results_nlsq/convergence_metrics)…, When generate_nlsq_plots raises, _generate_post_fit_plots must report that…, Minimal ConfigManager stand-in — the resolver only reads ``.config``. (+21 more)

### Community 106 - "MemoryPressureMonitor"
Cohesion: 0.13
Nodes (15): Regression: ``get_pressure_trend`` must not iterate ``_pressure_history`` bare.…, test_get_pressure_trend_survives_concurrent_appends(), Regression: pressure-state hysteresis dead-zone must preserve prior state.…, 0.85 -> 0.75 (dead zone) -> 0.5: recovery must fire exactly once. With…, test_deadzone_preserves_state_and_recovery_still_fires(), log_once(), MemoryPressureMonitor, Real-time memory pressure monitoring with adaptive responses. Monitors system… (+7 more)

### Community 107 - "test_transforms.py"
Cohesion: 0.06
Nodes (56): _laminar_index_map(), parametrize, Scientific tests for xpcsjax.optimization.nlsq.transforms. The shear transforms…, test_adjust_covariance_beta_unchanged(), test_adjust_covariance_log_jacobian(), test_adjust_covariance_no_state_or_empty(), test_build_physical_index_map(), test_format_x_scale_for_log() (+48 more)

### Community 108 - "generate_nlsq_plots"
Cohesion: 0.14
Nodes (26): _phi_filename(), _png_sha256(), Path, End-to-end tests for generate_nlsq_plots., Heterodyne now produces plots + NPZ + JSON via per-angle scaling reconstruction., One angle's compute failure leaves NaN; others render; NPZ still written., If datashader is missing, orchestrator warns and uses matplotlib. The…, plot_simulated_data and _generate_heatmap_plots were removed in favor of… (+18 more)

### Community 109 - "ShearSensitivityWeighting"
Cohesion: 0.11
Nodes (16): Array, _compute_weights_jax(), ndarray, partial, Shear-sensitivity weighted loss for anti-degeneracy defense. This class manages…, Initialize the weighter and precompute weights for the initial phi0. See the…, Compute angle weights for given phi0. Performance Optimization (Spec 001 -…, Update phi0 estimate from current parameters. Parameters ---------- params :… (+8 more)

### Community 110 - "test_artifacts.py"
Cohesion: 0.15
Nodes (28): floating, integer, Any, MonkeyPatch, Path, Tests for NPZ + JSON artifact serialization., If writing fails mid-stream, no stale .npz or .tmp should remain., _sample_arrays() (+20 more)

### Community 111 - "log_heterodyne_completion"
Cohesion: 0.14
Nodes (23): _chi2_args(), _completion_physics(), _het_result(), Regression tests for the 2026-06-17 debug-audit fixes (optimization/core)., test_averaged_physics_first_marker_reads_physics_from_head(), test_averaged_scaling_first_marker_reads_physics_from_tail(), test_gradient_chi2_does_not_raise_on_static_args(), test_hessian_chi2_does_not_raise_on_static_args() (+15 more)

### Community 112 - "test_quality_controller_smoke.py"
Cohesion: 0.09
Nodes (29): _baseline_config(), _minimal_data(), parametrize, Smoke tests for :mod:`xpcsjax.data.quality_controller`. The…, Each pipeline stage must return a QualityControlResult without raising., When the controller is disabled, every stage must return a minimal result…, The module-level convenience function constructs the controller and validates…, QualityLevel maps to the ``validation_level`` config string set in… (+21 more)

### Community 113 - "_fit_nlsq_heterodyne"
Cohesion: 0.08
Nodes (29): Logging-parity tests: two_component must emit the same setup-log narrative as…, Mirrors laminar's ``xpcsjax.device.cpu`` configuration banner., Mirrors laminar's ``memory_strategy_selection`` phase + threshold line., Logging must never break a fit even when the optional deps are absent., test_helpers_never_raise(), test_safe_configure_cpu_threading_emits_device_cpu_block(), test_safe_log_memory_strategy_emits_phase_and_threshold(), Return the validated analysis mode as a typed enum. Centralizes the scattered… (+21 more)

### Community 114 - "test_static_individual_invariant.py"
Cohesion: 0.12
Nodes (23): parametrize, Regression: static modes are pinned to the ``individual`` per-angle layout on…, Wiring: every large-data reduced-chi2 DOF computation in the wrapper must…, Laminar reduction is preserved (byte-identical routing for laminar)., Pin the wiring: the streaming fit must resolve its mode via the helper, never…, _static_cfg(), test_controller_disabled_for_static_so_no_compression(), test_resolve_static_pinned_forces_individual() (+15 more)

### Community 115 - "compute_c2_heterodyne"
Cohesion: 0.10
Nodes (33): _het_params(), Regression tests for the 2026-07-22 debug-audit fixes (Fix 1-3). Fix 1:…, compute_chi_squared's masked support must equal sum(compute_residuals**2) --…, Perturbing ONLY the t=0 row/col or the diagonal must not change chi2, since…, Static (<7-param) mode with a 0-d scalar t1 must not raise., With N > 10001 unique lag times, passing time_grid must change the (previously…, test_compute_g1_shear_static_mode_accepts_0d_scalar_t1(), test_compute_g1_total_elementwise_time_grid_avoids_truncation() (+25 more)

### Community 116 - "validate_xpcs_data"
Cohesion: 0.12
Nodes (31): _synthetic_xpcs_data(), test_incremental_validation_refreshes_changed_component_statistics(), test_incremental_validation_refreshes_physics_checks_on_q_change(), _clean_data(), _correlation_warnings(), ndarray, parametrize, Tests for xpcsjax.data.validation (M-2: closes the largest coverage gap).… (+23 more)

### Community 117 - "AdvancedMemoryManager"
Cohesion: 0.04
Nodes (50): Regression test for the mmap-backed virtual-memory allocator's success path.…, test_allocate_virtual_memory_returns_usable_buffer(), test_advanced_memory_manager_collected_without_gc_sweep(), test_memory_pool_max_buffers_floor_is_one_not_four(), AdvancedMemoryManager, AllocationError, _check_vm_path(), MemoryManagerError (+42 more)

### Community 118 - "RunController"
Cohesion: 0.09
Nodes (22): Path, Tests for the JAX-free figure-export helper (export.py). Two required cases:…, export_figures copies *.png/pdf from <result_dir>/plots recursively., Two PNGs with the same filename in different sub-dirs must both survive., export_figures must return [] and NOT raise when there is no plots/ dir., _on_export_figure shim delegates to RunController; export_figures is called., test_export_copies_figures_to_dest(), test_export_disambiguates_same_name_files() (+14 more)

### Community 119 - "test_heterodyne_data_prep.py"
Cohesion: 0.11
Nodes (29): parametrize, Coverage for heterodyne data-prep pure functions (audit finding #16). Exercises…, test_compute_weights_exclude_diagonal_zeros_diagonal(), test_compute_weights_inverse_variance_requires_sigma(), test_compute_weights_inverse_variance_shape_mismatch(), test_compute_weights_unknown_method(), test_degrees_of_freedom_normal(), test_degrees_of_freedom_underdetermined_floors_at_one() (+21 more)

### Community 120 - "test_jacobian.py"
Cohesion: 0.08
Nodes (39): fixture_data(), _polynomial_residual(), fixture, ndarray, Smoke tests for the Jacobian-stats utility module.…, Smooth polynomial → low but finite gradient noise across perturbations. The…, 20-point polynomial fixture: xdata + initial params at the truth., Happy path: well-conditioned polynomial returns a (3,3) J^T J and a length-3… (+31 more)

### Community 121 - "json_safe"
Cohesion: 0.14
Nodes (5): TestJsonSafeEdgeCases, TestJsonSafeFloatSanitization, TestJsonSafeNumpyTypes, json_safe(), Recursively convert numpy arrays and special types to JSON-safe types.…

### Community 122 - "MonkeyPatch"
Cohesion: 0.11
Nodes (28): MonkeyPatch, test_detect_shell_type_unknown_via_psutil_match(), test_detect_shell_type_unknown_when_probe_raises(), test_get_venv_path_conda_fallback(), test_get_venv_path_prefers_virtual_env(), test_get_venv_path_sys_prefix_fallback(), test_install_completion_activation_routes(), test_install_completion_activation_skips_outside_venv() (+20 more)

### Community 123 - "plots_view.py"
Cohesion: 0.07
Nodes (34): _SquareBase, test_diagonal_residual_traces_diag_against_t1(), test_residual_histogram_renders_and_tolerates_all_nan(), test_residual_map_accepts_array(), test_residuals_vs_fitted_decimates_large_cloud(), test_residuals_vs_fitted_tolerates_all_nan(), _PhiSection, ndarray (+26 more)

### Community 124 - "config_generator.py"
Cohesion: 0.12
Nodes (24): Regression tests for config_generator scalar substitution (audit C2).…, test_overrides_produce_parseable_yaml(), _build_parser(), main(), ArgumentParser, Configuration file generator for xpcsjax NLSQ analysis. Provides the ``xpcsjax-…, Public factory alias for the config-generator parser. Returns -------…, Run the ``xpcsjax-config`` console script. Parses arguments and dispatches to… (+16 more)

### Community 125 - "test_plot_dispatch_logging.py"
Cohesion: 0.19
Nodes (17): _FakeConfigManager, _FakeResult, _make_data(), fixture, LogCaptureFixture, MonkeyPatch, Logging-behavior tests for ``xpcsjax.cli.plot_dispatch``. These tests pin the…, A SECOND separate call must emit its own WARNING (no cross-call collapse). With… (+9 more)

### Community 126 - "test_memory_manager_cache_gating.py"
Cohesion: 0.11
Nodes (26): _make_manager(), skipif, Emergency cleanup must not thrash JAX recompiles under live-array pressure. RCA…, A live-array regime logs pressure warnings at DEBUG, not WARNING. During a…, Critical pressure under live arrays stays visible (WARNING) but not CRITICAL., Outside the live-array regime the original WARNING/CRITICAL are preserved., A first zero-result GC must short-circuit the old 3x collect loop., Repeated warnings must NOT compound gc thresholds toward 0 (agy F2). The old… (+18 more)

### Community 127 - "DataInspectDialog"
Cohesion: 0.10
Nodes (21): _make_h5(), Tests for DataInspectDialog — the wiring for the previously-orphaned…, HDF5 names may contain spaces; the label they're rendered into is not a…, A 3-D dataset with a zero-length first axis makes the auto-mode read raise…, A corrupt-but-signature-valid HDF5 file can raise RuntimeError/KeyError/…, test_corrupt_file_error_shown_not_crash(), test_empty_leading_axis_shows_error_not_crash(), test_lists_datasets() (+13 more)

### Community 128 - "test_strategy_chunking.py"
Cohesion: 0.11
Nodes (26): _balanced_dataset(), Any, parametrize, Tests for xpcsjax.optimization.nlsq.strategies.chunking. All pure functions…, _stratify(), test_adaptive_chunk_size_clamped_to_max(), test_adaptive_chunk_size_clamped_to_min(), test_adaptive_chunk_size_docstring_scenario_pinned() (+18 more)

### Community 129 - "ndarray"
Cohesion: 0.09
Nodes (14): log_calls, ndarray, Compute g1 correlation function for this model., Get default parameter values., Convert parameter array to named dictionary., Compute diffusion contribution to g1. g₁_diff = exp[-q²/2 ∫|t₂-t₁| D(t')dt'], Default values for typical XPCS measurements., Compute shear contribution to g1. g₁_shear = [sinc(Φ)]² where Φ = (qL/2π)… (+6 more)

### Community 130 - ".fit"
Cohesion: 0.13
Nodes (21): NLSQStrategy, test_assess_convergence_no_progress(), test_assess_convergence_nonfinite(), test_assess_convergence_poor_fit(), test_assess_convergence_success(), test_optimizer_kwargs_includes_explicit_max_nfev(), test_optimizer_kwargs_omits_none_max_nfev(), _assess_convergence() (+13 more)

### Community 131 - "test_strategy_executors.py"
Cohesion: 0.08
Nodes (41): _enable_streaming(), _FakeOpt, _logger(), Any, MonkeyPatch, ndarray, parametrize, Tests for xpcsjax.optimization.nlsq.strategies.executors. The executors wrap… (+33 more)

### Community 132 - "ConfigTextEditorDialog"
Cohesion: 0.09
Nodes (18): Malformed YAML must not be written to disk — the on-disk config stays intact., Valid YAML still saves and accepts the dialog (validation isn't over-strict)., agy HIGH: a failed load must disable Save so a blank editor can't truncate the…, A write OSError must surface and keep the dialog open — never escape the slot…, test_config_text_editor_accepts_valid_yaml_on_save(), test_config_text_editor_disables_save_on_load_failure(), test_config_text_editor_rejects_invalid_yaml_on_save(), test_config_text_editor_round_trip() (+10 more)

### Community 133 - ".n_optimized"
Cohesion: 0.12
Nodes (21): slice, parametrize, test_averaged_and_individual_are_not_frozen(), test_blocks_partition_the_vector_scaling_first(), test_canonical_rejects_unresolved_mode(), test_constant_mode_is_frozen_with_empty_scaling_head(), test_group_indices(), test_legacy_fourier_constructor_is_untouched() (+13 more)

### Community 134 - "test_heterodyne_tied_parameters.py"
Cohesion: 0.14
Nodes (24): _base_config(), Tests for heterodyne tied_parameters (equality-constrained physics params). See…, A grouped-format vary: true for a tied child must NOT undo the tie --…, child also listed in active_parameters: tie wins, warning logged, not a hard…, child's configured bounds differing from its tied parent's bounds must log a…, A malformed mapping VALUE (unhashable, e.g. an empty list) must raise the…, The 'also listed as varying' warning must be scoped to an EXPLICIT…, test_tied_parameters_absent_is_noop() (+16 more)

### Community 135 - "test_cache_safety.py"
Cohesion: 0.12
Nodes (25): _bare_loader(), _good_payload(), MonkeyPatch, ndarray, Path, XPCSDataLoader, Trust-boundary regression tests for :meth:`XPCSDataLoader._load_from_cache`.…, _peek_npz_array_header must report the same shape/dtype as a full read.… (+17 more)

### Community 136 - "load_project"
Cohesion: 0.14
Nodes (22): Round-trip tests for .xpcsproj save/load., The ``..`` guard applies to result_dir too, not just config_path., Absolute paths are the documented invariant (GUI file-dialog paths) -- not a…, A failure during the swap must leave the prior .xpcsproj intact (no torn write)., A crafted/hand-edited .xpcsproj can't use ``..`` to point outside its own dir.…, Windows-style ``..\\`` segments must be normalized and rejected like ``../``., test_load_accepts_absolute_path_by_design(), test_load_rejects_backslash_style_traversal() (+14 more)

### Community 137 - "test_heterodyne_cmaes_warmstart_success_gate.py"
Cohesion: 0.13
Nodes (20): _cfg(), parametrize, skipif, Parity: heterodyne JOINT CMA-ES auto-skip must gate on warm-start SUCCESS.…, n_phi=3 averaged warm-start converges ⇒ auto-skip ⇒ finite covariance., n_phi=2 individual warm-start converges (max_nfev=500) ⇒ finite covariance., ``escape='cmaes'`` kept purely via ``nlsq_refined`` ⇒ marginal/failed, not…, Control: ``escape='cmaes'`` kept via a REAL convergence reason ⇒ still… (+12 more)

### Community 138 - "ndarray"
Cohesion: 0.14
Nodes (15): test_validate_bounds_consistency_upper_length_mismatch(), test_validate_initial_params_within_bounds_direct(), test_validate_array_dimensions_accepts_matching_shapes(), test_validate_array_dimensions_rejects_empty_and_mismatched(), test_validate_bounds_consistency_accepts_sorted_bounds(), test_validate_bounds_consistency_rejects_inverted_or_misshaped(), ndarray, Validate all input data. (+7 more)

### Community 139 - "AnalysisSummaryLogger"
Cohesion: 0.10
Nodes (12): test_summary_logger_as_dict_sanitizes_nan(), test_summary_logger_records_and_logs(), AnalysisSummaryLogger, Structured logging for analysis completion summaries. Tracks phase timings,…, Initialize the summary logger for an analysis run. Parameters ---------- run_id…, Mark a phase as completed. Silently ignores a ``name`` that was never started.…, Record a named metric (e.g. ``chi_squared``). Parameters ---------- name Metric…, Record an output file path. Parameters ---------- path Path to the output file. (+4 more)

### Community 140 - "test_cpu_config.py"
Cohesion: 0.12
Nodes (22): _fake_cpuinfo(), _fake_lscpu(), isolated_configure(), fixture, MonkeyPatch, parametrize, Regression tests for the CPU HPC configuration helpers. The thread-reservation…, A MemoryError from the CPU benchmark surfaces as the documented error dict… (+14 more)

### Community 141 - "parameter_names.py"
Cohesion: 0.10
Nodes (22): Smoke tests for the canonical parameter-name constants.…, Static isotropic = 2 scaling + 3 physical = 5 names., Laminar flow = 2 scaling + 3 physical + 4 flow = 9 names., ``get_parameter_names(mode)`` must dispatch to the matching constant list.…, No parameter name appears in both the scaling and physical blocks — otherwise…, test_get_parameter_names_dispatches_on_mode(), test_laminar_flow_has_9_params(), test_scaling_and_physical_blocks_are_disjoint() (+14 more)

### Community 142 - "test_memory_manager_logging.py"
Cohesion: 0.08
Nodes (28): _make_manager(), fixture, Memory-manager cleanup/monitoring must be observed, not silently swallowed.…, REL-1: repeated calls with the SAME malformed pool_id emit exactly one log., The HAS_V2_LOGGING=False fallback shim must re-raise when policy='reraise'. If…, The fallback shim must swallow when policy='suppress' (existing behaviour)., Exercise the fallback shim directly under forced HAS_V2_LOGGING=False.…, A virtual_memory_path containing '..' must be rejected before any directory is… (+20 more)

### Community 143 - "ProjectDialogHandler"
Cohesion: 0.05
Nodes (26): workflow LOW/agy NIT: a non-blank malformed numeric surfaces (not silently…, test_create_config_dialog_collects_inputs(), test_create_config_dialog_omits_blank_optionals(), test_create_config_dialog_raises_on_invalid_numeric(), CreateConfigDialog, _parse_optional(), File-menu config dialogs: Create Config (from templates) and Edit Config…, Return the chosen analysis mode. (+18 more)

### Community 144 - "_NullLogger"
Cohesion: 0.14
Nodes (14): _CapturingOptimizer, _FakeStratifiedData, _laminar_dataset(), _NullLogger, Any, ndarray, Shared fakes/fixtures for laminar-flow hybrid-streaming regression tests.…, Stand-in for NLSQ's ``AdaptiveHybridStreamingOptimizer``: captures the… (+6 more)

### Community 145 - "test_escape_disabled_hint.py"
Cohesion: 0.18
Nodes (15): Actionable hint when a heterodyne joint fit fails with NO global escape…, A failed escape-less fit logs a hint naming cmaes.enable and n_seeds., Missing ``success`` defaults to converged (no hint); the helper never raises., test_log_hint_emits_actionable_cmaes_message(), test_log_hint_robust_to_missing_attributes(), test_log_hint_silent_when_escape_already_enabled(), test_log_hint_silent_when_fit_succeeded(), test_no_hint_when_cmaes_already_enabled() (+7 more)

### Community 146 - "test_hybrid_streaming_retry.py"
Cohesion: 0.17
Nodes (22): _AlwaysFailsOptimizer, _enable(), _fake_hybrid_streaming_config(), _FlakyOptimizer, _logger(), Any, MonkeyPatch, ndarray (+14 more)

### Community 147 - "test_homodyne_engine_preservation.py"
Cohesion: 0.11
Nodes (22): load_or_init_golden(), Any, Path, Golden-snapshot load/init helper for the parity preservation suites. Mechanism:…, Return the golden arrays at ``path``. Args: path: Destination ``.npz`` path…, _build_laminar_fit(), _build_stratified_residual_fn(), ndarray (+14 more)

### Community 148 - "test_runtime_shell.py"
Cohesion: 0.14
Nodes (20): _bash_executable(), parametrize, Path, Tests for xpcsjax.runtime.shell and the runtime package re-exports., Resolve the real Git-Bash executable, not the WSL launcher stub. Windows…, Source xla_config.bash with ``mode`` in an isolated HOME. An empty ``mode``…, _source_xla_config(), test_get_completion_script_returns_absolute_existing_path() (+12 more)

### Community 149 - "FitQueueController"
Cohesion: 0.06
Nodes (31): _FakeHandle, QObject, Regression tests for the click-path-audit findings (state-conflict bugs). Each…, Minimal WorkerHandle stand-in: never spawns a real process (mirrors…, test_add_dataset_does_not_retarget_while_run_active(), test_dataset_row_selection_updates_active_dataset(), test_load_config_preserves_prior_sidebar_selection(), test_on_open_project_surfaces_malformed_file_as_warning_not_crash() (+23 more)

### Community 150 - "_unpack_result_params"
Cohesion: 0.14
Nodes (14): Regression: homodyne per-angle results are ``[c_0..N-1, o_0..N-1, phys]``. The…, Heterodyne result with wrong param count should raise ValueError., Per-angle layout: [c_0..N-1, o_0..N-1, 14 physical]., Homodyne result with <3 params should raise ValueError., test_unpack_heterodyne_per_angle_layout(), test_unpack_heterodyne_size_mismatch_raises(), test_unpack_homodyne(), test_unpack_homodyne_per_angle_layout_offset_not_misread() (+6 more)

### Community 151 - "maps.py"
Cohesion: 0.10
Nodes (31): _make_h5(), test_c2_preview_missing_dataset_returns_none(), test_c2_preview_missing_group_returns_none(), test_c2_preview_nested_group_at_key_returns_none(), test_c2_preview_reconstructs_group_half_matrix(), test_c2_preview_slices_and_downsamples(), test_c2_preview_unknown_type_group_path_returns_none(), test_metadata_lists_datasets_without_loading() (+23 more)

### Community 152 - ".from_optimization_block"
Cohesion: 0.11
Nodes (19): test_safe_float(), test_safe_int(), Explicit YAML nulls mean "unset" -- fall back to defaults, don't crash., `bool(None)` is False -- a null must not turn the safety check off., test_stratification_null_bool_does_not_silently_disable_memory_safety(), test_stratification_null_scalars_fall_back_to_defaults(), Any, Convert *value* to int, returning *default* on failure. Parameters ----------… (+11 more)

### Community 153 - "test_cmaes_trigger.py"
Cohesion: 0.12
Nodes (24): _laminar_cmaes_config(), fixture, parametrize, CMA-ES auto-triggers at scale_ratio >= 1000 (homodyne default). XPCS multi-…, All resolved per-angle modes expand back to the dense per-angle 2*n_phi +…, A CMA-ES solve that actually ran must not be discarded into a failed result.…, A converged fixed-constant CMA-ES solve whose covariance is ``None`` must not…, Realistic XPCS multi-scale bounds (D0 ~ 1e4 vs gamma_dot ~ 1e-3) must trigger.… (+16 more)

### Community 154 - "get_adaptive_memory_threshold"
Cohesion: 0.12
Nodes (27): MonkeyPatch, Concurrency-aware memory-budget routing (OOM-overcommit prevention). Background…, A fit that is STANDARD when alone escalates to OUT_OF_CORE under load., Item-3 regression guard: PYTEST_XDIST_WORKER_COUNT shrinks the budget. This is…, test_detect_fit_concurrency_default_is_one(), test_detect_fit_concurrency_explicit_arg_wins(), test_detect_fit_concurrency_fit_env_beats_xdist(), test_detect_fit_concurrency_floors_at_one() (+19 more)

### Community 155 - "RecoveryStrategyApplicator"
Cohesion: 0.12
Nodes (21): Exception, _strategy_name(), test_convergence_error_strategy_order(), test_numerical_error_strategy_order(), test_passthrough_strategies_return_copy(), test_perturb_parameters_reproducible_and_shaped(), test_perturb_zero_param_uses_additive_fallback(), test_should_retry() (+13 more)

### Community 156 - "StratifiedResidualFunctionJIT"
Cohesion: 0.20
Nodes (20): _full_grid_chunk(), SimpleNamespace, Tests for xpcsjax.optimization.nlsq.strategies.residual_jit.…, _stratified(), test_dt_none_uses_fallback_and_warns(), test_empty_chunks_raises(), test_fixed_scaling_constant_mode(), test_get_diagnostics() (+12 more)

### Community 157 - "._write_json"
Cohesion: 0.20
Nodes (7): test_write_json_emits_strict_json_for_non_finite(), Any, ndarray, Path, Write NPZ file in background., Write JSON file in background., Submit an arbitrary callable for background execution.

### Community 158 - "make_cfgmgr_and_data"
Cohesion: 0.07
Nodes (38): make_cfgmgr_and_data(), Build a real ``ConfigManager`` plus a heterodyne-loader data dict. Returns…, parametrize, Routing tests for the heterodyne standard-tier stratification gate. The gate…, LARGE memory tier + hybrid_streaming.enable=true → hybrid path, not stratified-…, per_angle_mode=individual + >=1M points → stratified-LS solver IS called., >=1M points in ``constant`` per-angle mode → stratified-LS solver IS called.…, Flat enable_cmaes=true (no nested cmaes block) + >=1M → stratified-LS NOT… (+30 more)

### Community 159 - "test_debug_audit_2026_07_23_sigma_weighting.py"
Cohesion: 0.08
Nodes (20): _CapturingAdaptiveOptimizerSpy, _FakeStratifiedDataWithSigma, _HierarchicalOptimizerSpy, Regression: the L2 hierarchical loss must honor per-point sigma weighting,…, Stand-in for HierarchicalOptimizer: captures loss_fn/grad_fn and returns a stub…, _sigma_weighted_mse (the arithmetic building block _hier_loss/_loss_jax call)…, Audit follow-up (2026-07-23): the plan's initial fix only edited the `else`…, Stand-in for NLSQ's AdaptiveHybridStreamingOptimizer: captures the fit() kwargs… (+12 more)

### Community 160 - "test_gradient_diagnostics.py"
Cohesion: 0.19
Nodes (20): _laminar_params(), _make_data(), SimpleNamespace, Coverage tests for `xpcsjax.optimization.nlsq.gradient_diagnostics`. Closes the…, Baseline params should get x_scale ≈ 1.0 (they ARE the baseline). The…, _static_params(), test_compute_gradient_norms_laminar_returns_all_seven_keys(), test_compute_gradient_norms_static_returns_three_keys() (+12 more)

### Community 161 - "test_heterodyne_tied_residuals.py"
Cohesion: 0.13
Nodes (21): ParameterManager, Direct residual-closure tests: every wired heterodyne residual function must…, heterodyne_core.py: _fit_cmaes's model_func (per-angle CMA-ES escape)., heterodyne_core.py: _fit_local's jax_residual_fn., heterodyne_core.py: _make_numpy_residual_fn's residual_fn., heterodyne_stratified_ls.py: residual_fn (scaling-first: scaling head, physics…, strategies/heterodyne_hybrid_streaming.py: model_fn (scaling-first: physics is…, Sanity check exercised again here (redundant with Task 2's test, kept as the… (+13 more)

### Community 162 - "test_parallel_accumulator.py"
Cohesion: 0.18
Nodes (20): _int_chunks(), _kernels(), MonkeyPatch, parametrize, Scientific tests for xpcsjax.optimization.nlsq.parallel_accumulator. Three…, Integer-valued (JtJ, Jtr, chi2) chunks — exact under float summation., _static_physics_config(), test_accumulate_sequential_empty_raises() (+12 more)

### Community 163 - "test_phase5_standard_resolver.py"
Cohesion: 0.14
Nodes (19): _fit(), Phase 5 — averaged uses EXPANDED constrained-model DOF (2*n_phi + n_physics)., test_averaged_dof_basis_is_expanded(), test_constant_dof_basis_is_physics_only(), test_individual_dof_basis_is_optimizer(), Phase 5 — averaged/constant results expand back to the dense per-angle layout., test_averaged_result_is_dense_per_angle(), test_constant_result_is_dense_per_angle() (+11 more)

### Community 164 - "execute_optimization_with_fallback"
Cohesion: 0.09
Nodes (26): Audit [2026-07-22], updated [2026-07-23] (PR #15 review, Finding #3 scope…, Audit [2026-07-23] (PR #15 review): the plain (enable_recovery=False, non-…, Audit [2026-07-23]: a STREAMING soft-failure (success=False, no exception) must…, Audit [2026-07-23]: enable_recovery=True's execute_with_recovery returning…, test_fallback_no_recovery_reports_failed_on_stagnation(), test_plain_soft_failure_escalates_to_next_strategy(), test_recovery_soft_failure_escalates_to_next_strategy(), test_streaming_soft_failure_escalates_to_chunked() (+18 more)

### Community 165 - "ndarray"
Cohesion: 0.11
Nodes (12): ndarray, ValidationResult, Get initial parameter values for optimization. Returns the config-specified…, Get all 14 parameter values. Returns a read-only cached array…, Get bounds for the varying physics parameters. Returns ------- tuple of…, Expand varying parameters to full 14-parameter array. Ordinary fixed parameters…, Expand a reduced joint optimizer result to the full-14-physics layout. The…, Extract the varying parameters from a full array. Parameters ----------… (+4 more)

### Community 166 - "validate_backend"
Cohesion: 0.09
Nodes (23): _create_gradient_fallback(), _create_hessian_fallback(), _create_no_gradient_fallback(), _create_no_hessian_fallback(), get_device_info(), _get_performance_recommendations(), get_performance_summary(), grad() (+15 more)

### Community 167 - "data_prep.py"
Cohesion: 0.11
Nodes (21): test_build_parameter_labels_no_per_angle(), test_build_parameter_labels_per_angle(), build_parameter_labels(), classify_parameter_status(), convert_bounds_to_nlsq_format(), ExpandedParameters, PreparedData, Any (+13 more)

### Community 168 - "References and Citations"
Cohesion: 0.13
Nodes (21): Andrade 1910 (Viscosity / rheology), Bradbury et al. 2018 (JAX), Duri et al. 2005 (Time-resolved correlation), Hansen 2016 (CMA-ES), He et al. 2024 (Transport coefficient, PNAS), He et al. 2025 (Bridging microscopic dynamics and rheology, PNAS), Kubo 1966 (Fluctuation-dissipation theorem), Lumma et al. 2000 (Area detector photon correlation) (+13 more)

### Community 169 - "HomodyneModel"
Cohesion: 0.06
Nodes (32): HomodyneModel must reject negative end_frame before constructing., end_frame=-1 must raise ValueError before any JAX computation., Error message must reference 'sentinel' so callers understand the fix., A properly resolved end_frame must not raise., TestHomodyneModelInitValidation, fixture, Shared pytest fixtures for the parity test suite., Small synthetic laminar_flow fit that routes through the live NLSQWrapper… (+24 more)

### Community 170 - "test_cache_loader_security.py"
Cohesion: 0.15
Nodes (22): cache_engine(), Any, fixture, MonkeyPatch, Path, skipif, Defense-in-depth regression tests for the trusted-cache loader. The /double-…, Path-containment gate: an explicit ``..``-traversal path fails. Even if a key-… (+14 more)

### Community 171 - "_Any"
Cohesion: 0.10
Nodes (13): load_xpcs_config(), _Any, Load an XPCS configuration file and return the parsed mapping. Convenience…, Return the current configuration mapping. Returns ------- dict The live…, Set a configuration value addressed by a dot-notation key. Intermediate…, Report whether the configured analysis mode is a static mode. Returns -------…, Construct the physics model class for this config's analysis mode. Thin wrapper…, Return the angle-filtering configuration block. Returns ------- dict The… (+5 more)

### Community 172 - "test_system_validator.py"
Cohesion: 0.15
Nodes (16): MonkeyPatch, Tests for xpcsjax.runtime.utils.system_validator. Covers the version-parsing…, test_config_templates_probe_finds_all(), test_cpu_info_probe_reports(), test_dependency_versions_probe_passes(), test_jax_installation_probe_x64_enabled(), test_main_returns_one_on_error(), test_python_version_probe_passes() (+8 more)

### Community 173 - "NLSQ vs xpcsjax ownership split contract"
Cohesion: 0.11
Nodes (20): select_nlsq_strategy (memory routing), Angle-stratified chunking (strategies/stratified_ls.py), 5-layer anti-degeneracy controller, Bounds and parameter transforms (transforms.py), CMA-ES escape (cmaes_wrapper, evosax BIPOP backend), nlsq.CurveFit (JIT-cached trust-region entry point), LHS multistart (multistart.py), NLSQ vs xpcsjax ownership split contract (+12 more)

### Community 174 - "XpcsDataset"
Cohesion: 0.27
Nodes (9): dict, Typed XpcsDataset at the load/fit boundary. Quality-gate type-design finding:…, _raw(), test_is_dict_subclass_backward_compatible(), test_missing_correlation_raises_clear_error(), test_typed_accessors_resolve_canonical_and_alias_keys(), Typed container for loaded XPCS experimental data. ``XpcsDataset`` is the named…, A loaded XPCS dataset: a ``dict`` with typed, alias-resolving accessors. (+1 more)

### Community 175 - "physics_nlsq.py"
Cohesion: 0.18
Nodes (20): _grid_reference_diffusion(), _grid_reference_shear(), parametrize, Regression tests for the degenerate 1x1-grid branch in physics_nlsq.py. The…, test_diffusion_degenerate_matches_grid(), test_shear_degenerate_matches_grid(), _compute_g1_diffusion_meshgrid(), _compute_g1_shear_meshgrid() (+12 more)

### Community 176 - "test_fit_queue.py"
Cohesion: 0.19
Nodes (13): _FakeHandle, QObject, _queue(), pytest-qt tests for the bounded-concurrency fit queue (fake handles)., test_bounded_concurrency_runs_one_at_a_time(), test_cancel_active_frees_slot_and_starts_next(), test_cancel_active_removes_partial_output_dir(), test_cancel_and_shutdown() (+5 more)

### Community 177 - "service/config.py"
Cohesion: 0.11
Nodes (28): agy#2 (service side): scalar initial_parameters yields an error, not…, test_validate_config_scalar_initial_parameters_reports_not_crashes(), _probe_import(), Tests for the argparse-free config loader (xpcsjax/service/config.py)., test_config_service_is_jax_free(), test_load_config_applies_mode_and_output(), test_load_config_no_overrides_leaves_config(), _ip() (+20 more)

### Community 178 - "test_main_window.py"
Cohesion: 0.11
Nodes (17): QCloseEvent, _FakeHandle, QObject, pytest-qt smoke tests for MainWindow (logic-free view, controller-less…, Minimal WorkerHandle stand-in: never spawns a real process., Closing a project must not orphan queued/active fit workers (codex review).…, Two runs for the SAME dataset must write to distinct dirs (no overwrite)., test_close_event_calls_queue_shutdown() (+9 more)

### Community 179 - "json_serializer"
Cohesion: 0.17
Nodes (8): TestJsonSerializer, I/O operations for xpcsjax XPCS analysis. This module provides functions for…, json_serializer(), Any, JSON utility functions for xpcsjax I/O operations. This module provides helper…, JSON serializer for numpy arrays and other objects. Use as the `default`…, Convert non-finite floats to JSON-safe representations. JSON spec does not…, _sanitize_float()

### Community 180 - "fit_computation.py"
Cohesion: 0.17
Nodes (17): compute_g2_batch(), compute_g2_batch_with_per_angle_scaling(), compute_theoretical_fits(), extract_parameters_from_result(), get_physical_param_count(), normalize_analysis_mode(), Any, ndarray (+9 more)

### Community 181 - "ParameterInfo"
Cohesion: 0.14
Nodes (8): ParameterInfo, Reject inverted bounds at construction (frozen-dataclass safe). ``lower_bound >…, Alias for ``lower_bound`` (upstream heterodyne API)., Alias for ``upper_bound`` (upstream heterodyne API)., Alias for ``units`` (upstream heterodyne API uses singular)., Check if value is within bounds (upstream heterodyne API)., Clip value to bounds (upstream heterodyne API)., Metadata for a single parameter. Attributes ---------- name : str Canonical…

### Community 182 - "HeterodyneModel"
Cohesion: 0.06
Nodes (32): _build_synthetic_c2(), ndarray, Path, Task 30: end-to-end heterodyne smoke fit on synthetic two-component data.…, A real ``individual`` joint fit returns scaling-first params + conserves SSR., Self-contained heterodyne config sufficient for HeterodyneModel.from_config.…, Forward-evaluate the model at each phi to build the c2 stack., End-to-end heterodyne fit on synthetic data must converge and recover… (+24 more)

### Community 183 - "Any"
Cohesion: 0.08
Nodes (23): 2026-07-22 audit Fix 2: MemoryMapManager must not evict (close) a file handle…, test_memory_map_manager_refcount_blocks_concurrent_eviction(), log_calls(), log_performance(), MemoryMapManager, Any, F, Initialize the performance engine from configuration. Builds the memory-map… (+15 more)

### Community 184 - "test_plots_view.py"
Cohesion: 0.14
Nodes (16): parametrize, Tests for interactive PyQtGraph plot widgets (plots_view)., Colorbar construction must degrade gracefully, like the image item's own…, test_map_applies_color_lookup_table(), test_map_axes_labelled_t1_t2(), test_map_keeps_equal_t1_t2_range_without_aspect_lock(), test_map_view_survives_unresolvable_colormap(), test_phi_grid_pins_scrollbars_to_keep_square_tiles() (+8 more)

### Community 185 - "test_quality_gate_fixes.py"
Cohesion: 0.15
Nodes (18): parametrize, Regression tests for the quality-gate audit fixes. Locks in the behavioral…, StrEnum result must still compare/serialize as its string value., Construct a minimal OptimizationResult, overriding selected fields., A genuinely failed fit may carry empty parameters — not an error., _result(), test_converged_result_with_empty_parameters_is_rejected(), test_converged_result_with_nonfinite_parameters_is_rejected() (+10 more)

### Community 186 - "ValidationResult"
Cohesion: 0.12
Nodes (12): test_result_to_dict_roundtrip(), test_validation_result_defaults(), Check that the running Python is >= 3.12. Returns ------- ValidationResult INFO…, Verify JAX imports, exposes devices, and has x64 precision enabled. Returns…, Import xpcsjax and resolve every public lazy-loaded symbol. Returns -------…, Verify every shipped YAML config template is present on disk. Returns -------…, Report CPU core count and system RAM (informational only). Returns -------…, Check that xpcsjax's ``XLA_FLAGS`` configuration was applied. Returns -------… (+4 more)

### Community 187 - "save_nlsq_npz_file"
Cohesion: 0.25
Nodes (8): _make_npz_arrays(), Build minimal valid arrays for save_nlsq_npz_file., JAX arrays must be coerced without error., TestSaveNlsqNpzFile, ndarray, Path, Save NPZ file with experimental/theoretical data and metadata. Parameters…, save_nlsq_npz_file()

### Community 188 - "compute_diagonal_overlay_stats"
Cohesion: 0.16
Nodes (16): Tests for compute_diagonal_overlay_stats., test_diagonal_overlay_out_of_bounds_raises(), test_diagonal_overlay_rmse_matches_manual(), test_diagonal_overlay_shapes_match(), test_diagonal_overlay_variance_ignores_single_inf(), test_compute_diagonal_overlay_stats_2d_input_raises(), test_compute_diagonal_overlay_stats_shape_mismatch_raises(), compute_diagonal_overlay_stats() (+8 more)

### Community 189 - "heterodyne_parameter_space.py"
Cohesion: 0.13
Nodes (16): Build default bounds lookup from the registry, then merge config overrides., Merge config-overridden bounds from ParameterSpace into _default_bounds.…, _apply_initial_parameters(), _apply_parameter_space_bounds(), _apply_tied_parameters(), Any, Parameter space definition with bounds for heterodyne NLSQ optimization., Serialize this space to a dict compatible with :meth:`from_config`. Produces… (+8 more)

### Community 190 - "fit_with_out_of_core_accumulation"
Cohesion: 0.13
Nodes (18): Regression: explicit ``per_angle_mode="averaged"`` expands DOF like ``"auto"``.…, test_explicit_averaged_matches_auto_expansion(), test_should_use_parallel_accumulation(), test_should_use_parallel_compute(), Determine if parallel chunk COMPUTE is worthwhile. Parameters ----------…, Determine if parallel accumulation is worthwhile. Parameters ----------…, should_use_parallel_accumulation(), should_use_parallel_compute() (+10 more)

### Community 191 - "BatchStatistics"
Cohesion: 0.13
Nodes (11): BatchStatistics, Any, Batch-level statistics tracking for streaming optimization. This module…, Calculate success rate from recent batches in buffer. Returns ------- float…, Calculate average loss from recent successful batches. Returns ------- float…, Calculate average iterations from recent batches. Returns ------- float Average…, Circular buffer for tracking batch-level statistics. Maintains statistics for…, Return comprehensive statistics dictionary. Returns ------- dict Dictionary… (+3 more)

### Community 192 - "load_viz_bundle"
Cohesion: 0.22
Nodes (12): NpzFile, Tests for the JAX-free viz bundle (writer-shape + loader)., test_exp_only_when_no_fitted_surface(), test_load_full_bundle_from_fitted_artifact(), test_load_missing_returns_none(), _write_fitted(), _get(), load_viz_bundle() (+4 more)

### Community 193 - "build_failed_result"
Cohesion: 0.10
Nodes (23): parametrize, test_build_failed_result_with_initial(), test_build_failed_result_without_initial(), test_build_from_arrays_with_and_without_jacobian(), test_build_from_scipy_with_jacobian(), test_build_from_scipy_without_jacobian(), test_compute_covariance_regularizes_near_singular(), test_compute_covariance_well_conditioned() (+15 more)

### Community 194 - "test_debug_audit_2026_06_18.py"
Cohesion: 0.10
Nodes (17): test_config_defaults(), LogCaptureFixture, parametrize, Regression tests for the 2026-06-18 whole-codebase debug-audit fixes. Each test…, For output_format='both', the durable NPZ must be written before the JSON, so a…, test_config_manager_null_experimental_data(), test_config_manager_null_initial_parameters_per_angle(), test_config_manager_null_optimization_angle_ranges() (+9 more)

### Community 195 - "MemoryStats"
Cohesion: 0.10
Nodes (16): _as_weak_callable(), log_calls(), MemoryStats, Any, Return comprehensive memory statistics. Returns ------- dict Nested dictionary…, Wrap a bound method so holding it doesn't keep its owner alive. Callers…, Comprehensive memory statistics and monitoring., Refresh the system memory fields from :mod:`psutil` in place. (+8 more)

### Community 196 - "performance_engine.py"
Cohesion: 0.10
Nodes (19): AdaptiveChunker, CacheError, ChunkInfo, get_logger(), MemoryPressureError, PerformanceEngineError, Exception, Advanced performance engine for massive XPCS datasets. High-performance data… (+11 more)

### Community 197 - "PhysicsFactors"
Cohesion: 0.11
Nodes (12): create_physics_factors_from_config_dict(), PhysicsFactors, Pre-computation and caching of physics factors for XPCS calculations.…, Validate physics factors after initialization., Create PhysicsFactors from experimental configuration. This is the recommended…, Validate physics factors for physical consistency. Checks: 1. All values are…, Convert to tuple for JIT-compatible function calls. Returns the two pre-…, Convert to dictionary for serialization or inspection. Returns ------- dict… (+4 more)

### Community 198 - "test_no_pickle_loads.py"
Cohesion: 0.19
Nodes (16): expr, _find_violations(), _is_np_load(), _iter_source_files(), Path, Regression guard: no unsafe ``np.load`` calls inside ``xpcsjax/``. The NPZ…, Sanity: variable-smuggled allow_pickle is flagged., Return True if ``node`` is ``np.load``, ``numpy.load``, or bare ``load``. (+8 more)

### Community 199 - "TestHeterodyneComputeResidual"
Cohesion: 0.17
Nodes (10): fixture, HeterodyneModel, ndarray, HeterodyneModel.compute_residual returns a flat 1-D residual array., compute_residual must return a 1-D array., Length must be n_phi * N * N., Residual against a synthetic all-ones c2 must be finite., When c2_exp == c2_model, residual must be all-zero. (+2 more)

### Community 200 - "test_diagnostics_map.py"
Cohesion: 0.14
Nodes (17): Tests for the JAX-free layer-status map + banner classifier., test_classify_banner_ignores_ordinary_log_lines(), test_classify_banner_recognizes_engine_prefixes(), test_layer_status_l5_inactive_sentinels_are_false(), test_layer_status_maps_diagnostics_keys(), classify_banner(), _l5_active(), layer_status_from_diagnostics() (+9 more)

### Community 201 - "test_anti_degeneracy_layers.py"
Cohesion: 0.06
Nodes (23): Verify the surviving anti-degeneracy layers ported over from homodyne. Task 29…, Homodyne AntiDegeneracyConfig.from_dict must honor config-file values over the…, ``regularization.auto_tune_lambda: False`` must reach the built regularizer.…, Static check: the controller class source must mention every layer name. If a…, ``execute_layers`` round-trips through the homodyne solver ``NLSQConfig``. The…, ``from_dict({})`` must give ``execute_layers is False``., A nested ``anti_degeneracy.execute_layers`` value must be parsed., ``to_dict()["anti_degeneracy"]["execute_layers"]`` echoes the field. (+15 more)

### Community 202 - "test_debug_audit_2026_08_05.py"
Cohesion: 0.13
Nodes (17): Regression tests for the 2026-08-05 data-module deep-RCA debug-audit fixes.…, test_apply_angle_filtering_explicit_none_values_do_not_crash(), test_calculate_matrix_quality_score_never_returns_nan(), test_close_all_closes_handle_not_in_use(), test_close_all_skips_handle_still_checked_out(), test_create_chunked_iterator_rejects_misaligned_phi(), test_create_chunked_iterator_slices_aligned_phi_correctly(), test_migrate_cache_template_all_old_style_migrates_both() (+9 more)

### Community 203 - "test_heterodyne_cmaes_warmstart_auto_skip.py"
Cohesion: 0.15
Nodes (17): _cfg(), skipif, Parity: heterodyne JOINT CMA-ES escapes honor ``cmaes_warmstart_auto_skip``.…, The auto-skip knob is CMA-ES-specific (matches laminar + the knob name)., dof ≤ 0 (more params than data) never skips — guards a meaningless χ²/dof., n_phi=3 → averaged escape (the user's 3-angle case) auto-skips CMA-ES., n_phi=2 → individual escape (``_fit_joint_cmaes_multi_phi``) auto-skips., SSR/dof below threshold ⇒ skip: tag the auto-skip and never call CMA-ES. (+9 more)

### Community 204 - "_version_at_least"
Cohesion: 0.20
Nodes (10): parametrize, test_parse_version(), test_version_at_least(), _is_final_release(), _parse_version(), Parse a PEP 440-ish version string into an int tuple for comparison. Strips…, Return whether ``version`` is a final release (no pre-release/dev suffix).…, Return whether ``actual`` is at least ``minimum``. Both versions are normalized… (+2 more)

### Community 205 - ".__init__"
Cohesion: 0.15
Nodes (9): Any, Logger, ndarray, Extract q, L, dt from chunks (should be same for all chunks)., Extract unique phi, t1, t2 values from ALL chunks. CRITICAL: Must extract from…, Create padded arrays with uniform size across all chunks. Returns ------- tuple…, Compute residuals for a single padded chunk. This function is designed to be…, Compute residuals (interface for NLSQ ``least_squares``). This method is JIT-… (+1 more)

### Community 206 - "test_stratified_max_iter_grading.py"
Cohesion: 0.21
Nodes (14): NLSQWrapper, _good_chi2_inputs(), R2 (status-grading parity): laminar stratified-LS must grade a max_nfev-limited…, status==0 (SciPy max_nfev code) + finite reduced chi^2 -> max_iter, not failed., When no status code is threaded, the SciPy message string also triggers it., A non-budget failure (e.g. status=-1, no max_nfev reason) is not upgraded., A converged solve stays converged regardless of the threaded reason/status., The relabel must not perturb parameters / chi^2 / covariance. (+6 more)

### Community 207 - "xla_config.py"
Cohesion: 0.12
Nodes (17): parametrize, test_build_parser_returns_parser(), test_post_install_shell_choices_preserved(), test_xla_config_threads_roundtrip(), build_parser(), configure_xla(), get_cpu_info(), main() (+9 more)

### Community 208 - "test_heterodyne_grouped_coercion.py"
Cohesion: 0.12
Nodes (5): Regression test for grouped-format value/bounds coercion (audit C10). The…, A config specifying both spellings of the same parameter must raise, not…, min == max is a fixed parameter, not an error (mirrors the registry)., test_equal_bounds_allowed(), test_grouped_format_rejects_alias_and_canonical_collision()

### Community 209 - "test_cache_no_pickle_exec.py"
Cohesion: 0.17
Nodes (15): cache_engine(), _Evil, _plant_pickle(), Any, fixture, MonkeyPatch, Path, SEC-1 regression: the disk cache must never execute pickle payloads. The… (+7 more)

### Community 210 - "result_builder.py"
Cohesion: 0.16
Nodes (16): Convert MultiStartResult to OptimizationResult for CLI compatibility. Returns…, compute_quality_metrics(), compute_uncertainties(), determine_convergence_status(), normalize_nlsq_result(), Any, ndarray, QualityMetrics (+8 more)

### Community 211 - "test_layer5_gating.py"
Cohesion: 0.16
Nodes (16): _make_controller(), parametrize, ShearSensitivityWeighting (anti-degeneracy Layer 5) is gated by analysis mode.…, Build a controller with minimal-but-valid arguments., Layer 5 is active ONLY for laminar_flow (the mode with a shear rate)., Layer 5 is inactive for static modes — no flow direction, no shear peak., Layer 5 is inactive for two_component (heterodyne) mode., The 'heterodyne' synonym must produce the same gating. (+8 more)

### Community 212 - "test_debug_audit_regressions.py"
Cohesion: 0.11
Nodes (20): _make_aps_old_hdf5(), parametrize, Regression tests for the 2026-06-10 whole-codebase debug audit. Each test pins…, Audit [26]: get_group_indices('scaling') must resolve, not KeyError. The…, 2026-07-22 audit Fix 1: the APS-old quality-filtering branch must probe-then-…, Audit [2026-07-22]: the sequential per-angle fallback's reduced-chi2…, Audit [8]: the active-parameter fallback for two_component must return the…, Audit [25]: with analysis_mode absent, the .analysis_mode property and the… (+12 more)

### Community 213 - "validators.py"
Cohesion: 0.15
Nodes (16): test_validate_numeric_range_rejects_nan_when_wrapped_and_unbounded(), test_validate_numeric_range_rejects_nan_with_require_positive(), test_validate_positive_value_rejects_nan(), Regression test: validate_by_rules' phi_range rule must accept wrapped ranges…, test_validate_by_rules_accepts_wrapped_phi_range(), test_validate_by_rules_still_rejects_out_of_bounds_phi_range(), Any, Configuration validators for XPCS data loading. Focused validator functions for… (+8 more)

### Community 214 - "test_codex_review_fixes.py"
Cohesion: 0.15
Nodes (17): _laminar_fit(), parametrize, Regression tests for the two confirmed Codex adversarial-review findings. Both…, End-to-end guard for BOTH transform fixes: a laminar fit with config-enabled…, The JIT-safe jnp inverse must be numerically identical to the numpy inverse…, The single DOF authority the out-of-core / hybrid-streaming / stratified-LS…, With ``scaling_head_size`` given, physics indices start at the head end and…, Without ``scaling_head_size`` the legacy dense ``2*n_phi`` head is preserved… (+9 more)

### Community 215 - "_logger_that_raises_on_log"
Cohesion: 0.17
Nodes (10): _logger_that_raises_on_log(), Logger, Return a fresh logger whose only handler always raises on emit., log_calls: a raising handler must never abort the decorated function., Logger raises on the *entry* emit; function must still return., Logger raises on the *success* emit; function must still return., Even when the logger raises, a real function exception still propagates., include_args=True path: raising logger must not abort the call. (+2 more)

### Community 216 - "test_gradient_monitor.py"
Cohesion: 0.19
Nodes (14): _monitor(), Tests for xpcsjax.optimization.nlsq.gradient_monitor. The monitor is a pure…, test_check_collapse_after_consecutive_triggers(), test_check_disabled_returns_ok(), test_check_healthy_gradient_is_ok(), test_check_rearms_after_recovery(), test_check_skips_off_interval(), test_check_tracks_best_params() (+6 more)

### Community 217 - "post_install.py"
Cohesion: 0.14
Nodes (23): test_completion_and_xla_source_paths_exist(), test_fish_completion_is_nonfatal_noop_but_xla_stays(), test_install_xla_config_scripts(), get_completion_source_path(), get_venv_path(), get_xla_config_source_path(), install_bash_completion(), _install_completion_bash_activation() (+15 more)

### Community 218 - "test_persist.py"
Cohesion: 0.13
Nodes (15): _probe_import(), Persist service: import-path equivalence + JAX-free guard., Off-shape uncertainties/covariance must be written at the documented shapes.…, nlsq_result.npz must round-trip with allow_pickle=False (no SEC-1 regression).…, A 0-D result.parameters (__post_init__ permits it) must not IndexError.…, 0-D numpy arrays (scalar arrays) must coerce, not crash. Regression:…, test_extract_parameters_handles_zero_dim_result(), test_json_safe_handles_zero_dim_ndarray() (+7 more)

### Community 219 - "test_memory_routing.py"
Cohesion: 0.17
Nodes (15): Direct unit tests for memory-aware NLSQ strategy routing. Localizes router…, Normalize the returned value (enum, string, dataclass) to upper-case name., Small datasets fit in memory — STANDARD strategy., When peak Jacobian memory exceeds the adaptive threshold, the router escalates.…, memory_fraction below 0.1 or above 0.9 is clamped (with a warning)., Audit finding #17: n_params<=0 forces peak_memory_gb to 0.0 (can't estimate a…, Audit finding #17: exercise the OUT_OF_CORE branch specifically — peak Jacobian…, Smoke check: the router accepts XPCS-typical sizes without crashing. (+7 more)

### Community 220 - "test_config_unwrap.py"
Cohesion: 0.19
Nodes (13): captured_nlsq(), _fake_data(), fixture, Regression test for the heterodyne config unwrap in ``_fit_nlsq_heterodyne``.…, Already-flat dicts (legacy/tests) must still parse correctly., ``analysis_mode`` placed only in the nested NLSQ section must reach NLSQConfig.…, Minimal ConfigManager replacement holding only ``self.config``., Patch HeterodyneModel + fit_nlsq_multi_phi to capture the NLSQConfig. (+5 more)

### Community 221 - "test_laminar_streaming_diag.py"
Cohesion: 0.18
Nodes (14): _build_sequential_laminar_fit(), Diagnostics-parity tests for laminar non-in-memory result builders. The in-…, Regression: the Site 4 (sequential per-angle) covariance-rescale call…, Reuse the small synthetic laminar fixture but force the SEQUENTIAL per-angle…, Site 4 (sequential per-angle) result carries the symmetric anti-degeneracy…, test_block_honest_active_from_streaming_info(), test_block_inactive_for_stratified_controller_only_info(), test_block_markers_when_no_info() (+6 more)

### Community 222 - "test_laminar_no_worse_phase6.py"
Cohesion: 0.33
Nodes (8): _homodyne_config(), parametrize, Phase 6 no-worse-SSR gate: deleting the truncated-basis mode on the laminar…, Self-consistent g2-at-truth drives residuals -> ~0; the individual solve must…, _stratified_info(), test_fourier_rejected_all_laminar_paths(), test_stratified_individual_ssr_tripwire(), test_stratified_ssr_finite()

### Community 223 - "load_dataset"
Cohesion: 0.13
Nodes (25): _cm(), SimpleNamespace, Tests for the argparse-free data service (xpcsjax/service/data.py)., An empty phi array must warn-and-skip, not raise from ``np.argmin``.…, End-to-end path the caller-side ``is not None`` guard failed to cover., test_load_dataset_empty_phi_does_not_crash(), test_load_dataset_no_subset_keeps_all(), test_load_dataset_raises_when_no_correlation_matrix() (+17 more)

### Community 224 - "_validate_loaded_arrays"
Cohesion: 0.32
Nodes (14): _good_data(), DATA-2 / threat-03: hard-fail validation of loaded correlation arrays. The…, test_accepts_all_nan_q_list(), test_accepts_nan_in_q_list(), test_accepts_non_monotonic_q_list(), test_accepts_valid_data(), test_frame_guard_is_invoked_for_3d_buffer(), test_rejects_inf_in_q_list() (+6 more)

### Community 225 - "logged_errors"
Cohesion: 0.14
Nodes (10): _contextmanager, log_exception(), logged_errors(), Handle memory pressure recovery., Clean up any virtual memory files., Shutdown memory manager and cleanup resources., Exit the context manager, releasing resources via :meth:`shutdown`., Destructor to ensure cleanup when garbage collected. (+2 more)

### Community 226 - "completion.sh"
Cohesion: 0.34
Nodes (13): _filedir(), _init_completion(), mapfile(), completion.sh script, _xpcsjax(), _xpcsjax_cleanup(), _xpcsjax_config(), _xpcsjax_config_xla() (+5 more)

### Community 227 - "test_debug_audit_2026_07_23_negative_correlation_repair.py"
Cohesion: 0.22
Nodes (12): _data_with_one_skipped_matrix(), _data_with_one_skipped_matrix_robust(), Regression: negative-correlation repair must not clamp matrices that were…, End-to-end (design spec Testing item 6): thread the REAL _normalize_data output…, Integer-dtype c2_exp must not truncate the 1e-6 floor to exactly 0. ``arr[mask]…, ROBUST method's zero-IQR skip branch must also be tracked per-matrix, mirroring…, test_normalize_data_tracks_per_matrix_mask(), test_normalize_data_tracks_per_matrix_mask_robust() (+4 more)

### Community 228 - "ndarray"
Cohesion: 0.19
Nodes (11): compute_averaged_scaling(), estimate_per_angle_scaling(), Logger, LoggerAdapter, ndarray, Return the full scaling parameter array. Returns ------- np.ndarray Array of…, Return only the varying scaling parameter values. Returns ------- np.ndarray…, Update scaling parameters from optimizer output. Parameters ----------… (+3 more)

### Community 229 - "test_laminar_execute_layers.py"
Cohesion: 0.22
Nodes (12): _build_laminar_stratified_data(), _fit(), Laminar stratified-LS ``execute_layers`` (Phase 3 step 11) — direct-call…, R2a: the stratified-LS ``info`` carries the SciPy termination reason + status…, Flag OFF: no L2 executes; honest inactive markers (byte-identical path)., Flag ON: L2 executes (+ L3), keep-better holds, objective is data-only., L2 alone (no regularization configured) still executes and keeps better., Synthetic ≥-chunk laminar dataset: 5 angles, self-consistent g2 at truth. (+4 more)

### Community 230 - "NLSQOptimizationError"
Cohesion: 0.17
Nodes (8): NLSQOptimizationError, Exception, ndarray, Initialize convergence error. Parameters ---------- message : str Detailed…, Initialize numerical error. Parameters ---------- message : str Detailed error…, Base exception for all NLSQ optimization errors. This is the base class for all…, Initialize base optimization error. Parameters ---------- message : str…, Return formatted error message with context.

### Community 231 - "Fit pipeline (load -> optimise -> save -> plot)"
Cohesion: 0.17
Nodes (12): cli.config_handling stage, cli.data_pipeline stage, Fit pipeline (load -> optimise -> save -> plot), Flat argument parser (flag-driven mode selection), cli.optimization_runner stage, cli.plot_dispatch stage, cli.result_saving stage, Standalone-plot path (skips optimisation) (+4 more)

### Community 232 - "test_post_install_fish.py"
Cohesion: 0.36
Nodes (11): _appended_block(), _make_fake_venv(), Path, Regression tests for fish-shell XLA activation under conda. Adversarial-review…, If fish is installed, the generated activate.fish must parse cleanly. Guards…, test_fish_activation_is_idempotent(), test_fish_activation_missing_script_returns_false(), test_fish_activation_resolves_conda_prefix() (+3 more)

### Community 233 - "ResultPresenter"
Cohesion: 0.16
Nodes (9): Any, QObject, Render a fit failure in the text panel, with a color-coded header. Parameters…, Populate the inspector dock with *summary* (or clear on None). Parameters…, Owns the result/error/inspector presentation bodies (operates on MainWindow…, Initialise and attach to *main_window* as Qt parent. Parameters ----------…, Render the finished-fit summary (a ResultSummary or None) in the text panel.…, Render the result: per-phi grid when a bundle exists, text otherwise.… (+1 more)

### Community 234 - "test_validation_crash_coverage.py"
Cohesion: 0.29
Nodes (11): _fresh_report(), _raise_boom(), F8 TEST-1 GAP-3: crash-logging regression tests for the four missing validator…, A crash inside ``_validate_statistical_properties`` must log ERROR and fail the…, A crash inside ``_compute_data_statistics`` must log ERROR and fail the report., A crash inside ``_validate_physics_parameters`` must log ERROR and fail the…, A crash inside ``_validate_correlation_matrices`` must log ERROR and fail the…, test_compute_data_statistics_crash_logs_error_and_invalidates_report() (+3 more)

### Community 235 - "fit_nlsq"
Cohesion: 0.13
Nodes (19): Task 2 gate: ``on_iteration`` observer threads through the NLSQ engine. The…, fit_nlsq accepts on_iteration and completes without error. Any callback firings…, fit_nlsq() with on_iteration=None must be bit-identical to the default. This is…, A raising observer must be silently swallowed; the fit must complete., fit_nlsq on two_component config: on_iteration accepted, never called. The…, Return (config, data) for the smallest laminar_flow fit. Reuses the exact…, test_default_none_does_not_change_result(), test_on_iteration_accepted_and_wellformed() (+11 more)

### Community 236 - "OOCSharedArrays"
Cohesion: 0.21
Nodes (8): shm_required, test_shared_arrays_roundtrip_and_cleanup(), test_shared_arrays_without_sigma(), OOCSharedArrays, Shared memory manager for OOC flat data arrays. Parameters ---------- phi_flat,…, Close and unlink all shared memory blocks., Enter the context manager, returning self., Exit the context manager, unlinking all shared memory blocks.

### Community 237 - "test_phase5_model_function_modes.py"
Cohesion: 0.30
Nodes (11): _data_obj(), _make_fn(), _phys(), Phase 5 — the JIT model_function slices per resolved mode (no crash, correct…, # NOTE: plan named ``xpcsjax.core.analysis.AnalysisMode`` / ``NLSQFitter``; the, Averaged with (c,o) must equal individual with all angles = (c,o)., Raw (non-stratified) grid data object the standard path builds., test_averaged_equals_individual_when_uniform() (+3 more)

### Community 238 - "test_model_adapter.py"
Cohesion: 0.18
Nodes (11): Tests for the model-agnostic ``PointEvaluator`` adapter. Phase 1.1 introduces a…, The heterodyne adapter satisfies the runtime-checkable Protocol., ``HomodynePointEvaluator.eval_points`` must equal the raw kernel exactly., The adapter satisfies the runtime-checkable Protocol., ``HeterodynePointEvaluator`` must plug into the stratification ENGINE. The…, test_heterodyne_evaluator_is_a_point_evaluator(), test_heterodyne_evaluator_returns_per_angle_meshgrid(), test_homodyne_evaluator_is_a_point_evaluator() (+3 more)

### Community 239 - "test_docs_structure.py"
Cohesion: 0.23
Nodes (11): _missing_api_pages(), Every top-level xpcsjax submodule needs a Sphinx API page (ADR-0001).…, Every importable top-level xpcsjax submodule (has __init__.py), sorted., Submodules (minus `excluded`) absent from `existing_page_names` — pure, no I/O., Each top-level xpcsjax submodule needs docs/source/api/{name}.rst (ADR-0001)., Regression proof the check actually fires — synthetic inputs, no filesystem.…, A stale EXCLUDED_PACKAGES entry (e.g. a renamed/removed package) should be…, test_every_submodule_has_an_api_page() (+3 more)

### Community 240 - "save_nlsq_json_files"
Cohesion: 0.35
Nodes (5): A ``Path.stat()`` failure on the post-write size check (the best-effort…, TestSaveNlsqJsonFiles, Any, Save 3 JSON files: parameters, analysis results, convergence metrics.…, save_nlsq_json_files()

### Community 241 - "viz/conftest.py"
Cohesion: 0.27
Nodes (12): _agg_backend(), converged_heterodyne_result(), converged_homodyne_result(), heterodyne_model(), homodyne_model(), minimal_homodyne_config(), Any, fixture (+4 more)

### Community 242 - "Laminar Anti-Degeneracy 5-Layer Defense"
Cohesion: 0.18
Nodes (12): Laminar Anti-Degeneracy 5-Layer Defense, CMA-ES Global Optimization (laminar), Laminar execute_layers (L2/L3 escape on >=1M stratified-LS), L4 Gradient-Collapse Detection (laminar), L2 Hierarchical Alternating Optimization (laminar), Hybrid Streaming Optimizer (laminar), Laminar per_angle_mode (constant/auto/individual scaling), L3 Adaptive CV-based Regularization (laminar) (+4 more)

### Community 243 - ".get_cache_stats"
Cohesion: 0.18
Nodes (7): PerformanceMetrics, Return comprehensive cache statistics across all tiers. Returns ------- dict…, Real-time performance monitoring metrics., Update real-time performance metrics., Update named metric fields and append a history snapshot. Parameters ----------…, Compute the recent linear trend for a metric. Fits a line to the last…, Return a comprehensive performance report. Returns ------- dict Current…

### Community 244 - "test_cache_q_validation.py"
Cohesion: 0.24
Nodes (11): _loader_with_q(), LogCaptureFixture, XPCSDataLoader, Regression test for cache q-vector validation (audit C1). A q-keyed selective…, A cache built for a different dt must not silently reuse its t1/t2 axes., Pre-existing caches without dt fingerprinting must still load (warn-only)., test_dt_mismatch_is_rejected(), test_matching_dt_is_accepted() (+3 more)

### Community 245 - "test_logging_wiring.py"
Cohesion: 0.20
Nodes (11): _isolate_logging(), Any, fixture, MonkeyPatch, Path, Regression test for config-driven file logging on the CLI path. Root cause this…, The default template (``file.enabled: false``) must not write a file., Restore the xpcsjax logger's handlers/level after the test. (+3 more)

### Community 246 - "test_heterodyne_config_bounds_override.py"
Cohesion: 0.25
Nodes (10): Regression: heterodyne ``parameter_space.bounds`` list overrides are honored.…, ``parameter_space.bounds`` with the template name ``v_beta`` overrides the…, Absent an explicit override, ``beta`` keeps the conservative registry default…, The ``phi0_het`` template name also translates onto canonical ``phi0``., An unrecognised bound name is skipped, not fatal (defensive parity)., _resolved_bounds(), test_parameter_space_bounds_default_unchanged_without_override(), test_parameter_space_bounds_override_phi0_het_translation() (+2 more)

### Community 247 - "heterodyne_parameter_manager.py"
Cohesion: 0.19
Nodes (12): _manager(), ParameterManager, Tests for ParameterManager.expand_reduced_result., Fully-untied 'constant' mode: n_scaling=0, physics_first irrelevant., A physics param excluded via active_parameters (not tied) must be NaN in the…, test_expand_reduced_result_fixed_physics_param_gets_nan(), test_expand_reduced_result_physics_first_with_scaling(), test_expand_reduced_result_scaling_first_with_scaling() (+4 more)

### Community 248 - "TestChiSquaredZeroSigma"
Cohesion: 0.26
Nodes (7): fixture, ndarray, All-zero sigma means all pixels excluded → chi-squared = 0., All-positive sigma must give same result before and after the fix path., After BUG2 fix, zero-sigma pixels are excluded (contribute 0), not Inf., A single zero-sigma element must not produce Inf chi-squared., TestChiSquaredZeroSigma

### Community 249 - "test_jax_backend.py"
Cohesion: 0.09
Nodes (21): Tests for xpcsjax.core.jax_backend identified by Gemini round-2 review. Covers…, The pre-computed-factors hot path must produce bit-identical output to the…, Batched basic correction must preserve 1x1 matrices like the scalar path., After BUG1 fix the hash key includes interior quartile samples., A concrete dt still produces a finite sinc² surface (guard is None-only)., ShearModel.compute_g1 default (dt=None) must raise the clear guard, not a…, Arrays with same (len, first, last) but different midpoint produce distinct…, Identical array inputs must return the same meshgrid object (cache hit). (+13 more)

### Community 250 - "test_debug_audit_2026_07_23_diagonal_skip.py"
Cohesion: 0.25
Nodes (10): Regression: the loader's mandatory diagonal correction must not silently…, End-to-end (design spec Testing item 2): drive the REAL load_experimental_data…, # NOTE: PreprocessingPipeline.process() itself does not consult the, End-to-end: apply_diagonal_correction_batch must NOT be called again when the…, _synthetic_data(), test_correct_diagonal_stage_marks_data_as_corrected(), test_disabled_preprocessing_does_not_set_marker(), test_loader_applies_configured_diagonal_correction_end_to_end() (+2 more)

### Community 251 - "test_anti_degeneracy_transforms.py"
Cohesion: 0.31
Nodes (10): _build(), _make_controller(), _per_angle_params(), ndarray, Round-trip coverage for anti-degeneracy parameter transforms (audit finding…, test_constant_collapse_uses_nanmean_and_preserves_physical(), test_constant_round_trip_is_exact_for_constant_scaling(), test_controller_rejects_removed_tokens_after_phase7() (+2 more)

### Community 252 - "test_l4_callback_layout.py"
Cohesion: 0.29
Nodes (8): _make(), Regression test for the L4 gradient-collapse monitor index layout.…, _StubConfig, _StubModel, _StubParamManager, test_disabled_monitoring_returns_none(), test_physics_first_layout_partitions_physics_as_head(), test_scaling_first_layout_partitions_physics_as_tail()

### Community 253 - "FitQualityConfig"
Cohesion: 0.18
Nodes (11): parametrize, test_classify_fit_quality_bands(), test_classify_fit_quality_respects_custom_thresholds(), test_fit_quality_config_defaults_match_homodyne(), test_fit_quality_config_from_dict_falls_back_on_missing_keys(), test_fit_quality_config_from_none_returns_defaults(), classify_fit_quality(), FitQualityConfig (+3 more)

### Community 254 - "validate_no_nan_inf"
Cohesion: 0.18
Nodes (10): test_validate_no_nan_inf_with_iteration_and_context(), test_validate_fit_quality_report_to_dict_keys(), test_validate_no_nan_inf_accepts_finite(), test_validate_no_nan_inf_rejects_nan_and_inf(), FitQualityReport, Any, Report from fit quality validation. Attributes ---------- passed : bool True if…, Convert to dictionary for saving in results. (+2 more)

### Community 255 - "run_validation"
Cohesion: 0.22
Nodes (10): CaptureFixture, test_print_report_renders_all_tags(), test_run_validation_human_report(), test_run_validation_json(), test_validate_verbose_prints(), _print_report(), Run all validation tests. Each test is isolated: an uncaught exception is…, Print a human-readable report (boxed header, per-test lines, summary).… (+2 more)

### Community 256 - "InputValidator"
Cohesion: 0.17
Nodes (10): test_input_validator_bounds_inconsistent_and_out_of_range(), test_input_validator_dimension_mismatch_recorded(), test_input_validator_errors_property_is_copy(), test_input_validator_non_strict_returns_false_and_records_errors(), test_input_validator_passes_on_clean_input(), test_input_validator_strict_raises_on_bad_input(), InputValidator, Validator for NLSQ optimization input data. (+2 more)

### Community 257 - "test_config_jax_free.py"
Cohesion: 0.29
Nodes (9): _imports_jax(), _probe_import(), Regression guard: importing xpcsjax.config must not load JAX (F1)., Import ``module`` in a fresh interpreter; return 0/1/2 (see contract above)., Return True iff importing ``module`` cleanly loads jax. Raises AssertionError…, test_importing_config_does_not_load_jax(), test_importing_parameter_manager_does_not_load_jax(), test_importing_registry_and_types_does_not_load_jax() (+1 more)

### Community 258 - "XPCSDataFormatError"
Cohesion: 0.08
Nodes (38): _FakeDS, Quality-gate finding #5: the APS-U loader builds an unbounded intermediate list…, Minimal h5py-dataset stand-in: exposes shape + dtype, no data read., test_guard_noop_when_no_valid_bins_in_range(), test_guard_passes_for_legitimate_small_input(), test_guard_rejects_budget_exceeded(), test_guard_rejects_non_square(), test_guard_rejects_oversized_frame_count() (+30 more)

### Community 259 - "test_run_controller.py"
Cohesion: 0.29
Nodes (9): Path, Tests for RunController.on_cancel's confirm-before-cancel guard. Regression…, Answering Yes to the confirmation dialog actually cancels the run., Answering No (or dismissing) the confirmation dialog must NOT cancel the run., With no run selected, on_cancel must not reach the confirmation/cancel path., test_cancel_confirmed_yes_proceeds(), test_cancel_declined_no_does_not_cancel(), test_cancel_no_selection_shows_message_not_queue_cancel() (+1 more)

### Community 260 - "PhysicsFactors"
Cohesion: 0.18
Nodes (8): create_physics_factors(), PhysicsFactors, ndarray, Pre-computed physics factors for efficient correlation computation., Pre-computed physics factors that do not depend on fit parameters. These are…, Validate that ``q`` and ``dt`` are strictly positive., Return ``q * cos(phi_total)`` for the cross-term phase. Parameters ----------…, Create physics factors from experimental parameters. Parameters ----------…

### Community 261 - "heterodyne_scaling_utils.py"
Cohesion: 0.20
Nodes (6): _BoundsAdapter, Per-angle scaling utilities for heterodyne XPCS analysis. Provides functions…, Look up a scaling parameter's ParameterInfo via the xpcsjax registry. Wraps…, Mapping-shaped access to scaling parameter info. Mirrors the upstream…, _scaling_param_info(), _ScalingParamProxy

### Community 262 - "TestNoScipyLeastSquares"
Cohesion: 0.20
Nodes (7): parametrize, Guard test: no direct ``scipy.optimize.least_squares`` in the NLSQ path.…, ScipyNLSQAdapter (the retired fallback) must not reappear in adapter.py., Verify scipy.optimize.least_squares is absent from the NLSQ path., No NLSQ-path file imports ``scipy.optimize.least_squares``., No NLSQ-path file calls scipy's ``least_squares(...)``. Distinguishes…, TestNoScipyLeastSquares

### Community 263 - "test_lazy_imports.py"
Cohesion: 0.20
Nodes (9): Verify top-level imports are lazy and that homodyne's env setup is mirrored., v0.1 public API symbols importable as of Phase 4 (Task 20). `HeterodyneModel`…, HeterodyneModel is a public lazy export as of Phase 6 (Task 27 + Task 28)., `import xpcsjax` must set the env vars homodyne sets at import time., Importing xpcsjax must not eagerly load jax — CLI arg parsing stays instant.…, test_env_setup_mirrors_homodyne(), test_heterodyne_model_exported(), test_public_exports_phase4() (+1 more)

### Community 264 - "TestContextFilterOnLogger"
Cohesion: 0.25
Nodes (5): ContextFilter must be attached to the named xpcsjax logger so context fields…, After configure_logging, the named xpcsjax logger has a ContextFilter., Multiple configure_logging calls must not double-install ContextFilter on the…, ContextFilter.filter() sets all fields to None when context is empty., TestContextFilterOnLogger

### Community 265 - "TestTypeBoundary"
Cohesion: 0.17
Nodes (7): set_log_context and log_context must only accept the 4 known fields;…, All four known keys must be accepted without error., log_context() context manager accepts all 4 known keys., The module must define a literal __all__ listing public symbols., Key public symbols must appear in __all__., log_once must accept a proper logging.Logger (not just Any)., TestTypeBoundary

### Community 266 - "PerAngleScaling"
Cohesion: 0.17
Nodes (7): PerAngleScaling, Total number of per-angle scaling parameters (2 * n_angles)., Number of varying scaling parameters., Indices of varying scaling parameters in the scaling array., Return bounds for the varying scaling parameters. Returns ------- tuple of…, Return the contrast and offset for a specific angle. Parameters ----------…, Per-angle contrast and offset parameter manager. Manages arrays of…

### Community 267 - ".validate_parameters"
Cohesion: 0.17
Nodes (7): Any, ndarray, Validate parameters for NaN/Inf and bounds violations after update. This is…, Validate loss value for NaN/Inf after loss computation. This is validation…, Update parameter bounds for validation. Parameters ---------- bounds : tuple of…, Initialize numerical validator. Parameters ---------- enable_validation : bool,…, Validate gradients for NaN/Inf after Jacobian computation. This is validation…

### Community 268 - "fit_nlsq(data, config.yaml)"
Cohesion: 0.22
Nodes (9): OptimizationResult, fit_nlsq, 5-layer anti-degeneracy controller (quickstart step 3), CMA-ES escape (quickstart step 5, config-gated), ConfigManager (loads/validates YAML), fit_nlsq(data, config.yaml), load_xpcs_data(config.yaml), OptimizationResult (parameters, uncertainties, chi_squared) (+1 more)

### Community 269 - "AntiDegeneracyController (5-layer composed controller)"
Cohesion: 0.22
Nodes (9): L1 Per-Angle Reparameterization (all modes), L5 Shear-Sensitivity Weighting (laminar_flow only), cma_es_escape cross-reference (separate config-gated escape), AntiDegeneracyController (5-layer composed controller), Layer 1 - Per-angle reparameterisation (PerAngleScalingPlan), Layer 2 - Hierarchical optimisation (HierarchicalOptimizer), Layer 3 - Adaptive regularisation (AdaptiveRegularizer), Layer 4 - Gradient collapse monitor (GradientCollapseMonitor) (+1 more)

### Community 270 - "get_or_create_fitter"
Cohesion: 0.22
Nodes (11): _clear_cache(), fixture, test_cache_distinct_keys(), test_cache_miss_then_hit(), test_clear_model_cache_resets(), clear_model_cache(), get_cache_stats(), get_or_create_fitter() (+3 more)

### Community 271 - "main_window.py"
Cohesion: 0.13
Nodes (15): Tests for the JAX-free failure->message mapping., test_oom_message_is_friendly(), test_plain_message_passthrough(), test_traceback_is_summarized_with_details_retained(), present_failure(), Map raw failure text to a user-facing (title, message, details) triple., Return ``(title, friendly_message, details)`` for a failure string. Parameters…, Re-evaluate dynamic-property selectors after a property change. Qt only re-… (+7 more)

### Community 272 - "test_stratified_ls_averaged_covariance_transform.py"
Cohesion: 0.25
Nodes (8): _broadcast_jacobian_transform(), ndarray, Regression: the averaged-mode inverse covariance transform in…, Exact transform now used in ``fit_with_stratified_least_squares``'s averaged-…, n_phi=2, n_physical=1: pcov is 3x3 over [contrast, offset, D0]., Wiring: the averaged-mode inverse block must use the J_full broadcast…, test_hand_computed_n_phi2_n_physical1(), test_source_no_longer_uses_identity_diagonal_shortcut()

### Community 273 - "ResultValidator"
Cohesion: 0.18
Nodes (9): test_result_validator_consistency_failure_recorded(), test_result_validator_happy_path_true(), test_result_validator_strict_raises_out_of_bounds(), test_result_validator_warnings_property_is_copy(), test_result_validator_records_warnings_for_bad_covariance(), Validator for NLSQ optimization results., Initialize ResultValidator. Parameters ---------- strict_mode : bool, optional…, Get list of validation warnings from last validate_all() call. (+1 more)

### Community 274 - "test_logging_quality_gate.py"
Cohesion: 0.22
Nodes (5): _RaisingHandler, Quality-gate regression tests for xpcsjax/utils/logging.py. Each section…, A handler whose emit() always raises — simulates a broken log backend., log_once with two DISTINCT keys must each emit exactly once., TestLogOnceTwoDistinctKeys

### Community 275 - "_ReaderThread"
Cohesion: 0.22
Nodes (7): QThread, A terminal event still queued when the grace deadline expires must be recovered…, test_reader_final_drain_recovers_terminal_after_grace(), Spawn the worker process and begin draining its events., Drains the event queue onto a Qt signal; synthesizes Died on abnormal exit., Emit any still-queued events; return True if a terminal was among them.…, _ReaderThread

### Community 276 - "PointEvaluator"
Cohesion: 0.17
Nodes (8): PointEvaluator, Any, Protocol, Pass through to ``compute_c2_heterodyne`` (meshgrid path). The engine supplies…, Return a flat ``(P,)`` theory vector at the scattered triples.…, Structural interface for a per-angle g2 point evaluator. Implementations map a…, Return the scaled g2 surface for ``phi`` over the ``(t1, t2)`` grid., Pass through to ``compute_g2_scaled`` with the kernel's arg order. Note the re-…

### Community 277 - "._apply_strategy"
Cohesion: 0.28
Nodes (6): Any, Exception, ndarray, Apply the specified recovery strategy. Parameters ---------- strategy_name :…, Add random perturbation to parameters. Parameters ---------- params :…, Get recovery strategy for the given error and attempt. Parameters ----------…

### Community 278 - ".wait_all"
Cohesion: 0.22
Nodes (6): _current_run_id(), BaseException, Wait for all pending writes. Returns list of errors. TimeoutError is not…, Wait for pending writes and shut down. Idempotent. ``drain_timeout`` bounds the…, Read the active ``run_id`` from the log-context registry, if any. Used to scope…, Exit the context manager, draining and shutting down the pool.

### Community 279 - "plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2)"
Cohesion: 0.43
Nodes (8): Reduced chi-squared goodness-of-fit (chi2_red = 0.906), Experimental Data panel (C2 heatmap), plot_nlsq_fit Baseline Figure (NLSQ Fit Results, 3-panel two-time C2), Fitted Model panel (NLSQ C2 heatmap), plot_nlsq_fit visualization function, NLSQ curve fit (XPCS correlation fit), Residuals panel (data minus model, diverging colormap), Two-time correlation C2(t1,t2) at phi=45 deg

### Community 280 - "NLSQ Residual Diagnostics Baseline (phi=45deg)"
Cohesion: 0.32
Nodes (8): Diagonal Residuals vs Time, NLSQ Residual Diagnostics Baseline (phi=45deg), NLSQ Fit (residual source), plot_residual_map (viz function), Residual Distribution with Normal Overlay, Two-Time Residual Map (t1 vs t2 heatmap), Residuals vs Fitted Value, Matplotlib Baseline Image Validation

### Community 281 - "NLSQ owns trust-region solve; xpcsjax owns strategy"
Cohesion: 0.25
Nodes (8): graphify-out/ knowledge graph, nlsq.MemoryBudgetSelector / fit() not used by xpcsjax, NLSQ owns trust-region solve; xpcsjax owns strategy, nlsq.WorkflowSelector removed in NLSQ v0.6.0, Bumping the NLSQ floor procedure, nlsq.MemoryBudgetSelector tripwire (not used), nlsq.fit tripwire (unified fit API not used), nlsq.WorkflowSelector tripwire (removed v0.6.0)

### Community 282 - "test_optimization_logging.py"
Cohesion: 0.15
Nodes (12): xpcsjax.device API, API Reference Index, xpcsjax.io API, xpcsjax.utils API, xpcsjax.viz API, Optional-component init failures in :mod:`xpcsjax.data.optimization` must be…, A crash importing PerformanceEngine logs at WARNING with context and -> None., A crash importing AdvancedMemoryManager logs at WARNING and degrades to None. (+4 more)

### Community 283 - ".__init__"
Cohesion: 0.29
Nodes (4): R, T, Return ``self`` (this loader is its own iterator)., Return the prefetched item and kick off the next background load. Joins the in-…

### Community 284 - "test_homodyne_models.py"
Cohesion: 0.20
Nodes (9): Smoke tests for HomodyneModel classes (DiffusionModel, CombinedModel).…, Static diffusion mode has 3 parameters with the expected names., Laminar flow mode has 7 parameters with the expected names., Static analysis mode collapses to the 3 diffusion parameters., compute_g1 must return a finite array for in-bounds default params. Smoke check…, test_combined_model_param_count(), test_combined_model_static_mode_param_count(), test_diffusion_model_g1_runs() (+1 more)

### Community 285 - "data/test_debug_audit_2026_06_17.py"
Cohesion: 0.24
Nodes (9): Regression tests for the 2026-06-17 debug-audit fixes (data/utils layer). Each…, test_frame_range_lower_bound_checked_when_end_is_none(), test_frame_range_valid_start_with_end_none_passes(), test_log_summary_survives_non_float_metric(), test_normalize_angle_preserves_minus_180_array(), test_normalize_angle_preserves_minus_180_scalar(), test_prefetch_loader_yields_none_items(), Validate frame range parameters. Parameters ---------- start_frame Starting… (+1 more)

### Community 286 - "quality_controller.py"
Cohesion: 0.11
Nodes (23): _bare_loader(), XPCSDataLoader, DATA-1: degraded-fallback paths must leave a detectable signal. The quality-…, An instance with __init__ bypassed — we only exercise the helper., A dataset that fails the FINAL_DATA quality gate must set ``_degraded`` even…, test_final_qc_gate_failure_sets_degraded(), test_record_degradation_accumulates(), test_record_degradation_appends_and_logs_error() (+15 more)

### Community 287 - "test_freeze_safety.py"
Cohesion: 0.32
Nodes (5): _extract_collect_all_names(), The GUI entry must be freeze-safe and resolvable as a console script., Lowercase package names from the spec's ``collect_all()`` for-loop tuple.…, test_collect_all_extraction_ignores_comment_text(), test_pyinstaller_spec_covers_runtime_deps()

### Community 288 - "LogRecord"
Cohesion: 0.29
Nodes (5): LogRecord, A record emitted before configure() carries the context fields. The…, JSONFormatter must handle a circular-reference context dict without raising and…, Circular ref injected via a record attribute must also be handled., TestJSONFormatterCircularRef

### Community 289 - "TestLogPerformanceNeverRaise"
Cohesion: 0.25
Nodes (5): log_performance: a raising handler must never abort the decorated function., Performance log raises on success; function must still return., Below-threshold path (no log emit) still works with a bad logger., Real function exception still propagates when logger also raises., TestLogPerformanceNeverRaise

### Community 290 - "estimate_per_angle_scaling_from_quantile"
Cohesion: 0.22
Nodes (10): _make_synthetic_c2(), Build a tiny synthetic two-time correlation stack for unit tests., Dual-region quantile estimator recovers per-angle contrast and offset. Small-…, A phi index with no samples in `phi_indices` is a malformed input., Locks in the dual-region rationale: diagonal-only input cannot recover offset.…, test_quantile_estimator_diagonal_only_fails_to_recover_offset(), test_quantile_estimator_raises_on_empty_phi_cell(), test_quantile_estimator_recovers_synthetic_contrast() (+2 more)

### Community 291 - "ScalingConfig"
Cohesion: 0.31
Nodes (7): Characterization tests for PerAngleScaling pack/unpack helpers. Quality-gate…, test_constant_mode_propagates_first_angle_to_all(), test_individual_mode_varying_roundtrip_is_identity(), Create a model from a configuration dictionary. Parameters ---------- config :…, Create a manager from a :class:`ScalingConfig`. Parameters ---------- config :…, Configuration for per-angle scaling behavior. Attributes ---------- n_angles :…, ScalingConfig

### Community 292 - "get_stratified_chunk_iterator"
Cohesion: 0.20
Nodes (8): test_get_stratified_chunk_iterator(), test_stratified_index_iterator(), get_stratified_chunk_iterator(), Iterator that yields index chunks for stratified data access. This iterator…, Yield each stratified chunk's index slice in order., Return the number of stratified chunks., Create an iterator yielding stratified index chunks. Parameters ---------- phi…, StratifiedIndexIterator

### Community 293 - "xla_config.bash"
Cohesion: 0.36
Nodes (4): xla_config.bash script, _xpcsjax_configure_xla(), _xpcsjax_save_xla_mode(), _xpcsjax_xla_setup()

### Community 294 - "Issue tracker: GitHub"
Cohesion: 0.29
Nodes (6): Conventions, Issue tracker: GitHub, Pull requests as a triage surface, Wayfinding operations, When a skill says "fetch the relevant ticket", When a skill says "publish to the issue tracker"

### Community 295 - "xpcsjax-post-install console script"
Cohesion: 0.29
Nodes (7): Package JAX env setup (JAX_ENABLE_X64, XLA_FLAGS, NLSQ_SKIP_GPU_CHECK), Shell completion (bash/zsh; fish non-fatal no-op), XLA activation scripts (bash/zsh/fish), xpcsjax.cli.xla_config helper / xpcsjax-config-xla, XLA_FLAGS read-once-at-backend-init ordering, xpcsjax-cleanup console script, xpcsjax-post-install console script

### Community 296 - "fit_with_stratified_least_squares"
Cohesion: 0.22
Nodes (9): LeastSquares, _get_nlsq_engine(), Get or create global NLSQ LeastSquares engine., fit_with_stratified_least_squares(), Any, Logger, LoggerAdapter, ndarray (+1 more)

### Community 297 - "Public API Reference"
Cohesion: 0.33
Nodes (7): Public API Reference, ConfigManager, fit_nlsq (single-entry NLSQ wrapper), HeterodyneModel, HomodyneModel, Lazy __getattr__ public-export mechanism, load_xpcs_data

### Community 298 - "_build_inputs"
Cohesion: 0.33
Nodes (6): _build_inputs(), Any, ndarray, Heterodyne residual-layout parity gate (corpus-loading, not generating).…, Reconstruct the deterministic inputs — must match the generator exactly., test_xpcsjax_matches_upstream_residual_layout()

### Community 299 - "test_phase5_vector_build.py"
Cohesion: 0.43
Nodes (6): Phase 5 — optimizer vector + bounds lengths per resolved mode., _run(), test_fit_vector_length_observed_averaged_expands_to_dense(), test_fit_vector_length_observed_individual(), test_vector_length_per_mode(), _vlen()

### Community 300 - "test_wrapper_finalize_result.py"
Cohesion: 0.33
Nodes (6): _build_laminar_config(), PR #37 review: the standard in-memory NLSQ path's chi_squared/pcov fixes had…, Regression sentinel for the diagnostics-overwrites-pcov bug. A live end-to-end…, Regression sentinel for the ``_finalize_result`` chi_squared bug. With 5e-4…, test_post_process_results_never_reassigns_pcov_from_diagnostics(), test_standard_path_chi_squared_is_from_true_residuals_not_model_output()

### Community 301 - "test_adversarial_review_nonfinite_coercion.py"
Cohesion: 0.28
Nodes (8): parametrize, Regression tests for the 2026-06-20 adversarial-review (codex) findings.…, test_bounds_reject_nonfinite(), test_finite_bounds_still_load(), test_finite_initial_parameters_still_load(), test_initial_parameter_value_rejects_nonfinite(), test_per_angle_scaling_rejects_nonfinite(), test_quality_score_still_flags_genuinely_overnormalized()

### Community 302 - "TimedContext"
Cohesion: 0.25
Nodes (5): test_timed_context_measures_elapsed(), Context manager for timing optimizer calls. Usage:: timer = TimedContext() with…, Start the timer and return the context manager., Stop the timer, recording the elapsed seconds on :attr:`elapsed`., TimedContext

### Community 303 - "plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg)"
Cohesion: 0.40
Nodes (6): C2 Value Colorbar / Range Annotation, Correlation Diagonal Ridge (t1=t2), plot_simulated_data Baseline (Simulated C2 Two-Time Map at phi=45deg), Scattering Angle phi = 45 deg, C2(t1,t2) Two-Time Correlation Map, plot_simulated_data Visualization Function

### Community 304 - "Automated structural doc-coverage check, content accuracy stays manual"
Cohesion: 0.33
Nodes (5): Automated structural doc-coverage check, content accuracy stays manual, Consequences, Considered options, Context, Decision

### Community 305 - "Domain Docs"
Cohesion: 0.33
Nodes (5): Before exploring, read these, Domain Docs, File structure, Flag ADR conflicts, Use the glossary's vocabulary

### Community 306 - "test_aps_u_empty_selection.py"
Cohesion: 0.47
Nodes (5): _loader(), XPCSDataLoader, Regression test for APS-U empty (q,phi) selection (audit C9). When phi…, test_empty_selection_raises(), test_nonempty_selection_returned_unchanged()

### Community 307 - "test_docs_no_fourier.py"
Cohesion: 0.29
Nodes (6): parametrize, Path, Phase 7: the Sphinx docs carry no references to the REMOVED per-angle Fourier…, No rst under docs/source references the deleted Fourier-scaling feature., test_all_rst_have_no_deleted_feature_tokens(), test_rst_has_no_removed_mode()

### Community 308 - "compute_transport_integral_matrix"
Cohesion: 0.31
Nodes (8): Characterization tests for the heterodyne velocity/transport integral kernels.…, test_transport_integral_constant_rate_is_abs_gap_and_symmetric(), test_velocity_integral_constant_velocity_is_linear_signed_gap(), test_velocity_integral_is_antisymmetric_with_zero_diagonal(), compute_transport_integral_matrix(), compute_velocity_integral_matrix(), Compute the velocity integral matrix (NLSQ meshgrid path, JIT-compiled).…, Compute the transport integral matrix (NLSQ meshgrid path, JIT-compiled).…

### Community 309 - "TestPropagateInversion"
Cohesion: 0.33
Nodes (4): Default no-arg configure() must not force propagate=True when there is no…, With no managed handler, the production branch must NOT set propagate=True., With a managed handler, the production branch sets propagate=False., TestPropagateInversion

### Community 310 - "adapter.py"
Cohesion: 0.11
Nodes (19): AdapterConfig, NLSQAdapterBase, ABC, Any, ndarray, Abstract base class for NLSQ adapters (FR-012). Provides shared methods for…, Compute parameter covariance matrix from Jacobian. Uses the standard formula:…, Abstract base class for NLSQ optimization adapters. Provides shared methods for… (+11 more)

### Community 311 - "system_validator.py"
Cohesion: 0.17
Nodes (12): test_main_returns_zero_when_no_errors(), Runtime utilities for the xpcsjax package. Provides: *…, Runtime utilities for xpcsjax., build_parser(), main(), ArgumentParser, Enum, System validation utilities for xpcsjax installation. This module provides… (+4 more)

### Community 312 - "_LAZY_EXPORTS table + module __getattr__"
Cohesion: 0.40
Nodes (5): Lazy Public API, _LAZY_EXPORTS table + module __getattr__, HeterodyneModel, HomodyneModel, load_xpcs_data

### Community 313 - "estimate_contrast_offset_from_quantiles"
Cohesion: 0.33
Nodes (8): _lag_separated_dataset(), Regression tests for xpcsjax.core.heterodyne_scaling_utils.…, C2 that genuinely decays with lag, so lag-separation differs from a global…, The added dt mask must be a no-op when every value is finite., test_all_finite_inputs_unchanged(), test_nan_delta_t_is_dropped_not_poisoning_thresholds(), estimate_contrast_offset_from_quantiles(), Estimate contrast and offset from C2 data using quantile analysis. Uses the…

### Community 314 - "test_io.py"
Cohesion: 0.22
Nodes (4): Tests for xpcsjax/io module: json_utils and nlsq_writers., Verify json_safe output is always valid JSON (no NaN/Inf tokens)., TestJsonSafeContainers, TestJsonSafeRoundTrip

### Community 315 - "test_validation_integrity_logging.py"
Cohesion: 0.50
Nodes (4): _raise_boom(), Data-integrity regression: unexpected validator-body crashes must be loud.…, A crash inside ``_validate_array_shapes`` must log ERROR and fail the report., test_array_shapes_crash_logs_error_and_invalidates_report()

### Community 316 - "nlsq/validation.py"
Cohesion: 0.25
Nodes (7): test_classify_parameter_status_all_three_states(), test_is_physical_param(), _classify_parameter_status(), _is_physical_param(), Validation utilities for NLSQ optimization. Consolidates three modules from the…, Classify each parameter's status relative to bounds., Return True if the label is for a physical (non per-angle-scaling) parameter.

### Community 317 - ".compute_g1"
Cohesion: 0.25
Nodes (5): Any, ndarray, Compute the heterodyne two-time correlation surface(s). Strictly speaking this…, Return registry default values for the 14 heterodyne params. Returns -------…, Compute the flat residual vector ``model - data`` for NLSQ. Parameters…

### Community 318 - "safe_exp"
Cohesion: 0.33
Nodes (5): jit, ndarray, Canonical numerically-safe math primitives shared across physics backends. This…, Overflow-protected exponential, canonical for all physics paths. Clips the…, safe_exp()

### Community 319 - "XPCSJAX Static Anisotropic Template"
Cohesion: 0.50
Nodes (4): anti_degeneracy_controller.py, XPCSJAX Static Anisotropic Template, XPCSJAX Static Isotropic Template, xpcs_loader

### Community 320 - "User Guide: Data Loading"
Cohesion: 0.50
Nodes (4): data_type aps_old / aps_u, load_xpcs_data Loader, User Guide: Data Loading, User Guide: Index

### Community 321 - "tests/conftest.py"
Cohesion: 0.50
Nodes (3): pytest_addoption(), Root pytest configuration., Block the pytest-qt plugin when PySide6 is absent (bare [dev] / CLI envs).…

### Community 323 - "ndarray"
Cohesion: 0.29
Nodes (8): _generate_plots_datashader(), _plot_single_angle_datashader(), ndarray, Path, Picklable worker: render one angle's 3-panel comparison via Datashader. Mirrors…, Render per-angle 3-panel comparisons via Datashader. Pool topology mirrors…, Pool worker initializer — pin JAX to CPU + lazy allocator + headless mpl., _worker_init_cpu_only()

### Community 324 - "test_no_removed_per_angle_tokens_in_tests_or_package"
Cohesion: 0.40
Nodes (4): skipif, Phase 7 exit gate: the removed per-angle tokens must not survive anywhere. This…, ``rg -n -w <tokens> tests/ xpcsjax/`` returns zero lines after teardown., test_no_removed_per_angle_tokens_in_tests_or_package()

### Community 326 - "_FakeModel"
Cohesion: 0.29
Nodes (5): _FakeModel, Minimal stand-in model for the fit-comparison render-logging tests., Regression test for the pre-existing bug: a fixed (non-tied) physics param must…, test_build_hybrid_streaming_result_expands_fixed_physics_param(), test_build_hybrid_streaming_result_mirrors_tied_child()

### Community 327 - "test_result_accessors.py"
Cohesion: 0.48
Nodes (6): Typed accessors on OptimizationResult (quality-gate type-design fixes). F8:…, _result(), test_accessors_raise_clearly_when_n_physics_unknown(), test_global_escape_typed_read(), test_physics_and_scaling_split_when_n_physics_known(), test_physics_parameters_empty_when_n_physics_zero()

### Community 328 - "test_install_shell_completion_routes"
Cohesion: 0.33
Nodes (6): parametrize, test_detect_shell_type_from_env(), test_install_shell_completion_routes(), test_validate_xla_mode(), Validate an XLA mode string. Accepts 'auto', 'nlsq', or an integer. Returns the…, _validate_xla_mode()

### Community 329 - "test_loader_integration.py"
Cohesion: 0.40
Nodes (5): Round-trip load test using a self-contained synthetic NPZ cache. Historically…, Write a minimal, valid 1-D-time-axis NPZ cache the loader can read directly.…, Load a synthetic NPZ cache end-to-end and assert the XPCS data invariants., test_load_synthetic_npz_cache_roundtrip(), _write_synthetic_npz_cache()

### Community 330 - "test_heterodyne_tied_gradient_coupling.py"
Cohesion: 0.40
Nodes (5): ParameterManager, Proves the tied-parameter mechanism is REAL coupling (gradient sums both usages…, Within THIS hand-rolled closure, d(loss)/d(D0_sample_free_var) must equal the…, test_tied_gradient_sums_both_usages(), _tied_param_manager()

### Community 333 - "Decision Record: CPU-only Execution"
Cohesion: 0.67
Nodes (3): Decision Record: CPU-only Execution, Compilation is the bottleneck, not compute, Float64 erases consumer-GPU advantage

### Community 334 - "Lumma et al. 2000 — Two-time correlation matrix estimator"
Cohesion: 0.67
Nodes (3): Duri et al. 2005 — Time-resolved-correlation of heterogeneous dynamics, Lumma et al. 2000 — Two-time correlation matrix estimator, Sutton 2008 — Review of X-ray intensity fluctuation spectroscopy

### Community 335 - "validate_optimized_params"
Cohesion: 0.33
Nodes (6): test_validate_optimized_params_below_lower(), test_validate_optimized_params_accepts_in_bounds(), test_validate_optimized_params_rejects_non_finite(), test_validate_optimized_params_rejects_out_of_bounds(), Validate that optimized parameters are finite and within bounds., validate_optimized_params()

### Community 336 - "log_quantile_scaling"
Cohesion: 0.33
Nodes (6): test_log_quantile_scaling_never_raises_on_empty(), log_gradient_sanity_check(), log_quantile_scaling(), ndarray, Mirror laminar "Quantile-based per-angle estimation complete" log block. Logs…, Mirror the laminar "GRADIENT SANITY CHECK" pre-solve diagnostic (80 wide). Pure…

### Community 346 - "isolated_env"
Cohesion: 0.40
Nodes (5): fake_venv(), isolated_env(), fixture, A venv-shaped directory with empty bash + fish activate scripts., Point HOME at a tmp dir and clear venv markers so config paths are sandboxed.

### Community 382 - "TestG2ScaledContrastZero"
Cohesion: 0.40
Nodes (3): Physics identity: g₂ = offset + 0·g₁² = offset when contrast=0., Sanity check: contrast=1 must not be constant., TestG2ScaledContrastZero

### Community 448 - "_jax_jacobian"
Cohesion: 0.50
Nodes (4): Regression: ``_jax_jacobian`` works under ``jax.jacfwd``. The inner…, test_jax_jacobian_returns_real_jacobian_under_jacfwd(), _jax_jacobian(), Compute Jacobian using JAX autodiff. Uses forward-mode autodiff (jacfwd) which…

### Community 470 - "_install_xla_bash_activation"
Cohesion: 0.50
Nodes (4): test_install_xla_bash_activation_injection_and_idempotency(), test_install_xla_bash_activation_missing_script(), _install_xla_bash_activation(), Install XLA config to bash/zsh activate script. The XLA *mode* is intentionally…

### Community 529 - "_reset_log_once"
Cohesion: 0.67
Nodes (3): fixture, Each rate-limit assertion needs a clean log_once cache., _reset_log_once()

### Community 541 - "Any"
Cohesion: 0.67
Nodes (3): Any, Per-point residuals divided by sigma, EXCLUDING invalid points. Mirrors the…, _sigma_weighted_residuals()

### Community 594 - "physics.py"
Cohesion: 0.20
Nodes (11): estimate_correlation_time(), get_default_parameters(), get_parameter_info(), parameter_bounds(), Any, Physical constants and parameter validation for homodyne XPCS analysis.…, # NOTE: These are reference values. The PRIMARY bounds used by NLSQ, Get standard parameter bounds for all model types. Returns ------- dict Mapping… (+3 more)

### Community 595 - "validate_parameters"
Cohesion: 0.40
Nodes (5): clip_parameters(), ndarray, Validate parameter values against bounds with tolerance. This is the legacy…, Clip parameters to stay within bounds. Parameters ---------- params :…, validate_parameters()

### Community 599 - "._first"
Cohesion: 0.32
Nodes (6): Any, NDArray, Return the first present alias key as a NumPy array. Parameters ---------- keys…, Experimental correlation data (``c2_exp`` / ``c2``)., Phi angles (``phi_angles_list`` / ``phi_angles`` / ``phi``)., Compute residuals for all chunks using ``vmap`` (JIT-compiled). Parameters…

## Knowledge Gaps
- **259 isolated node(s):** `completion.sh script`, `Context`, `Decision`, `Considered options`, `Consequences` (+254 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **283 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `get_logger` to `ParameterManager`, `heterodyne_core.py`, `logging.py`, `make_synthetic_two_component`, `XPCSDataLoader`, `heterodyne_scaling_utils.py`, `xpcsjax/data/__init__.py`, `NLSQResult`, `ConfigManager`, `test_memory_manager_logging.py`, `NLSQWrapper`, `data/config.py`, `heterodyne_constant_mode.py`, `DataQualityReport`, `AnalysisMode`, `select_nlsq_strategy`, `parallel_accumulator.py`, `StratifiedResidualFunction`, `jax_backend.py`, `quality_controller.py`, `cmaes_wrapper.py`, `test_gradient_diagnostics.py`, `core.py`, `execute_optimization_with_fallback`, `diagonal_correction.py`, `multistart.py`, `log_exception`, `heterodyne_logging.py`, `simulated.py`, `physics_nlsq.py`, `service/config.py`, `normalize_angle_to_symmetric_range`, `fit_computation.py`, `adapter.py`, `DatasetOptimizer`, `fit_heterodyne_stratified_least_squares`, `nlsq/validation.py`, `heterodyne_parameter_space.py`, `fit_with_out_of_core_accumulation`, `heterodyne_hybrid_streaming.py`, `nlsq_plots.py`, `sequential.py`, `performance_engine.py`, `PhysicsFactors`, `executors.py`, `save_results_npz`, `result_presenter.py`, `configure_logging`, `.__init__`, `physics.py`, `result_builder.py`, `_logger_that_raises_on_log`, `test_heterodyne_data_prep.py`, `test_heterodyne_stratified_anti_degeneracy_parity.py`, `test_output_resolution.py`, `_fit_nlsq_heterodyne`, `test_logging_wiring.py`, `heterodyne_parameter_manager.py`, `config_generator.py`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `ConfigManager` connect `ConfigManager` to `ParameterManager`, `logging.py`, `make_synthetic_two_component`, `NLSQConfig`, `fit_two_component_via_engine`, `_build_laminar_fit`, `heterodyne_constant_mode.py`, `test_homodyne_engine_preservation.py`, `AnalysisMode`, `select_nlsq_strategy`, `test_cmaes_trigger.py`, `make_cfgmgr_and_data`, `cmaes_wrapper.py`, `test_heterodyne_hybrid_streaming.py`, `core.py`, `test_phase5_standard_resolver.py`, `test_heterodyne_tied_result_assembly.py`, `HomodyneModel`, `FitOverrides`, `_Any`, `_write_config`, `test_adversarial_review_nonfinite_coercion.py`, `test_wrapper_finalize_result.py`, `simulated.py`, `service/config.py`, `HeterodyneModel`, `test_debug_audit_2026_06_18.py`, `save_results_npz`, `ParameterSpace`, `test_heterodyne_tied_gradient_coupling.py`, `test_debug_audit_regressions.py`, `test_codex_review_fixes.py`, `test_engine_heterodyne_fit_parity.py`, `test_laminar_no_worse_phase6.py`, `load_dataset`, `test_laminar_execute_layers.py`, `test_output_resolution.py`, `fit_nlsq`, `_fit_nlsq_heterodyne`, `test_static_individual_invariant.py`, `test_config_analysis_mode.py`, `config_generator.py`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `AnalysisMode` connect `AnalysisMode` to `ParameterManager`, `logging.py`, `make_synthetic_two_component`, `NLSQConfig`, `AnalysisSummaryLogger`, `ConfigManager`, `parameter_names.py`, `TwoComponentModel`, `NLSQWrapper`, `.from_optimization_block`, `get_logger`, `data/test_debug_audit_2026_06_17.py`, `cmaes_wrapper.py`, `test_gradient_diagnostics.py`, `core.py`, `fit_with_stratified_least_squares`, `HomodyneModel`, `AntiDegeneracyController`, `test_adversarial_review_nonfinite_coercion.py`, `test_heterodyne_config.py`, `HeterodynePointEvaluator`, `heterodyne_logging.py`, `service/config.py`, `fit_computation.py`, `adapter.py`, `fit_heterodyne_stratified_least_squares`, `test_quality_gate_fixes.py`, `test_low_level_plots.py`, `nlsq_plots.py`, `test_debug_audit_2026_06_18.py`, `ParameterSpace`, `_PhysicsModelProtocol`, `HierarchicalOptimizer`, `test_debug_audit_regressions.py`, `ParameterRegistry`, `_PhaseRecord`, `NLSQAdapter`, `test_review_regressions.py`, `test_laminar_execute_layers.py`, `.from_config`, `test_transforms.py`, `test_phase5_model_function_modes.py`, `test_artifacts.py`, `_fit_nlsq_heterodyne`, `test_static_individual_invariant.py`, `AdvancedMemoryManager`, `test_config_analysis_mode.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ConfigManager` (e.g. with `ParameterManager` and `AnalysisMode`) actually correct?**
  _`ConfigManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `AnalysisMode` (e.g. with `ConfigManager` and `ConstraintSeverity`) actually correct?**
  _`AnalysisMode` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `OptimizationResult` (e.g. with `AdapterConfig` and `CachedModel`) actually correct?**
  _`OptimizationResult` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `completion.sh script`, `Context`, `Decision` to the rest of the system?**
  _259 weakly-connected nodes found - possible documentation gaps or missing edges._