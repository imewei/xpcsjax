"""NLSQ Optimization Subpackage for xpcsjax.

This subpackage contains all NLSQ (Non-Linear Least Squares) optimization
components organized into logical modules.

Structure:
- core.py: Main fit_nlsq_jax function and NLSQResult class
- wrapper.py: NLSQWrapper adapter class (legacy)
- adapter.py: NLSQAdapter using NLSQ's CurveFit class (v2.11.0+)
- memory.py: Memory management utilities (extracted Jan 2026)
- parameter_utils.py: Parameter utilities (extracted Jan 2026)
- jacobian.py: Jacobian computation utilities
- results.py: OptimizationResult and related dataclasses
- transforms.py: Parameter transformation utilities
- data_prep.py: Data preparation utilities
- result_builder.py: Result building utilities
- fit_computation.py: Fit computation utilities
- multistart.py: Multi-start optimization with LHS (v2.6.0)
- anti_degeneracy_adapter.py: NLSQ integration for anti-degeneracy (v2.11.0+)
- strategies/: Optimization strategy modules
  - chunking.py: Angle-stratified chunking for large datasets
  - residual.py: StratifiedResidualFunction for per-angle optimization
  - residual_jit.py: JIT-compiled stratified residual function
  - sequential.py: Sequential per-angle optimization
  - executors.py: Strategy pattern executors

NLSQ Integration (v2.11.0+):
- Uses NLSQ's CurveFit class for JIT compilation caching
- Uses xpcsjax's select_nlsq_strategy() for memory-aware strategy selection
- Integrates with MultiStartOrchestrator for global optimization
- Anti-degeneracy layers remain in xpcsjax (physics-specific)

Note: xpcsjax uses NLSQ's curve_fit() directly, not the fit() unified API.
Memory strategy selection is handled by xpcsjax's own select_nlsq_strategy()
function rather than NLSQ's MemoryBudgetSelector.
"""

# =============================================================================
# NLSQ Package Imports (v0.6.10+)
# Core curve fitting API with CurveFit class for JIT caching
# =============================================================================

# Core NLSQ imports (always available)
try:
    from nlsq import CurveFit, curve_fit

    NLSQ_CURVEFIT_AVAILABLE = True
except ImportError:
    CurveFit = None  # type: ignore[assignment, misc]
    curve_fit = None  # type: ignore[assignment]
    NLSQ_CURVEFIT_AVAILABLE = False

# OptimizationGoal is still available in NLSQ 0.6.4 (FAST, ROBUST, QUALITY, etc.)
try:
    from nlsq.core.workflow import OptimizationGoal

    NLSQ_GOAL_AVAILABLE = True
except ImportError:
    OptimizationGoal = None  # type: ignore[misc, assignment]
    NLSQ_GOAL_AVAILABLE = False

# Global optimization (NLSQ v0.4+)
try:
    from nlsq.global_optimization import (
        GlobalOptimizationConfig,
        MultiStartOrchestrator,
    )

    NLSQ_GLOBAL_OPT_AVAILABLE = True
except ImportError:
    GlobalOptimizationConfig = None  # type: ignore[misc, assignment]
    MultiStartOrchestrator = None  # type: ignore[misc, assignment]
    NLSQ_GLOBAL_OPT_AVAILABLE = False

# CMA-ES Global Optimization (NLSQ v0.6.3+)
# Requires evosax for JAX-accelerated evolution strategies
try:
    from nlsq.global_optimization import (
        CMAES_PRESETS,
        CMAESConfig,
        CMAESDiagnostics,
        CMAESOptimizer,
        MethodSelector,
        auto_configure_cmaes_memory,
        compute_default_popsize,
        estimate_cmaes_memory_gb,
        is_evosax_available,
    )

    # Check if evosax is actually installed
    NLSQ_CMAES_AVAILABLE = is_evosax_available()
except ImportError:
    CMAES_PRESETS = None  # type: ignore[assignment]
    CMAESConfig = None  # type: ignore[misc, assignment]
    CMAESDiagnostics = None  # type: ignore[misc, assignment]
    CMAESOptimizer = None
    MethodSelector = None
    auto_configure_cmaes_memory = None  # type: ignore[assignment]
    compute_default_popsize = None  # type: ignore[assignment]
    estimate_cmaes_memory_gb = None  # type: ignore[assignment]
    is_evosax_available = None  # type: ignore[assignment]
    NLSQ_CMAES_AVAILABLE = False

# Stability and recovery (NLSQ v0.4+)
try:
    from nlsq.stability import (
        NumericalStabilityGuard,
    )
    from nlsq.stability import (
        OptimizationRecovery as NLSQOptimizationRecovery,
    )

    NLSQ_STABILITY_AVAILABLE = True
except ImportError:
    NumericalStabilityGuard = None  # type: ignore[misc, assignment]
    NLSQOptimizationRecovery = None  # type: ignore[misc, assignment]
    NLSQ_STABILITY_AVAILABLE = False

# Caching and memory management (NLSQ v0.4+)
try:
    from nlsq.caching import MemoryManager as NLSQMemoryManager
    from nlsq.caching import get_memory_manager

    NLSQ_CACHING_AVAILABLE = True
except ImportError:
    NLSQMemoryManager = None  # type: ignore[misc, assignment]
    get_memory_manager = None  # type: ignore[assignment]
    NLSQ_CACHING_AVAILABLE = False

# Result types (NLSQ v0.4+)
try:
    from nlsq.result import CurveFitResult

    NLSQ_RESULT_AVAILABLE = True
except ImportError:
    CurveFitResult = None  # type: ignore[misc, assignment]
    NLSQ_RESULT_AVAILABLE = False

# Streaming optimizer (NLSQ v0.3.2+)
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    NLSQ_STREAMING_AVAILABLE = True
except ImportError:
    AdaptiveHybridStreamingOptimizer = None
    HybridStreamingConfig = None
    NLSQ_STREAMING_AVAILABLE = False

# =============================================================================
# xpcsjax NLSQ Module Imports
# =============================================================================

# NLSQAdapter using CurveFit class (v2.11.0+)
from xpcsjax.optimization.nlsq.adapter import (  # noqa: E402
    AdapterConfig,
    NLSQAdapter,
    clear_model_cache,
    get_adapter,
    get_cache_stats,
    get_or_create_model,
    is_adapter_available,
)

# Architecture refactoring (v2.14.0): NLSQAdapterBase
from xpcsjax.optimization.nlsq.adapter_base import NLSQAdapterBase  # noqa: E402

# Anti-degeneracy defense system (v2.9.0)
from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (  # noqa: E402
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)

# CMA-ES global optimization wrapper (v2.15.0 / NLSQ 0.6.4+)
# Note: NLSQ_CMAES_AVAILABLE from cmaes_wrapper is the canonical source
from xpcsjax.optimization.nlsq.cmaes_wrapper import (  # noqa: E402
    CMAES_AVAILABLE as NLSQ_CMAES_AVAILABLE,  # overrides global_optimization import
)
from xpcsjax.optimization.nlsq.cmaes_wrapper import (  # noqa: E402
    CMAESResult,
    CMAESWrapper,
    CMAESWrapperConfig,
    fit_with_cmaes,
)
from xpcsjax.optimization.nlsq.config import (  # noqa: E402
    HybridRecoveryConfig,
    NLSQConfig,
)
from xpcsjax.optimization.nlsq.core import (  # noqa: E402
    JAX_AVAILABLE,
    NLSQ_AVAILABLE,
    NLSQResult,
    _get_param_names,
    fit_nlsq_cmaes,
    fit_nlsq_jax,
    fit_nlsq_multistart,
)

# New refactored modules (Dec 2025)
from xpcsjax.optimization.nlsq.data_prep import (  # noqa: E402
    ExpandedParameters,
    PreparedData,
    build_parameter_labels,
    classify_parameter_status,
    convert_bounds_to_nlsq_format,
    expand_per_angle_parameters,
    validate_bounds,
    validate_initial_params,
)
from xpcsjax.optimization.nlsq.fit_computation import (  # noqa: E402
    compute_theoretical_fits,
    extract_parameters_from_result,
    get_physical_param_count,
    normalize_analysis_mode,
)

# Memory management utilities (extracted Jan 2026)
from xpcsjax.optimization.nlsq.memory import (  # noqa: E402
    DEFAULT_MEMORY_FRACTION,
    FALLBACK_THRESHOLD_GB,
    NLSQStrategy,
    StrategyDecision,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    get_adaptive_memory_threshold,
    select_nlsq_strategy,
)

# Multi-start optimization (v2.6.0)
# NOTE: Subsampling is explicitly NOT supported per project requirements.
# Numerical precision and reproducibility take priority over computational speed.
from xpcsjax.optimization.nlsq.multistart import (  # noqa: E402
    MultiStartConfig,
    MultiStartResult,
    SingleStartResult,
    check_zero_volume_bounds,
    detect_degeneracy,
    generate_lhs_starts,
    generate_random_starts,
    include_custom_starts,
    run_multistart_nlsq,
    screen_starts,
    validate_n_starts_for_lhs,
)
from xpcsjax.optimization.nlsq.parameter_index_mapper import (  # noqa: E402
    ParameterIndexMapper,
)

# Parameter utilities (extracted Jan 2026)
from xpcsjax.optimization.nlsq.parameter_utils import (  # noqa: E402
    build_parameter_labels as build_parameter_labels_utils,
)
from xpcsjax.optimization.nlsq.parameter_utils import (  # noqa: E402
    classify_parameter_status as classify_parameter_status_utils,
)
from xpcsjax.optimization.nlsq.parameter_utils import (  # noqa: E402
    compute_consistent_per_angle_init,
    compute_jacobian_stats,
    sample_xdata,
)
from xpcsjax.optimization.nlsq.result_builder import (  # noqa: E402
    QualityMetrics,
    ResultBuilder,
    compute_quality_metrics,
    compute_uncertainties,
    determine_convergence_status,
    normalize_nlsq_result,
)
from xpcsjax.optimization.nlsq.results import (  # noqa: E402
    FunctionEvaluationCounter,
    OptimizationResult,
)
from xpcsjax.optimization.nlsq.strategies.chunking import (  # noqa: E402
    StratificationDiagnostics,
    analyze_angle_distribution,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_stratification_memory,
    format_diagnostics_report,
    should_use_stratification,
)
from xpcsjax.optimization.nlsq.strategies.executors import (  # noqa: E402
    ExecutionResult,
    LargeDatasetExecutor,
    OptimizationExecutor,
    StandardExecutor,
    StreamingExecutor,
    get_executor,
)
from xpcsjax.optimization.nlsq.strategies.residual import (  # noqa: E402
    StratifiedResidualFunction,
    create_stratified_residual_function,
)
from xpcsjax.optimization.nlsq.strategies.residual_jit import (  # noqa: E402
    StratifiedResidualFunctionJIT,
)
from xpcsjax.optimization.nlsq.strategies.sequential import (  # noqa: E402
    JAC_SAMPLE_SIZE,
    optimize_per_angle_sequential,
)
from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper  # noqa: E402

__all__ = [
    # NLSQ Package Integration (v2.11.0+)
    # Core NLSQ classes
    "CurveFit",
    "curve_fit",
    "NLSQ_CURVEFIT_AVAILABLE",
    # OptimizationGoal (still available in NLSQ 0.6.4)
    "OptimizationGoal",  # FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY
    "NLSQ_GOAL_AVAILABLE",
    # Global optimization (Multi-Start)
    "GlobalOptimizationConfig",
    "MultiStartOrchestrator",
    "NLSQ_GLOBAL_OPT_AVAILABLE",
    # CMA-ES Global Optimization (NLSQ 0.6.3+)
    "CMAES_PRESETS",
    "CMAESConfig",
    "CMAESDiagnostics",
    "CMAESOptimizer",
    "MethodSelector",
    "auto_configure_cmaes_memory",
    "compute_default_popsize",
    "estimate_cmaes_memory_gb",
    "is_evosax_available",
    "NLSQ_CMAES_AVAILABLE",
    # Stability and recovery
    "NumericalStabilityGuard",
    "NLSQOptimizationRecovery",
    "NLSQ_STABILITY_AVAILABLE",
    # Caching
    "NLSQMemoryManager",
    "get_memory_manager",
    "NLSQ_CACHING_AVAILABLE",
    # Result types
    "CurveFitResult",
    "NLSQ_RESULT_AVAILABLE",
    # Streaming
    "AdaptiveHybridStreamingOptimizer",
    "HybridStreamingConfig",
    "NLSQ_STREAMING_AVAILABLE",
    # xpcsjax Core
    "fit_nlsq_jax",
    "fit_nlsq_multistart",
    "fit_nlsq_cmaes",
    "fit_with_cmaes",
    "CMAESWrapper",
    "CMAESWrapperConfig",
    "CMAESResult",
    "NLSQResult",
    "JAX_AVAILABLE",
    "NLSQ_AVAILABLE",
    "_get_param_names",
    # Configuration (v2.11.0+)
    "NLSQConfig",
    "HybridRecoveryConfig",
    # Anti-degeneracy defense system (v2.9.0)
    "AntiDegeneracyConfig",
    "AntiDegeneracyController",
    "ParameterIndexMapper",
    # Multi-start (v2.6.0)
    # NOTE: No subsampling - numerical precision takes priority
    "MultiStartConfig",
    "MultiStartResult",
    "SingleStartResult",
    "generate_lhs_starts",
    "generate_random_starts",
    "screen_starts",
    "detect_degeneracy",
    "run_multistart_nlsq",
    "include_custom_starts",
    "check_zero_volume_bounds",
    "validate_n_starts_for_lhs",
    # Wrapper (legacy)
    "NLSQWrapper",
    "OptimizationResult",
    "FunctionEvaluationCounter",
    # Adapter (v2.11.0+ - recommended)
    "NLSQAdapter",
    "AdapterConfig",
    "get_adapter",
    "is_adapter_available",
    # Model caching (v2.11.0+)
    "get_or_create_model",
    "clear_model_cache",
    "get_cache_stats",
    # Chunking
    "StratificationDiagnostics",
    "analyze_angle_distribution",
    "compute_stratification_diagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "estimate_stratification_memory",
    "format_diagnostics_report",
    "should_use_stratification",
    # Residual
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    # Sequential
    "JAC_SAMPLE_SIZE",
    "optimize_per_angle_sequential",
    # Data Preparation (new in Dec 2025)
    "PreparedData",
    "ExpandedParameters",
    "expand_per_angle_parameters",
    "validate_bounds",
    "validate_initial_params",
    "convert_bounds_to_nlsq_format",
    "build_parameter_labels",
    "classify_parameter_status",
    # Result Building (new in Dec 2025)
    "QualityMetrics",
    "ResultBuilder",
    "compute_quality_metrics",
    "compute_uncertainties",
    "normalize_nlsq_result",
    "determine_convergence_status",
    # Fit Computation (new in Dec 2025)
    "compute_theoretical_fits",
    "normalize_analysis_mode",
    "get_physical_param_count",
    "extract_parameters_from_result",
    # Executors (new in Dec 2025)
    "ExecutionResult",
    "OptimizationExecutor",
    "StandardExecutor",
    "LargeDatasetExecutor",
    "StreamingExecutor",
    "get_executor",
    # Memory management and unified strategy selection (Jan 2026)
    "DEFAULT_MEMORY_FRACTION",
    "FALLBACK_THRESHOLD_GB",
    "NLSQStrategy",
    "StrategyDecision",
    "detect_total_system_memory",
    "estimate_peak_memory_gb",
    "get_adaptive_memory_threshold",
    "select_nlsq_strategy",
    # Parameter utilities (extracted Jan 2026)
    "build_parameter_labels_utils",
    "classify_parameter_status_utils",
    "compute_consistent_per_angle_init",
    "compute_jacobian_stats",
    "sample_xdata",
    # Architecture refactoring (v2.14.0)
    "NLSQAdapterBase",
]


# ============================================================================
# xpcsjax v0.1 single-entry public wrapper
# ============================================================================
from pathlib import Path as _Path  # noqa: E402 - public API section is below the verbatim port
from typing import TYPE_CHECKING, Any  # noqa: E402

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager


def fit_nlsq(
    data: dict[str, Any],
    config: "ConfigManager | str | _Path",
) -> "OptimizationResult":
    """Single-entry NLSQ fit for both physics models.

    Parameters
    ----------
    data : dict
        XPCS data dict returned by ``xpcsjax.data.load_xpcs_data`` (homodyne)
        or a heterodyne-style dict with keys ``c2_exp`` / ``c2`` and
        ``phi_angles_list`` / ``phi_angles`` (heterodyne).
    config : str | Path | ConfigManager
        Path to a YAML config file or a pre-built ConfigManager.

    Returns
    -------
    OptimizationResult
        Homodyne path returns ``OptimizationResult``.

        Heterodyne path returns ``OptimizationResult`` for every
        ``per_angle_mode`` (``"constant"``, ``"averaged"``, ``"fourier"``,
        ``"individual"``, ``"auto"``). Mode-specific per-angle data
        (``chi2_per_angle``, ``parameter_names``,
        ``contrast_per_angle`` / ``offset_per_angle``, etc.) lives under
        ``result.nlsq_diagnostics``. The ``individual`` mode additionally
        carries ``covariance_structure="block_diagonal_sequential"`` to
        signal that off-diagonal covariance entries are zero **by
        construction** (sequential per-angle fits with held-fixed
        scaling) rather than by fit.

        See :mod:`xpcsjax.optimization.nlsq.heterodyne_views` for post-hoc
        per-angle reconstruction helpers (``reconstruct_per_angle_scaling``,
        ``per_angle_chi2``).
    """
    if isinstance(config, (str, _Path)):
        from xpcsjax.config import ConfigManager
        config = ConfigManager(str(config))

    mode = ""
    if hasattr(config, "config") and config.config:
        mode = config.config.get("analysis_mode", "")
        if not mode:
            opt = config.config.get("optimization", {})
            if isinstance(opt, dict):
                nlsq_sect = opt.get("nlsq", {})
                if isinstance(nlsq_sect, dict):
                    mode = nlsq_sect.get("analysis_mode", "")
    mode = str(mode).lower().replace("-", "_")
    if mode in ("two_component", "heterodyne"):
        return _fit_nlsq_heterodyne(data, config)

    # Homodyne path — unchanged.
    return fit_nlsq_jax(data, config)


def _fit_nlsq_heterodyne(
    data: dict[str, Any],
    config: "ConfigManager | str | _Path",
) -> "OptimizationResult":
    """Dispatch the heterodyne multi-phi fit through the ported orchestration.

    The xpcsjax homodyne loader does not produce a heterodyne-style data dict
    (the source heterodyne cache uses ``c2`` / ``phi`` keys), so this helper
    accepts both layouts:

    - ``c2_exp`` / ``c2``: experimental 3-D stack ``(n_phi, N, N)`` or
      single-angle ``(N, N)``.
    - ``phi_angles_list`` / ``phi_angles`` / ``phi``: 1-D array of detector
      angles (degrees).

    All other physics inputs (``t, q, dt``) are sourced from the
    :class:`HeterodyneModel` constructed from the YAML config.
    """
    import numpy as _np

    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_config import (
        NLSQConfig as _HeterodyneNLSQConfig,
    )
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    # Raw YAML dict for HeterodyneModel.from_config and NLSQConfig.from_dict
    yaml_dict = config.config if hasattr(config, "config") else dict(config)

    model = HeterodyneModel.from_config(yaml_dict)

    # NLSQConfig.from_dict expects a flat NLSQ section (max_iterations,
    # enable_cmaes, recovery, …). Production YAMLs nest it under
    # ``optimization.nlsq``; without unwrapping, every nested setting is
    # silently dropped and the solver runs with defaults.
    opt_block = yaml_dict.get("optimization", {}) if isinstance(yaml_dict, dict) else {}
    nested_nlsq = opt_block.get("nlsq") if isinstance(opt_block, dict) else None
    if isinstance(nested_nlsq, dict) and nested_nlsq:
        nlsq_dict = dict(nested_nlsq)
        top_mode = yaml_dict.get("analysis_mode") if isinstance(yaml_dict, dict) else None
        if top_mode is not None and "analysis_mode" not in nlsq_dict:
            nlsq_dict["analysis_mode"] = top_mode
    else:
        nlsq_dict = yaml_dict

    nlsq_cfg = _HeterodyneNLSQConfig.from_dict(nlsq_dict)

    # Extract c2 + phi from data dict, accepting either heterodyne or
    # xpcsjax-loader key names.
    if "c2_exp" in data:
        c2 = _np.asarray(data["c2_exp"])
    elif "c2" in data:
        c2 = _np.asarray(data["c2"])
    else:
        raise KeyError(
            "heterodyne dispatch requires 'c2_exp' or 'c2' in the data dict; "
            f"got keys {list(data)}"
        )

    if "phi_angles_list" in data:
        phi = _np.asarray(data["phi_angles_list"], dtype=_np.float64)
    elif "phi_angles" in data:
        phi = _np.asarray(data["phi_angles"], dtype=_np.float64)
    elif "phi" in data:
        phi = _np.asarray(data["phi"], dtype=_np.float64)
    else:
        raise KeyError(
            "heterodyne dispatch requires 'phi_angles_list', 'phi_angles', "
            f"or 'phi' in the data dict; got keys {list(data)}"
        )

    # Optional weights for weighted least squares.
    weights = data.get("weights")
    if weights is not None:
        weights = _np.asarray(weights)

    # Sync the model's internal time axis with the data shape (the source
    # heterodyne pipeline drops the leading time point, shrinking N by 1).
    n_data = c2.shape[-1]
    n_model = int(model.t.shape[0])
    if n_data != n_model:
        # Build a numpy view of the trimmed time axis and re-attach.
        model.sync_time_axis(_np.arange(n_data, dtype=_np.float64))

    # Multi-start dispatch (mirrors homodyne core.py:374-392). Reads the nested
    # multi_start section; CMA-ES keeps precedence (it owns the enable_cmaes
    # branch inside fit_nlsq_multi_phi).
    ms_dict = nlsq_dict.get("multi_start", {}) if isinstance(nlsq_dict, dict) else {}
    cmaes_on = bool(
        nlsq_dict.get("cmaes", {}).get("enable", False)
        if isinstance(nlsq_dict, dict)
        else False
    )
    if isinstance(ms_dict, dict) and ms_dict.get("enable", False) and not cmaes_on:
        from xpcsjax.optimization.nlsq.heterodyne_multistart import (
            build_multistart_config,
            fit_nlsq_multistart_heterodyne,
        )

        ms_cfg = build_multistart_config(ms_dict)
        return fit_nlsq_multistart_heterodyne(model, c2, phi, nlsq_cfg, weights, ms_cfg)

    return fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights)


# Ensure fit_nlsq joins whatever __all__ the verbatim port already defined
try:
    __all__  # type: ignore[name-defined]  # noqa: B018 - existence probe for __all__
except NameError:
    __all__ = []
if "fit_nlsq" not in __all__:
    __all__.append("fit_nlsq")
