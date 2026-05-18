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

# NLSQ 0.6.3+ Workflow System Changes:
# - WorkflowSelector was removed in NLSQ v0.6.0
# - NLSQ now uses 3-preset workflows: "auto", "auto_global", "hpc"
# - xpcsjax uses its own select_nlsq_strategy() for memory-aware selection
# - OptimizationGoal still exists in NLSQ 0.6.4 (nlsq.core.workflow)
WorkflowSelector = None  # Removed in NLSQ v0.6.0
WorkflowTier = None  # Removed in NLSQ v0.6.0
NLSQDatasetSizeTier = None  # Removed in NLSQ v0.6.0
NLSQ_WORKFLOW_AVAILABLE = False  # WorkflowSelector removed in NLSQ v0.6.0

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

# NOTE: DatasetSizeStrategy, OptimizationStrategy, estimate_memory_requirements
# removed from public API in v2.12.0. Use NLSQ's WorkflowSelector instead.
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
    # Workflow selection (DEPRECATED in NLSQ v0.6.0)
    # WorkflowSelector, WorkflowTier, NLSQDatasetSizeTier were removed
    # Use xpcsjax's select_nlsq_strategy() from memory.py instead
    "WorkflowSelector",  # None - removed in NLSQ v0.6.0
    "WorkflowTier",  # None - removed in NLSQ v0.6.0
    "NLSQDatasetSizeTier",  # None - removed in NLSQ v0.6.0
    "NLSQ_WORKFLOW_AVAILABLE",  # False - WorkflowSelector removed
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
    # NOTE: DatasetSizeStrategy, OptimizationStrategy, estimate_memory_requirements
    # removed from public API in v2.12.0. Use NLSQ's WorkflowSelector instead.
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
from pathlib import Path as _Path


def fit_nlsq(data, config):
    """Single-entry NLSQ fit for both physics models.

    Parameters
    ----------
    data : dict
        XPCS data dict returned by ``xpcsjax.data.load_xpcs_data``.
    config : str | Path | ConfigManager
        Path to a YAML config file or a pre-built ConfigManager.

    Returns
    -------
    OptimizationResult
        Fit parameters, covariance, diagnostics, and metadata.
    """
    if isinstance(config, (str, _Path)):
        from xpcsjax.config import ConfigManager
        config = ConfigManager(str(config))
    return fit_nlsq_jax(data, config)


# Ensure fit_nlsq joins whatever __all__ the verbatim port already defined
try:
    __all__  # type: ignore[name-defined]
except NameError:
    __all__ = []
if "fit_nlsq" not in __all__:
    __all__.append("fit_nlsq")
