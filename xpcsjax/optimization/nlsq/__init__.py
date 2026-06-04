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


def _estimate_heterodyne_points(c2: "Any", phi: "Any") -> int:
    """Total scalar count of a heterodyne ``c2`` stack.

    Accepts the in-memory layouts produced by the heterodyne loader: a 2-D
    single-angle ``(N, N)`` correlation matrix or a 3-D ``(n_phi, N, N)`` stack.
    ``phi`` is accepted for signature symmetry with the stratification gate but
    is not needed for the count (the angle axis is already the leading dim).
    """
    import numpy as _np

    arr = _np.asarray(c2)
    if arr.ndim == 2:
        return int(arr.shape[0] * arr.shape[1])
    return int(arr.shape[0] * arr.shape[1] * arr.shape[2])


def _safe_log_heterodyne_start(nlsq_cfg: Any, analysis_mode: str, n_phi: int) -> None:
    """Best-effort opening ``NLSQ OPTIMIZATION`` banner for the heterodyne path.

    Logging must never break a fit: guarded so stubbed dispatch unit tests
    (which may replace ``heterodyne_core`` or pass a sentinel config) skip the
    banner instead of raising.
    """
    try:
        from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_start

        per_angle_mode = str(getattr(nlsq_cfg, "per_angle_mode", "auto"))
        log_heterodyne_start(analysis_mode, per_angle_mode, n_phi)
    except Exception:  # nosec B110 # pragma: no cover - logging is non-critical, never fatal
        pass


def _safe_configure_cpu_threading() -> None:
    """Best-effort CPU/HPC threading configuration for the heterodyne path.

    Mirrors the homodyne ``fit_nlsq_jax`` call (``core.py``) so ``two_component``
    fits emit the same ``xpcsjax.device.cpu`` configuration banner as
    ``laminar_flow``. The call is numerically inert — it only configures thread
    counts and logs — and is guarded so a missing optional dependency or an
    already-initialised JAX backend never breaks a fit.
    """
    try:
        from xpcsjax.device.cpu import configure_cpu_threading

        configure_cpu_threading()
    except Exception:  # nosec B110 # pragma: no cover - setup/logging is non-critical
        pass


def _safe_log_memory_strategy() -> None:
    """Best-effort ``memory_strategy_selection`` phase + adaptive-threshold log.

    Emits the same ``xpcsjax.optimization.nlsq.memory`` block the homodyne path
    logs (the ``Phase 'memory_strategy_selection'`` banner plus the
    ``Adaptive memory threshold`` line) so ``two_component`` matches
    ``laminar_flow``. Only the *threshold* is computed and logged — no strategy
    decision is taken here (heterodyne routing uses its own ``heterodyne_memory``
    module), which deliberately avoids emitting a misleading ``Auto-switching``
    line. Guarded so it never breaks a fit.
    """
    try:
        from xpcsjax.optimization.nlsq import memory as _memory
        from xpcsjax.utils.logging import log_phase

        with log_phase("memory_strategy_selection", logger=_memory.logger):
            _memory.get_adaptive_memory_threshold(_memory.DEFAULT_MEMORY_FRACTION)
    except Exception:  # nosec B110 # pragma: no cover - logging is non-critical
        pass


def _safe_log_heterodyne_initial_params(yaml_dict: Any) -> None:
    """Log the homodyne-parity ``Using initial parameters from configuration`` line.

    Only emitted when the merged config actually carries a non-empty
    ``initial_parameters`` block (which the heterodyne flat-format override path
    consumes), so the line is never logged untruthfully. Logged under the
    ``heterodyne_core`` namespace — the heterodyne analogue of homodyne's
    ``nlsq.core`` preamble. Guarded so it never breaks a fit.
    """
    try:
        if not (isinstance(yaml_dict, dict) and yaml_dict.get("initial_parameters")):
            return
        from xpcsjax.utils.logging import get_logger

        get_logger("xpcsjax.optimization.nlsq.heterodyne_core").info(
            "Using initial parameters from configuration"
        )
    except Exception:  # nosec B110 # pragma: no cover - logging is non-critical
        pass


def _seed42_angle_reorder(
    c2: Any, phi: Any, weights: Any, n_points: int
) -> tuple[Any, Any, Any, Any]:
    """Apply the homodyne-parity seed-42 angle-stratified pre-shuffle.

    Mirrors ``laminar_flow``'s 100k–1M regime (``wrapper.py``): a per-angle fit
    of >100k points that stays in-memory reorganizes and seed-42 pre-shuffles its
    data *before* the solve. Heterodyne data is batched by angle
    (``(n_phi, N, N)``), so the **angle axis** is the natural stratification unit;
    permuting it is **objective-invariant** — the joint SSR is a sum over angles,
    so the fit *objective* is unchanged. The fitted parameter vector is unchanged
    up to floating-point summation order; at a degenerate minimum equi-objective
    parameter sets may still differ (the same property laminar_flow's shuffle
    has). Returns the reordered ``(c2, phi, weights)`` plus the inverse
    permutation used by :func:`_restore_angle_order` to map per-angle outputs
    back to the caller's angle order.
    """
    import numpy as _np

    from xpcsjax.utils.logging import get_logger as _get_logger

    log = _get_logger("xpcsjax.optimization.nlsq.heterodyne_logging")
    phi_arr = _np.asarray(phi)
    n_phi = int(phi_arr.shape[0])
    perm = _np.random.default_rng(42).permutation(n_phi)
    inv = _np.empty_like(perm)
    inv[perm] = _np.arange(n_phi)

    log.info(
        "Applying angle-stratified reordering: %s points across %d angles "
        "(heterodyne 100k–1M in-memory regime, mirrors laminar_flow)",
        f"{int(n_points):,}",
        n_phi,
    )
    log.info("Pre-shuffled angle order (seed=42) — objective-invariant reorder")

    c2_r = _np.asarray(c2)[perm]
    phi_r = phi_arr[perm]
    # Only per-angle (3-D ``(n_phi, N, N)``) weights are angle-indexed. A shared
    # 2-D ``(N, N)`` weight array broadcasts across angles and must NOT be
    # permuted — doing so would permute its time axis (and IndexError when
    # n_phi > N). ``None`` and 2-D pass through unchanged.
    if weights is not None and _np.asarray(weights).ndim == 3:
        weights_r = _np.asarray(weights)[perm]
    else:
        weights_r = weights

    log.info("Angle-stratified reorder complete: %d angles reorganized", n_phi)
    return c2_r, phi_r, weights_r, inv


def _restore_angle_order(result: Any, inv_perm: Any) -> None:
    """Restore per-angle result fields to the caller's original angle order.

    The fit ran on seed-42-reordered angles, so every angle-indexed diagnostic
    (``chi2_per_angle``, ``phi_angles``, and any ``*_per_angle*`` array such as
    the per-angle scaling estimates) comes back in shuffled order. Inverting the
    permutation realigns them all with the caller's input angle order. Only used
    for the ``averaged`` mode, whose fitted vector is two GLOBAL scaling params
    (no angle-ordered parameter/covariance tail), so realigning the diagnostics
    is sufficient and no parameter or covariance un-permutation is needed.
    Best-effort: never raises.
    """
    try:
        import numpy as _np

        diag = getattr(result, "nlsq_diagnostics", None)
        if not isinstance(diag, dict):
            return
        inv = _np.asarray(inv_perm)
        n = int(inv.shape[0])
        for key, val in list(diag.items()):
            # Realign every angle-indexed diagnostic, not just chi2/phi.
            if key not in ("chi2_per_angle", "phi_angles") and "per_angle" not in key:
                continue
            if val is None:
                continue
            try:
                arr = _np.asarray(val)
            except Exception:  # noqa: BLE001 - non-array diagnostic value; skip
                continue
            if arr.ndim >= 1 and arr.shape[0] == n and arr.dtype != object:
                diag[key] = arr[inv]
    except Exception:  # nosec B110 # pragma: no cover - diagnostics realignment is non-critical
        pass


def _safe_log_heterodyne_completion(result: Any, model: Any, n_phi: int) -> None:
    """Best-effort homodyne-parity completion logging for the heterodyne path.

    Logging must never break a fit: the call is guarded so that dispatch unit
    tests using lightweight stubs (a fake model without ``param_manager``, a
    sentinel result, or a replaced ``heterodyne_core`` module) skip the block
    instead of raising. Real runs always carry a model + ``OptimizationResult``
    with the required attributes and log the full block.
    """
    try:
        from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_completion

        param_manager = getattr(model, "param_manager", None)
        if param_manager is None:
            return
        log_heterodyne_completion(
            result,
            list(param_manager.varying_names),
            int(param_manager.n_varying),
            n_phi,
        )
    except Exception:  # nosec B110 # pragma: no cover - logging is non-critical, never fatal
        pass


def _fit_nlsq_heterodyne(
    data: dict[str, Any],
    config: "ConfigManager",
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

    # Raw YAML dict for HeterodyneModel.from_config and NLSQConfig.from_dict.
    # fit_nlsq has already coerced str/Path → ConfigManager, so config.config
    # is always present here.
    yaml_dict = config.config

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
            f"heterodyne dispatch requires 'c2_exp' or 'c2' in the data dict; got keys {list(data)}"
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

    _safe_log_heterodyne_start(
        nlsq_cfg,
        str(getattr(nlsq_cfg, "analysis_mode", "two_component") or "two_component"),
        len(phi),
    )

    # Setup-log parity with the homodyne/laminar path (``core.fit_nlsq_jax``):
    # configure CPU/HPC threading, note config-sourced initial parameters, and
    # emit the adaptive memory-threshold + ``memory_strategy_selection`` phase.
    # All three are best-effort and numerically inert — heterodyne routing is
    # unchanged; this only restores the matching log narrative.
    _safe_configure_cpu_threading()
    _safe_log_heterodyne_initial_params(yaml_dict)
    _safe_log_memory_strategy()

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
    # Read CMA-ES enablement from the PARSED config so BOTH the nested
    # (``cmaes.enable``) and FLAT (``enable_cmaes``) YAML forms are honored —
    # NLSQConfig.from_dict folds the nested block into the flat ``enable_cmaes``
    # field. Reading the raw nested dict alone would miss a flat
    # ``optimization.nlsq.enable_cmaes: true`` and let multistart/hybrid/
    # stratified-LS intercept before CMA-ES. This single change fixes the
    # multistart, hybrid, AND stratified-LS precedence gates.
    cmaes_on = bool(getattr(nlsq_cfg, "enable_cmaes", False))
    if isinstance(ms_dict, dict) and ms_dict.get("enable", False) and not cmaes_on:
        from xpcsjax.optimization.nlsq.heterodyne_logging import (
            log_strategy_selection as _log_strategy,
        )
        from xpcsjax.optimization.nlsq.heterodyne_multistart import (
            build_multistart_config,
            fit_nlsq_multistart_heterodyne,
        )

        _log_strategy("multi_start", "multi_start.enable=True")
        ms_cfg = build_multistart_config(ms_dict)
        result = fit_nlsq_multistart_heterodyne(model, c2, phi, nlsq_cfg, weights, ms_cfg)
        _safe_log_heterodyne_completion(result, model, len(phi))
        return result

    # Hybrid-streaming dispatch (Phase 2). Mirrors homodyne wrapper.py:1119-1165:
    # fire only when the memory tier requires streaming (LARGE/STREAMING) so the
    # template's default hybrid_streaming.enable: true does not stream small data.
    # Precedence: cmaes > multi_start > hybrid_streaming > local.
    # `enable` defaults to False (homodyne parity, wrapper.py:1109): a config that
    # omits the hybrid_streaming section opts out, so the gate never touches
    # select_nlsq_strategy / model.param_manager for non-hybrid fits. The shipped
    # template sets enable: true explicitly, so template users still get it.
    hybrid_dict = nlsq_dict.get("hybrid_streaming", {}) if isinstance(nlsq_dict, dict) else {}
    if not cmaes_on and isinstance(hybrid_dict, dict) and hybrid_dict.get("enable", False):
        from xpcsjax.optimization.nlsq.heterodyne_memory import (
            NLSQStrategy,
            select_nlsq_strategy,
        )
        from xpcsjax.utils.logging import get_logger as _get_logger

        _logger = _get_logger(__name__)
        decision = select_nlsq_strategy(
            int(_np.asarray(c2).size), int(model.param_manager.n_varying)
        )
        if decision.strategy in (NLSQStrategy.LARGE, NLSQStrategy.STREAMING):
            from xpcsjax.optimization.nlsq.heterodyne_logging import (
                log_strategy_selection as _log_strategy,
            )
            from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
                build_hybrid_streaming_result,
            )
            from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
                build_heterodyne_stratified_data,
            )
            from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
                fit_with_stratified_hybrid_streaming_heterodyne,
            )

            _log_strategy("hybrid_streaming", f"memory tier {decision.strategy.name}")
            strat = build_heterodyne_stratified_data(model, c2, phi, weights)
            lower, upper = model.param_manager.get_bounds()
            popt, pcov, info = fit_with_stratified_hybrid_streaming_heterodyne(
                stratified_data=strat,
                model=model,
                physical_param_names=list(model.param_manager.varying_names),
                initial_params=_np.asarray(
                    model.param_manager.get_initial_values(), dtype=_np.float64
                ),
                bounds=(_np.asarray(lower), _np.asarray(upper)),
                hybrid_config=hybrid_dict,
                anti_degeneracy_config=(
                    nlsq_dict.get("anti_degeneracy", {}) if isinstance(nlsq_dict, dict) else {}
                ),
            )
            result = build_hybrid_streaming_result(
                model=model,
                popt=popt,
                pcov=pcov,
                info=info,
                phi_angles=phi,
            )
            _safe_log_heterodyne_completion(result, model, len(phi))
            return result
        _logger.debug(
            "hybrid_streaming enabled but memory tier is %s (< LARGE); using standard joint fit",
            decision.strategy,
        )

    # Standard tier: homodyne-mirrored stratification gate.
    # Mirror homodyne: stratify when per-angle + >100k points + balanced angles;
    # engage the stratified-LS solver only at >=1M (homodyne's stratified-LS gate).
    from xpcsjax.optimization.nlsq.heterodyne_config import (
        StratificationConfig as _StratificationConfig,
    )
    from xpcsjax.optimization.nlsq.strategies.chunking import (
        analyze_angle_distribution,
        should_use_stratification,
    )

    # Stratification lives at ``optimization.stratification`` -- a SIBLING of
    # ``optimization.nlsq`` (homodyne wrapper.py:_apply_stratification_if_needed),
    # so it must be read from ``opt_block``, NOT from the unwrapped ``nlsq_dict``.
    strat_cfg = _StratificationConfig.from_optimization_block(opt_block)

    n_points = _estimate_heterodyne_points(c2, phi)
    # ``imbalance`` here is computed from the per-angle ``phi`` list (each unique
    # angle counted once), mirroring upstream homodyne's gate
    # (wrapper.py:_apply_stratification_if_needed → ``analyze_angle_distribution(phi)``)
    # 1:1 — homodyne feeds the same per-angle array, not a per-point distribution.
    # It DOES drive a real routing decision below (the configured
    # ``max_imbalance_ratio`` gate), but the TRUE per-point angle imbalance is
    # applied inside the stratified path itself: ``reorder_for_stratification`` →
    # ``create_angle_stratified_indices`` runs ``analyze_angle_distribution`` over
    # the filtered per-point ``phi_idx_filtered`` support. Switching this gate to a
    # per-point distribution would diverge from homodyne (and would require
    # materializing the flat support before the routing decision), so it is left as
    # the homodyne-parity per-angle imbalance.
    imbalance = float(analyze_angle_distribution(_np.asarray(phi)).imbalance_ratio)
    # Pass ``imbalance_ratio=0.0`` so ``should_use_stratification``'s hard-coded
    # 5.0 cutoff never fires here; we apply the CONFIGURED threshold ourselves
    # below. This lets ``max_imbalance_ratio`` move the cutoff in BOTH directions
    # (tighten below 5.0 OR loosen above it), not merely tighten it. The
    # point-count (>100k) and single-angle checks inside should_use_stratification
    # still apply.
    use_strat, _reason = should_use_stratification(
        n_points=n_points,
        n_angles=len(phi),
        per_angle_scaling=True,
        imbalance_ratio=0.0,
    )
    # ``enabled: false`` disables; ``enabled: true`` and ``"auto"`` are equivalent
    # for heterodyne -- there is no separate force-on path because the ``>=1M``
    # stratified-LS solver gate below is the real control (stratification only
    # changes the solver, which engages at >=1M regardless of this flag).
    if strat_cfg.is_disabled():
        use_strat = False
    # Apply the CONFIGURED imbalance threshold (this is now the SOLE imbalance
    # gate, honoring ``optimization.stratification.max_imbalance_ratio`` exactly).
    if use_strat and imbalance > strat_cfg.max_imbalance_ratio:
        from xpcsjax.utils.logging import get_logger as _get_logger

        _get_logger(__name__).debug(
            "angle imbalance %.2f > configured max_imbalance_ratio %.2f; stratified-LS skipped",
            imbalance,
            strat_cfg.max_imbalance_ratio,
        )
        use_strat = False

    # Only resolve the effective per-angle mode when the stratified-LS gate could
    # actually fire (CMA-ES off, stratification chosen, and >= 1M points). This
    # keeps the heterodyne_core import lazy so dispatch unit tests that stub
    # heterodyne_core (and never reach the 1M gate) are unaffected.
    if not cmaes_on and use_strat and n_points >= 1_000_000:
        # Only ``averaged`` and ``fourier`` use the JOINT stratified-LS objective.
        # ``individual`` is a SEQUENTIAL per-angle algorithm
        # (heterodyne_core._aggregate_individual_results); routing it through
        # stratified-LS would silently change the objective at the 1M boundary.
        # ``constant`` (and anything else) likewise stays in-memory.
        from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode

        effective_mode = _resolve_effective_mode(nlsq_cfg, len(phi))
    else:
        effective_mode = None

    _stratified_ls_fallback = False
    if effective_mode in ("averaged", "fourier"):
        try:
            from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as _hsl
            from xpcsjax.optimization.nlsq.heterodyne_logging import (
                log_strategy_selection as _log_strategy,
            )

            _log_strategy(
                "stratified_least_squares",
                f"{int(n_points):,} points >= 1M, per_angle_mode={effective_mode}",
            )
            result = _hsl.fit_heterodyne_stratified_least_squares(
                model=model,
                c2=c2,
                phi=phi,
                config=nlsq_cfg,
                weights=weights,
                target_chunk_size=strat_cfg.target_chunk_size,
                shuffle=True,
                use_index_based=strat_cfg.use_index_based,
                check_memory_safety=strat_cfg.check_memory_safety,
            )
            _safe_log_heterodyne_completion(result, model, len(phi))
            return result
        # Phase-2: intentionally left — implements the keep-better/fallback contract; conversion would risk parity.
        except Exception as exc:  # best-effort: never let stratification break a fit
            from xpcsjax.utils.logging import get_logger as _get_logger

            _get_logger(__name__).warning(
                "Heterodyne stratified-LS failed (%s); falling back to in-memory joint fit.",
                exc,
            )
            # Surface the fallback so the caller can distinguish "stratified
            # succeeded" from "stratified failed → OOM-prone in-memory path".
            _stratified_ls_fallback = True
    elif effective_mode is not None:
        # effective_mode resolved (>=1M, stratification chosen, CMA-ES off) but is
        # individual/constant — those use the in-memory path, not stratified-LS.
        # "No silent caps": at >=1M the in-memory joint fit is the OOM-prone path,
        # and the ONLY reason stratification was skipped is that this mode lacks a
        # stratified expander. Surface that at WARNING so a large fit silently
        # taking the higher-memory path is VISIBLE. Below 1M stratification would
        # not have engaged anyway, so keep it at debug there.
        from xpcsjax.utils.logging import get_logger as _get_logger

        if n_points >= 1_000_000:
            _get_logger(__name__).warning(
                "per_angle_mode=%s at %d points (>=1M): stratified-LS skipped "
                "because this mode's stratified expander is not wired; using the "
                "higher-memory in-memory joint fit (potential OOM risk).",
                effective_mode,
                int(n_points),
            )
        else:
            _get_logger(__name__).debug(
                "per_angle_mode=%s uses in-memory path; stratified-LS skipped",
                effective_mode,
            )

    from xpcsjax.optimization.nlsq.heterodyne_logging import (
        log_strategy_selection as _log_strategy,
    )

    _log_strategy("standard", f"{int(n_points):,} points (in-memory joint fit)")

    # Homodyne-parity 100k–1M shuffle regime: laminar_flow reorganizes + seed-42
    # pre-shuffles >=100k-point per-angle fits that stay in-memory (stratified-LS
    # only engages at >=1M). Mirror it on the heterodyne in-memory path via an
    # OBJECTIVE-invariant angle-axis reorder, scoped to the **averaged** mode
    # ONLY: it collapses scaling to two GLOBAL params, so the fit objective is
    # unchanged by angle order and the only angle-indexed outputs are diagnostics
    # (realigned to the caller's order after the fit). 'constant' freezes
    # PER-ANGLE scaling and writes model.scaling in angle order; individual/
    # fourier carry an angle-ordered param+covariance tail — those are
    # intentionally NOT reordered here (they would need per-angle state /
    # covariance un-permutation; deferred).
    _angle_inv = None
    if use_strat and 100_000 <= n_points < 1_000_000 and not cmaes_on:
        from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode

        if _resolve_effective_mode(nlsq_cfg, len(phi)) == "averaged":
            c2, phi, weights, _angle_inv = _seed42_angle_reorder(c2, phi, weights, n_points)

    result = fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights)
    if _angle_inv is not None:
        _restore_angle_order(result, _angle_inv)
    if _stratified_ls_fallback and result.nlsq_diagnostics is not None:
        result.nlsq_diagnostics["stratified_ls_fallback"] = True
    _safe_log_heterodyne_completion(result, model, len(phi))
    return result


# Ensure fit_nlsq joins whatever __all__ the verbatim port already defined
try:
    __all__  # type: ignore[name-defined]  # noqa: B018 - existence probe for __all__
except NameError:
    __all__ = []
if "fit_nlsq" not in __all__:
    __all__.append("fit_nlsq")
