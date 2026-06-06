"""NLSQ (non-linear least squares) optimization subpackage for xpcsjax.

Houses the entire NLSQ curve-fitting stack and exposes
:func:`fit_nlsq`, the v0.1 single-entry public wrapper that fits both the
homodyne and heterodyne physics models. xpcsjax is NLSQ-only by design; no
Bayesian / MCMC pathway exists here.

The package owns the fit *strategy* (memory-aware routing, the anti-degeneracy
defense layers, CMA-ES / LHS-multistart global escapes, bounds and parameter
transforms, angle-stratified chunking) while delegating the trust-region solve
and JIT cache to the upstream ``nlsq`` library. xpcsjax calls ``nlsq``'s
``curve_fit()`` / ``CurveFit`` directly rather than its higher-level ``fit()``
API, and routes memory itself via :func:`select_nlsq_strategy` instead of
``nlsq``'s ``MemoryBudgetSelector``.

Notes
-----
Module map:

- ``core.py`` — :func:`fit_nlsq_jax` (homodyne entry point) and ``NLSQResult``.
- ``wrapper.py`` — ``NLSQWrapper`` adapter (legacy).
- ``adapter.py`` — ``NLSQAdapter`` over ``nlsq``'s ``CurveFit`` for JIT caching.
- ``memory.py`` — memory estimation and :func:`select_nlsq_strategy`.
- ``parameter_utils.py`` — parameter labelling, per-angle init, Jacobian stats.
- ``results.py`` — :class:`~results.OptimizationResult` and result dataclasses.
- ``transforms.py`` — parameter transformation utilities.
- ``data_prep.py`` — data validation and per-angle parameter expansion.
- ``result_builder.py`` — result assembly and quality-metric computation.
- ``fit_computation.py`` — theoretical-fit and parameter-extraction helpers.
- ``multistart.py`` — LHS multistart global optimization (no subsampling).
- ``anti_degeneracy_controller.py`` — the 5-layer anti-degeneracy controller.
- ``cmaes_wrapper.py`` — CMA-ES global-optimization escape.
- ``heterodyne_*`` — the ``two_component`` model fit orchestration.
- ``strategies/`` — chunking, stratified residual functions, sequential
  per-angle optimization, and strategy-pattern executors.
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
    """Run the xpcsjax v0.1 NLSQ fit for either physics model.

    Single public entry point for non-linear least squares curve fitting in
    xpcsjax. It reads ``analysis_mode`` from ``config`` and dispatches:
    ``two_component`` / ``heterodyne`` modes go to the heterodyne orchestration,
    everything else (``laminar_flow``, ``static_anisotropic``,
    ``static_isotropic``) goes to the homodyne path
    (:func:`xpcsjax.optimization.nlsq.core.fit_nlsq_jax`). Within each path,
    memory-aware strategy routing, the anti-degeneracy controller, and any
    configured global escapes (CMA-ES / multistart) are selected automatically
    from the config — there is no second optimizer pathway and no Bayesian /
    MCMC route (out of scope for xpcsjax by design).

    Parameters
    ----------
    data : dict
        XPCS data dict. For the homodyne path, the dict returned by
        :func:`xpcsjax.data.load_xpcs_data`. For the heterodyne path, a dict
        carrying the correlation stack under ``c2_exp`` or ``c2`` (a
        ``(n_phi, N, N)`` stack or a single-angle ``(N, N)`` matrix) and the
        detector angles under ``phi_angles_list``, ``phi_angles``, or ``phi``
        (degrees). An optional ``weights`` array enables weighted least squares.
    config : ConfigManager or str or pathlib.Path
        A pre-built :class:`xpcsjax.config.ConfigManager`, or a path to a YAML
        config file (coerced to a ``ConfigManager``).

    Returns
    -------
    OptimizationResult
        The completed fit. Both paths return the same
        :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` type.

        The heterodyne path returns one for every per-angle scaling mode —
        ``"constant"``, ``"fourier"``, ``"individual"``, and ``"auto"`` (which
        resolves internally to ``"averaged"`` for ``n_phi >= 3`` or
        ``"individual"`` otherwise). Mode-specific per-angle data
        (``chi2_per_angle``, ``parameter_names``,
        ``contrast_per_angle`` / ``offset_per_angle``, etc.) lives under
        :attr:`result.nlsq_diagnostics <OptimizationResult.nlsq_diagnostics>`.
        The ``individual`` mode additionally carries
        ``covariance_structure="block_diagonal_sequential"`` to signal that
        off-diagonal covariance entries are zero **by construction**
        (sequential per-angle fits with held-fixed scaling) rather than by fit.

    Raises
    ------
    KeyError
        If the heterodyne path is selected but the ``data`` dict lacks a
        correlation stack (``c2_exp`` / ``c2``) or an angle array
        (``phi_angles_list`` / ``phi_angles`` / ``phi``).

    See Also
    --------
    OptimizationResult : The result type returned by this function.
    xpcsjax.optimization.nlsq.core.fit_nlsq_jax : Homodyne fit entry point.
    xpcsjax.optimization.nlsq.heterodyne_views : Post-hoc per-angle
        reconstruction helpers (``reconstruct_per_angle_scaling``,
        ``per_angle_chi2``).

    Notes
    -----
    Mode dispatch is case- and separator-insensitive: ``"two-component"`` is
    normalized to ``"two_component"`` before routing.

    Examples
    --------
    >>> from xpcsjax import fit_nlsq, load_xpcs_data
    >>> data = load_xpcs_data("config_laminar.yaml")  # doctest: +SKIP
    >>> result = fit_nlsq(data, "config_laminar.yaml")  # doctest: +SKIP
    >>> result.success  # doctest: +SKIP
    True
    >>> result.reduced_chi_squared  # doctest: +SKIP
    1.07
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
            ad_dict = (
                nlsq_dict.get("anti_degeneracy")
                if isinstance(nlsq_dict, dict)
                else None
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
                anti_degeneracy_dict=ad_dict,
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

    # In-memory joint fit (<1M, non-escape). The three in-scope per-angle scaling
    # modes (fixed_constant / individual / auto_averaged — production constant /
    # individual / auto-at-n_phi>=3) route through the SHARED stratification
    # engine via fit_two_component_via_engine (Task #16b). The engine route does
    # its own frame-0 exclusion + stratification, so the superseded seed-42
    # angle-shuffle regime is removed. fourier stays on fit_nlsq_multi_phi (the
    # engine route raises NotImplementedError for it). Best-effort: any routing
    # failure falls back to fit_nlsq_multi_phi so a fit never breaks.
    #
    # NOTE: this CHANGES two_component in-memory in-scope-mode results by ~1e-3
    # vs the old direct fit_nlsq_multi_phi path — the accepted no-worse contract
    # (engine SSR <= production SSR), NOT bit-identical. See CLAUDE.md.
    # Best-effort: the whole mode-resolution + engine attempt is guarded so any
    # failure (including the heterodyne_core import being stubbed out by a
    # dispatch unit test that never reaches a real fit) falls back to
    # fit_nlsq_multi_phi. The import stays inside the try so it does NOT eagerly
    # touch heterodyne_core for stubbed callers.
    result = None
    try:
        from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode

        _effective_mode = _resolve_effective_mode(nlsq_cfg, len(phi))
        if _effective_mode != "fourier":
            from xpcsjax.optimization.nlsq.heterodyne_engine_route import (
                fit_two_component_via_engine,
            )

            result = fit_two_component_via_engine(model, c2, phi, nlsq_cfg, weights)
    except Exception as _engine_exc:  # noqa: BLE001 - best-effort engine route
        from xpcsjax.utils.logging import get_logger as _get_logger

        _get_logger(__name__).warning(
            "Engine-route two_component fit failed (%s: %s); falling back to fit_nlsq_multi_phi.",
            type(_engine_exc).__name__,
            _engine_exc,
        )
        result = None

    if result is None:
        result = fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights)
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
