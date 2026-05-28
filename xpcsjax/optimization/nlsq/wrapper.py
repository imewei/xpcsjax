"""NLSQ Wrapper for Homodyne Optimization.

Role and When to Use (v2.11.0+)
-------------------------------

**NLSQWrapper** (this module) is the **stable fallback adapter** for:
- Complex optimizations requiring full anti-degeneracy integration
- laminar_flow mode with many phi angles (> 6)
- Large datasets (> 100M points) requiring streaming/chunking strategies
- Custom transforms or advanced recovery mechanisms
- Production stability where reliability is critical

Use **NLSQAdapter** instead for:
- Standard optimizations (static_isotropic mode)
- Small to medium datasets (< 10M points)
- Multi-start optimization (model caching provides 3-5× speedup)
- Performance-critical workflows requiring JIT compilation

**Key Differences:**

* Model caching: NLSQWrapper=None, NLSQAdapter=Built-in
* JIT compilation: NLSQWrapper=Manual, NLSQAdapter=Auto
* Workflow auto-select: NLSQWrapper=Custom, NLSQAdapter=Via NLSQ
* Anti-degeneracy layers: NLSQWrapper=Full, NLSQAdapter=Via fit()
* Recovery system: NLSQWrapper=3-attempt, NLSQAdapter=NLSQ native
* Streaming support: NLSQWrapper=Full custom, NLSQAdapter=Via NLSQ

**Decision Guide:**

1. If you need robust streaming for 100M+ points: Use NLSQWrapper
2. If you need full anti-degeneracy control: Use NLSQWrapper
3. If you need maximum speed for multi-start optimization: Use NLSQAdapter
4. Default recommendation: NLSQAdapter with automatic fallback to NLSQWrapper

This module provides an adapter layer between homodyne's optimization API
and the NLSQ package's trust-region nonlinear least squares interface.

The NLSQWrapper class implements the Adapter pattern to translate:
- Homodyne's multi-dimensional XPCS data → NLSQ's flattened array format
- Homodyne's parameter bounds tuple → NLSQ's (lower, upper) format
- NLSQ's (popt, pcov) output → Homodyne's OptimizationResult dataclass

Key Features:
- Automatic dataset size detection and strategy selection
- Angle-stratified chunking for per-angle parameter compatibility (v2.2+)
- Intelligent error recovery with 3-attempt retry strategy (T022-T024)
- Actionable error diagnostics with 5 error categories
- CPU-optimized execution through JAX
- Progress logging and convergence diagnostics
- Scientifically validated (7/7 validation tests passed, T036-T041)
- Serves as fallback when NLSQAdapter fails

Per-Angle Scaling Fix (v2.2):
- Fixes silent optimization failures with per-angle parameters on large datasets
- Applies angle-stratified chunking when: per_angle_scaling=True AND n_points>100k
- Ensures every NLSQ chunk contains all phi angles → gradients always well-defined
- <1% performance overhead (0.15s for 3M points)
- Reference: ultra-think-20251106-012247

Production Status:
- Production-ready with comprehensive error recovery
- Scientifically validated (100% test pass rate)
- Parameter recovery accuracy: 2-14% on core parameters
- Sub-linear performance scaling with dataset size
- Per-angle scaling compatible with large datasets (v2.2+)

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Validation: See tests/validation/test_scientific_validation.py (T036-T041)
- Documentation: See CHANGELOG.md and CLAUDE.md for detailed status
"""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# ruff: noqa: I001
# Import order is INTENTIONAL: nlsq must be imported BEFORE JAX
# This enables automatic x64 (double precision) configuration per NLSQ best practices
# Reference: https://nlsq.readthedocs.io/en/latest/guides/advanced_features.html
from nlsq import curve_fit, curve_fit_large

# Try importing AdaptiveHybridStreamingOptimizer (available in NLSQ >= 0.3.2)
# This is the preferred streaming optimizer - the old StreamingOptimizer was removed in NLSQ 0.4.0
# Fixes: 1) Shear-term weak gradients, 2) Slow convergence, 3) Crude covariance
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    STREAMING_AVAILABLE = True  # For backwards compatibility
    HYBRID_STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    HYBRID_STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None
    HybridStreamingConfig = None

import logging

from xpcsjax.utils.logging import get_logger

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.batch_statistics import BatchStatistics
from xpcsjax.optimization.nlsq.adapter_base import NLSQAdapterBase
from xpcsjax.optimization.nlsq.results import (
    FunctionEvaluationCounter,
    OptimizationResult,
    UseSequentialOptimization,
)
from xpcsjax.optimization.nlsq.strategies.chunking import (
    StratificationDiagnostics,
    analyze_angle_distribution,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_stratification_memory,
    format_diagnostics_report,
    should_use_stratification,
)
from xpcsjax.optimization.nlsq.strategies.residual import (
    StratifiedResidualFunction,
    create_stratified_residual_function,
)
from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)

# Fallback chain logic (extracted to fallback_chain.py)
from xpcsjax.optimization.nlsq.fallback_chain import (
    OptimizationStrategy,
    execute_optimization_with_fallback,
    get_fallback_strategy,
    handle_nlsq_result,
)
from xpcsjax.optimization.nlsq.recovery import (
    diagnose_error,
    execute_with_recovery,
    safe_uncertainties_from_pcov,
)
from xpcsjax.optimization.nlsq.strategies.out_of_core import (
    fit_with_out_of_core_accumulation,
)
from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
    create_stratified_chunks,
    fit_with_stratified_least_squares,
)
from xpcsjax.optimization.nlsq.strategies.hybrid_streaming import (
    estimate_memory_for_stratified_ls,
    fit_with_hybrid_streaming_optimizer,
    fit_with_stratified_hybrid_streaming,
    fit_with_streaming_optimizer_stratified_deprecated,
    should_use_streaming,
)


from xpcsjax.optimization.nlsq.strategies.sequential import (  # noqa: E402
    JAC_SAMPLE_SIZE,
    optimize_per_angle_sequential,
)
from xpcsjax.core.physics_nlsq import compute_g2_scaled  # noqa: E402
from xpcsjax.core.physics_utils import apply_diagonal_correction  # noqa: E402
from xpcsjax.optimization.nlsq.transforms import (  # noqa: E402
    adjust_covariance_for_transforms,
    apply_forward_shear_transforms_to_bounds,
    apply_forward_shear_transforms_to_vector,
    apply_inverse_shear_transforms_to_vector,
    build_per_parameter_x_scale,
    build_physical_index_map,
    format_x_scale_for_log,
    normalize_x_scale_map,
    parse_shear_transform_config,
    wrap_model_function_with_transforms,
    wrap_stratified_function_with_transforms,
)
from xpcsjax.optimization.numerical_validation import NumericalValidator  # noqa: E402
from xpcsjax.optimization.recovery_strategies import (  # noqa: E402
    RecoveryStrategyApplicator,
)

# Anti-Degeneracy Defense System v2.9.0

# Memory management utilities (extracted to memory.py for reduced complexity)
from xpcsjax.optimization.nlsq.memory import (  # noqa: E402
    get_adaptive_memory_threshold,
    NLSQStrategy,
    select_nlsq_strategy,
)

# Parameter utilities (extracted to parameter_utils.py for reduced complexity)
from xpcsjax.optimization.nlsq.parameter_utils import (  # noqa: E402
    build_parameter_labels as _build_parameter_labels,
    classify_parameter_status as _classify_parameter_status,
    sample_xdata as _sample_xdata,
    compute_jacobian_stats as _compute_jacobian_stats,
    compute_consistent_per_angle_init as _compute_consistent_per_angle_init,
)

# Module-level logger
_memory_logger = get_logger(__name__)


def _extract_n_points(data: Any) -> int:
    """Extract number of data points from various data formats.

    Handles XPCSData objects, numpy arrays, lists, and other iterables.

    Parameters
    ----------
    data : Any
        Data object with g2 attribute or array-like

    Returns
    -------
    int
        Number of data points (0 if cannot determine)
    """
    # Try g2 attribute (XPCSData)
    if hasattr(data, "g2"):
        g2 = data.g2
        if hasattr(g2, "size"):
            return int(g2.size)
        if hasattr(g2, "__len__"):
            return len(g2)
    # Try direct array-like
    if hasattr(data, "size"):
        return int(data.size)
    if hasattr(data, "__len__"):
        return len(data)
    return 0


def create_multistart_warmup_func(
    model_func: Callable[..., np.ndarray],
    xdata: np.ndarray,
    ydata: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    warmup_learning_rate: float = 0.001,
    normalize: bool = True,
    chunk_size: int = 50_000,
) -> Callable[[dict[str, Any], np.ndarray, int], Any]:
    """Create a warmup-only fit function for multi-start Phase 1 strategy.

    This function creates a warmup_fit_func compatible with the multi-start
    optimization module Phase 1 strategy. It uses the L-BFGS warmup phase
    from the NLSQ AdaptiveHybridStreamingOptimizer to quickly explore the
    parameter space without full Gauss-Newton refinement.

    Parameters
    ----------
    model_func : Callable
        Model function with signature: ``func(x, *params) -> predictions``
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data (observations)
    bounds : tuple[np.ndarray, np.ndarray] | None, optional
        Parameter bounds as (lower, upper)
    warmup_learning_rate : float, default=0.001
        L-BFGS line search scale for warmup phase
    normalize : bool, default=True
        Whether to use parameter normalization (recommended for scale imbalance)
    chunk_size : int, default=50000
        Points per chunk for streaming computation

    Returns
    -------
    warmup_fit_func : Callable
        Function with signature: (data, initial_params, n_iterations) -> SingleStartResult
        Compatible with run_multistart_nlsq() warmup_fit_func parameter.

    Raises
    ------
    RuntimeError
        If AdaptiveHybridStreamingOptimizer is not available (NLSQ < 0.3.2)

    Examples
    --------
    >>> from xpcsjax.optimization.nlsq.wrapper import create_multistart_warmup_func
    >>> from xpcsjax.optimization.nlsq.multistart import run_multistart_nlsq
    >>>
    >>> # Create warmup function
    >>> warmup_func = create_multistart_warmup_func(
    ...     model_func=my_model,
    ...     xdata=x_data,
    ...     ydata=y_data,
    ...     bounds=(lower, upper),
    ... )
    >>>
    >>> # Use with multi-start
    >>> result = run_multistart_nlsq(
    ...     data=my_data,
    ...     bounds=bounds,
    ...     config=config,
    ...     single_fit_func=full_fit_func,
    ...     warmup_fit_func=warmup_func,  # For Phase 1 strategy
    ... )

    Notes
    -----
    This function integrates with the Phase 1 multi-start strategy which:
    1. Runs parallel L-BFGS warmup from multiple starting points
    2. Selects the best warmup result
    3. Performs full Gauss-Newton refinement from the best starting point

    This approach is memory-efficient for very large datasets (>100M points)
    and provides good exploration of the parameter space.

    See Also
    --------
    xpcsjax.optimization.nlsq.multistart.run_multistart_nlsq : Main multi-start function
    xpcsjax.optimization.nlsq.multistart._run_phase1_strategy : Phase 1 strategy implementation
    """
    from xpcsjax.optimization.nlsq.multistart import SingleStartResult

    if not HYBRID_STREAMING_AVAILABLE:
        raise RuntimeError(
            "AdaptiveHybridStreamingOptimizer not available. "
            "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
        )

    def warmup_fit_func(
        data: dict[str, Any],
        initial_params: np.ndarray,
        n_iterations: int,
    ) -> SingleStartResult:
        """Run warmup-only optimization from a starting point.

        Parameters
        ----------
        data : dict
            Data dictionary (not used directly; uses captured xdata/ydata)
        initial_params : np.ndarray
            Initial parameter values
        n_iterations : int
            Number of L-BFGS warmup iterations

        Returns
        -------
        SingleStartResult
            Optimization result with warmup parameters and cost
        """
        import time

        start_time = time.perf_counter()

        try:
            # Configure for warmup-only: skip Gauss-Newton phase
            config = HybridStreamingConfig(
                normalize=normalize,
                normalization_strategy="bounds",
                warmup_iterations=n_iterations,
                max_warmup_iterations=n_iterations,  # Force stop at n_iterations
                warmup_learning_rate=warmup_learning_rate,
                gauss_newton_max_iterations=0,  # Skip GN phase
                gauss_newton_tol=1e-8,
                chunk_size=chunk_size,
                validate_numerics=True,
            )

            optimizer = AdaptiveHybridStreamingOptimizer(config)

            # Run warmup-only optimization
            result = optimizer.fit(
                data_source=(xdata, ydata),
                func=model_func,
                p0=initial_params,
                bounds=bounds,
                verbose=0,  # Quiet mode
            )

            # Extract results
            final_params = np.asarray(result["x"])

            # Compute chi-squared from final cost
            # The optimizer returns cost as 0.5 * sum(residuals^2)
            diagnostics = result.get("streaming_diagnostics", {})
            warmup_diag = diagnostics.get("warmup_diagnostics", {})
            final_loss = warmup_diag.get("final_loss", float("inf"))

            # Convert loss to chi-squared (loss = 0.5 * chi_sq for LSQ)
            chi_squared = (
                2.0 * final_loss if final_loss != float("inf") else float("inf")
            )

            wall_time = time.perf_counter() - start_time

            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=final_params,
                chi_squared=chi_squared,
                success=result.get("success", False),
                n_iterations=n_iterations,
                wall_time=wall_time,
                message="L-BFGS warmup completed",
            )

        except (ValueError, RuntimeError, OSError) as e:
            wall_time = time.perf_counter() - start_time
            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=initial_params,
                chi_squared=float("inf"),
                success=False,
                n_iterations=0,
                wall_time=wall_time,
                message=f"Warmup failed: {str(e)}",
            )

    return warmup_fit_func


def _safe_uncertainties_from_pcov(pcov: np.ndarray, n_params: int) -> np.ndarray:
    """Extract uncertainties with diagonal regularization for singular pcov."""
    return safe_uncertainties_from_pcov(pcov, n_params)


class NLSQWrapper(NLSQAdapterBase):
    """Adapter class for NLSQ package integration with homodyne optimization.

    This class translates between homodyne's optimization API and the NLSQ
    package's curve_fit interface, handling:
    - Data format transformations
    - Parameter validation and bounds checking
    - Automatic strategy selection for large datasets
    - Hybrid error handling and recovery

    Usage:
        wrapper = NLSQWrapper(enable_large_dataset=True)
        result = wrapper.fit(data, config, initial_params, bounds, analysis_mode)
    """

    def __init__(
        self,
        enable_large_dataset: bool = True,
        enable_recovery: bool = True,
        enable_numerical_validation: bool = True,
        max_retries: int = 2,
        fast_mode: bool = False,
    ) -> None:
        """Initialize NLSQWrapper.

        Args:
            enable_large_dataset: Use curve_fit_large for datasets >1M points
            enable_recovery: Enable automatic error recovery strategies
            enable_numerical_validation: Enable NaN/Inf validation at 3 critical points
            max_retries: Maximum retry attempts per batch (default: 2)
            fast_mode: Disable non-essential checks for < 1% overhead (Task 5.5)
        """
        self.enable_large_dataset = enable_large_dataset
        self.enable_recovery = enable_recovery
        self.enable_numerical_validation = enable_numerical_validation and not fast_mode
        self.max_retries = max_retries
        self.fast_mode = fast_mode

        # Initialize streaming optimization components
        self.batch_statistics = BatchStatistics(max_size=100)
        self.recovery_applicator = RecoveryStrategyApplicator(max_retries=max_retries)
        self.numerical_validator = NumericalValidator(
            enable_validation=enable_numerical_validation and not fast_mode
        )

        # Best parameter tracking
        self.best_params = None
        self.best_loss = float("inf")
        self.best_batch_idx = -1

    @staticmethod
    def _get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]:
        """Get physical parameter names for a given analysis mode.

        Args:
            analysis_mode: 'static_isotropic' or 'laminar_flow'

        Returns:
            List of physical parameter names (excludes scaling parameters)

        Raises:
            ValueError: If analysis_mode is not recognized
        """
        normalized_mode = analysis_mode.lower()

        if normalized_mode in {"static_anisotropic", "static_isotropic"}:
            return ["D0", "alpha", "D_offset"]
        elif normalized_mode == "laminar_flow":
            return [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",  # Canonical name (was gamma_dot_0)
                "beta",
                "gamma_dot_t_offset",  # Canonical name (was gamma_dot_offset)
                "phi0",
            ]
        else:
            raise ValueError(
                f"Unknown analysis_mode: '{analysis_mode}'. "
                f"Expected 'static_anisotropic', 'static_isotropic', or 'laminar_flow'"
            )

    @staticmethod
    def _extract_nlsq_settings(config: Any) -> dict[str, Any]:
        """Return NLSQ-specific settings from the config tree (if present)."""

        config_dict = None
        if hasattr(config, "config") and isinstance(config.config, dict):
            config_dict = config.config
        elif isinstance(config, dict):
            config_dict = config

        if not config_dict:
            return {}

        nlsq_settings = config_dict.get("optimization", {}).get("nlsq", {})
        return cast(dict[str, Any], nlsq_settings)

    @staticmethod
    def _handle_nlsq_result(
        result: Any,
        strategy: OptimizationStrategy,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Normalize NLSQ return values to consistent format.

        Delegates to fallback_chain.handle_nlsq_result().
        """
        return handle_nlsq_result(result, strategy)

    def _get_fallback_strategy(
        self, current_strategy: OptimizationStrategy
    ) -> OptimizationStrategy | None:
        """Get fallback strategy when current strategy fails.

        Delegates to fallback_chain.get_fallback_strategy().
        """
        return get_fallback_strategy(current_strategy)

    def fit(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        analysis_mode: AnalysisMode = AnalysisMode.STATIC_ISOTROPIC,
        per_angle_scaling: bool = True,  # REQUIRED: per-angle is physically correct
        diagnostics_enabled: bool = False,
        shear_transforms: dict[str, Any] | None = None,
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
    ) -> OptimizationResult:
        """Execute NLSQ optimization with automatic strategy selection and per-angle scaling.

        Args:
            data: XPCS experimental data
            config: Configuration manager with optimization settings
            initial_params: Initial parameter guess (auto-loaded if None)
            bounds: Parameter bounds as (lower, upper) tuple
            analysis_mode: 'static_isotropic' or 'laminar_flow'
            per_angle_scaling: MUST be True. Per-angle contrast/offset parameters are physically correct
                             as each scattering angle has different optical properties and detector responses.
                             Legacy scalar mode (False) is no longer supported.

        Returns:
            OptimizationResult with converged parameters and diagnostics

        Raises:
            ValueError: If bounds are invalid (lower > upper) or if per_angle_scaling=False
        """
        import time

        # nlsq imported at module level (line 36) for automatic x64 configuration

        logger = get_logger(__name__)

        # BREAKING CHANGE (Nov 2025): Validate per-angle scaling is enabled
        # Legacy scalar contrast/offset mode is not physically meaningful
        if not per_angle_scaling:
            logger.error(
                "Legacy scalar contrast/offset mode (per_angle_scaling=False) is no longer supported. "
                "Single contrast/offset parameters are not physically meaningful as each scattering "
                "angle has different optical properties and detector responses. "
                "Per-angle scaling is required for physically correct NLSQ optimization."
            )
            raise ValueError(
                "per_angle_scaling=False is deprecated and removed. "
                "Use per_angle_scaling=True (default) for physically correct behavior."
            )

        # Start timing
        start_time = time.time()

        physical_param_names = self._get_physical_param_names(analysis_mode)

        nlsq_settings = self._extract_nlsq_settings(config)
        loss_name = nlsq_settings.get("loss", "soft_l1")
        trust_region_scale = float(nlsq_settings.get("trust_region_scale", 1.0))
        if trust_region_scale <= 0:
            trust_region_scale = 1.0
        x_scale_override = nlsq_settings.get("x_scale")
        x_scale_value = x_scale_override if x_scale_override is not None else "jac"
        x_scale_map_config = normalize_x_scale_map(nlsq_settings.get("x_scale_map"))
        diagnostics_cfg = nlsq_settings.get("diagnostics", {})
        diagnostics_enabled = diagnostics_enabled or bool(
            diagnostics_cfg.get("enable", False),
        )
        diagnostics_sample_size = int(diagnostics_cfg.get("sample_size", 2048))
        diagnostics_payload = (
            {"solver_settings": {"loss": loss_name}} if diagnostics_enabled else None
        )
        transform_cfg = parse_shear_transform_config(shear_transforms)

        # Step 0.5: Unified Memory-Based Strategy Selection (v2.13.0)
        # Uses pure memory estimation - no legacy point thresholds.
        n_est_points = _extract_n_points(data)
        n_params = len(initial_params) if initial_params is not None else 0

        strategy_decision = select_nlsq_strategy(n_est_points, n_params)
        logger.info(
            f"Strategy selection: {strategy_decision.strategy.value} "
            f"({strategy_decision.reason})"
        )

        # Handle HYBRID_STREAMING (extreme scale - index array > 75% RAM)
        if strategy_decision.strategy == NLSQStrategy.HYBRID_STREAMING:
            if not HYBRID_STREAMING_AVAILABLE:
                logger.critical(
                    "AdaptiveHybridStreamingOptimizer required for extreme-scale "
                    f"dataset ({n_est_points:,} points) but not available."
                )
                raise MemoryError(
                    f"Dataset too large for RAM (index={strategy_decision.index_memory_gb:.1f} GB > "
                    f"threshold={strategy_decision.threshold_gb:.1f} GB) and Streaming unavailable."
                )
            # Streaming path continues below (handled by existing streaming logic)
            logger.warning(
                f"Extreme-scale dataset: {strategy_decision.reason}. "
                "Proceeding with Adaptive Hybrid Streaming."
            )

        # Handle OUT_OF_CORE (large scale - peak memory > 75% RAM)
        elif strategy_decision.strategy == NLSQStrategy.OUT_OF_CORE:
            if initial_params is None:
                raise ValueError("initial_params required for out-of-core optimization")

            validated_params = self._validate_initial_params(initial_params, bounds)
            nlsq_bounds = self._convert_bounds(bounds)

            # Default to False (User requirement: Never subsample data)
            use_fast_mode = self.fast_mode or config.config.get("optimization", {}).get(
                "fast_chi2_mode", False
            )

            # Extract anti-degeneracy config (will warn that it's not supported for out-of-core)
            ooc_anti_degeneracy_config = None
            if config is not None and hasattr(config, "config"):
                ooc_nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                ooc_anti_degeneracy_config = ooc_nlsq_config.get("anti_degeneracy", {})

            popt, pcov, info = self._fit_with_out_of_core_accumulation(
                stratified_data=None,
                data=data,
                per_angle_scaling=per_angle_scaling,
                physical_param_names=physical_param_names,
                initial_params=validated_params,
                bounds=nlsq_bounds,
                logger=logger,
                config=config,
                fast_chi2_mode=use_fast_mode,
                anti_degeneracy_config=ooc_anti_degeneracy_config,
            )

            execution_time = time.time() - start_time
            uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))
            # Effective DOF for reduced chi-squared: in auto_averaged mode the
            # optimizer works on a compressed param vector but the true model
            # DOF is 2*n_phi + n_physical (one contrast+offset per angle).
            _ooc_init_n_params_effective: int | None = None
            if per_angle_scaling and config is not None and hasattr(config, "config"):
                _ooc_init_ad = (
                    config.config.get("optimization", {})
                    .get("nlsq", {})
                    .get("anti_degeneracy", {})
                )
                _ooc_init_mode = _ooc_init_ad.get("per_angle_mode", "auto")
                _ooc_init_thresh = _ooc_init_ad.get("constant_scaling_threshold", 3)
                # Use data.phi to count unique angles — reading n_phi from len(popt)
                # is incorrect in auto_averaged mode where popt has compressed length
                # (e.g. 9 for laminar_flow: 7 physical + 2 averaged) rather than
                # the expanded length (2*n_phi + n_physical = 53 for 23 angles).
                # Inferring (len(popt) - n_physical) // 2 gives 1, not 23, so the
                # threshold check 1 >= 3 never fires and the DOF fix is silently skipped.
                _ooc_init_n_phi = len(np.unique(np.asarray(data.phi)))
                _ooc_init_n_physical = len(physical_param_names)
                if _ooc_init_mode == "auto" and _ooc_init_n_phi >= _ooc_init_thresh:
                    _ooc_init_n_params_effective = (
                        2 * _ooc_init_n_phi + _ooc_init_n_physical
                    )
                elif _ooc_init_mode == "constant":
                    _ooc_init_n_params_effective = _ooc_init_n_physical
            _ooc_init_dof = (
                _ooc_init_n_params_effective
                if _ooc_init_n_params_effective is not None
                else len(popt)
            )
            reduced_chi2 = info.get("chi_squared", 0.0) / max(
                1, n_est_points - _ooc_init_dof
            )
            # Derive quality_flag from reduced chi-squared (same thresholds
            # as _create_fit_result) instead of hardcoding "good".
            if reduced_chi2 < 1.5:
                _ooc_init_quality = "good"
            elif reduced_chi2 < 3.0:
                _ooc_init_quality = "marginal"
            else:
                _ooc_init_quality = "poor"

            return OptimizationResult(
                parameters=popt,
                uncertainties=uncertainties,
                covariance=pcov,
                chi_squared=info.get("chi_squared", 0.0),
                reduced_chi_squared=reduced_chi2,
                convergence_status=info.get("convergence_status", "unknown"),
                iterations=info.get("iterations", 0),
                execution_time=execution_time,
                device_info={
                    "device": "cpu_accumulated",
                    "strategy": "out_of_core",
                    "fast_mode": use_fast_mode,
                    "decision": strategy_decision.reason,
                },
                recovery_actions=["out_of_core_delegation"],
                quality_flag=_ooc_init_quality,
            )

        # STANDARD strategy falls through to existing optimization path

        # Step 1: Apply angle-stratified chunking if needed (BEFORE data preparation)
        # This fixes per-angle parameter incompatibility with NLSQ chunking (ultra-think-20251106-012247)
        stratified_data = self._apply_stratification_if_needed(
            data, per_angle_scaling, config, logger
        )

        # Extract stratification diagnostics if available
        stratification_diagnostics = None
        if hasattr(stratified_data, "stratification_diagnostics"):
            stratification_diagnostics = stratified_data.stratification_diagnostics

        # Check if sequential optimization is required
        transform_state: dict[str, Any] | None = None

        if isinstance(stratified_data, UseSequentialOptimization):
            logger.info(
                f"Using sequential per-angle optimization: {stratified_data.reason}"
            )
            return self._run_sequential_optimization(
                stratified_data.data,
                config,
                initial_params,
                bounds,
                analysis_mode,
                per_angle_scaling,
                logger,
                start_time,
                x_scale_value=x_scale_value,
                transform_cfg=transform_cfg,
                physical_param_names=physical_param_names,
                per_angle_scaling_initial=per_angle_scaling_initial,
            )

        # NEW: Check if stratified least_squares should be used (v2.2.0 double-chunking fix)
        # Conditions:
        # 1. Stratified data was created (has phi_flat attribute)
        # 2. Per-angle scaling is enabled
        # 3. Dataset is large enough to benefit (>1M points)
        #
        # FIXED (Nov 13, 2025): Use JIT-compatible StratifiedResidualFunctionJIT
        # Solution: Padded vmap implementation with static shapes
        # - Pads chunks to uniform size (enables JIT compilation)
        # - Uses jax.vmap for parallel chunk processing (no Python loops)
        # - Masks padded values in final residuals
        # Performance: ~1% memory overhead, 10-100x speedup from vectorization
        use_stratified_least_squares = (
            hasattr(stratified_data, "phi_flat")
            and per_angle_scaling
            and hasattr(stratified_data, "g2_flat")
            and len(stratified_data.g2_flat) >= 1_000_000
        )
        if use_stratified_least_squares:
            logger.info("=" * 80)
            logger.info("STRATIFIED LEAST-SQUARES PATH ACTIVATED (v2.2.1)")
            logger.info("Solving double-chunking problem with NLSQ's least_squares()")
            logger.info("=" * 80)

            # Validate initial parameters
            if initial_params is None:
                raise ValueError("initial_params must be provided")
            validated_params = self._validate_initial_params(initial_params, bounds)

            # Convert bounds
            nlsq_bounds = self._convert_bounds(bounds)

            # Validate bounds consistency
            if nlsq_bounds is not None:
                lower, upper = nlsq_bounds
                if np.any(lower > upper):
                    invalid_indices = np.where(lower > upper)[0]
                    raise ValueError(
                        f"Invalid bounds at indices {invalid_indices}: "
                        f"lower > upper. Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}"
                    )

            # Get physical parameter names for this analysis mode
            physical_param_names = self._get_physical_param_names(analysis_mode)
            logger.info(
                f"Physical parameters for {analysis_mode}: {physical_param_names}"
            )

            # FIX: Expand scaling parameters for per-angle scaling
            # When per_angle_scaling=True with N angles, we need:
            # - All physical parameters (7 for laminar_flow, 3 for static)
            # - N contrast parameters (one per angle)
            # - N offset parameters (one per angle)
            # Total: n_physical + 2*N parameters
            #
            # Config provides: n_physical + 2 parameters (single contrast, single offset)
            # We must expand: [contrast, offset] → [c0, c1, ..., cN-1, o0, o1, ..., oN-1]

            if per_angle_scaling:
                # Determine number of angles from stratified data
                n_angles = len(np.unique(stratified_data.phi_flat))
                n_physical = len(physical_param_names)

                logger.info("Expanding scaling parameters for per-angle scaling:")
                logger.info(f"  Angles: {n_angles}")
                logger.info(f"  Physical parameters: {n_physical}")
                logger.info(
                    f"  Input parameters: {len(validated_params)} (expected: {n_physical + 2})"
                )

                # Validate input parameter count
                expected_input = (
                    n_physical + 2
                )  # Physical params + single contrast + single offset
                if len(validated_params) != expected_input:
                    raise ValueError(
                        f"Parameter count mismatch for per-angle scaling: "
                        f"got {len(validated_params)}, expected {expected_input} "
                        f"({n_physical} physical + 2 scaling). "
                        f"For {n_angles} angles, will expand to {n_physical + 2 * n_angles} parameters."
                    )

                # Expand compact [contrast, offset, physical...] to per-angle format
                # matching StratifiedResidualFunction order:
                #   [contrast_per_angle, offset_per_angle, physical_params]
                from xpcsjax.optimization.nlsq.data_prep import (
                    expand_per_angle_parameters,
                )

                expanded = expand_per_angle_parameters(
                    validated_params,
                    nlsq_bounds,
                    n_angles,
                    n_physical,
                    logger=logger,
                )
                validated_params = expanded.params
                nlsq_bounds = expanded.bounds

            # Parameter count validation (CRITICAL)
            # Per-angle scaling is always enabled (legacy mode removed Nov 2025)
            n_physical = len(physical_param_names)
            n_angles = len(np.unique(stratified_data.phi_flat))
            expected_params = n_physical + 2 * n_angles

            if len(validated_params) != expected_params:
                raise ValueError(
                    f"Parameter count mismatch: got {len(validated_params)}, "
                    f"expected {expected_params} "
                    f"(physical={n_physical}, per_angle_scaling=True, "
                    f"n_angles={n_angles})"
                )

            logger.info(
                f"Parameter validation passed: {len(validated_params)} parameters"
            )

            # Step: Re-run unified strategy selection with EFFECTIVE parameter count
            # (v2.14.0, v2.22.0 fix: anti-degeneracy pre-check)
            #
            # The expanded param count (e.g. 53 for 23 angles individual) may be much
            # larger than the effective count after anti-degeneracy mode selection
            # (e.g. 9 for auto_averaged). Using the expanded count for memory estimation
            # can unnecessarily trigger out-of-core routing, which bypasses the
            # anti-degeneracy defense system entirely — causing parameter absorption
            # degeneracy and false convergence.
            #
            # Fix: Pre-check what anti-degeneracy would select, and use the effective
            # param count for memory routing. The actual anti-degeneracy transformation
            # still happens inside _fit_with_stratified_least_squares().
            n_total_points = len(stratified_data.g2_flat)
            actual_n_params = len(validated_params)
            effective_n_params = actual_n_params  # Default: no reduction

            if per_angle_scaling and config is not None and hasattr(config, "config"):
                nlsq_cfg = config.config.get("optimization", {}).get("nlsq", {})
                ad_cfg = nlsq_cfg.get("anti_degeneracy", {})
                ad_per_angle_mode = ad_cfg.get("per_angle_mode", "auto")
                ad_threshold = ad_cfg.get("constant_scaling_threshold", 3)
                n_angles_check = len(np.unique(stratified_data.phi_flat))

                if ad_per_angle_mode == "auto" and n_angles_check >= ad_threshold:
                    # auto_averaged: 2 averaged scaling params replace 2*n_angles
                    effective_n_params = n_physical + 2
                    logger.info(
                        f"Anti-Degeneracy pre-check: auto -> auto_averaged "
                        f"(n_phi={n_angles_check} >= threshold={ad_threshold}). "
                        f"Effective params: {effective_n_params} "
                        f"(expanded: {actual_n_params})"
                    )
                elif ad_per_angle_mode == "constant":
                    # constant: scaling fixed, only physical params optimized
                    effective_n_params = n_physical
                    logger.info(
                        f"Anti-Degeneracy pre-check: constant mode. "
                        f"Effective params: {effective_n_params} "
                        f"(expanded: {actual_n_params})"
                    )

            strategy_recheck = select_nlsq_strategy(n_total_points, effective_n_params)

            logger.info(
                f"Strategy re-check (with {effective_n_params} effective params, "
                f"{actual_n_params} expanded): "
                f"{strategy_recheck.strategy.value} ({strategy_recheck.reason})"
            )

            # Route to OUT_OF_CORE if peak memory exceeds threshold
            if strategy_recheck.strategy == NLSQStrategy.OUT_OF_CORE:
                # Safety check: warn if anti-degeneracy would have prevented this
                if effective_n_params < actual_n_params:
                    logger.warning(
                        f"Out-of-core triggered with {actual_n_params} expanded params, "
                        f"but anti-degeneracy would reduce to {effective_n_params}. "
                        f"This should not happen - the pre-check should have used "
                        f"effective params for memory estimation. Check routing logic."
                    )
                logger.info("=" * 80)
                logger.info("OUT-OF-CORE ACCUMULATION MODE (Re-check)")
                logger.info(
                    f"Peak memory ({strategy_recheck.peak_memory_gb:.1f} GB) exceeds "
                    f"threshold ({strategy_recheck.threshold_gb:.1f} GB)"
                )
                logger.info("Using chunk-wise J^T J accumulation for memory efficiency")
                logger.info("=" * 80)

                # Default to False (User requirement: Never subsample data)
                use_fast_mode = self.fast_mode or (
                    config.config.get("optimization", {}).get("fast_chi2_mode", False)
                    if config is not None and hasattr(config, "config")
                    else False
                )

                # Extract anti-degeneracy config (will warn that it's not supported for out-of-core)
                recheck_anti_degeneracy_config = None
                if config is not None and hasattr(config, "config"):
                    recheck_nlsq_config = config.config.get("optimization", {}).get(
                        "nlsq", {}
                    )
                    recheck_anti_degeneracy_config = recheck_nlsq_config.get(
                        "anti_degeneracy", {}
                    )

                popt, pcov, info = self._fit_with_out_of_core_accumulation(
                    stratified_data=stratified_data,
                    data=data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=logger,
                    config=config,
                    fast_chi2_mode=use_fast_mode,
                    anti_degeneracy_config=recheck_anti_degeneracy_config,
                )

                execution_time = time.time() - start_time
                uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))
                # Effective DOF for reduced chi-squared: in auto_averaged mode the
                # optimizer works on a compressed param vector but the true model
                # DOF is 2*n_phi + n_physical (one contrast+offset per angle).
                _ooc_n_params_effective: int | None = None
                if (
                    per_angle_scaling
                    and config is not None
                    and hasattr(config, "config")
                ):
                    _ooc_ad = (
                        config.config.get("optimization", {})
                        .get("nlsq", {})
                        .get("anti_degeneracy", {})
                    )
                    _ooc_mode = _ooc_ad.get("per_angle_mode", "auto")
                    _ooc_thresh = _ooc_ad.get("constant_scaling_threshold", 3)
                    if _ooc_mode == "auto" and n_angles_check >= _ooc_thresh:
                        _ooc_n_params_effective = 2 * n_angles_check + n_physical
                    elif _ooc_mode == "constant":
                        _ooc_n_params_effective = n_physical
                _ooc_dof = (
                    _ooc_n_params_effective
                    if _ooc_n_params_effective is not None
                    else len(popt)
                )
                reduced_chi2 = info.get("chi_squared", 0.0) / max(
                    1, n_total_points - _ooc_dof
                )
                # Derive quality_flag from reduced chi-squared (same thresholds
                # as _create_fit_result) instead of hardcoding "good".
                if reduced_chi2 < 1.5:
                    _ooc_recheck_quality = "good"
                elif reduced_chi2 < 3.0:
                    _ooc_recheck_quality = "marginal"
                else:
                    _ooc_recheck_quality = "poor"

                return OptimizationResult(
                    parameters=popt,
                    uncertainties=uncertainties,
                    covariance=pcov,
                    chi_squared=info.get("chi_squared", 0.0),
                    reduced_chi_squared=reduced_chi2,
                    convergence_status=info.get("convergence_status", "unknown"),
                    iterations=info.get("iterations", 0),
                    execution_time=execution_time,
                    device_info={
                        "device": "cpu_accumulated",
                        "strategy": "out_of_core",
                        "fast_mode": use_fast_mode,
                        "decision": strategy_recheck.reason,
                    },
                    recovery_actions=["out_of_core_recheck_delegation"],
                    quality_flag=_ooc_recheck_quality,
                )

            # Route to HYBRID_STREAMING if index array exceeds threshold (extreme scale)
            if strategy_recheck.strategy == NLSQStrategy.HYBRID_STREAMING:
                if not HYBRID_STREAMING_AVAILABLE:
                    logger.critical(
                        "AdaptiveHybridStreamingOptimizer required for extreme-scale "
                        f"dataset ({n_total_points:,} points) but not available."
                    )
                    raise MemoryError(
                        f"Dataset too large for RAM (index={strategy_recheck.index_memory_gb:.1f} GB > "
                        f"threshold={strategy_recheck.threshold_gb:.1f} GB) and Streaming unavailable."
                    )
                logger.warning(
                    f"Extreme-scale dataset: {strategy_recheck.reason}. "
                    "Proceeding with Adaptive Hybrid Streaming."
                )
                # Fall through to streaming path below (use_streaming_mode will be set)

            # Extract target chunk size from config
            target_chunk_size = 100_000  # Default
            hybrid_streaming_config = None
            use_streaming_mode = False
            use_hybrid_streaming = False

            # Compute adaptive memory threshold (v2.7.0+)
            # Default: 75% of total system memory instead of fixed 16 GB
            memory_fraction: float | None = None  # Will use default or env var
            memory_threshold_gb: float | None = None  # Will be computed adaptively

            if config is not None and hasattr(config, "config"):
                strat_config = config.config.get("optimization", {}).get(
                    "stratification", {}
                )
                target_chunk_size = strat_config.get("target_chunk_size", 100_000)

                # Extract streaming configuration
                nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                hybrid_streaming_config = nlsq_config.get("hybrid_streaming", {})

                # Support for explicit memory_threshold_gb (backwards compatible)
                # or memory_fraction (new adaptive approach)
                if "memory_threshold_gb" in nlsq_config:
                    memory_threshold_gb = nlsq_config["memory_threshold_gb"]
                if "memory_fraction" in nlsq_config:
                    memory_fraction = nlsq_config["memory_fraction"]

            # Compute adaptive threshold if not explicitly set
            if memory_threshold_gb is None:
                memory_threshold_gb, threshold_info = get_adaptive_memory_threshold(
                    memory_fraction=memory_fraction
                )
                logger.debug(
                    f"Using adaptive memory threshold: {memory_threshold_gb:.1f} GB "
                    f"(fraction={threshold_info['memory_fraction']}, "
                    f"total={threshold_info['total_memory_gb']:.1f} GB, "
                    f"source={threshold_info['source']})"
                )
            else:
                logger.debug(
                    f"Using explicit memory threshold from config: {memory_threshold_gb:.1f} GB"
                )

            # Check for hybrid streaming mode (preferred for large datasets)
            if hybrid_streaming_config is not None:
                use_hybrid_streaming = hybrid_streaming_config.get("enable", False)

            # Check for forced streaming mode from config
            # Also set from strategy_recheck if it returned HYBRID_STREAMING
            if config is not None and hasattr(config, "config"):
                nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                use_streaming_mode = nlsq_config.get("use_streaming", False)

            # Set streaming mode if strategy_recheck returned HYBRID_STREAMING (extreme scale)
            # This unified decision replaces the legacy _should_use_streaming() check
            if strategy_recheck.strategy == NLSQStrategy.HYBRID_STREAMING:
                logger.info("=" * 80)
                logger.info("HYBRID STREAMING MODE (Strategy Re-check)")
                logger.info(
                    f"Index array ({strategy_recheck.index_memory_gb:.1f} GB) exceeds "
                    f"threshold ({strategy_recheck.threshold_gb:.1f} GB)"
                )
                logger.info("=" * 80)
                use_streaming_mode = True

            # Log strategy decision for STANDARD (in-memory) path
            if not use_streaming_mode:
                logger.info(
                    f"Memory check: {strategy_recheck.reason}. "
                    "Proceeding with in-memory stratified least-squares."
                )

            # Use streaming optimizer if needed
            if use_streaming_mode:
                # Prefer AdaptiveHybridStreamingOptimizer when available
                # It fixes shear-term gradients, convergence, and covariance issues
                # Use hybrid if: (1) explicitly enabled, OR (2) basic streaming unavailable
                use_hybrid = HYBRID_STREAMING_AVAILABLE and (
                    use_hybrid_streaming or not STREAMING_AVAILABLE
                )

                if use_hybrid:
                    logger.info("=" * 80)
                    logger.info("ADAPTIVE HYBRID STREAMING MODE (Preferred)")
                    logger.info(
                        "Using NLSQ AdaptiveHybridStreamingOptimizer for better "
                        "convergence and parameter estimation"
                    )
                    logger.info("=" * 80)
                    # Extract anti-degeneracy config for defense system v2.9.0
                    anti_degeneracy_config = nlsq_config.get("anti_degeneracy", {})
                    try:
                        popt, pcov, info = self._fit_with_stratified_hybrid_streaming(
                            stratified_data=stratified_data,
                            per_angle_scaling=per_angle_scaling,
                            physical_param_names=physical_param_names,
                            initial_params=validated_params,
                            bounds=nlsq_bounds,
                            logger=logger,
                            hybrid_config=hybrid_streaming_config,
                            anti_degeneracy_config=anti_degeneracy_config,
                        )

                        # Compute final residuals for result creation
                        chunked_data = self._create_stratified_chunks(
                            stratified_data, target_chunk_size
                        )
                        residual_fn = create_stratified_residual_function(
                            stratified_data=chunked_data,
                            per_angle_scaling=per_angle_scaling,
                            physical_param_names=physical_param_names,
                            logger=cast(logging.Logger | None, logger),
                            validate=False,
                        )
                        final_residuals = residual_fn(popt)
                        n_data = len(final_residuals)

                        # Get execution time
                        execution_time = time.time() - start_time

                        # Compute effective DOF for reduced_chi_squared.
                        # In auto_averaged mode, popt has compressed length (e.g. 9),
                        # but the true model DOF is 2*n_phi + n_physical (e.g. 53).
                        _hs_n_params_effective: int | None = None
                        if per_angle_scaling and anti_degeneracy_config:
                            _hs_ad_mode = anti_degeneracy_config.get(
                                "per_angle_mode", "auto"
                            )
                            _hs_thresh = anti_degeneracy_config.get(
                                "constant_scaling_threshold", 3
                            )
                            if _hs_ad_mode == "auto" and n_angles_check >= _hs_thresh:
                                _hs_n_params_effective = 2 * n_angles_check + n_physical
                            elif _hs_ad_mode == "constant":
                                _hs_n_params_effective = n_physical

                        # Create result
                        result = self._create_fit_result(
                            popt=popt,
                            pcov=pcov,
                            residuals=final_residuals,
                            n_data=n_data,
                            iterations=info.get("nit", 0),
                            execution_time=execution_time,
                            convergence_status=(
                                "converged" if info.get("success", False) else "failed"
                            ),
                            recovery_actions=["hybrid_streaming_optimizer_method"],
                            streaming_diagnostics=info.get(
                                "hybrid_streaming_diagnostics"
                            ),
                            stratification_diagnostics=stratification_diagnostics,
                            diagnostics_payload=None,
                            n_params_effective=_hs_n_params_effective,
                        )

                        logger.info("=" * 80)
                        logger.info("HYBRID STREAMING OPTIMIZATION COMPLETE")
                        logger.info(
                            f"Final chi2: {result.chi_squared:.4e}, "
                            f"Reduced chi2: {result.reduced_chi_squared:.4f}"
                        )
                        logger.info("=" * 80)

                        return result

                    except (ValueError, RuntimeError, MemoryError, OSError) as e:
                        logger.warning(
                            f"Hybrid streaming optimization failed: {e}\n"
                            f"Falling back to stratified least-squares..."
                        )
                        # Fall through to stratified least-squares

                if not STREAMING_AVAILABLE:
                    # AdaptiveHybridStreamingOptimizer not available (NLSQ < 0.3.2)
                    logger.error(
                        "Streaming mode requested but AdaptiveHybridStreamingOptimizer "
                        "not available. Upgrade NLSQ to >= 0.3.2. "
                        "Falling back to stratified least-squares."
                    )
                    # Fall through to stratified least-squares

            # Extract NLSQ config dict for tolerance propagation and anti-degeneracy
            nlsq_config_dict = None
            anti_degeneracy_config = None
            if config is not None and hasattr(config, "config"):
                nlsq_config_dict = config.config.get("optimization", {}).get("nlsq", {})
                anti_degeneracy_config = nlsq_config_dict.get("anti_degeneracy", {})
                if anti_degeneracy_config:
                    logger.info(
                        f"Anti-Degeneracy config loaded: per_angle_mode="
                        f"{anti_degeneracy_config.get('per_angle_mode', 'auto')}"
                    )

            # Call stratified least_squares optimization
            try:
                popt, pcov, info = self._fit_with_stratified_least_squares(
                    stratified_data=stratified_data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=logger,
                    target_chunk_size=target_chunk_size,
                    anti_degeneracy_config=anti_degeneracy_config,
                    nlsq_config_dict=nlsq_config_dict,
                    analysis_mode=analysis_mode,
                )

                # Compute final residuals for result creation
                # We need to recreate the residual function to compute final residuals
                chunked_data = self._create_stratified_chunks(
                    stratified_data, target_chunk_size
                )
                residual_fn = create_stratified_residual_function(
                    stratified_data=chunked_data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    logger=cast(logging.Logger | None, logger),
                    validate=False,  # Already validated
                )
                final_residuals = residual_fn(popt)
                n_data = len(final_residuals)

                # Get execution time
                execution_time = time.time() - start_time

                # Compute effective DOF for reduced_chi_squared.
                # In auto_averaged mode, popt has compressed length (e.g. 9),
                # but the true model DOF is 2*n_phi + n_physical (e.g. 53).
                _sls_n_params_effective: int | None = None
                if per_angle_scaling and anti_degeneracy_config:
                    _sls_ad_mode = anti_degeneracy_config.get("per_angle_mode", "auto")
                    _sls_thresh = anti_degeneracy_config.get(
                        "constant_scaling_threshold", 3
                    )
                    if _sls_ad_mode == "auto" and n_angles_check >= _sls_thresh:
                        _sls_n_params_effective = 2 * n_angles_check + n_physical
                    elif _sls_ad_mode == "constant":
                        _sls_n_params_effective = n_physical

                # Create result
                result = self._create_fit_result(
                    popt=popt,
                    pcov=pcov,
                    residuals=final_residuals,
                    n_data=n_data,
                    iterations=info.get("nit", 0),
                    execution_time=execution_time,
                    convergence_status=(
                        "converged" if info.get("success", False) else "failed"
                    ),
                    recovery_actions=["stratified_least_squares_method"],
                    streaming_diagnostics=None,
                    stratification_diagnostics=stratification_diagnostics,
                    diagnostics_payload=None,
                    n_params_effective=_sls_n_params_effective,
                )

                logger.info("=" * 80)
                logger.info("STRATIFIED LEAST-SQUARES COMPLETE")
                logger.info(
                    f"Final chi2: {result.chi_squared:.4e}, Reduced chi2: {result.reduced_chi_squared:.4f}"
                )
                logger.info("=" * 80)

                return result

            except (ValueError, RuntimeError, MemoryError, OSError) as e:
                logger.error(
                    f"Stratified least_squares failed: {e}\n"
                    f"Falling back to standard curve_fit_large path..."
                )
                # Fall through to standard optimization path below

        # Step 2: Prepare data
        logger.info(f"Preparing data for {analysis_mode} optimization...")
        xdata, ydata = self._prepare_xy_data(stratified_data)
        n_data = len(ydata)
        logger.info(f"Data prepared: {n_data} points")

        # Note: Memory estimation is deferred to NLSQ's estimate_memory_requirements()
        # which provides accurate Jacobian sizing based on actual parameter count.
        if n_data > 10_000_000:
            logger.warning(
                f"Very large dataset: {n_data:,} points. "
                f"NLSQ will use memory-efficient strategies automatically."
            )
        elif n_data > 1_000_000:
            logger.info(
                f"Large dataset: {n_data:,} points. Memory managed automatically."
            )

        # Step 3: Validate initial parameters
        if initial_params is None:
            raise ValueError(
                "initial_params must be provided (auto-loading not yet implemented)",
            )

        validated_params = self._validate_initial_params(initial_params, bounds)

        # Step 4: Convert bounds
        nlsq_bounds = self._convert_bounds(bounds)

        # Step 5: Validate bounds consistency (FR-006)
        if nlsq_bounds is not None:
            lower, upper = nlsq_bounds
            if np.any(lower > upper):
                invalid_indices = np.where(lower > upper)[0]
                raise ValueError(
                    f"Invalid bounds at indices {invalid_indices}: "
                    f"lower > upper. Bounds must satisfy lower <= upper elementwise. "
                    f"Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}",
                )

        # Step 6: Create residual function with per-angle scaling
        logger.info(
            f"Creating residual function (per_angle_scaling={per_angle_scaling})..."
        )
        residual_fn = self._create_residual_function(
            stratified_data, analysis_mode, per_angle_scaling
        )
        base_residual_fn = residual_fn
        physical_param_names = self._get_physical_param_names(analysis_mode)
        phi_values = np.asarray(stratified_data.phi)
        n_phi_unique = len(np.unique(phi_values)) if phi_values.size else 0

        per_angle_contrast_override: np.ndarray | None = None
        per_angle_offset_override: np.ndarray | None = None
        if per_angle_scaling_initial:
            contrast_override = per_angle_scaling_initial.get("contrast")
            if contrast_override is not None:
                try:
                    arr = np.asarray(contrast_override, dtype=np.float64)
                    if arr.size == n_phi_unique:
                        per_angle_contrast_override = arr.copy()
                    else:
                        logger.warning(
                            "per_angle_scaling contrast override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_unique,
                        )
                except (TypeError, ValueError):
                    logger.warning("Invalid per-angle contrast override; ignoring")
            offset_override = per_angle_scaling_initial.get("offset")
            if offset_override is not None:
                try:
                    arr = np.asarray(offset_override, dtype=np.float64)
                    if arr.size == n_phi_unique:
                        per_angle_offset_override = arr.copy()
                    else:
                        logger.warning(
                            "per_angle_scaling offset override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_unique,
                        )
                except (TypeError, ValueError):
                    logger.warning("Invalid per-angle offset override; ignoring")

        # Step 6.5: Expand parameters for per-angle scaling if needed
        # This is CRITICAL: the residual function expects per-angle parameters,
        # but validated_params is still in compact form [contrast, offset, *physical]
        if per_angle_scaling:
            n_phi = n_phi_unique

            # Expand parameters from compact to per-angle form
            # Input:  [contrast, offset, *physical] (e.g., 5 params)
            # Output: [contrast_0, ..., contrast_{n-1}, offset_0, ..., offset_{n-1}, *physical]
            #         (e.g., 2*n_phi + 3 params for static_isotropic with n_phi angles)

            contrast_single = validated_params[0]
            offset_single = validated_params[1]
            physical_params = validated_params[2:]

            # Replicate contrast and offset for each angle
            # CRITICAL FIX (v2.7.1): For laminar_flow mode, use consistent initialization
            # to prevent per-angle params from absorbing the shear signal
            is_laminar_flow = "gamma_dot_t0" in physical_param_names
            use_consistent_init = (
                is_laminar_flow
                and per_angle_contrast_override is None
                and per_angle_offset_override is None
                and n_phi > 3  # Only for many angles where absorption is a problem
            )

            if use_consistent_init:
                logger.info(
                    "Computing consistent per-angle initialization for laminar_flow mode..."
                )
                try:
                    contrast_per_angle, offset_per_angle = (
                        _compute_consistent_per_angle_init(
                            stratified_data=stratified_data,
                            physical_params=physical_params,
                            physical_param_names=physical_param_names,
                            default_contrast=contrast_single,
                            default_offset=offset_single,
                            logger=logger,
                        )
                    )
                except (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    AttributeError,
                    np.linalg.LinAlgError,
                ) as e:
                    logger.warning(
                        f"Failed to compute consistent per-angle init: {e}\n"
                        "Falling back to uniform replication."
                    )
                    contrast_per_angle = np.full(n_phi, contrast_single)
                    offset_per_angle = np.full(n_phi, offset_single)
            else:
                if per_angle_contrast_override is not None:
                    contrast_per_angle = per_angle_contrast_override
                else:
                    contrast_per_angle = np.full(n_phi, contrast_single)
                if per_angle_offset_override is not None:
                    offset_per_angle = per_angle_offset_override
                else:
                    offset_per_angle = np.full(n_phi, offset_single)

            # Concatenate: [contrasts, offsets, physical]
            validated_params = np.concatenate(
                [contrast_per_angle, offset_per_angle, physical_params]
            )

            logger.info(
                f"Expanded parameters for per-angle scaling:\n"
                f"  {n_phi} phi angles detected\n"
                f"  Parameters: compact {2 + len(physical_params)} -> per-angle {len(validated_params)}\n"
                f"  Structure: [{n_phi} contrasts, {n_phi} offsets, {len(physical_params)} physical]"
            )

            # Also expand bounds if they exist
            if nlsq_bounds is not None:
                lower, upper = nlsq_bounds

                # Extract compact bounds
                contrast_lower, offset_lower = lower[0], lower[1]
                contrast_upper, offset_upper = upper[0], upper[1]
                physical_lower = lower[2:]
                physical_upper = upper[2:]

                # Expand to per-angle bounds
                contrast_lower_per_angle = np.full(n_phi, contrast_lower)
                contrast_upper_per_angle = np.full(n_phi, contrast_upper)
                offset_lower_per_angle = np.full(n_phi, offset_lower)
                offset_upper_per_angle = np.full(n_phi, offset_upper)

                # Concatenate expanded bounds
                expanded_lower = np.concatenate(
                    [contrast_lower_per_angle, offset_lower_per_angle, physical_lower]
                )
                expanded_upper = np.concatenate(
                    [contrast_upper_per_angle, offset_upper_per_angle, physical_upper]
                )

                nlsq_bounds = (expanded_lower, expanded_upper)

                logger.info(
                    f"Expanded bounds for per-angle scaling:\n"
                    f"  Bounds: compact {2 + len(physical_lower)} -> per-angle {len(expanded_lower)}"
                )

        n_angles_for_map = n_phi_unique if per_angle_scaling else 1
        physical_index_map = build_physical_index_map(
            per_angle_scaling,
            n_angles_for_map,
            physical_param_names,
        )
        validated_params, transform_state = apply_forward_shear_transforms_to_vector(
            validated_params,
            physical_index_map,
            transform_cfg,
        )
        if transform_state:
            nlsq_bounds = apply_forward_shear_transforms_to_bounds(
                nlsq_bounds,
                transform_state,
            )

        solver_residual_fn = base_residual_fn
        if transform_state:
            if isinstance(base_residual_fn, StratifiedResidualFunction):
                solver_residual_fn = wrap_stratified_function_with_transforms(
                    base_residual_fn,
                    transform_state,
                )
            else:
                solver_residual_fn = wrap_model_function_with_transforms(
                    base_residual_fn,
                    transform_state,
                )

        param_labels = _build_parameter_labels(
            per_angle_scaling,
            n_phi_unique if per_angle_scaling else 0,
            physical_param_names,
        )

        per_param_x_scale = build_per_parameter_x_scale(
            per_angle_scaling,
            n_phi_unique if per_angle_scaling else 0,
            physical_param_names,
            analysis_mode,
            x_scale_map_config,
        )
        if per_param_x_scale is not None:
            x_scale_value = per_param_x_scale

        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {
                "solver_settings": {"loss": loss_name}
            }
            solver_settings = diagnostics_payload.setdefault(
                "solver_settings", {"loss": loss_name}
            )
            solver_settings["x_scale"] = (
                x_scale_value.tolist()
                if isinstance(x_scale_value, np.ndarray)
                else x_scale_value
            )
            logger.info(
                "Diagnostics enabled: loss=%s, x_scale=%s, sample_size=%d",
                loss_name,
                format_x_scale_for_log(x_scale_value),
                diagnostics_sample_size,
            )

        diagnostics_sample_x: np.ndarray | None = None
        sample_scaling = 1.0
        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {}
            diagnostics_sample_x = _sample_xdata(xdata, diagnostics_sample_size)
            if diagnostics_sample_x.size == 0:
                diagnostics_sample_x = xdata
            sample_scaling = max(1.0, xdata.size / max(diagnostics_sample_x.size, 1))
            initial_jtj, initial_norms = _compute_jacobian_stats(
                solver_residual_fn,
                diagnostics_sample_x,
                validated_params,
                sample_scaling,
            )
            if initial_norms is not None:
                diagnostics_payload.setdefault("initial_jacobian_norms", {})
                diagnostics_payload["initial_jacobian_norms"] = dict(
                    zip(param_labels, initial_norms.tolist(), strict=False),
                )
                logger.info(
                    "Initial Jacobian column norms: %s",
                    ", ".join(
                        f"{label}={norm:.3e}"
                        for label, norm in diagnostics_payload[
                            "initial_jacobian_norms"
                        ].items()
                    ),
                )

        residual_counter: FunctionEvaluationCounter | None = None
        wrapped_residual_fn: StratifiedResidualFunction | FunctionEvaluationCounter
        if diagnostics_enabled:
            residual_counter = FunctionEvaluationCounter(solver_residual_fn)
            wrapped_residual_fn = residual_counter
        else:
            wrapped_residual_fn = solver_residual_fn

        # Step 7: Select optimization strategy using memory-based selection (v2.13.0)
        # Uses unified select_nlsq_strategy() instead of deprecated DatasetSizeStrategy
        n_parameters = len(validated_params)

        # Map NLSQStrategy to local OptimizationStrategy for fallback chain
        from xpcsjax.optimization.nlsq.strategies.chunking import (
            estimate_nlsq_optimization_memory,
        )

        memory_stats = estimate_nlsq_optimization_memory(n_data, n_parameters)
        logger.info(
            f"Memory estimate: {memory_stats['peak_gb']:.2f} GB peak, "
            f"{memory_stats.get('available_gb', 0):.2f} GB available"
        )

        if not memory_stats.get("is_safe", True):
            logger.warning(
                f"Memory usage may be high ({memory_stats['peak_gb']:.2f} GB). "
                f"Using memory-efficient strategy."
            )

        # Check for strategy override in config
        strategy_override = None
        if config is not None and hasattr(config, "config"):
            perf_config = config.config.get("performance", {})
            strategy_override = perf_config.get("strategy_override")

        # Select strategy: use override if provided, else use memory-based selection
        if strategy_override:
            try:
                strategy = OptimizationStrategy(strategy_override)
                logger.info(f"Using overridden strategy: {strategy.value}")
            except ValueError:
                logger.warning(
                    f"Invalid strategy override '{strategy_override}', using auto"
                )
                strategy_override = None

        if not strategy_override:
            # Map memory-based decision to OptimizationStrategy for fallback chain
            decision = select_nlsq_strategy(n_data, n_parameters)
            if decision.strategy == NLSQStrategy.HYBRID_STREAMING:
                strategy = OptimizationStrategy.STREAMING
            elif decision.strategy == NLSQStrategy.OUT_OF_CORE:
                strategy = OptimizationStrategy.CHUNKED
            else:
                # STANDARD: use size-based selection for STANDARD/LARGE distinction
                if n_data < 1_000_000:
                    strategy = OptimizationStrategy.STANDARD
                elif n_data < 10_000_000:
                    strategy = OptimizationStrategy.LARGE
                else:
                    strategy = OptimizationStrategy.CHUNKED

        logger.info(
            f"Selected {strategy.value} strategy for {n_data:,} points "
            f"(peak memory: {memory_stats['peak_gb']:.2f} GB)"
        )

        # Step 8: Execute optimization with strategy fallback
        popt, pcov, info, recovery_actions, convergence_status = (
            self._execute_optimization_with_fallback(
                strategy=strategy,
                wrapped_residual_fn=wrapped_residual_fn,
                xdata=xdata,
                ydata=ydata,
                validated_params=validated_params,
                nlsq_bounds=nlsq_bounds,
                loss_name=loss_name,
                x_scale_value=x_scale_value,
                config=config,
                start_time=start_time,
                logger=logger,
            )
        )

        # Compute effective DOF for the diagnostics covariance scaling (s²).
        # In auto_averaged mode the optimizer works on a compressed 9-param vector
        # (contrast_avg, offset_avg, physical×7), but the physics model consumes
        # 2*n_phi + n_physical effective degrees of freedom (one contrast+offset per
        # angle, constrained to an averaged value).  Using the compressed count (9)
        # would underestimate s² and produce artificially tight diagnostic pcov.
        n_physical = len(physical_param_names)
        n_dof_effective: int | None = None
        if per_angle_scaling and config is not None and hasattr(config, "config"):
            _nlsq_cfg = config.config.get("optimization", {}).get("nlsq", {})
            _ad_cfg = _nlsq_cfg.get("anti_degeneracy", {})
            _ad_mode = _ad_cfg.get("per_angle_mode", "auto")
            _ad_threshold = _ad_cfg.get("constant_scaling_threshold", 3)
            if _ad_mode == "auto" and n_phi_unique >= _ad_threshold:
                # auto_averaged: expanded DOF = 2*n_phi + n_physical (e.g. 53)
                n_dof_effective = 2 * n_phi_unique + n_physical
            elif _ad_mode == "constant":
                # constant: only physical params are optimised; scaling is fixed
                n_dof_effective = n_physical

        return self._post_process_results(
            popt=popt,
            pcov=pcov,
            info=info,
            transform_state=transform_state,
            validated_params=validated_params,
            residual_counter=residual_counter,
            base_residual_fn=base_residual_fn,
            xdata=xdata,
            n_data=n_data,
            start_time=start_time,
            nlsq_bounds=nlsq_bounds,
            convergence_status=convergence_status,
            recovery_actions=recovery_actions,
            stratification_diagnostics=stratification_diagnostics,
            diagnostics_state={
                "enabled": diagnostics_enabled,
                "payload": diagnostics_payload,
                "sample_x": diagnostics_sample_x,
                "solver_residual_fn": solver_residual_fn,
                "sample_scaling": sample_scaling,
                "param_labels": param_labels,
            },
            logger=logger,
            n_dof_effective=n_dof_effective,
        )

    def _execute_optimization_with_fallback(
        self,
        strategy: OptimizationStrategy,
        wrapped_residual_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        validated_params: np.ndarray,
        nlsq_bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | str,
        config: Any,
        start_time: float,
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[str], str]:
        """Execute optimization with strategy fallback.

        Delegates to fallback_chain.execute_optimization_with_fallback().
        """
        return execute_optimization_with_fallback(
            strategy=strategy,
            wrapped_residual_fn=wrapped_residual_fn,
            xdata=xdata,
            ydata=ydata,
            validated_params=validated_params,
            nlsq_bounds=nlsq_bounds,
            loss_name=loss_name,
            x_scale_value=x_scale_value,
            config=config,
            start_time=start_time,
            log=logger,
            enable_recovery=self.enable_recovery,
            execute_with_recovery_fn=self._execute_with_recovery,
            fit_with_hybrid_streaming_fn=self._fit_with_hybrid_streaming_optimizer,
            streaming_available=STREAMING_AVAILABLE,
            curve_fit_fn=curve_fit,
            curve_fit_large_fn=curve_fit_large,
            fast_mode=self.fast_mode,
        )

    def _post_process_results(
        self,
        popt: np.ndarray,
        pcov: np.ndarray | None,
        info: dict[str, Any],
        transform_state: Any,
        validated_params: np.ndarray,
        residual_counter: Any,
        base_residual_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        n_data: int,
        start_time: float,
        nlsq_bounds: tuple[np.ndarray, np.ndarray] | None,
        convergence_status: str,
        recovery_actions: list[str],
        stratification_diagnostics: Any,
        diagnostics_state: dict[str, Any],
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
        n_dof_effective: int | None = None,
    ) -> OptimizationResult:
        """Post-process optimization outputs into final result.

        Applies inverse transforms, computes final residuals and costs,
        runs optional diagnostics, determines success, and creates result.
        """
        import time

        # Unpack diagnostics state
        diagnostics_enabled = diagnostics_state.get("enabled", False)
        diagnostics_payload = diagnostics_state.get("payload")
        diagnostics_sample_x = diagnostics_state.get("sample_x")
        solver_residual_fn = diagnostics_state.get("solver_residual_fn")
        sample_scaling = diagnostics_state.get("sample_scaling")
        param_labels = diagnostics_state.get("param_labels")

        # Apply inverse shear transforms
        solver_params = np.asarray(popt, dtype=float)
        if transform_state:
            physical_params = apply_inverse_shear_transforms_to_vector(
                solver_params,
                transform_state,
            )
            popt = physical_params
            if pcov is not None:
                pcov = adjust_covariance_for_transforms(
                    np.asarray(pcov, dtype=float),
                    solver_params,
                    physical_params,
                    transform_state,
                )
        else:
            popt = np.asarray(popt, dtype=float)

        # Count function evaluations
        reported_nfev: int = cast(int, info.get("nfev", -1))
        corrected_nfev = (
            residual_counter.count if residual_counter is not None else reported_nfev
        )
        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {}
            diagnostics_payload["nfev_reported"] = reported_nfev
            diagnostics_payload["nfev_actual"] = corrected_nfev
            logger.info(
                "Diagnostics: nfev reported=%s actual=%s",
                reported_nfev,
                corrected_nfev,
            )

        # Compute final residuals using the base function (avoid counter side-effects).
        # StratifiedResidualFunction/JIT takes (params), not (xdata, *params).
        if isinstance(
            base_residual_fn,
            (StratifiedResidualFunction, StratifiedResidualFunctionJIT),
        ):
            final_residuals = base_residual_fn(popt)
        else:
            final_residuals = base_residual_fn(xdata, *popt)

        reported_iterations = -1
        if isinstance(info, dict):
            reported_iterations = info.get("nit", info.get("nfev", -1))
        iterations = max(0, corrected_nfev)

        if reported_iterations == -1:
            logger.debug(
                "Iteration count not available from NLSQ (curve_fit_large does not return this info)"
            )

        execution_time = time.time() - start_time

        # Optional diagnostics: Jacobian stats, parameter status, covariance refinement
        if diagnostics_enabled and diagnostics_sample_x is not None:
            assert diagnostics_payload is not None
            final_jtj, final_norms = _compute_jacobian_stats(
                solver_residual_fn,
                diagnostics_sample_x,
                solver_params,
                sample_scaling,
            )
            if final_norms is not None:
                diagnostics_payload["final_jacobian_norms"] = dict(
                    zip(param_labels, final_norms.tolist(), strict=False),
                )
                logger.info(
                    "Final Jacobian column norms: %s",
                    ", ".join(
                        f"{label}={norm:.3e}"
                        for label, norm in diagnostics_payload[
                            "final_jacobian_norms"
                        ].items()
                    ),
                )
            if nlsq_bounds is not None:
                statuses = _classify_parameter_status(
                    popt,
                    nlsq_bounds[0],
                    nlsq_bounds[1],
                )
                diagnostics_payload["parameter_status"] = dict(
                    zip(param_labels, statuses, strict=False),
                )
                clips = [
                    label
                    for label, st in diagnostics_payload["parameter_status"].items()
                    if st != "active"
                ]
                if clips:
                    logger.warning(
                        "Diagnostics: parameters at bounds -> %s",
                        ", ".join(clips),
                    )
            if final_jtj is not None:
                n_diag_data = len(final_residuals)
                # Use n_dof_effective when provided (e.g. auto_averaged mode where
                # the compressed optimizer vector has fewer entries than the true
                # model DOF: 2*n_phi + n_physical >> len(popt)).  Falling back to
                # len(popt) would underestimate s² and produce artificially tight
                # diagnostic covariances.
                n_diag_params = (
                    n_dof_effective if n_dof_effective is not None else len(popt)
                )
                s2_diag = float(np.sum(final_residuals**2)) / max(
                    n_diag_data - n_diag_params, 1
                )
                pcov = s2_diag * np.linalg.pinv(final_jtj, rcond=1e-10)
                diagnostics_payload["jtj_condition"] = (
                    float(np.linalg.cond(final_jtj)) if final_jtj.size > 0 else None
                )

        # Determine optimization success
        initial_cost = info.get("initial_cost", 0) if isinstance(info, dict) else 0
        final_cost = np.sum(final_residuals**2)

        function_evals = iterations
        cost_reduction = (
            (initial_cost - final_cost) / initial_cost if initial_cost > 0 else 0
        )
        params_changed = not np.allclose(popt, validated_params, rtol=1e-8)

        optimization_ran = function_evals > 10 or params_changed
        optimization_improved = cost_reduction > 0.05

        if optimization_ran and optimization_improved:
            status_indicator = "SUCCESS"
            status_msg = "Optimization succeeded"
        elif optimization_ran and not optimization_improved:
            status_indicator = "MARGINAL"
            status_msg = "Optimization ran but minimal improvement"
        else:
            status_indicator = "FAILED"
            status_msg = "Optimization failed (0 iterations, no cost reduction)"

        logger.info(
            f"{status_indicator}: {status_msg} in {execution_time:.2f}s\n"
            f"  Function evaluations: {function_evals}\n"
            f"  Cost: {initial_cost:.4e} -> {final_cost:.4e} ({cost_reduction * 100:+.1f}%)\n"
            f"  Iterations reported: {reported_iterations} (NLSQ may report 0)"
        )
        if recovery_actions:
            logger.info(f"Recovery actions applied: {len(recovery_actions)}")

        # Extract streaming diagnostics
        streaming_diagnostics = None
        if "batch_statistics" in info:
            streaming_diagnostics = info["batch_statistics"]
        elif "streaming_diagnostics" in info:
            streaming_diagnostics = info["streaming_diagnostics"]

        # Create result
        result = self._create_fit_result(
            popt=popt,
            pcov=pcov,
            residuals=final_residuals,
            n_data=n_data,
            iterations=iterations,
            execution_time=execution_time,
            convergence_status=convergence_status,
            recovery_actions=recovery_actions,
            streaming_diagnostics=streaming_diagnostics,
            stratification_diagnostics=stratification_diagnostics,
            diagnostics_payload=diagnostics_payload if diagnostics_enabled else None,
            n_params_effective=n_dof_effective,
        )

        logger.info(
            f"Final chi-squared: {result.chi_squared:.4e}, "
            f"reduced chi-squared: {result.reduced_chi_squared:.4f}",
        )

        return result

    def _execute_with_recovery(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        strategy: OptimizationStrategy,
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
        loss_name: str,
        x_scale_value: float | str,
    ) -> tuple[np.ndarray, np.ndarray, dict, list[str], str]:
        """Execute optimization with automatic error recovery (T022-T024)."""
        return execute_with_recovery(
            residual_fn=residual_fn,
            xdata=xdata,
            ydata=ydata,
            initial_params=initial_params,
            bounds=bounds,
            strategy=strategy,
            log=logger,
            loss_name=loss_name,
            x_scale_value=x_scale_value,
            handle_nlsq_result_fn=self._handle_nlsq_result,
            curve_fit_fn=curve_fit,
            curve_fit_large_fn=curve_fit_large,
        )

    def _diagnose_error(
        self,
        error: Exception,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        attempt: int,
    ) -> dict[str, Any]:
        """Diagnose optimization error and provide actionable recovery strategy (T023)."""
        return diagnose_error(
            error=error,
            params=params,
            bounds=bounds,
            attempt=attempt,
        )

    def _prepare_xy_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Transform multi-dimensional XPCS data to flattened 1D arrays.

        Named distinctly from the base ``NLSQAdapterBase._prepare_data``
        (``(t1, t2, phi, g2, weights) -> dict``): this wrapper variant takes a
        single data object and returns flattened ``(xdata, ydata)``. The two
        are different operations, so this is intentionally not an override.

        Args:
            data: XPCSData with shape (n_phi, n_t1, n_t2) OR StratifiedData (already flattened)

        Returns:
            (xdata, ydata): Flattened independent variables and observations
        """
        # Validate data has required attributes
        if (
            not hasattr(data, "phi")
            or not hasattr(data, "t1")
            or not hasattr(data, "t2")
            or not hasattr(data, "g2")
        ):
            raise ValueError("Data must have 'phi', 't1', 't2', and 'g2' attributes")

        # Check if this is already stratified data (has phi_flat attribute)
        if hasattr(data, "phi_flat"):
            # Stratified data is already flattened - use directly
            g2_flat = np.asarray(data.g2_flat, dtype=np.float64)
            # Use int64 to avoid int32 overflow for large datasets
            # (n_phi * n_t1 * n_t2 can exceed 2.147B for 100+ angles × 5000+ time points).
            xdata = np.arange(len(g2_flat), dtype=np.int64)
            ydata = g2_flat
            return xdata, ydata

        # Original data path: needs meshgrid and flattening
        # Get dimensions
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # CRITICAL FIX (Nov 14, 2025): Extract 1D arrays from 2D meshgrids if needed
        # Same issue as in _apply_stratification_if_needed - cache loader returns 2D
        # but meshgrid expects 1D inputs
        if t1.ndim == 2:
            t1 = t1[:, 0] if t1.size > 0 else np.array([])
        if t2.ndim == 2:
            t2 = t2[0, :] if t2.size > 0 else np.array([])

        # Validate non-empty arrays
        if phi.size == 0 or t1.size == 0 or t2.size == 0:
            raise ValueError("Data arrays cannot be empty")

        # Create meshgrid with indexing='ij' to preserve correct ordering
        # This ensures phi varies slowest, t2 varies fastest
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")

        # Flatten all arrays to 1D
        # For NLSQ curve_fit interface, xdata is typically just indices
        # Use int64 to avoid int32 overflow for large datasets
        # (n_phi * n_t1 * n_t2 can exceed 2.147B for 100+ angles × 5000+ time points).
        xdata = np.arange(g2.size, dtype=np.int64)

        # Flatten observations
        ydata = g2.flatten().astype(np.float64, copy=False)

        return xdata, ydata

    def _apply_stratification_if_needed(
        self,
        data: Any,
        per_angle_scaling: bool,
        config: Any,
        logger: Any,
    ) -> Any:
        """Apply angle-stratified chunking if conditions require it.

        This method fixes the per-angle scaling + NLSQ chunking incompatibility
        identified in ultra-think-20251106-012247. When per-angle parameters are
        used (contrast[i], offset[i] for each phi angle), NLSQ's arbitrary chunking
        can create chunks without certain angles, resulting in zero gradients and
        silent optimization failures.

        Solution: Reorganize data so every chunk contains all phi angles, ensuring
        gradients are always well-defined.

        Args:
            data: XPCSData object with phi, t1, t2, g2 attributes
            per_angle_scaling: Whether per-angle parameters are enabled
            config: Configuration manager with stratification settings
            logger: Logger instance for diagnostics

        Returns:
            Data object (original or stratified copy) ready for optimization

        Notes:
            - No-op if conditions don't require stratification
            - Creates temporary 2x memory overhead during reorganization
            - <1% performance overhead (0.15s for 3M points)
            - Respects configuration overrides in optimization.stratification
        """
        # Extract stratification configuration with defaults
        strat_config = {}
        if config is not None and hasattr(config, "config"):
            opt_config = config.config.get("optimization", {})
            strat_config = opt_config.get("stratification", {})

        # Configuration defaults (matching YAML template)
        enabled = strat_config.get("enabled", "auto")  # "auto", true, false
        target_chunk_size = strat_config.get("target_chunk_size", 100_000)
        max_imbalance_ratio = strat_config.get("max_imbalance_ratio", 5.0)
        force_sequential = strat_config.get("force_sequential_fallback", False)
        check_memory = strat_config.get("check_memory_safety", True)
        use_index_based = strat_config.get("use_index_based", False)
        collect_diagnostics = strat_config.get("collect_diagnostics", False)
        log_diagnostics = strat_config.get("log_diagnostics", False)

        # Check if explicitly disabled
        if enabled is False or (
            isinstance(enabled, str) and enabled.lower() == "false"
        ):
            logger.info("Stratification disabled via configuration")
            return data

        # Check if we should fallback to sequential
        if force_sequential:
            logger.info("Sequential per-angle fallback forced via configuration")
            return UseSequentialOptimization(
                data=data, reason="force_sequential_fallback=true in configuration"
            )

        # Get data dimensions
        # Note: Per-angle scaling is always enabled (legacy mode removed Nov 2025)
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # CRITICAL FIX (Nov 14, 2025): Extract 1D arrays from 2D meshgrids if needed
        # ROOT CAUSE: After commit e5ac926, cache loader returns 2D meshgrids (600, 600)
        # but np.meshgrid() at line 2428 expects 1D input arrays (600,)
        # Calling meshgrid on already-meshgridded data produces wrong structure!
        # This was breaking alpha parameter gradient computation.
        if t1.ndim == 2:
            # t1_2d[i, j] = time[i] (constant along j), extract first column
            if t1.size > 0:
                t1 = t1[:, 0]
                logger.debug(
                    f"Extracted 1D t1 array from 2D meshgrid: shape {t1.shape}"
                )
            else:
                t1 = np.array([])
                logger.debug("Empty 2D t1 array converted to empty 1D array")
        if t2.ndim == 2:
            # t2_2d[i, j] = time[j] (constant along i), extract first row
            if t2.size > 0:
                t2 = t2[0, :]
                logger.debug(
                    f"Extracted 1D t2 array from 2D meshgrid: shape {t2.shape}"
                )
            else:
                t2 = np.array([])
                logger.debug("Empty 2D t2 array converted to empty 1D array")

        # Calculate total points (meshgrid creates n_phi × n_t1 × n_t2 points)
        n_points = len(phi) * len(t1) * len(t2)

        # Analyze angle distribution
        stats = analyze_angle_distribution(phi)

        # Decision logic (use configured max_imbalance_ratio)
        # Override the imbalance check with configured value
        should_stratify_auto, reason = should_use_stratification(
            n_points=n_points,
            n_angles=stats.n_angles,
            per_angle_scaling=per_angle_scaling,
            imbalance_ratio=stats.imbalance_ratio,
        )

        # Override with configuration if imbalance exceeds configured threshold
        if stats.imbalance_ratio > max_imbalance_ratio:
            # Extreme imbalance - use sequential optimization
            logger.info(
                f"Extreme angle imbalance detected ({stats.imbalance_ratio:.1f} > {max_imbalance_ratio:.1f})"
            )
            return UseSequentialOptimization(
                data=data,
                reason=f"Angle imbalance ratio ({stats.imbalance_ratio:.1f}) exceeds threshold ({max_imbalance_ratio:.1f})",
            )

        # Handle "auto" mode
        if enabled == "auto" or (
            isinstance(enabled, str) and enabled.lower() == "auto"
        ):
            should_stratify = should_stratify_auto
        else:
            # enabled is True (force on)
            should_stratify = True
            reason = "Stratification forced via configuration (enabled=true)"

        if not should_stratify:
            logger.info(f"Stratification skipped: {reason}")
            return data

        # Apply stratification
        logger.info(
            f"Applying angle-stratified chunking: {reason}\n"
            f"  Angles: {stats.n_angles}, Imbalance ratio: {stats.imbalance_ratio:.2f}\n"
            f"  Total points: {n_points:,}\n"
            f"  Target chunk size: {target_chunk_size:,}\n"
            f"  Use index-based: {use_index_based}"
        )

        # Check memory safety (if enabled in config)
        if check_memory:
            mem_stats = estimate_stratification_memory(
                n_points, use_index_based=use_index_based
            )
            if not mem_stats["is_safe"]:
                logger.warning(
                    f"Stratification may use significant memory: "
                    f"{mem_stats['peak_memory_mb']:.1f} MB peak. "
                    f"Consider: (1) setting use_index_based=true, or "
                    f"(2) setting force_sequential_fallback=true"
                )

        # Reorganize data arrays
        # Need to expand to full meshgrid first, then stratify
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")
        phi_flat = phi_grid.flatten()
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()
        g2_flat = g2.flatten()

        # Measure stratification execution time
        import time

        stratification_start = time.perf_counter()

        # Apply stratification based on mode
        if use_index_based:
            # Index-based stratification (zero-copy, ~1% memory overhead)
            logger.info("Using index-based stratification (zero-copy)")
            indices, chunk_sizes = create_angle_stratified_indices(
                phi_flat, target_chunk_size=target_chunk_size
            )

            # Apply indices to get stratified data
            phi_stratified = phi_flat[indices]
            t1_stratified = t1_flat[indices]
            t2_stratified = t2_flat[indices]
            g2_stratified = g2_flat[indices]

            # CRITICAL FIX (Jan 2026): For index-based stratification,
            # chunk_sizes are now explicitly returned.
        else:
            # Full copy stratification (2x memory overhead)
            logger.info("Using full-copy stratification")
            # Convert to JAX arrays for stratification
            phi_jax = jnp.array(phi_flat)
            t1_jax = jnp.array(t1_flat)
            t2_jax = jnp.array(t2_flat)
            g2_jax = jnp.array(g2_flat)

            # Apply stratification (use configured target_chunk_size)
            # CRITICAL FIX (Nov 10, 2025): Now returns chunk_sizes as 5th value
            # to preserve stratification boundaries during re-chunking
            (
                phi_stratified,
                t1_stratified,
                t2_stratified,
                g2_stratified,
                chunk_sizes,
            ) = create_angle_stratified_data(
                phi_jax, t1_jax, t2_jax, g2_jax, target_chunk_size=target_chunk_size
            )

            # Convert back to numpy
            phi_stratified = np.array(phi_stratified)
            t1_stratified = np.array(t1_stratified)
            t2_stratified = np.array(t2_stratified)
            g2_stratified = np.array(g2_stratified)

        # Measure execution time
        stratification_time_ms = (time.perf_counter() - stratification_start) * 1000.0

        # Compute diagnostics if requested
        diagnostics = None
        if collect_diagnostics:
            diagnostics = compute_stratification_diagnostics(
                phi_original=phi_flat,
                phi_stratified=phi_stratified,
                execution_time_ms=stratification_time_ms,
                use_index_based=use_index_based,
                target_chunk_size=target_chunk_size,
                chunk_sizes=chunk_sizes,  # Pass actual chunk boundaries
            )

            # Optionally log diagnostic report
            if log_diagnostics and diagnostics is not None:
                report = format_diagnostics_report(diagnostics)
                logger.info(f"\n{report}")

        # Create stratified data object (modify in-place or create copy)
        # We need to "unflatten" back to original shape for _prepare_xy_data to work
        # Actually, we can't easily unflatten to 3D grid, so instead we'll create
        # a modified data object that stores the flattened stratified arrays

        # Create a simple namespace object to hold stratified data
        class StratifiedData:
            def __init__(
                self,
                phi: np.ndarray,
                t1: np.ndarray,
                t2: np.ndarray,
                g2: np.ndarray,
                original_data: Any,
                diagnostics: Any = None,
                chunk_sizes: Any = None,
            ) -> None:
                # Store flattened stratified arrays
                self.phi_flat = phi
                self.t1_flat = t1
                self.t2_flat = t2
                self.g2_flat = g2

                # Also store unique values for backwards compatibility
                self.phi = np.unique(phi)
                self.t1 = np.unique(t1)
                self.t2 = np.unique(t2)

                # Store as 1D array (already flattened and stratified)
                self.g2 = g2

                # Copy critical metadata attributes from original data
                # These are required for residual function computation
                self.sigma = original_data.sigma  # Uncertainty/error bars (CRITICAL)
                self.q = original_data.q  # Wavevector magnitude (CRITICAL)
                self.L = original_data.L  # Sample-detector distance (CRITICAL)

                # Copy optional dt if present (time step)
                if hasattr(original_data, "dt"):
                    self.dt = original_data.dt

                # Store diagnostics if available
                self.stratification_diagnostics = diagnostics

                # CRITICAL FIX (Nov 10, 2025): Store original chunk sizes
                # to preserve stratification boundaries during re-chunking
                self.chunk_sizes = chunk_sizes

        # CRITICAL FIX (Dec 2025): Pre-shuffle stratified data before returning
        # This prevents the hybrid streaming optimizer from seeing angle-sequential data
        # during L-BFGS warmup, which would cause local minimum traps (gamma_dot_t0 -> 0)
        # The shuffle must happen HERE, not in _fit_with_stratified_hybrid_streaming,
        # because other code paths may also use the stratified data.
        # Fixed seed for reproducible stratified shuffling.
        # Not user-configurable — this ensures deterministic data ordering
        # for consistent NLSQ convergence across runs.
        shuffle_seed = 42
        rng = np.random.RandomState(shuffle_seed)  # noqa: NPY002 — keep for reproducibility
        perm = rng.permutation(len(phi_stratified))
        phi_stratified = phi_stratified[perm]
        t1_stratified = t1_stratified[perm]
        t2_stratified = t2_stratified[perm]
        g2_stratified = g2_stratified[perm]
        logger.info(
            f"Pre-shuffled stratified data (seed={shuffle_seed}) to prevent local minimum traps"
        )

        stratified_data = StratifiedData(
            phi_stratified,
            t1_stratified,
            t2_stratified,
            g2_stratified,
            data,  # Pass original data to copy metadata attributes
            diagnostics,
            chunk_sizes,  # CRITICAL FIX: Pass chunk sizes for boundary-aware re-chunking
        )

        logger.info(
            f"Stratification complete: {len(g2_stratified):,} points reorganized"
        )

        return stratified_data

    def _run_sequential_optimization(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        analysis_mode: AnalysisMode,
        per_angle_scaling: bool,
        logger: Any,
        start_time: float,
        x_scale_value: Any,
        transform_cfg: dict[str, Any],
        physical_param_names: list[str],
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
    ) -> OptimizationResult:
        """Run sequential per-angle optimization as a fallback strategy.

        This method optimizes each phi angle independently and combines results
        using inverse variance weighting. It's used when:
        - Angle imbalance ratio exceeds threshold (>5.0 by default)
        - force_sequential_fallback=true in configuration
        - Stratification cannot be applied

        Args:
            data: Original XPCS data object
            config: Configuration manager
            initial_params: Initial parameter guess
            bounds: Parameter bounds (lower, upper)
            analysis_mode: 'static_isotropic' or 'laminar_flow'
            per_angle_scaling: Whether per-angle parameters enabled
            logger: Logger instance
            start_time: Start time for execution timing

        Returns:
            OptimizationResult with combined parameters from all angles

        Raises:
            RuntimeError: If too few angles converge (<50% by default)
        """
        import time

        logger.info("=" * 80)
        logger.info("SEQUENTIAL PER-ANGLE OPTIMIZATION")
        logger.info("=" * 80)

        # Prepare data arrays
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # Create full meshgrid
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")
        phi_flat = phi_grid.flatten()
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()
        g2_flat = g2.flatten()

        from xpcsjax.config.parameter_manager import ParameterManager

        param_manager = ParameterManager(config.config, analysis_mode)
        base_param_names = param_manager.get_all_parameter_names()
        config_lower_bounds, config_upper_bounds = param_manager.get_bounds_as_arrays(
            base_param_names
        )
        config_lower_bounds = np.asarray(config_lower_bounds, dtype=float)
        config_upper_bounds = np.asarray(config_upper_bounds, dtype=float)

        # Load initial parameters if not provided
        if initial_params is None:
            initial_params = param_manager.get_initial_values()
            logger.info(
                f"Loaded initial parameters from config: {len(initial_params)} parameters"
            )

        # Load bounds if not provided
        if bounds is None:
            bounds = param_manager.get_parameter_bounds(base_param_names)
            logger.info("Loaded parameter bounds from config")

        if initial_params is not None:
            initial_params = np.asarray(initial_params, dtype=np.float64)

        if bounds is not None:
            bounds = (
                np.asarray(bounds[0], dtype=np.float64),
                np.asarray(bounds[1], dtype=np.float64),
            )
            try:
                logger.debug(
                    "Sequential bounds dtype: lower=%s upper=%s",
                    getattr(bounds[0], "dtype", type(bounds[0])),
                    getattr(bounds[1], "dtype", type(bounds[1])),
                )
                logger.debug(
                    "Sequential bounds values: lower=%s upper=%s",
                    np.array2string(bounds[0], precision=3),
                    np.array2string(bounds[1], precision=3),
                )
            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:  # pragma: no cover - logging safeguard
                logger.debug(f"Sequential bounds dtype logging failed: {exc}")

        # Create residual function using physics kernels
        # (apply_diagonal_correction and compute_g2_scaled imported at module level)

        phi_unique_all = np.unique(np.round(phi_flat, decimals=6))
        t1_unique_all = np.unique(np.asarray(t1))
        t2_unique_all = np.unique(np.asarray(t2))
        n_phi_total = len(phi_unique_all)

        per_angle_contrast_override: np.ndarray | None = None
        per_angle_offset_override: np.ndarray | None = None
        if per_angle_scaling_initial:
            contrast_override = per_angle_scaling_initial.get("contrast")
            if contrast_override is not None:
                try:
                    arr = np.asarray(contrast_override, dtype=np.float64)
                    if arr.size == n_phi_total:
                        per_angle_contrast_override = arr.copy()
                    else:
                        logger.warning(
                            "Sequential per-angle contrast override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_total,
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid sequential per-angle contrast override; ignoring"
                    )
            offset_override = per_angle_scaling_initial.get("offset")
            if offset_override is not None:
                try:
                    arr = np.asarray(offset_override, dtype=np.float64)
                    if arr.size == n_phi_total:
                        per_angle_offset_override = arr.copy()
                    else:
                        logger.warning(
                            "Sequential per-angle offset override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_total,
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid sequential per-angle offset override; ignoring"
                    )

        scalar_layout_len = len(physical_param_names) + 2
        expected_per_angle_len = 2 * n_phi_total + len(physical_param_names)

        def _expand_compact_layout(vector: np.ndarray) -> np.ndarray:
            """Replicate scalar contrast/offset entries across all angles."""

            arr = np.asarray(vector, dtype=np.float64)
            if (
                expected_per_angle_len == scalar_layout_len
                or arr.size == expected_per_angle_len
            ):
                return arr
            if n_phi_total == 0 or arr.size != scalar_layout_len:
                return arr
            contrast_val = arr[0]
            offset_val = arr[1]
            physical_vals = arr[2:]
            contrast_block = np.full(n_phi_total, contrast_val, dtype=np.float64)
            offset_block = np.full(n_phi_total, offset_val, dtype=np.float64)
            return np.concatenate([contrast_block, offset_block, physical_vals])

        solver_initial_params = initial_params.copy()
        solver_per_angle_scaling = False
        solver_per_angle_expanded = False

        if per_angle_scaling:
            if solver_initial_params.size == expected_per_angle_len:
                solver_per_angle_scaling = True
            elif (
                expected_per_angle_len > scalar_layout_len
                and solver_initial_params.size == scalar_layout_len
            ):
                solver_initial_params = _expand_compact_layout(solver_initial_params)
                solver_per_angle_scaling = True
                solver_per_angle_expanded = True
                logger.info(
                    "Expanded scalar contrast/offset to per-angle layout for sequential solver (%d angles)",
                    n_phi_total,
                )
                if bounds is not None and bounds[0].size == scalar_layout_len:
                    bounds = (
                        _expand_compact_layout(bounds[0]),
                        _expand_compact_layout(bounds[1]),
                    )
            else:
                logger.warning(
                    "Per-angle scaling requested but parameter vector has %d entries (expected %d); "
                    "sequential solver will operate with scalar scaling",
                    solver_initial_params.size,
                    expected_per_angle_len,
                )

        if (
            solver_per_angle_scaling
            and solver_initial_params.size == expected_per_angle_len
        ):
            if per_angle_contrast_override is not None:
                solver_initial_params[:n_phi_total] = per_angle_contrast_override
            if per_angle_offset_override is not None:
                solver_initial_params[n_phi_total : 2 * n_phi_total] = (
                    per_angle_offset_override
                )

        param_lower_bounds = config_lower_bounds.copy()
        param_upper_bounds = config_upper_bounds.copy()
        if solver_per_angle_scaling and expected_per_angle_len > scalar_layout_len:
            param_lower_bounds = _expand_compact_layout(param_lower_bounds)
            param_upper_bounds = _expand_compact_layout(param_upper_bounds)

        if solver_per_angle_scaling:
            param_names = _build_parameter_labels(
                True,
                n_phi_total,
                physical_param_names,
            )
        else:
            param_names = base_param_names

        def _maybe_expand_x_scale(value: Any) -> Any:
            if value is None or not solver_per_angle_scaling:
                return value
            if isinstance(value, (str, bytes, dict)):
                return value
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return value
            if array.ndim == 0:
                if expected_per_angle_len > 0:
                    return np.full(
                        expected_per_angle_len,
                        float(array),
                        dtype=np.float64,
                    )
                return float(array)
            if array.size == expected_per_angle_len:
                return array
            if (
                expected_per_angle_len > scalar_layout_len
                and array.size == scalar_layout_len
            ):
                return _expand_compact_layout(array)
            return value

        x_scale_value = _maybe_expand_x_scale(x_scale_value)
        sigma_source = getattr(data, "sigma", None)
        if sigma_source is None:
            sigma_array = np.ones(
                (n_phi_total, len(t1_unique_all), len(t2_unique_all)),
                dtype=np.float64,
            )
        else:
            sigma_array = np.asarray(sigma_source, dtype=np.float64)
            if not np.all(np.isfinite(sigma_array)):
                raise ValueError(
                    "sigma values must be finite; received NaN/inf entries"
                )
            if np.any(sigma_array <= 0):
                non_positive = float(np.count_nonzero(sigma_array <= 0))
                raise ValueError(
                    "sigma values must be strictly positive for least-squares "
                    "weighting; found "
                    f"{non_positive:.0f} non-positive entries"
                )

        q_value = float(getattr(data, "q", 1.0))
        L_value = float(getattr(data, "L", 1.0))
        dt_attr = getattr(data, "dt", None)
        dt_value = float(dt_attr) if dt_attr is not None else None

        t1_unique_jnp = jnp.asarray(t1_unique_all)
        t2_unique_jnp = jnp.asarray(t2_unique_all)

        physical_index_map = build_physical_index_map(
            solver_per_angle_scaling,
            n_phi_total if solver_per_angle_scaling else 0,
            physical_param_names,
        )

        transform_state: dict[str, Any] = {}
        if transform_cfg:
            solver_initial_params, transform_state = (
                apply_forward_shear_transforms_to_vector(
                    solver_initial_params,
                    physical_index_map,
                    transform_cfg,
                )
            )
            if bounds is not None:
                bounds = apply_forward_shear_transforms_to_bounds(
                    bounds,
                    transform_state,
                )

        def _compute_g2_grid_for_phi(
            phi_index: int,
            physical_params: np.ndarray,
            contrast_params: np.ndarray | float,
            offset_params: np.ndarray | float,
        ) -> np.ndarray:
            phi_val = float(phi_unique_all[phi_index])
            if solver_per_angle_scaling:
                contrast_val = float(contrast_params[phi_index])
                offset_val = float(offset_params[phi_index])
            else:
                contrast_val = float(contrast_params)
                offset_val = float(offset_params)

            g2_grid = compute_g2_scaled(
                params=jnp.asarray(physical_params),
                t1=t1_unique_jnp,
                t2=t2_unique_jnp,
                phi=phi_val,
                q=q_value,
                L=L_value,
                contrast=contrast_val,
                offset=offset_val,
                dt=dt_value,
            )
            g2_grid = jnp.squeeze(g2_grid, axis=0)
            g2_grid = apply_diagonal_correction(g2_grid)
            return np.asarray(g2_grid, dtype=np.float64)

        residual_debug_logged = False

        def residual_func(
            params: np.ndarray,
            phi_vals: np.ndarray,
            t1_vals: np.ndarray,
            t2_vals: np.ndarray,
            g2_vals: np.ndarray,
        ) -> np.ndarray:
            """Residual function compatible with sequential optimization."""

            params_np = np.asarray(params, dtype=np.float64)
            if transform_state:
                params_np = apply_inverse_shear_transforms_to_vector(
                    params_np,
                    transform_state,
                )
            phi_section = np.asarray(phi_vals, dtype=np.float64)
            t1_section = np.asarray(t1_vals, dtype=np.float64)
            t2_section = np.asarray(t2_vals, dtype=np.float64)
            g2_section = np.asarray(g2_vals, dtype=np.float64)

            if solver_per_angle_scaling:
                contrast_params = params_np[:n_phi_total]
                offset_params = params_np[n_phi_total : 2 * n_phi_total]
                physical_params = params_np[2 * n_phi_total :]
            else:
                contrast_params = float(params_np[0])
                offset_params = float(params_np[1])
                physical_params = params_np[2:]

            # Note: clip removed - sequential residual data comes from same source as
            # unique arrays (phi_flat, t1, t2), so all values are guaranteed to be in range.
            # The clip was causing optimization to converge to wrong local minima.
            # See: stratified LS fix in residual.py (D0=91342 vs 19253 issue).
            phi_indices = np.searchsorted(
                phi_unique_all, np.round(phi_section, decimals=6)
            )
            t1_indices = np.searchsorted(t1_unique_all, t1_section)
            t2_indices = np.searchsorted(t2_unique_all, t2_section)

            g2_model = np.empty_like(g2_section, dtype=np.float64)
            sigma_vals = np.empty_like(g2_section, dtype=np.float64)

            nonlocal residual_debug_logged
            if not residual_debug_logged:
                logger.debug(
                    "Sequential residual call: params_shape=%s, phi_unique=%d",
                    params_np.shape,
                    len(np.unique(phi_section)),
                )
                residual_debug_logged = True

            for phi_idx in np.unique(phi_indices):
                mask = phi_indices == phi_idx
                g2_grid = _compute_g2_grid_for_phi(
                    phi_idx, physical_params, contrast_params, offset_params
                )
                g2_model[mask] = g2_grid[t1_indices[mask], t2_indices[mask]]
                sigma_slice = sigma_array[phi_idx]
                sigma_vals[mask] = sigma_slice[t1_indices[mask], t2_indices[mask]]

            residuals = (g2_section - g2_model) / (sigma_vals + 1e-10)
            return residuals

        # Get optimizer configuration
        opt_config = config.config.get("optimization", {})
        nlsq_config = opt_config.get("nlsq", {})

        # Sequential-specific config. Default raised 0.5 -> 0.8 (CR-6): at 0.5,
        # up to half the angles could be unconverged while the overall result was
        # still stamped "converged". 0.8 requires a clear majority to converge.
        seq_config = opt_config.get("sequential", {})
        min_success_rate = seq_config.get("min_success_rate", 0.8)
        weighting = seq_config.get("weighting", "inverse_variance")

        # Run sequential optimization
        logger.info(
            f"Starting per-angle optimization with {len(np.unique(phi_flat))} angles..."
        )

        least_squares_kwargs: dict[str, Any] = {
            "max_nfev": nlsq_config.get("max_iterations", 1000),
            "ftol": nlsq_config.get("tolerance", 1e-8),
        }
        if "diff_step" in nlsq_config:
            least_squares_kwargs["diff_step"] = nlsq_config["diff_step"]
        if "f_scale" in nlsq_config:
            least_squares_kwargs["f_scale"] = nlsq_config["f_scale"]
        if x_scale_value is not None:
            least_squares_kwargs["x_scale"] = x_scale_value

        sequential_result = optimize_per_angle_sequential(
            phi=phi_flat,
            t1=t1_flat,
            t2=t2_flat,
            g2_exp=g2_flat,
            residual_func=residual_func,
            initial_params=solver_initial_params,
            bounds=bounds,
            weighting=weighting,
            min_success_rate=min_success_rate,
            parameter_names=param_names,
            **least_squares_kwargs,
        )

        # Convert SequentialResult to OptimizationResult
        execution_time = time.time() - start_time

        # Get device info
        device_info = {
            "type": "CPU",  # Sequential strategy runs on CPU via nlsq.CurveFit
            "backend": "nlsq.CurveFit",
            "strategy": "sequential_per_angle",
        }

        # Compute chi-squared
        combined_solver = sequential_result.combined_parameters.copy()
        combined_physical = combined_solver.copy()
        if transform_state:
            combined_physical = apply_inverse_shear_transforms_to_vector(
                combined_physical,
                transform_state,
            )

        final_residuals = residual_func(
            combined_physical, phi_flat, t1_flat, t2_flat, g2_flat
        )
        chi_squared = float(np.sum(final_residuals**2))
        n_data = len(phi_flat)
        n_params = len(sequential_result.combined_parameters)
        reduced_chi_squared = chi_squared / (n_data - n_params)

        # Diagnostics payload
        param_status = {}
        for idx, name in enumerate(param_names):
            value = combined_physical[idx]
            if np.isclose(value, param_lower_bounds[idx]):
                status = "at_lower_bound"
            elif np.isclose(value, param_upper_bounds[idx]):
                status = "at_upper_bound"
            else:
                status = "active"
            param_status[name] = status

        def _norm_array_to_dict(array: np.ndarray | None) -> dict[str, float] | None:
            if array is None:
                return None
            return {name: float(array[idx]) for idx, name in enumerate(param_names)}

        per_angle_jac = {}
        for angle_result in sequential_result.per_angle_results:
            angle_label = f"phi_{angle_result['phi_angle']:.2f}"
            per_angle_jac[angle_label] = {
                "initial": None,
                "final": None,
            }
            if angle_result.get("jac_initial_norms") is not None:
                per_angle_jac[angle_label]["initial"] = {
                    name: float(angle_result["jac_initial_norms"][idx])
                    for idx, name in enumerate(param_names)
                }
            if angle_result.get("jac_final_norms") is not None:
                per_angle_jac[angle_label]["final"] = {
                    name: float(angle_result["jac_final_norms"][idx])
                    for idx, name in enumerate(param_names)
                }

        total_nfev = sum(
            r.get("n_iterations", 0) for r in sequential_result.per_angle_results
        )

        diagnostics_payload = {
            "solver_settings": {
                "loss": nlsq_config.get("loss", "linear"),
                "x_scale": nlsq_config.get("x_scale", "jac"),
                "strategy": "sequential_per_angle",
                "jac_sample_size": JAC_SAMPLE_SIZE,
            },
            "nfev_reported": total_nfev,
            "nfev_actual": total_nfev,
            "parameter_status": param_status,
            "initial_jacobian_norms": _norm_array_to_dict(
                sequential_result.initial_jacobian_norms
            ),
            "final_jacobian_norms": _norm_array_to_dict(
                sequential_result.final_jacobian_norms
            ),
            "per_angle_jacobian_norms": per_angle_jac,
            "solver_per_angle_expanded": solver_per_angle_expanded,
            "chi_squared": chi_squared,
            "reduced_chi_squared": reduced_chi_squared,
        }

        # CR-6: enumerate which angles failed so a partially-converged result is
        # not opaque. A caller checking only convergence_status must still be able
        # to discover that some per-angle parameters are unconverged.
        failed_angle_indices = [
            i
            for i, r in enumerate(sequential_result.per_angle_results)
            if not r.get("success", False)
        ]
        diagnostics_payload["n_angles_failed"] = sequential_result.n_angles_failed
        diagnostics_payload["failed_angle_indices"] = failed_angle_indices
        diagnostics_payload["min_success_rate"] = min_success_rate

        # Determine convergence status
        if sequential_result.success_rate >= min_success_rate:
            convergence_status = "converged"
            quality_flag = (
                "good" if sequential_result.success_rate > 0.8 else "marginal"
            )
            if failed_angle_indices:
                logger.warning(
                    "Sequential fit marked converged at %.0f%% success, but %d "
                    "angle(s) failed; their parameters in the result are "
                    "unconverged. Failed angle indices: %s",
                    sequential_result.success_rate * 100,
                    sequential_result.n_angles_failed,
                    failed_angle_indices,
                )
        else:
            convergence_status = "failed"
            quality_flag = "poor"

        # Compute uncertainties from covariance. Guard against negative diagonal
        # entries (possible from a near-singular Hessian or numerical noise):
        # an unguarded np.sqrt would silently emit NaN uncertainties indistinguishable
        # from a valid zero. Mirror the np.maximum(diag, 0) guard in recovery.py.
        cov_diag = np.diag(sequential_result.combined_covariance)
        n_negative = int(np.sum(cov_diag < 0.0))
        if n_negative > 0:
            logger.warning(
                "%d negative covariance diagonal entr%s clipped to 0 before sqrt; "
                "uncertainties for those parameters are unreliable (near-singular fit).",
                n_negative,
                "y" if n_negative == 1 else "ies",
            )
        uncertainties = np.sqrt(np.maximum(cov_diag, 0.0))

        # Summary logging
        logger.info("=" * 80)
        logger.info("SEQUENTIAL OPTIMIZATION COMPLETE")
        logger.info(f"  Success rate: {sequential_result.success_rate:.1%}")
        logger.info(
            f"  Angles optimized: {sequential_result.n_angles_optimized}/{sequential_result.n_angles_optimized + sequential_result.n_angles_failed}"
        )
        logger.info(f"  Combined cost: {sequential_result.total_cost:.4f}")
        logger.info(f"  Reduced chi2: {reduced_chi_squared:.4f}")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Weighting: {weighting}")
        logger.info("=" * 80)

        return OptimizationResult(
            parameters=combined_physical,
            uncertainties=uncertainties,
            covariance=sequential_result.combined_covariance,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=sum(
                r["n_iterations"] for r in sequential_result.per_angle_results
            ),
            execution_time=execution_time,
            device_info=device_info,
            recovery_actions=[
                f"Sequential per-angle optimization: {sequential_result.n_angles_optimized} angles converged"
            ],
            quality_flag=quality_flag,
            nlsq_diagnostics=diagnostics_payload,
        )

    def _validate_initial_params(
        self,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Validate initial parameters are within bounds, clip if necessary.

        Args:
            params: Initial parameter guess
            bounds: (lower, upper) bounds tuple or None

        Returns:
            Validated/clipped parameter array

        Raises:
            ValueError: If params shape doesn't match bounds
        """
        params = np.asarray(params)

        # If no bounds, return params as-is
        if bounds is None:
            return params

        lower, upper = bounds
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Validate parameter count matches bounds
        if params.shape != lower.shape or params.shape != upper.shape:
            raise ValueError(
                f"Parameter shape mismatch: params={params.shape}, "
                f"lower={lower.shape}, upper={upper.shape}",
            )

        # Clip parameters to bounds
        clipped_params = np.clip(params, lower, upper)

        # Warn if any parameters were clipped
        if not np.allclose(params, clipped_params):
            clipped_indices = np.where(~np.isclose(params, clipped_params))[0]
            logger = get_logger(__name__)
            logger.warning(
                f"Initial parameters clipped to bounds at indices {clipped_indices}",
            )

        return clipped_params

    def _convert_bounds(
        self,
        homodyne_bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Convert homodyne bounds format to NLSQ format.

        Args:
            homodyne_bounds: (lower_array, upper_array) tuple or None

        Returns:
            NLSQ-compatible bounds tuple or None for unbounded optimization

        Raises:
            ValueError: If bounds are invalid (lower > upper)
        """
        # Handle None bounds (unbounded optimization)
        if homodyne_bounds is None:
            return None

        # Extract lower and upper bounds
        lower, upper = homodyne_bounds

        # Convert to numpy arrays if not already
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Validate bounds: lower <= upper elementwise (allow equal for fixed params)
        if np.any(lower > upper):
            invalid_indices = np.where(lower > upper)[0]
            raise ValueError(
                f"Invalid bounds: lower > upper at indices {invalid_indices}. "
                f"Bounds must satisfy lower <= upper elementwise.",
            )

        # NLSQ uses the same (lower, upper) tuple format as homodyne
        # Just return validated bounds
        return (lower, upper)

    def _create_residual_function(
        self, data: Any, analysis_mode: AnalysisMode, per_angle_scaling: bool = True
    ) -> Any:
        """Create JAX-compatible model function for NLSQ with per-angle scaling support.

        IMPORTANT: NLSQ's curve_fit_large expects a MODEL FUNCTION f(x, *params) -> y,
        NOT a residual function. NLSQ internally computes residuals = data - model.

        Args:
            data: XPCS experimental data
            analysis_mode: Analysis mode determining model computation
            per_angle_scaling: If True (default), use per-angle contrast/offset parameters.
                             This is the physically correct behavior.
                             If False, use legacy single contrast/offset for all angles.

        Returns:
            Model function with signature f(xdata, *params) -> ydata_theory
            where xdata is a dummy variable for NLSQ compatibility

        Raises:
            AttributeError: If data is missing required attributes
        """
        # Import NLSQ physics backend for g2 computation
        from xpcsjax.core.physics_nlsq import compute_g2_scaled

        # Validate data has required attributes
        required_attrs = ["phi", "t1", "t2", "g2", "sigma", "q", "L"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                raise AttributeError(
                    f"Data must have '{attr}' attribute for residual computation",
                )

        # Extract data attributes and convert to JAX arrays
        # CRITICAL FIX (Nov 11, 2025): Handle stratified vs non-stratified data differently
        #
        # Stratified data: phi_flat, t1_flat, t2_flat are all per-point arrays (same length)
        # Non-stratified data: phi, t1, t2 are unique grid values (different lengths)
        is_stratified = hasattr(data, "phi_flat")

        if is_stratified:
            # Stratified data: use per-point flat arrays
            phi = jnp.asarray(data.phi_flat)  # Shape: (n_data,)
            t1 = jnp.asarray(data.t1_flat)  # Shape: (n_data,)
            t2 = jnp.asarray(data.t2_flat)  # Shape: (n_data,)
        else:
            # Non-stratified data: use unique grid values
            phi = jnp.asarray(data.phi)  # Shape: (n_phi,)
            t1 = jnp.asarray(data.t1)  # Shape: (n_t1,)
            t2 = jnp.asarray(data.t2)  # Shape: (n_t2,)

        q = float(data.q)
        L = float(data.L)

        # Get dt from data — required for compute_g2_scaled (float, not Optional).
        dt = getattr(data, "dt", None)
        if dt is not None:
            dt = float(dt)
            # Validate dt before JIT compilation (avoid JAX tracing issues)
            if dt <= 0:
                raise ValueError(f"dt must be positive, got {dt}")
            if not np.isfinite(dt):
                raise ValueError(f"dt must be finite, got {dt}")
        else:
            # Fallback: derive from t1 minimum spacing
            t1_arr = np.asarray(data.t1)
            t1_unique = np.unique(t1_arr)
            if len(t1_unique) > 1:
                dt = float(np.min(np.diff(t1_unique)))
            else:
                dt = 0.001
            import warnings

            warnings.warn(
                f"data.dt missing; derived dt={dt:.6g} from t1 spacing",
                stacklevel=2,
            )

        # Pre-compute phi_unique for per-angle parameter mapping
        phi_unique = jnp.asarray(np.unique(np.asarray(phi)))
        n_phi = len(phi_unique)

        # Determine parameter structure based on analysis mode and per_angle_scaling
        # Legacy (per_angle_scaling=False): [contrast, offset, *physical_params]
        #   Static isotropic: 5 params total (2 scaling + 3 physical)
        #   Laminar flow: 9 params total (2 scaling + 7 physical)
        #
        # Per-angle (per_angle_scaling=True): [contrast_0, ..., contrast_{n_phi-1},
        #                                       offset_0, ..., offset_{n_phi-1}, *physical_params]
        #   Static isotropic: (2*n_phi + 3) params total
        #   Laminar flow: (2*n_phi + 7) params total

        def model_function(xdata: jnp.ndarray, *params_tuple: float) -> jnp.ndarray:
            """Compute theoretical g2 model for NLSQ optimization with per-angle scaling.

            IMPORTANT: xdata contains indices into the flattened data array.
            This function MUST respect xdata size for curve_fit_large chunking.
            When curve_fit_large chunks the data, xdata will be a subset of indices.

            NLSQ will internally compute residuals as: (ydata - model) / sigma

            Args:
                xdata: Array of indices into flattened g2 array.
                       Full dataset: [0, 1, ..., n-1]
                       Chunked: [0, 1, ..., chunk_size-1] (subset)
                *params_tuple: Unpacked parameters (per-angle scaling only)
                    - Format: [contrast_0, ..., contrast_{n_phi-1},
                              offset_0, ..., offset_{n_phi-1}, *physical]

            Returns:
                Theoretical g2 values at requested indices (size matches xdata)
            """
            # Convert params tuple to array (stack avoids retracing vs asarray)
            params_array = jnp.stack(params_tuple)

            # Extract per-angle scaling parameters (legacy mode removed Nov 2025)
            # Per-angle mode: first n_phi are contrasts, next n_phi are offsets
            contrast = params_array[:n_phi]  # Array of shape (n_phi,)
            offset = params_array[n_phi : 2 * n_phi]  # Array of shape (n_phi,)
            physical_params = params_array[2 * n_phi :]

            # Get requested data point indices
            # Use int64 to prevent overflow when n_phi * n_t1 * n_t2 > 2.147B.
            indices = jnp.asarray(xdata, dtype=jnp.int64)

            # CRITICAL FIX (Nov 11, 2025): Handle stratified vs non-stratified data
            if is_stratified:
                # STRATIFIED DATA PATH (per-point arrays)
                # Extract per-point values for requested indices
                phi_requested = phi[indices]  # Shape: (chunk_size,)
                t1_requested = t1[indices]  # Shape: (chunk_size,)
                t2_requested = t2[indices]  # Shape: (chunk_size,)

                # Map phi values to indices in phi_unique to get correct contrast/offset
                # Find which unique phi each requested phi corresponds to
                # Since phi values come from phi_unique, we can use searchsorted
                # CRITICAL: Keep all arrays in JAX (no np.asarray) for JIT compatibility
                # Note: clip removed - phi_requested is a subset of phi which was used to
                # build phi_unique, so all values are guaranteed to be in range.
                # The clip was causing optimization to converge to wrong local minima.
                phi_idx = jnp.searchsorted(
                    phi_unique, phi_requested
                )  # Shape: (chunk_size,)

                # Select per-angle contrast and offset for each data point
                contrast_requested = contrast[phi_idx]  # Shape: (chunk_size,)
                offset_requested = offset[phi_idx]  # Shape: (chunk_size,)

                # Compute g2 per-point using vmap
                # Each point has its own (phi, t1, t2, contrast, offset)
                compute_g2_per_point = jax.vmap(
                    lambda phi_val, t1_val, t2_val, c_val, o_val: compute_g2_scaled(
                        params=physical_params,
                        t1=jnp.array([t1_val]),  # Single value as 1D array
                        t2=jnp.array([t2_val]),
                        phi=phi_val,
                        q=q,
                        L=L,
                        contrast=c_val,
                        offset=o_val,
                        dt=dt,
                    )[0, 0],  # Extract scalar from (1, 1) output
                    in_axes=(0, 0, 0, 0, 0),  # Vmap over all arrays
                )

                g2_theory = compute_g2_per_point(
                    phi_requested,
                    t1_requested,
                    t2_requested,
                    contrast_requested,
                    offset_requested,
                )  # Shape: (chunk_size,) or possibly (chunk_size, 1)

                # Ensure 1D output by squeezing any trailing dimensions
                g2_theory = jnp.squeeze(g2_theory)

                return g2_theory

            else:
                # NON-STRATIFIED DATA PATH (grid-based computation)
                # Original grid-based logic for non-stratified data
                compute_g2_scaled_vmap = jax.vmap(
                    lambda phi_val, contrast_val, offset_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t1,  # 1D arrays
                            t2=t2,
                            phi=phi_val,  # Single phi value
                            q=q,
                            L=L,
                            contrast=contrast_val,  # Per-angle contrast
                            offset=offset_val,  # Per-angle offset
                            dt=dt,
                        ),
                        axis=0,  # Squeeze the phi dimension
                    ),
                    in_axes=(0, 0, 0),  # Vectorize over all three arrays
                )

                # Compute on grid for all unique angles
                g2_theory = compute_g2_scaled_vmap(phi_unique, contrast, offset)
                # Shape: (n_phi, n_t1, n_t2)

                # Apply diagonal correction
                from xpcsjax.core.jax_backend import apply_diagonal_correction

                apply_diagonal_vmap = jax.vmap(apply_diagonal_correction, in_axes=0)
                g2_theory = apply_diagonal_vmap(g2_theory)

                # Grid-based indexing for non-stratified data
                n_t1 = len(t1)
                n_t2 = len(t2)
                grid_size_per_angle = n_t1 * n_t2

                # Decompose flat indices into grid coordinates
                phi_idx = indices // grid_size_per_angle
                remaining = indices % grid_size_per_angle
                t1_idx = remaining // n_t2
                t2_idx = remaining % n_t2

                return g2_theory[phi_idx, t1_idx, t2_idx]

        return model_function

    def _update_best_parameters(
        self,
        params: np.ndarray,
        loss: float,
        batch_idx: int,
        logger: Any,
    ) -> None:
        """Update best parameters if current loss is better.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        loss : float
            Current loss value
        batch_idx : int
            Current batch index
        logger : logging.Logger
            Logger instance for reporting
        """
        if params is None:
            return  # Cannot update without parameters
        if loss < self.best_loss:
            prev_best = self.best_loss
            self.best_params = params.copy()
            self.best_loss = loss
            self.best_batch_idx = batch_idx
            logger.info(
                f"New best loss: {loss:.6e} at batch {batch_idx} "
                f"(improved from {prev_best:.6e})"
            )

    def _fit_with_hybrid_streaming_optimizer(
        self,
        residual_fn: Any,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        nlsq_config: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ AdaptiveHybridStreamingOptimizer."""
        return fit_with_hybrid_streaming_optimizer(
            residual_fn=residual_fn,
            xdata=xdata,
            ydata=ydata,
            initial_params=initial_params,
            bounds=bounds,
            logger=logger,
            nlsq_config=nlsq_config,
            fast_mode=self.fast_mode,
        )

    def _create_stratified_chunks(
        self,
        stratified_data: Any,
        target_chunk_size: int = 100_000,
    ) -> Any:
        """Convert stratified flat arrays into chunks for StratifiedResidualFunction."""
        return create_stratified_chunks(stratified_data, target_chunk_size)

    def _fit_with_stratified_least_squares(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        target_chunk_size: int = 100_000,
        anti_degeneracy_config: dict | None = None,
        nlsq_config_dict: dict | None = None,
        analysis_mode: AnalysisMode | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ's least_squares() with stratified residual function."""
        return fit_with_stratified_least_squares(
            stratified_data=stratified_data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=initial_params,
            bounds=bounds,
            log=logger,
            target_chunk_size=target_chunk_size,
            anti_degeneracy_config=anti_degeneracy_config,
            nlsq_config_dict=nlsq_config_dict,
            analysis_mode=analysis_mode,
        )

    def _fit_with_streaming_optimizer(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        streaming_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Deprecated: delegates to fit_with_streaming_optimizer_stratified_deprecated."""
        return fit_with_streaming_optimizer_stratified_deprecated(
            stratified_data=stratified_data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=initial_params,
            bounds=bounds,
            logger=logger,
            streaming_config=streaming_config,
        )

    # NOTE: Dead streaming optimizer code removed (NLSQ 0.4.0+ removed StreamingOptimizer)

    def _fit_with_out_of_core_accumulation(
        self,
        stratified_data: Any,
        data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        config: Any,
        fast_chi2_mode: bool = False,
        anti_degeneracy_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using Out-of-Core Global Accumulation for massive datasets."""
        return fit_with_out_of_core_accumulation(
            stratified_data=stratified_data,
            data=data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=initial_params,
            bounds=bounds,
            log=logger,
            config=config,
            fast_chi2_mode=fast_chi2_mode,
            anti_degeneracy_config=anti_degeneracy_config,
        )

    def _fit_with_stratified_hybrid_streaming(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        hybrid_config: dict | None = None,
        anti_degeneracy_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ AdaptiveHybridStreamingOptimizer for large datasets."""
        return fit_with_stratified_hybrid_streaming(
            stratified_data=stratified_data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=initial_params,
            bounds=bounds,
            logger=logger,
            hybrid_config=hybrid_config,
            anti_degeneracy_config=anti_degeneracy_config,
        )

    def _estimate_memory_for_stratified_ls(
        self,
        n_points: int,
        n_params: int,
        n_chunks: int,
    ) -> float:
        """Estimate peak memory usage for stratified least-squares optimization."""
        return estimate_memory_for_stratified_ls(n_points, n_params, n_chunks)

    def _should_use_streaming(
        self,
        n_points: int,
        n_params: int,
        n_chunks: int,
        memory_threshold_gb: float | None = None,
        memory_fraction: float | None = None,
    ) -> tuple[bool, float, str]:
        """Determine if streaming optimizer should be used based on memory estimate."""
        return should_use_streaming(
            n_points=n_points,
            n_params=n_params,
            n_chunks=n_chunks,
            memory_threshold_gb=memory_threshold_gb,
            memory_fraction=memory_fraction,
        )

    def _create_fit_result(
        self,
        popt: np.ndarray,
        pcov: np.ndarray,
        residuals: np.ndarray,
        n_data: int,
        iterations: int,
        execution_time: float,
        convergence_status: str = "converged",
        recovery_actions: list[str] | None = None,
        streaming_diagnostics: dict[str, Any] | None = None,
        stratification_diagnostics: StratificationDiagnostics | None = None,
        diagnostics_payload: dict[str, Any] | None = None,
        n_params_effective: int | None = None,
    ) -> OptimizationResult:
        """Convert NLSQ output to OptimizationResult.

        Args:
            popt: Optimized parameters
            pcov: Parameter covariance matrix
            residuals: Final residuals
            n_data: Number of data points
            iterations: Optimization iterations
            execution_time: Execution time in seconds
            convergence_status: Convergence status string
            recovery_actions: List of recovery actions taken
            streaming_diagnostics: Enhanced diagnostics for streaming optimization (Task 5.4)

        Returns:
            Complete OptimizationResult dataclass
        """

        # Convert to numpy arrays
        popt = np.asarray(popt)
        pcov = np.asarray(pcov)
        residuals = np.asarray(residuals)

        # Compute uncertainties from covariance diagonal
        uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))

        # Compute chi-squared
        chi_squared = float(np.sum(residuals**2))

        # Compute reduced chi-squared.
        # Use n_params_effective when provided — in auto_averaged mode the compressed
        # optimizer vector (e.g. 9 params) has fewer entries than the true model DOF
        # (e.g. 2*n_phi + n_physical = 53). Using len(popt) would underestimate the
        # true DOF, producing an artificially low reduced chi-squared and a falsely
        # optimistic quality_flag ("good" when the fit is "marginal" or "poor").
        n_params = n_params_effective if n_params_effective is not None else len(popt)
        degrees_of_freedom = n_data - n_params
        reduced_chi_squared = (
            chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.inf
        )

        # Get device information
        devices = jax.devices()
        device_info = {
            "platform": devices[0].platform,
            "device": str(devices[0]),
            "device_kind": devices[0].device_kind,
            "n_devices": len(devices),
        }

        # Determine quality flag based on reduced chi-squared
        if reduced_chi_squared < 1.5:
            quality_flag = "good"
        elif reduced_chi_squared < 3.0:
            quality_flag = "marginal"
        else:
            quality_flag = "poor"

        # Task 5.4: Build enhanced streaming diagnostics if batch statistics available
        enhanced_streaming_diagnostics = None
        if streaming_diagnostics is not None:
            # Start with provided diagnostics
            enhanced_streaming_diagnostics = streaming_diagnostics.copy()

            # Add batch statistics if available
            if (
                hasattr(self, "batch_statistics")
                and self.batch_statistics.total_batches > 0
            ):
                batch_stats = self.batch_statistics.get_statistics()

                # Extract key metrics for enhanced diagnostics
                enhanced_streaming_diagnostics.update(
                    {
                        "batch_success_rate": batch_stats["success_rate"],
                        "failed_batch_indices": [
                            b["batch_idx"]
                            for b in batch_stats["recent_batches"]
                            if not b["success"]
                        ],
                        "error_type_distribution": batch_stats["error_distribution"],
                        "average_iterations_per_batch": batch_stats[
                            "average_iterations"
                        ],
                        "total_batches_processed": batch_stats["total_batches"],
                    }
                )

        # Create result
        result = OptimizationResult(
            parameters=popt,
            uncertainties=uncertainties,
            covariance=pcov,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=iterations,
            execution_time=execution_time,
            device_info=device_info,
            recovery_actions=recovery_actions or [],
            quality_flag=quality_flag,
            streaming_diagnostics=enhanced_streaming_diagnostics,  # Task 5.4
            stratification_diagnostics=stratification_diagnostics,  # v2.2.1: Stratification diagnostics
            nlsq_diagnostics=diagnostics_payload,
        )

        return result
