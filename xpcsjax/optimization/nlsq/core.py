"""NLSQ: Primary Optimization Method for Homodyne
==================================================

NLSQ package-based trust-region nonlinear least squares solver for the scaled
optimization process. This is the primary optimization method providing
fast, reliable parameter estimation for homodyne analysis.

Core Equation: c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

Key Features:
- NLSQ trust-region solver (TRF/Levenberg-Marquardt) for robust optimization
- JAX JIT compilation for high performance
- Intelligent error recovery with 3-attempt retry strategy (T022-T024)
- Compatible with existing ParameterSpace and FitResult classes
- HPC-optimized for 36/128-core CPU nodes
- CPU-only (no GPU support since v2.3.0)
- Dataset size-aware optimization strategies

Performance (Validated T036-T041):
- Parameter recovery accuracy: 2-14% on core parameters
- Sub-linear time scaling: ~1.5s for 500-9,375 point datasets
- Numerical stability: <4% deviation across initial conditions
- Throughput: 317-5,977 points/second
- 100% convergence rate across all validation tests

Production Status:
- Scientifically validated (7/7 tests passed)
- Production-ready with error recovery
- Approved for scientific research and deployment

Migration from Optimistix:
- Replaced Optimistix with NLSQ package (github.com/imewei/NLSQ)
- NLSQWrapper provides unified interface with error recovery
- Maintains backward API compatibility
- User-facing Optimistix references removed from public APIs

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Validation Report: SCIENTIFIC_VALIDATION_REPORT.md
- Production Report: PRODUCTION_READINESS_REPORT.md
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

# JAX imports with fallback
F = TypeVar("F", bound=Callable[..., Any])

try:
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp: types.ModuleType = np  # type: ignore[no-redef]

    def jit(f: F) -> F:  # type: ignore[no-redef]  # noqa: UP047
        return f

    def vmap(f: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return f

    def grad(f: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[misc]
        return lambda x: np.zeros_like(x)


# Core homodyne imports
try:
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.config.parameter_registry import AnalysisMode
    from xpcsjax.core.fitting import ParameterSpace
    from xpcsjax.utils.logging import get_logger, log_performance

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False
    import logging
    from collections.abc import Mapping
    from logging import Logger, LoggerAdapter

    def get_logger(
        name: str | None = None, *, context: Mapping[str, Any] | None = None
    ) -> Logger | LoggerAdapter[Logger]:
        return logging.getLogger(name or __name__)

    def log_performance(
        logger: Logger | LoggerAdapter[Logger] | None = None,
        level: int = logging.INFO,
        threshold: float = 0.0,
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator


# Optional ParameterManager import (Phase 4.2)
try:
    from xpcsjax.config.parameter_manager import ParameterManager

    HAS_PARAMETER_MANAGER = True
except ImportError:
    HAS_PARAMETER_MANAGER = False
    ParameterManager = None  # type: ignore[assignment,misc]

# NLSQWrapper import for legacy implementation
try:
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

    HAS_NLSQ_WRAPPER = True
except ImportError:
    HAS_NLSQ_WRAPPER = False
    NLSQWrapper = None  # type: ignore[assignment,misc]
    WrapperOptimizationResult = None  # type: ignore[assignment,misc]

# Results module import (for return type)
try:
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    HAS_RESULTS_MODULE = True
except ImportError:
    HAS_RESULTS_MODULE = False
    OptimizationResult = None  # type: ignore[assignment,misc]

# NLSQAdapter import for new CurveFit-based implementation (v2.11.0+)
try:
    from xpcsjax.optimization.nlsq.adapter import (
        AdapterConfig,
        NLSQAdapter,
        is_adapter_available,
    )

    HAS_NLSQ_ADAPTER = is_adapter_available()
except ImportError:
    HAS_NLSQ_ADAPTER = False
    NLSQAdapter = None  # type: ignore[assignment,misc]
    AdapterConfig = None  # type: ignore[assignment,misc]

# Multi-start optimization import (v2.6.0)
try:
    from xpcsjax.optimization.nlsq.multistart import (
        MultiStartConfig,
        MultiStartResult,
        SingleStartResult,
        run_multistart_nlsq,
    )

    HAS_MULTISTART = True
except ImportError:
    HAS_MULTISTART = False
    MultiStartConfig = None  # type: ignore[assignment,misc]
    MultiStartResult = None  # type: ignore[assignment,misc]
    SingleStartResult = None  # type: ignore[assignment,misc]
    run_multistart_nlsq = None  # type: ignore[assignment]

# CMA-ES global optimization import (v2.15.0 / NLSQ 0.6.4+)
try:
    from xpcsjax.optimization.nlsq.cmaes_wrapper import (
        CMAES_AVAILABLE,
        CMAESResult,
        CMAESWrapper,
        CMAESWrapperConfig,
        fit_with_cmaes,
    )

    HAS_CMAES = CMAES_AVAILABLE
except ImportError:
    HAS_CMAES = False
    CMAESWrapper = None  # type: ignore[assignment,misc]
    CMAESWrapperConfig = None  # type: ignore[assignment,misc]
    CMAESResult = None  # type: ignore[assignment,misc]
    fit_with_cmaes = None  # type: ignore[assignment]

# CPU threading configuration (FR-005, T026)
try:
    from xpcsjax.device.cpu import configure_cpu_threading

    HAS_CPU_CONFIG = True
except ImportError:
    HAS_CPU_CONFIG = False
    configure_cpu_threading = None  # type: ignore[assignment]

# Anti-degeneracy controller import (v2.16.1+)
try:
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

    HAS_ANTI_DEGENERACY = True
except ImportError:
    HAS_ANTI_DEGENERACY = False
    AntiDegeneracyController = None  # type: ignore[assignment,misc]

# Export NLSQ availability for tests and external code
NLSQ_AVAILABLE = HAS_NLSQ_WRAPPER and JAX_AVAILABLE

logger = get_logger(__name__)

# Default sigma used when experimental uncertainties are not provided.
# Applied at lines ~715 and ~1775 when creating placeholder sigma arrays.
# Used in auto-skip normalization to undo the 1/sigma^2 chi2 inflation.
_DEFAULT_SIGMA = 0.01


class NLSQResult:
    """Result container for NLSQ optimization compatible with FitResult."""

    def __init__(
        self,
        parameters: dict[str, float],
        parameter_errors: dict[str, float],
        chi_squared: float,
        reduced_chi_squared: float,
        success: bool,
        message: str,
        n_iterations: int,
        optimization_time: float,
        method: str = "nlsq",
    ):
        self.parameters = parameters
        self.parameter_errors = parameter_errors
        self.chi_squared = chi_squared
        self.reduced_chi_squared = reduced_chi_squared
        self.success = success
        self.message = message
        self.n_iterations = n_iterations
        self.optimization_time = optimization_time
        self.method = method


@log_performance(threshold=1.0)
def fit_nlsq_jax(
    data: dict[str, Any],
    config: ConfigManager,
    initial_params: dict[str, float] | None = None,
    per_angle_scaling: bool = True,  # REQUIRED: per-angle is physically correct
    use_adapter: bool = False,  # Experimental: use NLSQAdapter (CurveFit) instead of NLSQWrapper
    _skip_global_selection: bool = False,  # Internal: skip global opt check (for fallback)
) -> OptimizationResult:
    """NLSQ trust-region nonlinear least squares optimization with per-angle scaling.

    Uses NLSQ package (github.com/imewei/NLSQ) for trust-region optimization.

    v2.11.0+: Experimental NLSQAdapter with CurveFit class available for
    improved JIT caching and automatic workflow selection. Set use_adapter=True
    to enable (default is False, uses NLSQWrapper).

    Primary optimization method implementing the scaled optimization process:
    c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

    Parameters
    ----------
    data : dict
        XPCS experimental data. Accepts two formats:

        **Format 1 (CLI/loader format)**:
        - 'phi_angles_list': phi angle array (mapped to 'phi')
        - 'c2_exp': experimental correlation data (n_phi, n_t1, n_t2) (mapped to 'g2')
        - 't1': first delay time array
        - 't2': second delay time array
        - 'wavevector_q_list': q-vector array (first element extracted as scalar 'q')
        - 'sigma': (optional) uncertainty array, defaults to 0.01 * ones_like(g2)
        - 'L': (optional) stator-rotor gap (rheology) or sample-detector distance (standard XPCS), defaults to config value or 2000000 Å (200 µm, typical rheology-XPCS gap)
        - 'dt': (optional) time step, defaults to config value or None

        **Format 2 (Direct format)**:
        - 'phi': phi angle array
        - 'g2': experimental correlation data (n_phi, n_t1, n_t2)
        - 't1': first delay time array
        - 't2': second delay time array
        - 'q': wavevector magnitude (scalar)
        - 'sigma': (optional) uncertainty array
        - 'L': (optional) stator-rotor gap or sample-detector distance [Å]
        - 'dt': (optional) time step [s]

    config : ConfigManager
        Configuration manager with optimization settings
    initial_params : dict, optional
        Initial parameter guesses. If None, uses defaults from config.
    per_angle_scaling : bool, default=True
        MUST be True. Per-angle contrast/offset parameters are physically correct as each
        scattering angle has different optical properties and detector responses.
        Legacy scalar mode (False) is no longer supported (removed Nov 2025).
    use_adapter : bool, default=False
        EXPERIMENTAL (v2.11.0+): If True, use NLSQAdapter with NLSQ's CurveFit
        class for improved JIT caching and automatic workflow selection.
        If False (default), use the stable NLSQWrapper implementation.

    Notes
    -----
    **Global Optimization Selection (v2.15.0+):**
    This function serves as the unified entry point for NLSQ optimization.
    When called, it first checks for global optimization methods:

    1. If ``cmaes.enable: true`` → delegates to ``fit_nlsq_cmaes()``
    2. If ``multi_start.enable: true`` → delegates to ``fit_nlsq_multistart()``
    3. Otherwise → runs local trust-region optimization

    The CMA-ES function will fall back to multi-start (if enabled) when the
    scale ratio is below the threshold, implementing the full fallback chain.

    Returns
    -------
    OptimizationResult
        Optimization result with parameters, uncertainties, and diagnostics

    Raises
    ------
    ImportError
        If NLSQ package is not available
    ValueError
        If data validation fails
    """
    # Determine which backend to use
    _use_adapter = use_adapter and HAS_NLSQ_ADAPTER

    if _use_adapter:
        logger.debug("Using NLSQAdapter (CurveFit class) for optimization")
    else:
        if use_adapter and not HAS_NLSQ_ADAPTER:
            logger.warning(
                "NLSQAdapter requested but not available, falling back to NLSQWrapper"
            )
        if not HAS_NLSQ_WRAPPER:
            raise ImportError(
                "NLSQWrapper is required for NLSQ optimization. "
                "Ensure xpcsjax.optimization.nlsq_wrapper is available.",
            )

    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION")
    logger.info("=" * 60)

    # Track whether sigma was provided or auto-generated (affects chi-squared interpretation)
    _sigma_is_default = isinstance(data, dict) and "sigma" not in data

    # ==========================================================================
    # Global Optimization Selection (v2.15.0+)
    # Priority: CMA-ES > Multi-Start > Local Optimization
    # ==========================================================================
    if not _skip_global_selection:
        # Handle both ConfigManager objects and plain dicts
        config_dict: dict[str, Any] = (
            config.config if hasattr(config, "config") else config
        )  # type: ignore[assignment]
        nlsq_dict = config_dict.get("optimization", {}).get("nlsq", {})

        # CMA-ES has highest priority (for multi-scale problems)
        cmaes_dict = nlsq_dict.get("cmaes", {})
        if cmaes_dict.get("enable", False):
            if HAS_CMAES:
                logger.info("CMA-ES enabled, delegating to fit_nlsq_cmaes")
                cmaes_result = fit_nlsq_cmaes(
                    data=data,
                    config=config,
                    initial_params=initial_params,
                    per_angle_scaling=per_angle_scaling,
                )
                cmaes_result.sigma_is_default = _sigma_is_default
                return cmaes_result
            else:
                logger.warning(
                    "[CMA-ES] Enabled in config but not available (evosax not installed). "
                    "Install with: pip install nlsq[evosax]. "
                    "Falling back to multi-start or local optimization."
                )

        # Multi-start is second priority
        multi_start_dict = nlsq_dict.get("multi_start", {})
        if multi_start_dict.get("enable", False):
            if HAS_MULTISTART:
                logger.info("Multi-start enabled, delegating to fit_nlsq_multistart")
                multistart_result = fit_nlsq_multistart(
                    data=data,
                    config=config,
                    initial_params=initial_params,
                    per_angle_scaling=per_angle_scaling,
                )
                ms_result = multistart_result.to_optimization_result()
                ms_result.sigma_is_default = _sigma_is_default
                return ms_result
            else:
                logger.warning(
                    "[Multi-Start] Enabled in config but not available. "
                    "Falling back to local optimization."
                )

        logger.debug("No global optimization enabled, using local optimization")

    # Performance Optimization (Spec 001 - FR-005, T026): Configure CPU threading for HPC
    if HAS_CPU_CONFIG:
        try:
            cpu_config = configure_cpu_threading()
            logger.debug(
                f"CPU threading configured: {cpu_config.get('threads_configured', 'auto')} threads"
            )
        except (OSError, RuntimeError, AttributeError) as e:
            logger.debug(f"CPU threading configuration skipped: {e}")

    # Determine analysis mode
    analysis_mode = _get_analysis_mode(config)
    logger.info(f"Analysis mode: {analysis_mode}")
    logger.info(f"Per-angle scaling: {per_angle_scaling}")

    # Set up initial parameters
    per_angle_scaling_initial: dict[str, list[float]] | None = None
    if initial_params is None:
        # Try to load from config first (pass data for contrast/offset estimation)
        initial_params_temp, per_angle_scaling_initial_temp = (
            _load_initial_params_from_config(config, analysis_mode, data)
        )
        initial_params = initial_params_temp
        per_angle_scaling_initial = per_angle_scaling_initial_temp
        if initial_params is None:
            # Fallback to defaults (estimate contrast/offset from data if available)
            initial_params = _get_default_initial_params(analysis_mode)
            if data is not None:
                contrast_est, offset_est = _estimate_contrast_offset_from_data(data)
                initial_params["contrast"] = contrast_est
                initial_params["offset"] = offset_est
            logger.info("Using default initial parameters")
        else:
            logger.info("Using initial parameters from configuration")
    else:
        # Make a copy so we don't mutate caller-provided dict
        initial_params = initial_params.copy()
        per_angle_scaling_initial_pop = initial_params.pop("per_angle_scaling", None)
        if isinstance(per_angle_scaling_initial_pop, dict):
            per_angle_scaling_initial = per_angle_scaling_initial_pop
        else:
            per_angle_scaling_initial = None

    # Convert initial params dict to array
    x0 = _params_to_array(initial_params, analysis_mode)

    # Set up parameter bounds
    # FIX: Use ParameterManager to load bounds from config (including custom user bounds)
    if HAS_PARAMETER_MANAGER:
        # Handle both ConfigManager objects and plain dicts
        if hasattr(config, "config"):
            config_dict_for_pm: Any = config.config  # ConfigManager object
        else:
            config_dict_for_pm = config  # Already a dict

        # Use ParameterManager to get bounds from config (properly loads custom bounds)
        param_manager = ParameterManager(
            config_dict=config_dict_for_pm, analysis_mode=analysis_mode
        )
        param_names = _get_param_names(analysis_mode)
        bounds_list = param_manager.get_parameter_bounds(param_names)
        # Convert ParameterManager format (list of dicts) to _bounds_to_arrays format (dict of tuples)
        bounds_dict = {b["name"]: (b["min"], b["max"]) for b in bounds_list}
    else:
        # Fallback to ParameterSpace with hardcoded defaults (for backward compatibility)
        param_space = ParameterSpace()
        bounds_dict = _get_parameter_bounds(analysis_mode, param_space)

    lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)
    bounds = (lower_bounds, upper_bounds)

    # Convert data dict to object if needed (NLSQWrapper expects object attributes)
    data = _normalize_data_to_object(data, config, logger)

    diagnostics_enabled = _is_nlsq_diagnostics_enabled(config)
    shear_transform_cfg = _extract_shear_transform_config(config)

    # ==========================================================================
    # T021-T024: Fallback mechanism from NLSQAdapter to NLSQWrapper
    # ==========================================================================
    adapter_error: Exception | None = None
    fallback_occurred = False
    result: Any  # Will be OptimizationResult from either adapter or wrapper

    # Create optimizer and run optimization
    if _use_adapter:
        # T021: Try NLSQAdapter first with CurveFit class (v2.11.0+)
        try:
            adapter_config = AdapterConfig(
                enable_cache=True,
                enable_jit=True,
                enable_recovery=True,
                enable_stability=True,
                goal="quality",  # XPCS requires precision
            )
            adapter = NLSQAdapter(config=adapter_config)
            logger.debug("Attempting optimization with NLSQAdapter")

            result = adapter.fit(
                data=data,
                config=config,
                initial_params=x0,  # type: ignore[arg-type]
                bounds=bounds,  # type: ignore[arg-type]
                analysis_mode=analysis_mode,
                per_angle_scaling=per_angle_scaling,
                diagnostics_enabled=diagnostics_enabled,
                shear_transforms=shear_transform_cfg,
                per_angle_scaling_initial=per_angle_scaling_initial,
            )

            # T023: Add fallback_occurred to device_info (adapter succeeded)
            result.device_info["fallback_occurred"] = False
            result.device_info["fallback_reason"] = None
            logger.info("NLSQAdapter optimization succeeded")

        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            OSError,
            MemoryError,
        ) as e:
            # T022: Log WARNING when fallback occurs
            adapter_error = e
            logger.warning("NLSQAdapter failed, falling back to NLSQWrapper: %s", e)
            fallback_occurred = True
            # Fall through to wrapper below

    # Use NLSQWrapper if: (1) use_adapter=False, or (2) adapter failed
    if not _use_adapter or fallback_occurred:
        try:
            # Use legacy NLSQWrapper
            # Note: enable_recovery=True provides automatic error recovery
            wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=True)
            logger.debug("Attempting optimization with NLSQWrapper")

            result = wrapper.fit(
                data=data,
                config=config,
                initial_params=x0,  # type: ignore[arg-type]
                bounds=bounds,  # type: ignore[arg-type]
                analysis_mode=analysis_mode,
                per_angle_scaling=per_angle_scaling,
                diagnostics_enabled=diagnostics_enabled,
                shear_transforms=shear_transform_cfg,
                per_angle_scaling_initial=per_angle_scaling_initial,
            )

            # T023: Add fallback info to device_info
            result.device_info["adapter"] = "NLSQWrapper"
            result.device_info["fallback_occurred"] = fallback_occurred
            result.device_info["fallback_reason"] = (
                str(adapter_error) if adapter_error else None
            )
            if fallback_occurred:
                logger.info("NLSQWrapper fallback optimization succeeded")
            else:
                logger.info("NLSQWrapper optimization succeeded")

        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            OSError,
            MemoryError,
        ) as wrapper_error:
            # T024: Both adapter and wrapper failed - return failed result
            logger.error(
                "Both NLSQAdapter and NLSQWrapper failed: adapter=%s, wrapper=%s",
                adapter_error,
                wrapper_error,
            )
            n_params = len(x0)
            result = OptimizationResult(
                parameters=np.asarray(x0),
                uncertainties=np.zeros(n_params),
                covariance=np.eye(n_params),
                chi_squared=float("inf"),
                reduced_chi_squared=float("inf"),
                convergence_status="failed",
                iterations=0,
                execution_time=0.0,
                device_info={
                    "device": "cpu",
                    "adapter": "NLSQWrapper",
                    "fallback_occurred": fallback_occurred,
                    "fallback_reason": str(adapter_error) if adapter_error else None,
                    "wrapper_error": str(wrapper_error),
                },
                recovery_actions=[],
                quality_flag="poor",
            )

    result.sigma_is_default = _sigma_is_default
    _log_optimization_results(result, analysis_mode, per_angle_scaling, logger)

    return result


def _log_optimization_results(
    result: Any,
    analysis_mode: AnalysisMode,
    per_angle_scaling: bool,
    logger: Any,
) -> None:
    """Log optimization results including parameters and uncertainties.

    Pure logging function — reads from result but does not modify state.
    """
    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Execution time: {result.execution_time:.3f}s")
    logger.info(f"chi2 = {result.chi_squared:.6e}")
    logger.info(f"Reduced chi2 = {result.reduced_chi_squared:.6f}")

    if hasattr(result, "parameters") and result.parameters is not None:
        physical_param_names = _get_physical_param_names(analysis_mode)
        n_physical = len(physical_param_names)
        n_params = len(result.parameters)

        if per_angle_scaling and n_params > (2 + n_physical):
            n_angles = (n_params - n_physical) // 2

            logger.info(f"Fitted parameters (per-angle scaling: {n_angles} angles):")
            logger.info("  Physical parameters:")
            physical_start_idx = 2 * n_angles

            unc_array = (
                result.uncertainties if hasattr(result, "uncertainties") else None
            )
            unc_size = len(unc_array) if unc_array is not None else 0
            if unc_array is not None and unc_size != n_params:
                logger.warning(
                    f"Uncertainty array size mismatch: expected {n_params}, got {unc_size}. "
                    "This may occur when Fourier covariance transformation failed."
                )
                unc_array = None

            for i, name in enumerate(physical_param_names):
                idx = physical_start_idx + i
                param_val = result.parameters[idx]
                unc_val = (
                    unc_array[idx]
                    if unc_array is not None and idx < len(unc_array)
                    else 0.0
                )
                logger.info(f"    {name}: {param_val:.6g} +/- {unc_val:.6g}")

            contrast_vals = result.parameters[:n_angles]
            offset_vals = result.parameters[n_angles : 2 * n_angles]
            logger.info(
                f"  Mean scaling: contrast={np.nanmean(contrast_vals):.4f}, "
                f"offset={np.nanmean(offset_vals):.4f}"
            )
        else:
            param_names = _get_param_names(analysis_mode)
            n_display = min(len(param_names), n_params)
            logger.info("Fitted parameters:")
            for i in range(n_display):
                param_val = result.parameters[i]
                unc_val = (
                    result.uncertainties[i]
                    if hasattr(result, "uncertainties")
                    and result.uncertainties is not None
                    and i < len(result.uncertainties)
                    else 0.0
                )
                logger.info(f"  {param_names[i]}: {param_val:.6g} +/- {unc_val:.6g}")

    logger.info("=" * 60)


def _normalize_data_to_object(data: Any, config: Any, logger: Any) -> Any:
    """Normalize data dict to object with expected attributes for NLSQWrapper.

    Handles: key mapping (CLI→internal names), q extraction from wavevector_q_list,
    sigma generation, 2D→1D time vector extraction, L loading, dt loading.
    If data is already an object (not a dict), only validates sigma.
    """

    def _ensure_positive_sigma(obj: Any) -> None:
        if not hasattr(obj, "sigma"):
            return
        sigma_array = np.asarray(obj.sigma, dtype=np.float64)
        if not np.all(np.isfinite(sigma_array)):
            raise ValueError("sigma values must be finite")
        if np.any(sigma_array <= 0):
            raise ValueError("sigma values must be strictly positive")
        obj.sigma = sigma_array

    if isinstance(data, dict):

        class DataObject:
            pass

        data_obj = DataObject()

        # Map CLI data structure keys to NLSQWrapper expected names
        key_mapping = {
            "phi_angles_list": "phi",
            "c2_exp": "g2",
        }

        for key, value in data.items():
            mapped_key = key_mapping.get(key, key)
            setattr(data_obj, mapped_key, value)

        # Extract scalar q from wavevector_q_list if present
        if hasattr(data_obj, "wavevector_q_list"):
            q_list = np.atleast_1d(np.asarray(data_obj.wavevector_q_list))
            if q_list.size > 0:
                data_obj.q = float(q_list[0])
                logger.debug(f"Extracted q = {data_obj.q:.6f} from wavevector_q_list")

        # Generate default sigma (uncertainty) if missing
        if not hasattr(data_obj, "sigma") and hasattr(data_obj, "g2"):
            g2_array = np.asarray(data_obj.g2)  # type: ignore[attr-defined]
            data_obj.sigma = _DEFAULT_SIGMA * np.ones_like(g2_array)  # type: ignore[attr-defined]
            data_obj.sigma_is_default = True  # type: ignore[attr-defined]
            logger.debug(f"Generated default sigma: shape {data_obj.sigma.shape}")  # type: ignore[attr-defined]
        else:
            data_obj.sigma_is_default = False  # type: ignore[attr-defined]
        _ensure_positive_sigma(data_obj)

        # Extract 1D time vectors from 2D meshgrids if needed
        if hasattr(data_obj, "t1"):
            t1 = np.asarray(data_obj.t1)
            if t1.ndim == 2:
                data_obj.t1 = t1[:, 0]
                logger.debug(
                    f"Extracted 1D t1 vector from 2D meshgrid: {t1.shape} -> {data_obj.t1.shape}",
                )
            elif t1.ndim != 1:
                raise ValueError(f"t1 must be 1D or 2D array, got shape {t1.shape}")

        if hasattr(data_obj, "t2"):
            t2 = np.asarray(data_obj.t2)
            if t2.ndim == 2:
                data_obj.t2 = t2[0, :]
                logger.debug(
                    f"Extracted 1D t2 vector from 2D meshgrid: {t2.shape} -> {data_obj.t2.shape}",
                )
            elif t2.ndim != 1:
                raise ValueError(f"t2 must be 1D or 2D array, got shape {t2.shape}")

        # Get characteristic length L from config
        if not hasattr(data_obj, "L"):
            try:
                analyzer_params = config.config.get("analyzer_parameters", {})  # type: ignore[union-attr]
                geometry = analyzer_params.get("geometry", {})

                if "stator_rotor_gap" in geometry:
                    data_obj.L = float(geometry["stator_rotor_gap"])  # type: ignore[attr-defined]
                    logger.debug(
                        f"Using stator_rotor_gap L = {data_obj.L:.1f} AA (from config.analyzer_parameters.geometry)",  # type: ignore[attr-defined]
                    )
                else:
                    exp_config = config.config.get("experimental_data", {})  # type: ignore[union-attr]
                    exp_geometry = exp_config.get("geometry", {})

                    if "stator_rotor_gap" in exp_geometry:
                        data_obj.L = float(exp_geometry["stator_rotor_gap"])  # type: ignore[attr-defined]
                        logger.debug(
                            f"Using stator_rotor_gap L = {data_obj.L:.1f} AA (from config.experimental_data.geometry)",  # type: ignore[attr-defined]
                        )
                    elif "sample_detector_distance" in exp_config:
                        data_obj.L = float(exp_config["sample_detector_distance"])  # type: ignore[attr-defined]
                        logger.debug(
                            f"Using sample_detector_distance L = {data_obj.L:.1f} AA (from config.experimental_data)",  # type: ignore[attr-defined]
                        )
                    else:
                        data_obj.L = 2000000.0  # type: ignore[attr-defined]
                        logger.warning(
                            f"No L parameter found in config, using default L = {data_obj.L:.1f} AA (200 um, typical rheology-XPCS gap)",  # type: ignore[attr-defined]
                        )
            except (AttributeError, TypeError, ValueError) as e:
                data_obj.L = 2000000.0  # type: ignore[attr-defined]
                logger.warning(
                    f"Error reading L from config: {e}, using default L = {data_obj.L:.1f} AA (200 um)",  # type: ignore[attr-defined]
                )

        # Get time step dt from config if available
        if not hasattr(data_obj, "dt"):
            try:
                analyzer_params = config.config.get("analyzer_parameters", {})  # type: ignore[union-attr]
                dt_value = analyzer_params.get("dt")

                if dt_value is None:
                    exp_config = config.config.get("experimental_data", {})  # type: ignore[union-attr]
                    dt_value = exp_config.get("dt")

                if dt_value is not None:
                    data_obj.dt = float(dt_value)  # type: ignore[attr-defined]
                    logger.debug(f"Using time step dt = {data_obj.dt:.6f} s")  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Error reading dt from config: {e}")

        return data_obj
    else:
        _ensure_positive_sigma(data)
        return data


def _validate_data(data: dict[str, Any]) -> None:
    """Validate experimental data structure (CLI or Direct format).

    CLI format uses ``wavevector_q_list``/``phi_angles_list``/``c2_exp``.
    Direct format uses ``q``/``phi``/``g2``. Both are accepted.
    """
    has_q = "wavevector_q_list" in data or "q" in data
    has_phi = "phi_angles_list" in data or "phi" in data
    has_c2 = "c2_exp" in data or "g2" in data
    missing: list[str] = []
    if not has_q:
        missing.append("wavevector_q_list or q")
    if not has_phi:
        missing.append("phi_angles_list or phi")
    if not has_c2:
        missing.append("c2_exp or g2")
    for key in ("t1", "t2"):
        if key not in data:
            missing.append(key)
    if missing:
        raise ValueError(f"Missing required data key(s): {missing}")

    # Accept either key: the missing-key check above allows "c2_exp" OR "g2",
    # so reading data["c2_exp"] directly would KeyError on g2-format callers.
    c2 = data.get("c2_exp", data.get("g2"))
    if c2 is not None and np.asarray(c2).shape[0] == 0:
        raise ValueError("Empty experimental data")


def _get_analysis_mode(config: ConfigManager) -> AnalysisMode:
    """Determine analysis mode from configuration.

    Returns the typed :class:`AnalysisMode` (a ``StrEnum``, so it compares equal
    to its string value everywhere downstream). ``ConfigManager`` validates the
    mode at construction, so the lookup value is always a recognised member.
    """
    if hasattr(config, "config") and config.config:
        return AnalysisMode(config.config.get("analysis_mode", "static_isotropic"))
    return AnalysisMode("static_isotropic")


def _is_nlsq_diagnostics_enabled(config: ConfigManager | dict[str, Any]) -> bool:
    """Return True if optimization.nlsq.diagnostics.enabled is truthy."""

    config_dict: dict[str, Any] | None = None
    if hasattr(config, "config") and config.config:
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return False

    return bool(
        config_dict.get("optimization", {})
        .get("nlsq", {})
        .get("diagnostics", {})
        .get("enabled", False)
    )


def _extract_shear_transform_config(
    config: ConfigManager | dict[str, Any],
) -> dict[str, Any]:
    config_dict: dict[str, Any] | None = None
    if hasattr(config, "config") and config.config:
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return {}

    return (
        config_dict.get("optimization", {}).get("nlsq", {}).get("shear_transforms", {})
    )


def _load_initial_params_from_config(
    config: ConfigManager,
    analysis_mode: AnalysisMode,
    data: dict[str, Any] | None = None,
) -> tuple[dict[str, float] | None, dict[str, list[float]] | None]:
    """Load initial parameters from configuration file.

    Handles parameter name mapping between config format and code format.
    Estimates contrast/offset from experimental data if not provided in config.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager with initial_parameters section
    analysis_mode : str
        Analysis mode (static_isotropic or laminar_flow)
    data : dict, optional
        Experimental data used to estimate contrast/offset if not in config

    Returns
    -------
    dict or None
        Dictionary of initial parameters, or None if not found in config
    """
    if not hasattr(config, "config") or not config.config:
        return None, None

    config_dict = config.config
    if "initial_parameters" not in config_dict:
        return None, None

    init_params = config_dict["initial_parameters"]
    if "parameter_names" not in init_params or "values" not in init_params:
        logger.warning(
            "Initial parameters in config missing 'parameter_names' or 'values'",
        )
        return None, None

    names = init_params["parameter_names"]
    values = init_params["values"]

    if len(names) != len(values):
        logger.warning(
            f"Parameter name/value count mismatch: {len(names)} names, {len(values)} values",
        )
        return None, None

    # Map config parameter names to code parameter names
    NAME_MAP = {
        "gamma_dot_0": "gamma_dot_t0",
        "gamma_dot_offset": "gamma_dot_t_offset",
        "phi_0": "phi0",
        "D0": "D0",
        "alpha": "alpha",
        "D_offset": "D_offset",
        "beta": "beta",
    }

    # Build parameter dictionary with name mapping
    params = {}
    for name, value in zip(names, values, strict=False):
        mapped_name = NAME_MAP.get(name, name)
        params[mapped_name] = float(value)

    # Add scaling parameters if missing
    # (config typically only includes physical parameters)
    # FIX (Nov 14, 2025): Use physically reasonable defaults instead of data estimation
    # PROBLEM: Data estimation from diagonal-corrected g2 gives wrong values
    #   - Estimated: contrast~0.055, offset~1.003 (from percentile + max)
    #   - Actual fitted: contrast~0.26, offset~0.77 (from previous successful runs)
    #   - Mismatch causes optimization to get stuck in wrong parameter space
    # SOLUTION: Use typical XPCS values as defaults
    if "contrast" not in params or "offset" not in params:
        # Use typical homodyne XPCS values (empirically validated)
        contrast_default = 0.3  # Typical range [0.1, 0.5] for homodyne detection
        offset_default = 0.8  # Typical range [0.5, 1.0] for baseline

        if "contrast" not in params:
            params["contrast"] = contrast_default
            logger.info(
                f"Using default contrast={contrast_default:.3f} (typical homodyne XPCS)"
            )
        if "offset" not in params:
            params["offset"] = offset_default
            logger.info(
                f"Using default offset={offset_default:.3f} (typical homodyne XPCS)"
            )

    # Validate parameter count matches analysis mode
    expected_count = 5 if "static" in analysis_mode.lower() else 9
    if len(params) != expected_count:
        logger.warning(
            f"Parameter count mismatch for {analysis_mode}: "
            f"got {len(params)}, expected {expected_count}",
        )
        # Don't return None - let validation/clipping handle it

    per_angle_scaling: dict[str, list[float]] | None = None
    per_angle_cfg = init_params.get("per_angle_scaling")
    if isinstance(per_angle_cfg, dict):
        contrast_vals = per_angle_cfg.get("contrast")
        offset_vals = per_angle_cfg.get("offset")
        try:
            contrast_array = (
                [float(x) for x in contrast_vals]
                if isinstance(contrast_vals, (list, tuple))
                else None
            )
            offset_array = (
                [float(x) for x in offset_vals]
                if isinstance(offset_vals, (list, tuple))
                else None
            )
        except (TypeError, ValueError):
            contrast_array = offset_array = None

        if contrast_array and offset_array and len(contrast_array) == len(offset_array):
            per_angle_scaling = {
                "contrast": contrast_array,
                "offset": offset_array,
            }
        elif contrast_array or offset_array:
            logger.warning(
                "per_angle_scaling in initial_parameters must provide equal-length contrast/offset arrays; ignoring overrides",
            )

    logger.debug(f"Loaded {len(params)} parameters from config: {list(params.keys())}")
    return params, per_angle_scaling


def _estimate_contrast_offset_from_data(
    data: Any,
) -> tuple[float, float]:
    """Estimate contrast and offset from experimental g2 data.

    For XPCS correlation function: c₂(φ,t₁,t₂) = offset + contrast × [c₁(φ,t₁,t₂)]²

    Parameters
    ----------
    data : dict or object
        Experimental data with 'g2' or 'c2_exp' key/attribute containing correlation data

    Returns
    -------
    contrast : float
        Estimated contrast parameter (amplitude of correlations)
    offset : float
        Estimated offset parameter (baseline of g2)
    """
    # Extract g2 data (try multiple possible key names)
    # Note: Cannot use `or` operator with numpy arrays as it evaluates truth value
    # Support both dict-like and object-like data access
    if isinstance(data, dict):
        g2 = data.get("g2")
        if g2 is None:
            g2 = data.get("c2_exp")
    else:
        g2 = getattr(data, "g2", None)
        if g2 is None:
            g2 = getattr(data, "c2_exp", None)

    if g2 is None:
        logger.warning(
            "Could not estimate contrast/offset: no 'g2' or 'c2_exp' in data. "
            "Using generic defaults (0.5, 1.0)"
        )
        return 0.5, 1.0

    # Convert to numpy array if needed
    g2_array = np.asarray(g2)

    # Estimate offset from baseline (5th percentile to avoid outliers)
    offset_est = float(np.nanpercentile(g2_array, 5))

    # Estimate contrast from amplitude (max - baseline)
    # For c2 = offset + contrast * [c1]^2, max occurs at c1^2=1
    max_g2 = float(np.nanmax(g2_array))
    contrast_est = max_g2 - offset_est

    # Sanity checks
    if contrast_est <= 0 or offset_est <= 0:
        logger.warning(
            f"Invalid estimated contrast={contrast_est:.3f} or offset={offset_est:.3f}. "
            f"Using generic defaults (0.5, 1.0)"
        )
        return 0.5, 1.0

    logger.info(
        f"Estimated scaling parameters from data: "
        f"contrast={contrast_est:.4f}, offset={offset_est:.4f} "
        f"(g2 range: [{np.nanmin(g2_array):.4f}, {np.nanmax(g2_array):.4f}])"
    )

    return contrast_est, offset_est


def _get_default_initial_params(analysis_mode: AnalysisMode) -> dict[str, float]:
    """Get default initial parameters for analysis mode.

    NOTE: This function provides generic physical parameter defaults.
    Contrast and offset should be estimated from experimental data
    using _estimate_contrast_offset_from_data() before calling this function.
    """
    # Static isotropic mode (3 parameters)
    if "static" in analysis_mode.lower():
        return {
            "contrast": 0.5,  # Generic default - should be replaced with data estimate
            "offset": 1.0,  # Generic default - should be replaced with data estimate
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
        }
    # Laminar flow mode (7 parameters)
    else:
        return {
            "contrast": 0.5,  # Generic default - should be replaced with data estimate
            "offset": 1.0,  # Generic default - should be replaced with data estimate
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.0,
            "gamma_dot_t_offset": 0.0,
            "phi0": 0.0,
        }


def _get_parameter_bounds(
    analysis_mode: AnalysisMode,
    param_space: ParameterSpace,
) -> dict[str, tuple[float, float]]:
    """Get parameter bounds for analysis mode."""
    bounds = {
        "contrast": param_space.contrast_bounds,
        "offset": param_space.offset_bounds,
        "D0": param_space.D0_bounds,
        "alpha": param_space.alpha_bounds,
        "D_offset": param_space.D_offset_bounds,
    }

    if "laminar" in analysis_mode.lower():
        bounds.update(
            {
                "gamma_dot_t0": param_space.gamma_dot_t0_bounds,
                "beta": param_space.beta_bounds,
                "gamma_dot_t_offset": param_space.gamma_dot_t_offset_bounds,
                "phi0": param_space.phi0_bounds,
            },
        )

    return bounds


def _get_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get parameter names for a given analysis mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode (e.g., 'static_anisotropic', 'static_isotropic', 'laminar_flow')

    Returns
    -------
    list[str]
        List of parameter names in the order they appear in the parameter array
    """
    if "static" in analysis_mode.lower():
        return ["contrast", "offset", "D0", "alpha", "D_offset"]
    else:
        return [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]


def _get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get physical parameter names for a given analysis mode.

    Unlike _get_param_names, this excludes scaling parameters (contrast, offset)
    and returns only the physical parameters.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode (e.g., 'static_anisotropic', 'static_isotropic', 'laminar_flow')

    Returns
    -------
    list[str]
        List of physical parameter names
    """
    if "static" in analysis_mode.lower():
        return ["D0", "alpha", "D_offset"]
    else:
        return [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]


def _params_to_array(params: dict[str, float], analysis_mode: AnalysisMode) -> jnp.ndarray:
    """Convert parameter dictionary to array."""
    if "static" in analysis_mode.lower():
        return jnp.array(
            [
                params["contrast"],
                params["offset"],
                params["D0"],
                params["alpha"],
                params["D_offset"],
            ],
        )
    else:
        return jnp.array(
            [
                params["contrast"],
                params["offset"],
                params["D0"],
                params["alpha"],
                params["D_offset"],
                params["gamma_dot_t0"],
                params["beta"],
                params["gamma_dot_t_offset"],
                params["phi0"],
            ],
        )


def _array_to_params(array: jnp.ndarray, analysis_mode: AnalysisMode) -> dict[str, Any]:
    """Convert parameter array to dictionary.

    Returns JAX arrays as-is to avoid tracing issues.
    Conversion to Python floats should only happen at the final step.
    """
    if "static" in analysis_mode.lower():
        return {
            "contrast": array[0],
            "offset": array[1],
            "D0": array[2],
            "alpha": array[3],
            "D_offset": array[4],
        }
    else:
        return {
            "contrast": array[0],
            "offset": array[1],
            "D0": array[2],
            "alpha": array[3],
            "D_offset": array[4],
            "gamma_dot_t0": array[5],
            "beta": array[6],
            "gamma_dot_t_offset": array[7],
            "phi0": array[8],
        }


def _bounds_to_arrays(
    bounds: dict[str, tuple[float, float]],
    analysis_mode: AnalysisMode,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert bounds dictionary to lower/upper bound arrays."""
    if "static" in analysis_mode.lower():
        param_order = ["contrast", "offset", "D0", "alpha", "D_offset"]
    else:
        param_order = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

    lower = jnp.array([bounds[key][0] for key in param_order])
    upper = jnp.array([bounds[key][1] for key in param_order])

    return lower, upper


def _get_optimizer_config(config: ConfigManager) -> dict[str, Any]:
    """Get NLSQ optimizer configuration from config."""
    default_config = {
        "method": "levenberg_marquardt",
        "max_iterations": 10000,
        "tolerance": 1e-8,
        "verbose": False,
    }

    if hasattr(config, "config") and config.config:
        lsq_config = config.config.get("optimization", {}).get("lsq", {})
        default_config.update(lsq_config)

    return default_config


def _check_convergence(result: Any) -> bool:
    """Check if NLSQ optimization converged."""
    return getattr(result, "success", False)


def _get_optimization_message(result: Any) -> str:
    """Get optimization status message from NLSQ result."""
    if hasattr(result, "result_flag") and result.result_flag is not None:
        return str(result.result_flag)
    elif result.success:
        return "Optimization converged successfully"
    else:
        return "Optimization failed to converge"


def _get_iteration_count(result: Any) -> int:
    """Get iteration count from NLSQ result."""
    if hasattr(result, "stats") and "num_steps" in result.stats:
        return int(result.stats["num_steps"])
    return 0


# =============================================================================
# Multi-Start Optimization Entry Point (v2.6.0)
# =============================================================================


class _SingleFitWorker:
    """Picklable worker class for parallel multi-start optimization.

    This class encapsulates the state needed for single NLSQ fits,
    making it suitable for use with ProcessPoolExecutor. The key is
    storing only picklable data (dict, str, bool) rather than complex
    objects like ConfigManager.

    The ConfigManager is reconstructed in the worker process using
    config_override, which avoids pickle issues with loggers, file
    handles, and cached objects.

    Attributes
    ----------
    config_dict : dict
        Raw configuration dictionary (picklable).
    config_file : str
        Path to the original config file.
    per_angle_scaling : bool
        Whether to use per-angle contrast/offset scaling.
    analysis_mode : str
        Analysis mode ("laminar_flow", "static_anisotropic", or "static_isotropic").
    """

    def __init__(
        self,
        config: Any,  # ConfigManager
        per_angle_scaling: bool,
        analysis_mode: AnalysisMode,
    ) -> None:
        # Extract picklable data from ConfigManager
        # This avoids pickle issues with loggers, file handles, etc.
        self.config_dict = config.config.copy()  # Raw dict is picklable
        self.config_file = config.config_file  # str is picklable
        self.per_angle_scaling = per_angle_scaling
        self.analysis_mode = analysis_mode

    def __call__(
        self, fit_data: dict[str, Any], start_params: np.ndarray
    ) -> SingleStartResult:
        """Run a single NLSQ fit.

        Parameters
        ----------
        fit_data : dict
            XPCS data dictionary.
        start_params : np.ndarray
            Starting parameter values as array.

        Returns
        -------
        SingleStartResult
            Result from this optimization run.
        """
        import time

        start_time = time.perf_counter()

        # Reconstruct ConfigManager in the worker process
        # Using config_override avoids file I/O and is faster
        from xpcsjax.config.manager import ConfigManager

        config = ConfigManager(
            config_file=self.config_file,
            config_override=self.config_dict,
        )

        # Convert array to dict
        param_names = _get_param_names(self.analysis_mode)
        params_dict = {
            name: float(start_params[i]) for i, name in enumerate(param_names)
        }

        try:
            result = fit_nlsq_jax(
                data=fit_data,
                config=config,
                initial_params=params_dict,
                per_angle_scaling=self.per_angle_scaling,
                _skip_global_selection=True,  # Prevent recursion from multistart
            )

            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=np.array(result.parameters),
                chi_squared=result.chi_squared,
                reduced_chi_squared=result.reduced_chi_squared,
                success=result.success,
                status=0,
                message=result.message,
                n_iterations=result.iterations,
                n_fev=result.iterations,
                wall_time=time.perf_counter() - start_time,
                covariance=result.covariance if hasattr(result, "covariance") else None,
            )
        except (ValueError, RuntimeError, TypeError, OSError) as e:
            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=start_params,
                chi_squared=np.inf,
                success=False,
                message=str(e),
                wall_time=time.perf_counter() - start_time,
            )


@log_performance(threshold=1.0)
def fit_nlsq_multistart(
    data: dict[str, Any],
    config: ConfigManager,
    initial_params: dict[str, float] | None = None,
    per_angle_scaling: bool = True,
) -> MultiStartResult:
    """Multi-start NLSQ optimization with Latin Hypercube Sampling.

    This function explores the parameter space using Latin Hypercube Sampling
    to avoid local minima. FULL strategy is always used regardless of dataset
    size - numerical precision and reproducibility take priority over speed.

    NOTE: Subsampling is explicitly NOT supported per project requirements.

    Parameters
    ----------
    data : dict[str, Any]
        XPCS experimental data with keys:
        - wavevector_q_list: Q-vector values
        - phi_angles_list: Azimuthal angles
        - t1, t2: Time coordinates
        - c2_exp: Experimental g2 correlation data
        - sigma (optional): Error weights
    config : ConfigManager
        Configuration manager with optimization.nlsq.multi_start settings.
    initial_params : dict[str, float], optional
        Initial parameter guess. If provided, included as one of the starts.
    per_angle_scaling : bool
        Whether to use per-angle contrast/offset scaling. Default: True.

    Returns
    -------
    MultiStartResult
        Aggregated results including:
        - best: Best result by chi-squared
        - all_results: All optimization attempts
        - strategy_used: "full" (only supported strategy)
        - n_unique_basins: Number of distinct local minima found
        - degeneracy_detected: Whether parameter degeneracy was detected

    Raises
    ------
    ImportError
        If multi-start module is not available.
    ValueError
        If multi-start is not enabled in configuration.

    Examples
    --------
    >>> config = ConfigManager("config.yaml")
    >>> # Ensure multi_start.enable: true in config
    >>> result = fit_nlsq_multistart(data, config)
    >>> print(f"Best chi2: {result.best.chi_squared:.4g}")
    >>> print(f"Strategy used: {result.strategy_used}")
    >>> if result.degeneracy_detected:
    ...     print(f"Warning: {result.n_unique_basins} distinct basins found")
    """
    if not HAS_MULTISTART:
        raise ImportError(
            "Multi-start optimization requires xpcsjax.optimization.nlsq.multistart. "
            "Ensure the multistart module is properly installed."
        )

    if not HAS_NLSQ_WRAPPER:
        raise ImportError("NLSQWrapper is required for multi-start optimization")

    # Extract multi-start config
    nlsq_dict = config.config.get("optimization", {}).get("nlsq", {})
    multi_start_dict = nlsq_dict.get("multi_start", {})

    if not multi_start_dict.get("enable", False):
        raise ValueError(
            "Multi-start optimization is not enabled. "
            "Set optimization.nlsq.multi_start.enable: true in config."
        )

    from xpcsjax.optimization.nlsq.config import NLSQConfig

    nlsq_config = NLSQConfig.from_dict(nlsq_dict)
    ms_config = MultiStartConfig.from_nlsq_config(nlsq_config)

    # Validate data
    _validate_data(data)

    # Get analysis mode and parameter setup
    analysis_mode = _get_analysis_mode(config)
    param_space = ParameterSpace() if HAS_CORE_MODULES else None

    # Load initial_params from config if not provided as argument
    # CRITICAL: User's initial parameters must be included in multi-start to ensure
    # the known-good solution is explored, especially for laminar_flow mode where
    # LHS starting points may not converge to the correct physical parameters
    if initial_params is None:
        initial_params, _ = _load_initial_params_from_config(
            config, analysis_mode, data
        )
        if initial_params is not None:
            logger.info(
                "Loaded initial parameters from configuration for multi-start optimization"
            )

    # Get bounds
    if HAS_PARAMETER_MANAGER:
        param_manager = ParameterManager(config.config, analysis_mode=analysis_mode)
        bounds_list = param_manager.get_parameter_bounds()
        bounds_dict = {b["name"]: (b["min"], b["max"]) for b in bounds_list}
    else:
        bounds_dict = _get_parameter_bounds(analysis_mode, param_space)

    lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)
    bounds_array = np.column_stack([lower_bounds, upper_bounds])

    # Create picklable single fit worker (replaces closure-based function)
    # This enables parallel execution with ProcessPoolExecutor
    single_fit_func = _SingleFitWorker(
        config=config,
        per_angle_scaling=per_angle_scaling,
        analysis_mode=analysis_mode,
    )

    # Create cost function for screening
    def cost_func(params: np.ndarray) -> float:
        """Quick cost evaluation for screening.

        Uses a heuristic based on distance from bounds center rather than
        full residual evaluation for efficiency during screening phase.
        """
        try:
            # Check if params are at bounds (return large cost)
            for i, (low, high) in enumerate(
                zip(lower_bounds, upper_bounds, strict=True)
            ):
                if params[i] <= low or params[i] >= high:
                    return 1e20

            # Approximate cost from parameter distance to center
            center = (lower_bounds + upper_bounds) / 2
            scale = upper_bounds - lower_bounds
            normalized_dist = np.sum(((params - center) / scale) ** 2)
            return normalized_dist
        except (ValueError, IndexError, TypeError, FloatingPointError):
            return 1e20

    # Prepare custom_starts with user's initial parameters (if provided)
    custom_starts = None
    if initial_params is not None:
        # Convert initial_params dict to array in correct order
        param_names = _get_param_names(analysis_mode)
        initial_array = np.array([initial_params[name] for name in param_names])
        custom_starts = [initial_array.tolist()]
        logger.info("Including user-specified initial parameters as custom start point")

    # Run multi-start optimization
    logger.info(
        f"Starting multi-start NLSQ with {ms_config.n_starts} starts, "
        f"strategy will be auto-selected based on dataset size"
    )

    result = run_multistart_nlsq(
        data=data,
        bounds=bounds_array,
        config=ms_config,
        single_fit_func=single_fit_func,
        cost_func=cost_func if ms_config.use_screening else None,
        custom_starts=custom_starts,
    )

    logger.info(
        f"Multi-start complete: strategy={result.strategy_used}, "
        f"best chi2={result.best.chi_squared:.4g}, "
        f"basins={result.n_unique_basins}"
    )

    return result


@log_performance(threshold=1.0)
def fit_nlsq_cmaes(
    data: dict[str, Any],
    config: ConfigManager,
    initial_params: dict[str, float] | None = None,
    per_angle_scaling: bool = True,
) -> OptimizationResult:
    """CMA-ES global optimization for multi-scale parameter problems.

    Uses NLSQ's CMAESOptimizer with evosax backend for global optimization.
    Particularly beneficial for laminar_flow mode where parameters have
    vastly different scales (e.g., D₀ ~ 1e4 vs γ̇₀ ~ 1e-3, scale ratio > 1e7).

    Features:
    - Covariance Matrix Adaptation for multi-scale parameters
    - BIPOP restart strategy for robust convergence
    - Memory batching/streaming for large datasets
    - Optional L-M refinement of CMA-ES solution

    Parameters
    ----------
    data : dict[str, Any]
        XPCS experimental data (same format as fit_nlsq_jax).
    config : ConfigManager
        Configuration manager with optimization.nlsq.cmaes settings.
    initial_params : dict[str, float], optional
        Initial parameter guess. Used as CMA-ES starting point.
    per_angle_scaling : bool
        Whether to use per-angle contrast/offset scaling. Default: True.

    Returns
    -------
    OptimizationResult
        Optimization result with parameters, uncertainties, and diagnostics.

    Raises
    ------
    ImportError
        If CMA-ES is not available (requires NLSQ 0.6.4+ with evosax).
    ValueError
        If CMA-ES is not enabled in configuration.

    Examples
    --------
    >>> config = ConfigManager("config.yaml")
    >>> # Ensure cmaes.enable: true in config
    >>> result = fit_nlsq_cmaes(data, config)
    >>> print(f"Chi2: {result.chi_squared:.4e}")
    >>> print(f"Method: {result.device_info['method']}")
    """
    import time

    if not HAS_CMAES:
        raise ImportError(
            "CMA-ES requires NLSQ 0.6.4+ with evosax backend. "
            "Install with: pip install nlsq[evosax]"
        )

    # Extract CMA-ES config
    nlsq_dict = config.config.get("optimization", {}).get("nlsq", {})
    cmaes_dict = nlsq_dict.get("cmaes", {})

    if not cmaes_dict.get("enable", False):
        raise ValueError(
            "CMA-ES optimization is not enabled. "
            "Set optimization.nlsq.cmaes.enable: true in config."
        )

    from xpcsjax.optimization.nlsq.config import NLSQConfig

    nlsq_config = NLSQConfig.from_dict(nlsq_dict)

    # Validate data
    _validate_data(data)

    # Get analysis mode and parameter setup
    analysis_mode = _get_analysis_mode(config)
    param_space = ParameterSpace() if HAS_CORE_MODULES else None

    logger.info("=" * 60)
    logger.info("CMA-ES GLOBAL OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Analysis mode: {analysis_mode}")
    logger.info(f"Preset: {nlsq_config.cmaes_preset}")

    # Set up initial parameters
    if initial_params is None:
        initial_params, _ = _load_initial_params_from_config(
            config, analysis_mode, data
        )
        if initial_params is None:
            initial_params = _get_default_initial_params(analysis_mode)
            logger.info("Using default initial parameters")
        else:
            logger.info("Using initial parameters from configuration")

    # Convert initial params to array
    x0 = _params_to_array(initial_params, analysis_mode)

    # Get bounds
    if HAS_PARAMETER_MANAGER:
        param_manager = ParameterManager(config.config, analysis_mode=analysis_mode)
        bounds_list = param_manager.get_parameter_bounds()
        bounds_dict = {b["name"]: (b["min"], b["max"]) for b in bounds_list}
    else:
        bounds_dict = _get_parameter_bounds(analysis_mode, param_space)

    lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)
    bounds = (lower_bounds, upper_bounds)

    # Create CMA-ES wrapper config
    cmaes_config = CMAESWrapperConfig.from_nlsq_config(nlsq_config)

    # Create CMA-ES wrapper and check if we should use it
    wrapper = CMAESWrapper(cmaes_config)

    if nlsq_config.cmaes_auto_select:
        if not wrapper.should_use_cmaes(bounds, nlsq_config.cmaes_scale_threshold):
            # Scale ratio too low for CMA-ES, check fallback options
            # Fall back to multi-start if enabled and available, otherwise local optimization
            multi_start_dict = nlsq_dict.get("multi_start", {})
            if multi_start_dict.get("enable", False) and HAS_MULTISTART:
                logger.info(
                    f"[CMA-ES] Scale ratio < {nlsq_config.cmaes_scale_threshold}, "
                    "falling back to multi-start optimization"
                )
                ms_result = fit_nlsq_multistart(
                    data=data,
                    config=config,
                    initial_params=initial_params,
                    per_angle_scaling=per_angle_scaling,
                )
                return ms_result.to_optimization_result()
            else:
                logger.info(
                    f"[CMA-ES] Scale ratio < {nlsq_config.cmaes_scale_threshold}, "
                    "falling back to local NLSQ optimization"
                )
                # Use _skip_global_selection=True to avoid infinite loop
                return fit_nlsq_jax(
                    data=data,
                    config=config,
                    initial_params=initial_params,
                    per_angle_scaling=per_angle_scaling,
                    _skip_global_selection=True,
                )

    # Prepare data arrays for CMA-ES
    # Need to build model function and flatten data
    start_time = time.time()

    try:
        # Use adapter's model building infrastructure
        from xpcsjax.optimization.nlsq.adapter import get_or_create_model

        # Prepare phi angles
        phi_key = "phi_angles_list" if "phi_angles_list" in data else "phi"
        phi_angles = np.asarray(data[phi_key])
        n_phi = len(phi_angles)

        # Get q value
        if "wavevector_q_list" in data:
            q = float(np.asarray(data["wavevector_q_list"])[0])
        else:
            q = float(data.get("q", 0.01))

        # Get model and function
        model, model_func, cache_hit = get_or_create_model(
            analysis_mode=analysis_mode,
            phi_angles=phi_angles,
            q=q,
            per_angle_scaling=per_angle_scaling,
            enable_jit=True,
        )

        logger.debug(f"Model cache {'hit' if cache_hit else 'miss'}")

        # Prepare data arrays
        t1_key = "t1"
        t2_key = "t2"
        g2_key = "c2_exp" if "c2_exp" in data else "g2"

        t1 = np.asarray(data[t1_key])
        t2 = np.asarray(data[t2_key])
        g2 = np.asarray(data[g2_key])

        # Handle 2D meshgrids
        if t1.ndim == 2:
            t1 = t1[:, 0]
        if t2.ndim == 2:
            t2 = t2[0, :]

        # Get sigma (uncertainty)
        _sigma_is_default = "sigma" not in data
        if not _sigma_is_default:
            sigma = np.asarray(data["sigma"])
        else:
            sigma = _DEFAULT_SIGMA * np.ones_like(g2)

        # Flatten data for CMA-ES
        # Flatten g2 and sigma
        ydata = g2.flatten()
        sigma_flat = sigma.flatten()

        n_data = len(ydata)
        logger.info(f"Data points: {n_data:,}")

        # Number of physical parameters
        n_physical = len(_get_physical_param_names(analysis_mode))

        # ==========================================================================
        # ANTI-DEGENERACY INTEGRATION (v2.18.0+)
        # ==========================================================================
        # For laminar_flow mode with per-angle scaling, the parameter space can be
        # degenerate: per-angle contrast/offset can absorb shear signals.
        #
        # Two modes:
        # - auto_averaged: Compute N quantile estimates, AVERAGE to 1 contrast + 1 offset,
        #                  OPTIMIZE these 2 along with 7 physical = 9 params
        # - fixed_constant: Compute N quantile estimates, use per-angle values DIRECTLY
        #                   as FIXED scaling, OPTIMIZE only 7 physical params
        # ==========================================================================
        use_constant_mode = False
        use_fixed_scaling = False
        use_averaged_scaling = False
        ad_controller = None
        is_laminar_flow = analysis_mode == "laminar_flow"
        # L1-L4 (Fourier reparam / hierarchical / regularization / gradient
        # monitoring) apply to ALL homodyne modes with per-angle scaling; only L5
        # (shear weighting) stays gated to laminar_flow. The controller's
        # _LAYER_GATES suppresses L5 internally for non-laminar modes, and the L5
        # application block below is independently is_laminar_flow-gated, so
        # constructing the controller for static modes is safe.
        is_homodyne = analysis_mode in {
            "static_anisotropic",
            "static_isotropic",
            "laminar_flow",
        }
        fixed_contrast_per_angle = None
        fixed_offset_per_angle = None

        if HAS_ANTI_DEGENERACY and per_angle_scaling and is_homodyne:
            # Load anti-degeneracy config
            anti_degeneracy_config = nlsq_dict.get("anti_degeneracy", {})

            if anti_degeneracy_config:
                phi_unique_rad = np.deg2rad(phi_angles)
                ad_controller = AntiDegeneracyController.from_config(
                    config_dict=anti_degeneracy_config,
                    n_phi=n_phi,
                    phi_angles=phi_unique_rad,
                    n_physical=n_physical,
                    per_angle_scaling=per_angle_scaling,
                    is_laminar_flow=is_laminar_flow,
                    analysis_mode=analysis_mode,
                )

                if ad_controller.is_enabled and ad_controller.use_constant:
                    use_constant_mode = True
                    # v2.18.0: Distinguish between fixed_constant and auto_averaged
                    use_fixed_scaling = ad_controller.use_fixed_scaling
                    use_averaged_scaling = ad_controller.use_averaged_scaling

                    if use_fixed_scaling:
                        logger.info("=" * 60)
                        logger.info(
                            "ANTI-DEGENERACY: Enabled for CMA-ES (Fixed Constant Mode)"
                        )
                        logger.info(
                            f"  Quantile estimation: N={n_phi} contrast + N={n_phi} offset values"
                        )
                        logger.info("  Per-angle values: FIXED (not optimized)")
                        logger.info("  Total parameters: 7 physical only")
                        logger.info("=" * 60)
                    else:
                        logger.info("=" * 60)
                        logger.info(
                            "ANTI-DEGENERACY: Enabled for CMA-ES (Auto Averaged Mode)"
                        )
                        logger.info(
                            f"  Quantile estimation: N={n_phi} contrast + N={n_phi} offset values"
                        )
                        logger.info("  Averaged to: 1 contrast + 1 offset (OPTIMIZED)")
                        logger.info(
                            "  Total parameters: 7 physical + 2 averaged scaling = 9"
                        )
                        logger.info("=" * 60)

        # Handle parameter expansion based on anti-degeneracy mode
        if per_angle_scaling and not use_constant_mode:
            # Standard behavior: expand to per-angle parameters (13 params for n_phi=3)
            if len(x0) == 2 + n_physical:
                from xpcsjax.optimization.nlsq.data_prep import (
                    expand_per_angle_parameters,
                )

                expanded = expand_per_angle_parameters(
                    x0,
                    bounds,
                    n_phi,
                    n_physical,
                    logger=logger,
                )
                x0 = expanded.params
                bounds = expanded.bounds
                lower_bounds, upper_bounds = bounds
            effective_per_angle_scaling = True
        elif use_constant_mode:
            # CONSTANT MODE (v2.18.0+): Compute per-angle scaling from quantiles
            # - use_fixed_scaling: per-angle values FIXED, optimize 7 physical only
            # - use_averaged_scaling: average to 2 values, optimize 9 params
            from xpcsjax.optimization.nlsq.parameter_utils import (
                compute_quantile_per_angle_scaling,
            )

            logger.info("=" * 60)
            logger.info("CONSTANT MODE: Computing per-angle scaling from quantiles")
            logger.info("=" * 60)

            # Create stratified data structure for quantile computation
            # Build flat arrays for all phi angles
            t1_mesh_temp, t2_mesh_temp = np.meshgrid(t1, t2, indexing="ij")
            n_time_points_temp = t1_mesh_temp.size

            # Build arrays with phi information
            g2_flat_all = []
            t1_flat_all = []
            t2_flat_all = []
            phi_flat_all = []

            for i_phi, phi_val in enumerate(phi_angles):
                # g2 shape is (n_phi, n_t1, n_t2)
                g2_slice = g2[i_phi].flatten()
                g2_flat_all.append(g2_slice)
                t1_flat_all.append(t1_mesh_temp.flatten())
                t2_flat_all.append(t2_mesh_temp.flatten())
                phi_flat_all.append(np.full(n_time_points_temp, phi_val))

            # Create simple data container for quantile estimation
            class SimpleStratifiedData:
                def __init__(
                    self, g2_flat: Any, phi_flat: Any, t1_flat: Any, t2_flat: Any
                ) -> None:
                    self.g2_flat = g2_flat
                    self.phi_flat = phi_flat
                    self.t1_flat = t1_flat
                    self.t2_flat = t2_flat

            stratified_for_quantile = SimpleStratifiedData(
                g2_flat=np.concatenate(g2_flat_all),
                phi_flat=np.concatenate(phi_flat_all),
                t1_flat=np.concatenate(t1_flat_all),
                t2_flat=np.concatenate(t2_flat_all),
            )

            # Get contrast/offset bounds from initial bounds
            contrast_bounds = (float(lower_bounds[0]), float(upper_bounds[0]))
            offset_bounds = (float(lower_bounds[1]), float(upper_bounds[1]))

            # Compute per-angle scaling from quantiles
            fixed_contrast_per_angle, fixed_offset_per_angle = (
                compute_quantile_per_angle_scaling(
                    stratified_data=stratified_for_quantile,
                    contrast_bounds=contrast_bounds,
                    offset_bounds=offset_bounds,
                    logger=logger,
                )
            )

            logger.info(
                f"Per-angle scaling computed:\n"
                f"  Contrast: mean={np.nanmean(fixed_contrast_per_angle):.4f}, "
                f"range=[{np.nanmin(fixed_contrast_per_angle):.4f}, {np.nanmax(fixed_contrast_per_angle):.4f}]\n"
                f"  Offset: mean={np.nanmean(fixed_offset_per_angle):.4f}, "
                f"range=[{np.nanmin(fixed_offset_per_angle):.4f}, {np.nanmax(fixed_offset_per_angle):.4f}]"
            )

            if use_fixed_scaling:
                # FIXED CONSTANT MODE: Use per-angle values DIRECTLY as FIXED
                # Optimize only 7 physical parameters
                logger.info("Fixed constant mode: per-angle scaling will be FIXED")

                # Extract physical parameters only from x0
                if len(x0) == 2 + n_physical:
                    # [contrast, offset, physical] format - extract physical only
                    physical_params = x0[2:]
                    x0 = physical_params.copy()
                    logger.info(f"Reduced to physical params only: {len(x0)} params")
                elif len(x0) == 2 * n_phi + n_physical:
                    # Per-angle format - extract physical only
                    physical_params = x0[2 * n_phi :]
                    x0 = physical_params.copy()
                    logger.info(f"Reduced per-angle to physical only: {len(x0)} params")
                elif len(x0) == n_physical:
                    # Already physical only
                    logger.info(f"Already physical params only: {len(x0)} params")

                # Update bounds for 7-parameter format: [*physical]
                if len(lower_bounds) == 2 + n_physical:
                    lower_bounds = lower_bounds[2:]
                    upper_bounds = upper_bounds[2:]
                elif len(lower_bounds) == 2 * n_phi + n_physical:
                    lower_bounds = lower_bounds[2 * n_phi :]
                    upper_bounds = upper_bounds[2 * n_phi :]

                bounds = (lower_bounds, upper_bounds)
                logger.info(
                    f"CMA-ES using fixed constant mode: {len(x0)} parameters (7 physical only)"
                )

            else:
                # AUTO AVERAGED MODE: Average to 2 values, optimize 9 params
                avg_contrast = float(np.nanmean(fixed_contrast_per_angle))
                avg_offset = float(np.nanmean(fixed_offset_per_angle))

                logger.info(
                    f"Auto averaged mode: scaling averaged to contrast={avg_contrast:.4f}, offset={avg_offset:.4f}"
                )

                # Build 9-parameter initial guess: [contrast_avg, offset_avg, *physical]
                if len(x0) == 2 + n_physical:
                    # Already in [contrast, offset, physical] format
                    physical_params = x0[2:]
                    x0 = np.concatenate([[avg_contrast], [avg_offset], physical_params])
                    logger.info(
                        f"Using averaged quantile estimates for scaling: contrast={avg_contrast:.4f}, offset={avg_offset:.4f}"
                    )
                elif len(x0) == 2 * n_phi + n_physical:
                    # Per-angle format: reduce to [contrast_avg, offset_avg, physical]
                    physical_params = x0[2 * n_phi :]
                    x0 = np.concatenate([[avg_contrast], [avg_offset], physical_params])
                    logger.info(f"Reduced per-angle to averaged: {len(x0)} params")
                elif len(x0) == n_physical:
                    # Physical only: prepend averaged scaling
                    x0 = np.concatenate([[avg_contrast], [avg_offset], x0])
                    logger.info(f"Prepended averaged scaling: {len(x0)} params")

                # Update bounds for 9-parameter format: [contrast, offset, *physical]
                if len(lower_bounds) == 2 + n_physical:
                    # Already correct format
                    pass
                elif len(lower_bounds) == 2 * n_phi + n_physical:
                    # Per-angle format: reduce to single scaling bounds
                    lower_bounds = np.concatenate(
                        [
                            [lower_bounds[0]],  # Single contrast bound
                            [lower_bounds[n_phi]],  # Single offset bound
                            lower_bounds[2 * n_phi :],  # Physical bounds
                        ]
                    )
                    upper_bounds = np.concatenate(
                        [
                            [upper_bounds[0]],
                            [upper_bounds[n_phi]],
                            upper_bounds[2 * n_phi :],
                        ]
                    )
                bounds = (lower_bounds, upper_bounds)
                logger.info(
                    f"CMA-ES using auto averaged mode: {len(x0)} parameters (9 = 7 physical + 2 averaged)"
                )

            effective_per_angle_scaling = False
        else:
            effective_per_angle_scaling = False

        # Create wrapped model function for CMA-ES
        # IMPORTANT: This function must be JAX-traceable for CMA-ES JIT compilation
        # Use JAX operations throughout to avoid TracerArrayConversionError
        t1_mesh, t2_mesh = np.meshgrid(t1, t2, indexing="ij")

        # Diagonal filtering for CMA-ES (v2.19.0: configurable)
        # At t1==t2, the experimental g2 has a diagonal correction applied at load
        # time but the CMA-ES theory function does not apply this correction,
        # creating systematic residual mismatch. Two approaches:
        # - "remove" (default): Filter diagonal points from data (clean, ~0.1% data loss)
        # - "none": Keep all points (matches stratified LS point count, but residual
        #   mismatch at diagonal persists for theory values)
        diagonal_mode = cmaes_dict.get("diagonal_filtering", "remove")

        idx1, idx2 = np.meshgrid(np.arange(len(t1)), np.arange(len(t2)), indexing="ij")
        if diagonal_mode == "remove":
            non_diag_single = (idx1 != idx2).flatten()
            non_diag_all = np.tile(non_diag_single, n_phi)

            n_before_diag = len(ydata)
            ydata = ydata[non_diag_all]
            sigma_flat = sigma_flat[non_diag_all]
            n_data = len(ydata)
            n_diag_removed = n_before_diag - n_data
            logger.info(
                f"Diagonal filtering: removed {n_diag_removed:,} points "
                f"({100 * n_diag_removed / n_before_diag:.1f}%)"
            )
        else:
            non_diag_single = np.ones(idx1.size, dtype=bool)
            non_diag_all = np.ones(len(ydata), dtype=bool)
            logger.info(
                f"Diagonal filtering: disabled (mode={diagonal_mode!r}), "
                f"keeping all {len(ydata):,} points"
            )

        # Pre-compute data arrays as JAX arrays for efficiency (post diagonal filter)
        t1_flat_np = t1_mesh.flatten()[non_diag_single]
        t2_flat_np = t2_mesh.flatten()[non_diag_single]
        n_time_points = len(t1_flat_np)
        t1_flat = jnp.array(t1_flat_np)
        t2_flat = jnp.array(t2_flat_np)

        # Build phi indices: each time grid repeats for all phi angles
        # phi_indices[k] gives which phi angle point k belongs to
        phi_indices = jnp.repeat(jnp.arange(n_phi), n_time_points)

        # Build phi values for each point
        phi_values = jnp.array(phi_angles)[phi_indices]

        # Tile time arrays for all phi angles
        t1_all = jnp.tile(t1_flat, n_phi)
        t2_all = jnp.tile(t2_flat, n_phi)

        # Get dt and L from config if available
        config_dict = config.config if hasattr(config, "config") else config
        dt_val = config_dict.get("analyzer_parameters", {}).get("dt", 0.1)

        # Get L from config (stator_rotor_gap or default 200 µm)
        analyzer_params = config_dict.get("analyzer_parameters", {})
        geometry = analyzer_params.get("geometry", {})
        L_val = float(geometry.get("stator_rotor_gap", 2000000.0))

        # Pre-compute physics factors (outside the traced function for efficiency)
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_val
        sinc_prefactor = 0.5 / np.pi * q * L_val * dt_val

        # ======================================================================
        # SHEAR-SENSITIVITY WEIGHTING FOR CMA-ES (v2.19.0, Fix #6)
        # ======================================================================
        # Apply angle-dependent weighting to sigma to emphasize shear-sensitive
        # angles (parallel/antiparallel to flow). Unlike stratified LS, CMA-ES
        # is derivative-free so gradient cancellation is not the concern.
        # Instead, weighting helps CMA-ES prioritize fit quality at angles
        # that are most informative for shear parameters.
        # ======================================================================
        if (
            is_laminar_flow
            and ad_controller is not None
            and ad_controller.is_enabled
            and hasattr(ad_controller, "shear_weighter")
            and ad_controller.shear_weighter is not None
        ):
            # Get phi0 from current x0 (may be NLSQ warm-started)
            # phi0 is stored in degrees in the parameter array (same as config/bounds)
            physical_params = x0[2:] if len(x0) > n_physical else x0
            phi0_idx = _get_physical_param_names(analysis_mode).index("phi0")
            phi0_current_deg = float(physical_params[phi0_idx])

            # Compute per-angle shear weights
            shear_weights = ad_controller.shear_weighter.get_weights(phi0_current_deg)
            shear_weights_np = np.asarray(shear_weights)

            # Broadcast per-angle weights to per-point weights
            # Each angle's weight applies to all its time points
            per_point_weights = np.repeat(shear_weights_np, n_time_points)

            # Apply weighting: divide sigma by sqrt(weight) so that
            # higher-weighted angles have smaller effective sigma
            # (larger contribution to chi-squared)
            sigma_flat = sigma_flat / np.sqrt(np.maximum(per_point_weights, 0.01))

            logger.info(
                f"[CMA-ES] Shear weighting applied: phi0={phi0_current_deg:.1f} deg, "
                f"weight range=[{np.min(shear_weights_np):.3f}, {np.max(shear_weights_np):.3f}]"
            )

        # Import the core JAX computation function that supports element-wise mode
        from xpcsjax.core.jax_backend import _compute_g1_total_core

        # Note: In constant mode (v2.18.0+), we have two sub-modes:
        # - auto_averaged: 9 parameters [contrast_avg, offset_avg, *physical]
        # - fixed_constant: 7 parameters [*physical] (scaling from pre-computed arrays)

        # Convert fixed per-angle scaling to JAX arrays if using fixed scaling
        fixed_contrast_jax = None
        fixed_offset_jax = None
        if use_fixed_scaling and fixed_contrast_per_angle is not None:
            fixed_contrast_jax = jnp.asarray(fixed_contrast_per_angle)
            fixed_offset_jax = jnp.asarray(fixed_offset_per_angle)

        def model_for_cmaes(xdata_unused: Any, *params: Any) -> Any:
            """JAX-traceable model function wrapper for CMA-ES.

            Uses pure JAX operations to allow JIT compilation by NLSQ's CMAESOptimizer.
            Element-wise mode is triggered automatically when len(t1) > 2000.

            In constant mode (v2.18.0+):
            - auto_averaged: 9 parameters [contrast, offset, *physical]
            - fixed_constant: 7 parameters [*physical] (scaling from fixed arrays)
            """
            params_array = jnp.asarray(params)

            # Extract per-angle scaling and physical params using JAX operations
            # Four modes:
            # 1. effective_per_angle_scaling=True: params = [contrast(n_phi), offset(n_phi), physical]
            # 2. use_fixed_scaling=True: params = [physical] (7 params, scaling from fixed arrays)
            # 3. use_averaged_scaling=True: params = [contrast, offset, physical] (9 params, broadcast)
            # 4. Neither: params = [contrast, offset, physical], broadcast to all angles
            if effective_per_angle_scaling:
                contrasts = params_array[:n_phi]
                offsets = params_array[n_phi : 2 * n_phi]
                physical = params_array[2 * n_phi :]
            elif use_fixed_scaling:
                # FIXED CONSTANT MODE (v2.18.0+): 7 physical params only
                # Use pre-computed fixed per-angle scaling arrays
                contrasts = fixed_contrast_jax
                offsets = fixed_offset_jax
                physical = params_array  # All params are physical
            elif use_averaged_scaling:
                # AUTO AVERAGED MODE (v2.18.0+): 9 parameters [contrast, offset, *physical]
                # Broadcast single contrast/offset to all angles
                contrasts = jnp.full(n_phi, params_array[0])
                offsets = jnp.full(n_phi, params_array[1])
                physical = params_array[2:]
            else:
                # Fallback: broadcast single contrast/offset to all angles
                contrasts = jnp.full(n_phi, params_array[0])
                offsets = jnp.full(n_phi, params_array[1])
                physical = params_array[2:]

            # Map per-angle contrast/offset to each data point
            contrast_per_point = contrasts[phi_indices]
            offset_per_point = offsets[phi_indices]

            # Use element-wise g1 computation (triggered when len > 2000)
            # This is a pure JAX function that can be JIT-traced
            g1_all = _compute_g1_total_core(
                physical,
                t1_all,
                t2_all,
                phi_values,
                wavevector_q_squared_half_dt,
                sinc_prefactor,
                dt_val,
            )

            # Compute g2 = offset + contrast * g1^2
            g2_all = offset_per_point + contrast_per_point * g1_all**2

            return g2_all

        # Create xdata placeholder (model_for_cmaes ignores it)
        # Use 1D array to match NLSQ curve_fit's expected shape for refinement
        xdata = np.zeros(n_data)

        # ======================================================================
        # PHASE 1: NLSQ WARM-START (v2.19.0)
        # ======================================================================
        # Run a quick NLSQ fit first to provide CMA-ES with an informed
        # starting point. This prevents CMA-ES from wasting generations
        # in poor regions of parameter space.
        # ======================================================================
        nlsq_warmstart_chi2 = float("inf")
        nlsq_warmstart_params = None
        nlsq_warmstart_cov = None

        warmstart_enabled = cmaes_dict.get("nlsq_warmstart", True)
        if warmstart_enabled:
            logger.info("[CMA-ES] Phase 1: Running NLSQ warm-start...")
            try:
                warmstart_result = wrapper._run_nlsq_refinement(
                    model_func=model_for_cmaes,
                    xdata=xdata,
                    ydata=ydata,
                    p0=x0,
                    bounds=bounds,
                    sigma=sigma_flat,
                )
                if (
                    warmstart_result["success"]
                    and warmstart_result["chi_squared"] is not None
                ):
                    nlsq_warmstart_chi2 = warmstart_result["chi_squared"]
                    nlsq_warmstart_params = warmstart_result["popt"]
                    nlsq_warmstart_cov = warmstart_result["pcov"]
                    # Use NLSQ solution as CMA-ES starting point
                    x0 = np.asarray(nlsq_warmstart_params)
                    logger.info(
                        f"[CMA-ES] NLSQ warm-start succeeded: chi2={nlsq_warmstart_chi2:.4e}, "
                        f"using as CMA-ES starting point"
                    )
                else:
                    logger.info(
                        "[CMA-ES] NLSQ warm-start did not improve fit, "
                        "using original starting point"
                    )
            except (ValueError, RuntimeError, TypeError, OSError, MemoryError) as e:
                logger.warning(f"[CMA-ES] NLSQ warm-start failed: {e}")

        # ======================================================================
        # PHASE 2: CMA-ES GLOBAL SEARCH (with auto-skip, v2.20.0)
        # ======================================================================
        # When warm-start achieves a good fit (reduced chi2 < threshold), skip
        # the expensive CMA-ES global search. CMA-ES with warm-start sigma is
        # a local refinement that rarely improves on a good NLSQ solution.
        skip_cmaes = False
        warmstart_skip_threshold = cmaes_dict.get(
            "warmstart_skip_threshold",
            getattr(nlsq_config, "cmaes_warmstart_skip_threshold", 5.0),
        )
        warmstart_auto_skip = cmaes_dict.get(
            "warmstart_auto_skip",
            getattr(nlsq_config, "cmaes_warmstart_auto_skip", True),
        )

        if (
            warmstart_auto_skip
            and nlsq_warmstart_params is not None
            and nlsq_warmstart_chi2 < float("inf")
        ):
            # Compute reduced chi-squared for auto-skip decision
            # Guard: if DOF <= 0 (more params than data), never skip CMA-ES
            n_data_eff = len(ydata) - len(x0)
            if n_data_eff <= 0:
                warmstart_reduced_chi2 = float("inf")
            else:
                # When sigma is a default placeholder, chi2 is inflated by
                # 1/sigma^2 relative to unweighted residuals. Undo this
                # inflation using the known base sigma value directly, not the
                # post-shear-weighted sigma_flat. Shear weighting is intentional
                # physics that should remain in chi2 — only the arbitrary
                # default inflation needs to be removed.
                effective_chi2 = nlsq_warmstart_chi2
                if _sigma_is_default:
                    effective_chi2 = nlsq_warmstart_chi2 * _DEFAULT_SIGMA**2
                warmstart_reduced_chi2 = effective_chi2 / n_data_eff
            if warmstart_reduced_chi2 < warmstart_skip_threshold:
                skip_cmaes = True
                logger.info(
                    f"[CMA-ES] Auto-skip: NLSQ warm-start reduced chi2="
                    f"{warmstart_reduced_chi2:.4f} < threshold="
                    f"{warmstart_skip_threshold:.1f}. Skipping CMA-ES global search."
                    f"{' (chi2 sigma-normalized for default sigma)' if _sigma_is_default else ''}"
                )

        if skip_cmaes:
            # Build a CMAESResult directly from warm-start
            cmaes_result = CMAESResult(
                parameters=nlsq_warmstart_params,
                covariance=nlsq_warmstart_cov,
                chi_squared=nlsq_warmstart_chi2,
                success=True,
                diagnostics={
                    "selected": "nlsq_warmstart_auto_skip",
                    "warmstart_reduced_chi2": warmstart_reduced_chi2,
                    "warmstart_raw_chi2": nlsq_warmstart_chi2,
                    "warmstart_skip_threshold": warmstart_skip_threshold,
                    "sigma_is_default": _sigma_is_default,
                    "cmaes_skipped": True,
                },
                method_used="nlsq_warmstart",
                nlsq_refined=True,
                message=(
                    f"CMA-ES skipped: warm-start reduced chi2="
                    f"{warmstart_reduced_chi2:.4f} < {warmstart_skip_threshold:.1f}"
                ),
            )
        else:
            logger.info("[CMA-ES] Phase 2: Running CMA-ES global optimization...")
            cmaes_result = wrapper.fit(
                model_func=model_for_cmaes,
                xdata=xdata,
                ydata=ydata,
                p0=x0,
                bounds=bounds,
                sigma=sigma_flat,
                warmstart_chi2=nlsq_warmstart_chi2,
            )

        # ======================================================================
        # PHASE 3: COMPARE AND SELECT BEST RESULT (v2.19.0)
        # ======================================================================
        # If NLSQ warm-start produced a better result than CMA-ES + refinement,
        # use the NLSQ result instead. This ensures CMA-ES never degrades
        # the solution quality compared to direct NLSQ.
        # ======================================================================
        if (
            nlsq_warmstart_params is not None
            and nlsq_warmstart_chi2 < cmaes_result.chi_squared
        ):
            logger.info(
                f"[CMA-ES] NLSQ warm-start result is better: "
                f"NLSQ chi2={nlsq_warmstart_chi2:.4e} < "
                f"CMA-ES chi2={cmaes_result.chi_squared:.4e}. "
                f"Using NLSQ solution."
            )
            # Replace CMA-ES result with NLSQ warm-start result
            cmaes_result = CMAESResult(
                parameters=nlsq_warmstart_params,
                covariance=nlsq_warmstart_cov,
                chi_squared=nlsq_warmstart_chi2,
                success=True,
                diagnostics={
                    **cmaes_result.diagnostics,
                    "selected": "nlsq_warmstart",
                    "cmaes_chi_squared": cmaes_result.chi_squared,
                    "nlsq_warmstart_chi_squared": nlsq_warmstart_chi2,
                },
                method_used="cmaes",
                nlsq_refined=True,
                message="NLSQ warm-start selected over CMA-ES (lower chi-squared)",
            )
        else:
            if nlsq_warmstart_params is not None:
                logger.info(
                    f"[CMA-ES] CMA-ES result is better: "
                    f"CMA-ES chi2={cmaes_result.chi_squared:.4e} <= "
                    f"NLSQ chi2={nlsq_warmstart_chi2:.4e}"
                )

        execution_time = time.time() - start_time

        # ==========================================================================
        # EXPAND CONSTANT MODE RESULTS (v2.17.0+)
        # ==========================================================================
        # If constant mode was used, expand 9 params [contrast, offset, *physical]
        # back to 2*n_phi + 7 for consistency with per_angle_scaling=True expectations.
        # The single contrast/offset are broadcast to all angles.
        # ==========================================================================
        final_params = np.asarray(cmaes_result.parameters)
        final_covariance = cmaes_result.covariance

        if use_constant_mode and per_angle_scaling:
            # Expand constant mode (9 params) to per-angle format
            from xpcsjax.optimization.nlsq.data_prep import (
                expand_per_angle_parameters,
            )

            n_before = len(final_params)
            expanded = expand_per_angle_parameters(
                final_params,
                None,
                n_phi,
                n_physical,
            )
            final_params = expanded.params

            logger.info(
                f"Expanding constant mode results: {n_before} -> "
                f"{len(final_params)} parameters (broadcast contrast={final_params[0]:.4f}, offset={final_params[n_phi]:.4f})"
            )

            # Expand covariance matrix if available
            # The covariance for single contrast/offset must be broadcast to per-angle
            if final_covariance is not None:
                _n_original = len(cmaes_result.parameters)  # noqa: F841
                n_expanded = 2 * n_phi + n_physical
                expanded_cov = np.zeros((n_expanded, n_expanded))

                # Original layout: [contrast, offset, physical(7)]
                # Expanded layout: [contrast(n_phi), offset(n_phi), physical(7)]

                # Contrast block: all entries = contrast_var (perfectly correlated
                # since all angles share a single source parameter)
                contrast_var = final_covariance[0, 0]
                expanded_cov[:n_phi, :n_phi] = contrast_var

                # Offset block: all entries = offset_var (perfectly correlated)
                offset_var = final_covariance[1, 1]
                expanded_cov[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = offset_var

                # Cross contrast-offset block
                contrast_offset_cov = final_covariance[0, 1]
                expanded_cov[:n_phi, n_phi : 2 * n_phi] = contrast_offset_cov
                expanded_cov[n_phi : 2 * n_phi, :n_phi] = contrast_offset_cov

                # Physical params block: direct slice copy
                # Original indices [2:2+n_physical] -> expanded indices [2*n_phi:]
                expanded_cov[2 * n_phi :, 2 * n_phi :] = final_covariance[
                    2 : 2 + n_physical, 2 : 2 + n_physical
                ]

                # Cross contrast-physical and offset-physical covariance
                # Each physical param has one covariance value broadcast to all angles
                for i in range(n_physical):
                    expanded_cov[:n_phi, 2 * n_phi + i] = final_covariance[0, 2 + i]
                    expanded_cov[2 * n_phi + i, :n_phi] = final_covariance[0, 2 + i]
                    expanded_cov[n_phi : 2 * n_phi, 2 * n_phi + i] = final_covariance[
                        1, 2 + i
                    ]
                    expanded_cov[2 * n_phi + i, n_phi : 2 * n_phi] = final_covariance[
                        1, 2 + i
                    ]

                final_covariance = expanded_cov

        # Convert CMAESResult to OptimizationResult
        n_params = len(final_params)
        dof = max(1, n_data - n_params)
        reduced_chi_squared = cmaes_result.chi_squared / dof

        # Determine quality flag using reduced chi-squared thresholds
        # consistent with NLSQWrapper's 3-level system (wrapper.py:3577-3583)
        if reduced_chi_squared < 1.5:
            quality_flag = "good"
        elif reduced_chi_squared < 3.0:
            quality_flag = "marginal"
        else:
            quality_flag = "poor"

        from xpcsjax.optimization.nlsq.result_builder import compute_uncertainties

        result = OptimizationResult(
            parameters=final_params,
            uncertainties=(
                compute_uncertainties(final_covariance)
                if final_covariance is not None
                else np.zeros(n_params)
            ),
            covariance=(
                final_covariance if final_covariance is not None else np.eye(n_params)
            ),
            chi_squared=cmaes_result.chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status="converged" if cmaes_result.success else "failed",
            iterations=cmaes_result.diagnostics.get("generations", 0),
            execution_time=execution_time,
            device_info={
                "device": "cpu",
                "method": "cmaes",
                "adapter": "CMAESWrapper",
                "preset": cmaes_config.preset,
                "nlsq_refined": cmaes_result.nlsq_refined,
                "restarts": cmaes_result.diagnostics.get("restarts", 0),
                "evaluations": cmaes_result.diagnostics.get("evaluations", 0),
                "anti_degeneracy_constant_mode": use_constant_mode,
                "fixed_per_angle_scaling": fixed_contrast_per_angle is not None,
                "cmaes_params": len(cmaes_result.parameters),
                "final_params": n_params,
            },
            recovery_actions=[],
            quality_flag=quality_flag,
        )

        logger.info("=" * 60)
        logger.info("CMA-ES OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        logger.info(f"Generations: {result.iterations}")
        logger.info(f"Execution time: {execution_time:.3f}s")
        logger.info(f"chi2 = {result.chi_squared:.6e}")
        logger.info(f"Reduced chi2 = {result.reduced_chi_squared:.6f}")
        logger.info(f"L-M refined: {cmaes_result.nlsq_refined}")
        if use_constant_mode:
            logger.info(
                f"Anti-degeneracy: constant mode with fixed per-angle scaling "
                f"({len(cmaes_result.parameters)} physical -> {n_params} total params)"
            )

        return result

    except (ValueError, RuntimeError, TypeError, OSError, MemoryError) as e:
        execution_time = time.time() - start_time
        logger.error(f"CMA-ES optimization failed: {e}")

        # Return failed result
        n_params = len(x0)
        return OptimizationResult(
            parameters=np.asarray(x0),
            uncertainties=np.zeros(n_params),
            covariance=np.eye(n_params),
            chi_squared=float("inf"),
            reduced_chi_squared=float("inf"),
            convergence_status="failed",
            iterations=0,
            execution_time=execution_time,
            device_info={
                "device": "cpu",
                "method": "cmaes",
                "adapter": "CMAESWrapper",
                "error": str(e),
            },
            recovery_actions=[],
            quality_flag="poor",
        )
