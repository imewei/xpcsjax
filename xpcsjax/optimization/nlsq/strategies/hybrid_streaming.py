"""Hybrid streaming optimization strategy for NLSQ optimization.

Extracted from wrapper.py to reduce file size and improve maintainability.

This module provides:
- Hybrid streaming optimizer (L-BFGS warmup + Gauss-Newton refinement)
- Stratified hybrid streaming with anti-degeneracy defense
- Memory estimation and streaming decision logic
- Deprecated streaming optimizer stubs
"""

from __future__ import annotations

import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.exceptions import NLSQOptimizationError
from xpcsjax.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from xpcsjax.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from xpcsjax.optimization.nlsq.hierarchical import (
    HierarchicalConfig,
    HierarchicalOptimizer,
)
from xpcsjax.optimization.nlsq.memory import get_adaptive_memory_threshold
from xpcsjax.optimization.nlsq.parameter_utils import (
    classify_parameter_status as _classify_parameter_status,
)
from xpcsjax.optimization.nlsq.parameter_utils import (
    compute_quantile_per_angle_scaling as _compute_quantile_per_angle_scaling,
)
from xpcsjax.optimization.nlsq.recovery import safe_uncertainties_from_pcov
from xpcsjax.optimization.nlsq.shear_weighting import (
    ShearSensitivityWeighting,
    ShearWeightingConfig,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid circular dependencies
_memory_logger = get_logger("xpcsjax.optimization.nlsq.memory")

# Try importing AdaptiveHybridStreamingOptimizer (available in NLSQ >= 0.3.2)
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    HYBRID_STREAMING_AVAILABLE = True
except ImportError:
    HYBRID_STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None
    HybridStreamingConfig = None


def fit_with_hybrid_streaming_optimizer(
    residual_fn: Any,
    xdata: np.ndarray,
    ydata: np.ndarray,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any,
    nlsq_config: Any = None,
    fast_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using NLSQ AdaptiveHybridStreamingOptimizer for large datasets.

    This method uses NLSQ's four-phase hybrid optimizer to fix three key issues:
    1. Shear-term weak gradients (scale imbalance) - via parameter normalization
    2. Slow convergence - via L-BFGS warmup + Gauss-Newton refinement
    3. Crude covariance - via exact J^T J accumulation + covariance transform

    Four Phases:
    - Phase 0: Parameter normalization setup (bounds-based)
    - Phase 1: L-BFGS warmup with adaptive switching
    - Phase 2: Streaming Gauss-Newton with exact J^T J accumulation
    - Phase 3: Denormalization and covariance transform

    Parameters
    ----------
    residual_fn : callable
        Residual function (StratifiedResidualFunction or similar)
    xdata : np.ndarray
        Independent variable data (flattened)
    ydata : np.ndarray
        Dependent variable data (flattened)
    initial_params : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray or None
        Parameter bounds (lower, upper)
    logger : logging.Logger
        Logger instance
    nlsq_config : NLSQConfig, optional
        NLSQ configuration with hybrid streaming settings

    Returns
    -------
    popt : np.ndarray
        Optimized parameters
    pcov : np.ndarray
        Covariance matrix (properly transformed to original space)
    info : dict
        Optimization information including phase diagnostics

    Raises
    ------
    RuntimeError
        If AdaptiveHybridStreamingOptimizer is not available
    NLSQOptimizationError
        If optimization fails
    """
    if not HYBRID_STREAMING_AVAILABLE:
        raise RuntimeError(
            "AdaptiveHybridStreamingOptimizer not available. "
            "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
        )

    logger.info("Initializing NLSQ AdaptiveHybridStreamingOptimizer...")
    logger.info("Fixes: 1) Shear-term gradients, 2) Convergence, 3) Covariance")

    # Create HybridStreamingConfig from NLSQConfig with 4-layer defense
    if nlsq_config is not None:
        config = HybridStreamingConfig(
            normalize=nlsq_config.hybrid_normalize,
            normalization_strategy=nlsq_config.hybrid_normalization_strategy,
            warmup_iterations=nlsq_config.hybrid_warmup_iterations,
            max_warmup_iterations=nlsq_config.hybrid_max_warmup_iterations,
            warmup_learning_rate=nlsq_config.hybrid_warmup_learning_rate,
            gauss_newton_max_iterations=nlsq_config.hybrid_gauss_newton_max_iterations,
            gauss_newton_tol=nlsq_config.hybrid_gauss_newton_tol,
            chunk_size=nlsq_config.hybrid_chunk_size,
            trust_region_initial=nlsq_config.hybrid_trust_region_initial,
            regularization_factor=nlsq_config.hybrid_regularization_factor,
            enable_checkpoints=nlsq_config.hybrid_enable_checkpoints,
            checkpoint_frequency=nlsq_config.hybrid_checkpoint_frequency,
            validate_numerics=nlsq_config.hybrid_validate_numerics,
            # 4-Layer Defense Strategy (NLSQ 0.3.6)
            enable_warm_start_detection=nlsq_config.hybrid_enable_warm_start_detection,
            warm_start_threshold=nlsq_config.hybrid_warm_start_threshold,
            enable_adaptive_warmup_lr=nlsq_config.hybrid_enable_adaptive_warmup_lr,
            warmup_lr_refinement=nlsq_config.hybrid_warmup_lr_refinement,
            warmup_lr_careful=nlsq_config.hybrid_warmup_lr_careful,
            enable_cost_guard=nlsq_config.hybrid_enable_cost_guard,
            cost_increase_tolerance=nlsq_config.hybrid_cost_increase_tolerance,
            enable_step_clipping=nlsq_config.hybrid_enable_step_clipping,
            max_warmup_step_size=nlsq_config.hybrid_max_warmup_step_size,
        )
    else:
        # Use NLSQ 0.3.6 defaults with 4-layer defense enabled
        config = HybridStreamingConfig(
            normalize=True,
            normalization_strategy="auto",
            warmup_iterations=200,
            max_warmup_iterations=500,
            gauss_newton_max_iterations=100,
            gauss_newton_tol=1e-8,
            chunk_size=10000,
            # 4-Layer Defense enabled by default
            enable_warm_start_detection=True,
            warm_start_threshold=0.01,
            enable_adaptive_warmup_lr=True,
            warmup_lr_refinement=1e-6,
            warmup_lr_careful=1e-5,
            enable_cost_guard=True,
            cost_increase_tolerance=0.05,
            enable_step_clipping=True,
            max_warmup_step_size=0.1,
        )

    logger.info(f"  Normalization: {config.normalization_strategy}")
    logger.info(f"  Warmup iterations: {config.warmup_iterations}")
    logger.info(f"  Gauss-Newton max: {config.gauss_newton_max_iterations}")
    logger.info(f"  Chunk size: {config.chunk_size}")

    # Initialize optimizer
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Create model function from residual function
    # The hybrid optimizer expects: func(x, *params) -> predictions
    # Our residual function computes: residuals = y - predictions
    # We need: predictions = y - residuals
    if hasattr(residual_fn, "jax_residual"):
        # Stratified residual function

        def model_fn(x: Any, *params: float) -> Any:
            params_array = jnp.asarray(params)
            residuals = residual_fn.jax_residual(params_array)
            return ydata - residuals

    else:
        # Standard residual function

        def model_fn(x: Any, *params: float) -> Any:
            residuals = residual_fn(x, *params)
            return ydata - residuals

    try:
        # Run optimization
        result = optimizer.fit(
            data_source=(xdata, ydata),
            func=model_fn,
            p0=initial_params,
            bounds=bounds,
            sigma=None,  # TODO: Add sigma support if needed
            verbose=1 if not fast_mode else 0,
        )

        # Extract results
        popt = np.asarray(result["x"])
        pcov = np.asarray(result.get("pcov", np.eye(len(popt))))
        perr = np.asarray(
            result.get("perr", safe_uncertainties_from_pcov(pcov, len(popt)))
        )

        # Build info dict with phase diagnostics
        info = {
            "success": result.get("success", False),
            "message": result.get("message", "Hybrid optimization completed"),
            "hybrid_streaming_diagnostics": result.get("streaming_diagnostics", {}),
            "perr": perr,
            "sigma_sq": result.get("streaming_diagnostics", {})
            .get("gauss_newton_diagnostics", {})
            .get("final_cost"),
            "phase_timings": result.get("streaming_diagnostics", {}).get(
                "phase_timings", {}
            ),
        }

        logger.info("Hybrid streaming optimization completed successfully")
        phase_timings = info.get("phase_timings", {})
        if phase_timings:
            logger.info(
                f"  Phase 0 (normalization): {phase_timings.get('phase0_normalization', 0):.3f}s"
            )
            logger.info(
                f"  Phase 1 (L-BFGS warmup): {phase_timings.get('phase1_warmup', 0):.3f}s"
            )
            logger.info(
                f"  Phase 2 (Gauss-Newton): {phase_timings.get('phase2_gauss_newton', 0):.3f}s"
            )
            logger.info(
                f"  Phase 3 (covariance): {phase_timings.get('phase3_finalize', 0):.3f}s"
            )

        return popt, pcov, info

    except (
        ValueError,
        RuntimeError,
        TypeError,
        AttributeError,
        OSError,
        MemoryError,
    ) as e:
        # T031: Log detailed warning explaining failure and lost capabilities
        logger.error(f"AdaptiveHybridStreamingOptimizer failed: {e}")
        logger.warning(
            "=" * 60 + "\n"
            "HYBRID OPTIMIZER FAILURE - Falling back to basic streaming\n"
            "=" * 60 + "\n"
            "The AdaptiveHybridStreamingOptimizer encountered an error.\n"
            "\n"
            "Capabilities lost with fallback:\n"
            "  - Parameter normalization (gradient equalization)\n"
            "  - L-BFGS warmup + Gauss-Newton hybrid convergence\n"
            "  - Exact J^T J covariance accumulation\n"
            "\n"
            "Fallback uses basic streaming optimizer which may:\n"
            "  - Converge slower (1000+ vs ~110 iterations)\n"
            "  - Miss shear parameters (imbalanced gradients)\n"
            "  - Produce less accurate uncertainties\n"
            "\n"
            f"Error details: {type(e).__name__}: {str(e)}\n"
            "=" * 60
        )
        # T030: TODO - Implement 3-attempt retry with HybridRecoveryConfig
        # For now, immediately raise to trigger fallback to streaming
        if isinstance(e, NLSQOptimizationError):
            raise
        else:
            raise NLSQOptimizationError(
                f"AdaptiveHybridStreamingOptimizer failed: {str(e)}",
                error_context={"original_error": type(e).__name__},
            ) from e


def fit_with_stratified_hybrid_streaming(
    stratified_data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any,
    hybrid_config: dict | None = None,
    anti_degeneracy_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using NLSQ AdaptiveHybridStreamingOptimizer for large datasets.

    This method implements the 4-phase hybrid optimization from NLSQ >=0.3.2:
    - Phase 0: Parameter normalization setup (bounds-based)
    - Phase 1: L-BFGS warmup with adaptive switching
    - Phase 2: Streaming Gauss-Newton with exact J^T J accumulation
    - Phase 3: Denormalization and covariance transform

    With Anti-Degeneracy Defense System v2.9.0 integration:
    - Layer 1: Fourier Reparameterization (reduces per-angle DoF)
    - Layer 2: Hierarchical Optimization (alternating stage fitting)
    - Layer 3: Adaptive CV-based Regularization (scales properly)
    - Layer 4: Gradient Collapse Detection (runtime monitoring)

    Key improvements over basic StreamingOptimizer:
    1. Shear-term weak gradients: Fixed via parameter normalization
    2. Slow convergence: Fixed via L-BFGS warmup + Gauss-Newton refinement
    3. Crude covariance: Fixed via exact J^T J accumulation
    4. Structural degeneracy: Fixed via anti-degeneracy defense layers

    Args:
        stratified_data: StratifiedData object with flat stratified arrays
        per_angle_scaling: Whether per-angle parameters are enabled
        physical_param_names: List of physical parameter names
        initial_params: Initial parameter guess
        bounds: Parameter bounds (lower, upper) tuple
        logger: Logger instance
        hybrid_config: Optional config dict with keys:
            - normalize: Enable parameter normalization (default: True)
            - normalization_strategy: "bounds" or "scale" (default: "bounds")
            - warmup_iterations: L-BFGS warmup iterations (default: 100)
            - max_warmup_iterations: Max L-BFGS iterations (default: 500)
            - warmup_learning_rate: L-BFGS line search scale (default: 0.001)
            - gauss_newton_max_iterations: GN iterations (default: 50)
            - gauss_newton_tol: Convergence tolerance (default: 1e-8)
            - chunk_size: Points per chunk for streaming (default: 50000)
        anti_degeneracy_config: Optional config dict for Anti-Degeneracy Defense:
            - per_angle_mode: "independent", "fourier", or "auto" (default: "auto")
            - fourier_order: Fourier harmonic order (default: 2)
            - fourier_auto_threshold: n_phi threshold for auto mode (default: 6)
            - hierarchical.enable: Enable hierarchical optimization (default: True)
            - regularization.mode: "absolute", "relative", or "auto" (default: "relative")
            - regularization.lambda: Base regularization strength (default: 1.0)
            - gradient_monitoring.enable: Enable gradient collapse detection (default: True)

    Returns:
        (popt, pcov, info) tuple

    Raises:
        RuntimeError: If AdaptiveHybridStreamingOptimizer is not available
    """

    if not HYBRID_STREAMING_AVAILABLE:
        raise RuntimeError(
            "AdaptiveHybridStreamingOptimizer not available. "
            "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
        )

    logger.info("Initializing NLSQ AdaptiveHybridStreamingOptimizer...")
    logger.info("Fixes: 1) Shear-term gradients, 2) Convergence, 3) Covariance")

    start_time = time.perf_counter()

    # Parse hybrid streaming configuration
    # Uses NLSQ 0.3.6 defaults which include 4-layer defense strategy
    config_dict = hybrid_config or {}
    normalize = config_dict.get("normalize", True)
    normalization_strategy = config_dict.get("normalization_strategy", "auto")
    # Standard warmup iterations - NLSQ 0.3.6 has 4-layer defense to prevent
    # divergence when starting from good parameters
    warmup_iterations = config_dict.get("warmup_iterations", 200)
    max_warmup_iterations = config_dict.get("max_warmup_iterations", 500)
    warmup_learning_rate = config_dict.get("warmup_learning_rate", 0.001)
    gauss_newton_max_iterations = config_dict.get("gauss_newton_max_iterations", 100)
    gauss_newton_tol = config_dict.get("gauss_newton_tol", 1e-8)
    chunk_size = config_dict.get("chunk_size", 10_000)
    trust_region_initial = config_dict.get("trust_region_initial", 1.0)
    regularization_factor = config_dict.get("regularization_factor", 1e-10)
    enable_checkpoints = config_dict.get("enable_checkpoints", True)
    checkpoint_frequency = config_dict.get("checkpoint_frequency", 100)
    validate_numerics = config_dict.get("validate_numerics", True)

    # Learning rate scheduling
    use_learning_rate_schedule = config_dict.get("use_learning_rate_schedule", False)
    lr_schedule_warmup_steps = config_dict.get(
        "lr_schedule_warmup_steps", warmup_iterations
    )
    lr_schedule_decay_steps = config_dict.get(
        "lr_schedule_decay_steps", max_warmup_iterations - warmup_iterations
    )
    lr_schedule_end_value = config_dict.get("lr_schedule_end_value", 0.0001)

    # 4-Layer Defense Strategy (NLSQ 0.3.6)
    # Prevents L-BFGS warmup from diverging when starting from good parameters
    # Layer 1: Warm Start Detection - skip warmup if already at good solution
    enable_warm_start_detection = config_dict.get("enable_warm_start_detection", True)
    warm_start_threshold = float(config_dict.get("warm_start_threshold", 0.01))
    # Layer 2: Adaptive Learning Rate - scale LR based on initial loss quality
    enable_adaptive_warmup_lr = config_dict.get("enable_adaptive_warmup_lr", True)
    warmup_lr_refinement = float(config_dict.get("warmup_lr_refinement", 1e-6))
    warmup_lr_careful = float(config_dict.get("warmup_lr_careful", 1e-5))
    # Layer 3: Cost-Increase Guard - abort if loss increases during warmup
    enable_cost_guard = config_dict.get("enable_cost_guard", True)
    cost_increase_tolerance = float(config_dict.get("cost_increase_tolerance", 0.05))
    # Layer 4: Step Clipping - limit max parameter change per L-BFGS iteration
    enable_step_clipping = config_dict.get("enable_step_clipping", True)
    max_warmup_step_size = float(config_dict.get("max_warmup_step_size", 0.1))

    # Group Variance Regularization (NLSQ 0.3.8)
    # Prevents per-angle parameters from absorbing angle-dependent physical signals
    enable_group_variance_regularization = config_dict.get(
        "enable_group_variance_regularization", False
    )
    group_variance_lambda = float(config_dict.get("group_variance_lambda", 0.01))
    # group_variance_indices will be auto-computed if not provided
    group_variance_indices_raw = config_dict.get("group_variance_indices", None)

    # Compute n_phi early for auto-computing group_variance_indices
    # Extract unique phi angles from stratified data
    all_phi_early = []
    if hasattr(stratified_data, "chunks") and len(stratified_data.chunks) > 0:
        for chunk in stratified_data.chunks:
            all_phi_early.extend(chunk.phi.tolist())
    else:
        all_phi_early = stratified_data.phi_flat.tolist()
    n_phi = len(set(all_phi_early))
    phi_unique = np.array(sorted(set(all_phi_early)))  # For shear weighting

    # is_laminar_flow (shear present) gates ONLY Layer 5 (shear weighting) and the
    # shear-specific bounds/popt handling below. Layers 1-4 (Fourier reparam,
    # hierarchical, adaptive regularization, gradient monitoring) gate on
    # per_angle_scaling alone so they fire for ALL analysis modes, not just
    # laminar_flow.
    is_laminar_flow = "gamma_dot_t0" in physical_param_names

    # =====================================================================
    # Anti-Degeneracy Defense System v2.9.0 Initialization
    # =====================================================================
    # CRITICAL FIX (Jan 2026): Define n_physical unconditionally FIRST
    # This variable is used by multiple conditional blocks (hierarchical,
    # gradient_monitor, shear_weighter). Previously it was only defined
    # inside conditional blocks, causing UnboundLocalError when those
    # conditions were false but shear_weighter tried to use it.
    n_physical = len(physical_param_names)

    # Parse anti-degeneracy configuration
    ad_config = anti_degeneracy_config or {}
    hierarchical_config = ad_config.get("hierarchical", {})
    regularization_config = ad_config.get("regularization", {})
    gradient_monitoring_config = ad_config.get("gradient_monitoring", {})

    # Layer 1: Fourier Reparameterization / Constant Scaling Configuration
    # v2.18.0: Distinct semantics for auto vs explicit constant mode
    per_angle_mode = ad_config.get("per_angle_mode", "auto")
    fourier_order = ad_config.get("fourier_order", 2)
    fourier_auto_threshold = ad_config.get("fourier_auto_threshold", 6)
    constant_scaling_threshold = ad_config.get("constant_scaling_threshold", 3)

    # Determine actual per-angle mode
    # v2.18.0: Distinct semantics:
    #   - auto (n_phi >= threshold): "auto_averaged" → 9 params, OPTIMIZED averaged scaling
    #   - constant (explicit): "fixed_constant" → 7 params, FIXED per-angle scaling
    #   - individual: per-angle scaling OPTIMIZED
    if per_angle_mode == "auto":
        if n_phi >= constant_scaling_threshold:
            # AUTO mode with large n_phi: optimize averaged scaling (9 params)
            # Computes N quantile estimates, averages to 1 contrast + 1 offset
            # These 2 averaged values ARE OPTIMIZED along with 7 physical params
            per_angle_mode_actual = "auto_averaged"
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Auto-selected 'auto_averaged' mode")
            logger.info(
                f"  Reason: n_phi ({n_phi}) >= "
                f"constant_scaling_threshold ({constant_scaling_threshold})"
            )
            logger.info("  Behavior: Quantile estimates -> AVERAGED -> OPTIMIZED")
            logger.info(
                "  Parameters: 7 physical + 2 averaged (contrast, offset) = 9 total"
            )
            logger.info("=" * 60)
        else:
            # Use individual per-angle parameters for few angles (N < 3)
            per_angle_mode_actual = "individual"
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Auto-selected 'individual' mode")
            logger.info(
                f"  Reason: n_phi ({n_phi}) < "
                f"constant_scaling_threshold ({constant_scaling_threshold})"
            )
            logger.info(
                f"  Parameters: 7 physical + {2 * n_phi} per-angle = {7 + 2 * n_phi} total"
            )
            logger.info("=" * 60)
    elif per_angle_mode == "constant":
        # EXPLICIT constant mode: FIXED per-angle scaling (7 params)
        # Computes N quantile estimates, uses per-angle values DIRECTLY (NOT averaged)
        # Only 7 physical params are optimized; scaling is FIXED
        per_angle_mode_actual = "fixed_constant"
        logger.info("=" * 60)
        logger.info(
            "ANTI-DEGENERACY DEFENSE: Explicit 'constant' mode -> fixed_constant"
        )
        logger.info(f"  n_phi: {n_phi}")
        logger.info(
            "  Behavior: Quantile estimates -> per-angle values FIXED (NOT optimized)"
        )
        logger.info("  Parameters: 7 physical only (scaling FIXED from quantiles)")
        logger.info("=" * 60)
    else:
        # Other explicit modes (fourier or individual)
        per_angle_mode_actual = per_angle_mode
        logger.debug(
            f"ANTI-DEGENERACY: Using explicit per_angle_mode: {per_angle_mode_actual}"
        )

    # T031: Determine mode flags
    # use_constant: True for both auto_averaged and fixed_constant (constant-style mapping)
    # use_fixed_scaling: True only for fixed_constant (scaling NOT optimized)
    # use_averaged_scaling: True only for auto_averaged (scaling optimized)
    use_constant = per_angle_mode_actual in ("auto_averaged", "fixed_constant")
    use_averaged_scaling = per_angle_mode_actual == "auto_averaged"
    # use_fixed_scaling will be set True after quantile estimation for fixed_constant mode

    # Initialize Fourier reparameterizer if using fourier mode
    fourier_reparameterizer = None
    if per_angle_mode_actual == "fourier" and per_angle_scaling:
        # Get unique phi angles in radians
        phi_unique_rad = np.deg2rad(np.array(sorted(set(all_phi_early))))

        # Extract user-configured bounds for contrast and offset from bounds tuple
        # Bounds layout: [contrast(n_phi), offset(n_phi), physical(7)]
        # Use first contrast/offset bound as the c0/o0 (mean) bounds
        c0_bounds = (0.1, 0.8)  # Default
        o0_bounds = (0.5, 1.5)  # Default
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            if len(lower_bounds) >= n_phi and len(upper_bounds) >= n_phi:
                # Extract contrast bounds from first contrast element
                c0_bounds = (float(lower_bounds[0]), float(upper_bounds[0]))
                # Extract offset bounds from first offset element
                o0_bounds = (float(lower_bounds[n_phi]), float(upper_bounds[n_phi]))
                logger.debug(
                    f"  Using user-configured Fourier bounds: "
                    f"c0={c0_bounds}, o0={o0_bounds}"
                )

        fourier_config = FourierReparamConfig(
            mode="fourier",
            fourier_order=fourier_order,
            auto_threshold=fourier_auto_threshold,
            c0_bounds=c0_bounds,
            o0_bounds=o0_bounds,
        )
        fourier_reparameterizer = FourierReparameterizer(phi_unique_rad, fourier_config)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Fourier Reparameterization")
        logger.info(f"  Mode: {per_angle_mode_actual}")
        logger.info(f"  n_phi: {n_phi}, Fourier order: {fourier_order}")
        logger.info(f"  Contrast bounds (c0): {c0_bounds}")
        logger.info(f"  Offset bounds (o0): {o0_bounds}")
        logger.info(
            f"  Parameter reduction: {2 * n_phi} -> {fourier_reparameterizer.n_coeffs}"
        )
        logger.info("=" * 60)
    elif per_angle_mode_actual == "fixed_constant" and per_angle_scaling:
        # fixed_constant mode: per-angle scaling is FIXED, not optimized
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Fixed Constant Mode (v2.18.0)")
        logger.info(f"  Mode: {per_angle_mode_actual}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info("  Method: Quantile-based per-angle scaling (FIXED, not optimized)")
        logger.info(
            "  Per-angle contrast/offset will be estimated from c2 data quantiles"
        )
        logger.info("  These values are FIXED (not optimized) during fitting")
        logger.info(f"  Parameter reduction: {2 * n_phi} -> 0 (physical only)")
        logger.info("=" * 60)
    elif per_angle_mode_actual == "auto_averaged" and per_angle_scaling:
        # auto_averaged mode: averaged scaling is OPTIMIZED (9 params)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Auto Averaged Mode (v2.18.0)")
        logger.info(f"  Mode: {per_angle_mode_actual}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info("  Method: Quantile estimates -> averaged -> OPTIMIZED")
        logger.info("  Initial values: averaged from per-angle quantile estimates")
        logger.info(
            f"  Parameter reduction: {2 * n_phi} -> 2 (averaged contrast + offset)"
        )
        logger.info("=" * 60)

    # =====================================================================
    # CONSTANT/AUTO_AVERAGED MODES (v2.18.0): Quantile-Based Scaling
    # =====================================================================
    # - fixed_constant: per-angle values are FIXED (not optimized), 7 params
    # - auto_averaged: averaged values are OPTIMIZED as initial values, 9 params
    # =====================================================================
    use_fixed_scaling = False
    fixed_contrast_per_angle: np.ndarray | None = None
    fixed_offset_per_angle: np.ndarray | None = None
    fixed_contrast_jax: jnp.ndarray | None = None
    fixed_offset_jax: jnp.ndarray | None = None
    # For auto_averaged mode: averaged values to use as initial optimization values
    averaged_contrast_init: float | None = None
    averaged_offset_init: float | None = None

    if use_constant and per_angle_scaling:
        logger.info("Computing quantile-based per-angle scaling estimates...")
        try:
            # Extract bounds for clipping
            contrast_bounds = (0.0, 1.0)  # Default
            offset_bounds = (0.5, 1.5)  # Default
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                if len(lower_bounds) >= n_phi and len(upper_bounds) >= n_phi:
                    contrast_bounds = (
                        float(lower_bounds[0]),
                        float(upper_bounds[0]),
                    )
                    offset_bounds = (
                        float(lower_bounds[n_phi]),
                        float(upper_bounds[n_phi]),
                    )

            # Compute quantile-based per-angle scaling
            fixed_contrast_per_angle, fixed_offset_per_angle = (
                _compute_quantile_per_angle_scaling(
                    stratified_data=stratified_data,
                    contrast_bounds=contrast_bounds,
                    offset_bounds=offset_bounds,
                    logger=logger,
                )
            )

            if (
                fixed_contrast_per_angle is not None
                and fixed_offset_per_angle is not None
            ):
                if per_angle_mode_actual == "fixed_constant":
                    # fixed_constant: Use per-angle values DIRECTLY as FIXED
                    use_fixed_scaling = True
                    fixed_contrast_jax = jnp.asarray(fixed_contrast_per_angle)
                    fixed_offset_jax = jnp.asarray(fixed_offset_per_angle)

                    logger.info(
                        "Fixed per-angle scaling computed (FIXED, not optimized):"
                    )
                    logger.info(
                        f"  Contrast: mean={np.nanmean(fixed_contrast_per_angle):.4f}, "
                        f"range=[{np.nanmin(fixed_contrast_per_angle):.4f}, "
                        f"{np.nanmax(fixed_contrast_per_angle):.4f}]"
                    )
                    logger.info(
                        f"  Offset: mean={np.nanmean(fixed_offset_per_angle):.4f}, "
                        f"range=[{np.nanmin(fixed_offset_per_angle):.4f}, "
                        f"{np.nanmax(fixed_offset_per_angle):.4f}]"
                    )
                elif per_angle_mode_actual == "auto_averaged":
                    # auto_averaged: AVERAGE per-angle values → use as INITIAL for optimization
                    averaged_contrast_init = float(np.nanmean(fixed_contrast_per_angle))
                    averaged_offset_init = float(np.nanmean(fixed_offset_per_angle))

                    logger.info(
                        "Averaged scaling computed (initial values for optimization):"
                    )
                    logger.info(f"  Averaged contrast: {averaged_contrast_init:.4f}")
                    logger.info(f"  Averaged offset: {averaged_offset_init:.4f}")
                    logger.info(
                        "  These will be OPTIMIZED along with 7 physical params (9 total)"
                    )

                    # Do NOT set use_fixed_scaling = True for auto_averaged
                    # The averaged values are just initial guesses for optimization
            else:  # pragma: no cover – defensive; function always returns arrays
                logger.warning(  # type: ignore[unreachable]
                    "Failed to compute quantile-based scaling, "
                    "falling back to standard constant mode (optimizing 2 params)"
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                f"Error computing quantile-based scaling: {e}, "
                f"falling back to standard constant mode"
            )
            use_fixed_scaling = False

    # Layer 2: Hierarchical Optimization Configuration
    # =====================================================================
    # CRITICAL FIX (Jan 2026): Auto-enable hierarchical when shear_weighting
    # is enabled. Shear weighting is ONLY applied inside hierarchical
    # optimizer's loss function. Without hierarchical, the gradient
    # cancellation for gamma_dot_t0 is NOT prevented!
    #
    # Root cause: The shear gradient ∂L/∂γ̇₀ ∝ Σ cos(φ₀-φ) cancels when
    # summing over angles spanning 360° (e.g., 23 angles → 94.6% cancellation).
    # Shear weighting emphasizes shear-sensitive angles to prevent this.
    # =====================================================================
    shear_weighting_config_early = ad_config.get("shear_weighting", {})
    shear_weighting_will_be_enabled = (
        shear_weighting_config_early.get("enable", True)
        and is_laminar_flow
        and n_phi > 3
    )

    enable_hierarchical = hierarchical_config.get("enable", True)

    # Override: shear weighting requires hierarchical optimization to function
    if shear_weighting_will_be_enabled and not enable_hierarchical:
        logger.warning("=" * 60)
        logger.warning(
            "ANTI-DEGENERACY: Shear weighting enabled but hierarchical disabled!"
        )
        logger.warning(
            "  Auto-enabling hierarchical optimization to apply shear weights."
        )
        logger.warning(
            "  Without this, gradient cancellation will collapse gamma_dot_t0."
        )
        logger.warning("=" * 60)
        enable_hierarchical = True

    hierarchical_optimizer = None
    # Skip hierarchical optimization in constant scaling mode:
    # - Constant mode already prevents per-angle absorption (2 DoF vs 46)
    # - HierarchicalOptimizer expects n_per_angle = 2*n_phi or n_coeffs (Fourier)
    # - Using hierarchical with constant mode causes index mismatch error
    if enable_hierarchical and per_angle_scaling and not use_constant:
        # n_physical defined unconditionally above
        hier_config = HierarchicalConfig(
            enable=True,
            max_outer_iterations=hierarchical_config.get("max_outer_iterations", 5),
            outer_tolerance=float(hierarchical_config.get("outer_tolerance", 1e-6)),
            physical_max_iterations=hierarchical_config.get(
                "physical_max_iterations", 100
            ),
            per_angle_max_iterations=hierarchical_config.get(
                "per_angle_max_iterations", 50
            ),
        )
        hierarchical_optimizer = HierarchicalOptimizer(
            config=hier_config,
            n_phi=n_phi,
            n_physical=n_physical,
            fourier_reparameterizer=fourier_reparameterizer,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
        logger.info(f"  Enabled: {enable_hierarchical}")
        logger.info(f"  Max outer iterations: {hier_config.max_outer_iterations}")
        logger.info(f"  Outer tolerance: {hier_config.outer_tolerance}")
        if shear_weighting_will_be_enabled:
            logger.info(
                "  Shear weighting: WILL BE APPLIED via hierarchical loss function"
            )
        logger.info("=" * 60)
    elif use_constant and enable_hierarchical and per_angle_scaling:
        # Log that hierarchical is skipped due to constant scaling mode
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
        logger.info(
            "  Skipped: constant scaling mode already prevents per-angle absorption"
        )
        logger.info(
            "  Reason: Only 2 per-angle DoF (vs 46), no need for hierarchical alternation"
        )
        logger.info("=" * 60)

    # Layer 3: Adaptive Relative Regularization Configuration
    # Replaces/enhances the basic group variance regularization with CV-based approach
    regularization_mode = regularization_config.get("mode", "relative")
    regularization_lambda = float(regularization_config.get("lambda", 1.0))
    target_cv = float(regularization_config.get("target_cv", 0.10))
    target_contribution = float(regularization_config.get("target_contribution", 0.10))
    max_cv = float(regularization_config.get("max_cv", 0.20))

    adaptive_regularizer = None
    if per_angle_scaling:
        # Compute mode-aware group indices
        # Group indices depend on per-angle mode: fixed_constant, auto_averaged, fourier, or individual
        if use_fixed_scaling:
            # fixed_constant: No scaling params to regularize (7 physical only)
            mode_group_indices = []
            logger.debug(
                "Fixed-constant mode: No per-angle regularization (scaling is fixed)"
            )
        elif use_averaged_scaling:
            # auto_averaged: 2 per-angle params (1 contrast + 1 offset) to regularize
            mode_group_indices = [(0, 1), (1, 2)]
            logger.debug(
                f"Auto-averaged regularization groups: {mode_group_indices} "
                f"(1 contrast + 1 offset)"
            )
        elif (
            fourier_reparameterizer is not None and fourier_reparameterizer.use_fourier
        ):
            n_coeffs_per_param = fourier_reparameterizer.n_coeffs_per_param
            mode_group_indices = [
                (0, n_coeffs_per_param),  # contrast Fourier coefficients
                (
                    n_coeffs_per_param,
                    2 * n_coeffs_per_param,
                ),  # offset Fourier coefficients
            ]
            logger.debug(
                f"Fourier-aware regularization groups: {mode_group_indices} "
                f"(n_coeffs_per_param={n_coeffs_per_param})"
            )
        else:
            mode_group_indices = None  # Use default: [(0, n_phi), (n_phi, 2*n_phi)]
            logger.debug(
                f"Using default regularization groups (Fourier mode not active): "
                f"fourier_reparameterizer={fourier_reparameterizer is not None}, "
                f"use_fourier={fourier_reparameterizer.use_fourier if fourier_reparameterizer else 'N/A'}"
            )

        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=regularization_mode,
            lambda_base=regularization_lambda,
            target_cv=target_cv,
            target_contribution=target_contribution,
            max_cv=max_cv,
            group_indices=mode_group_indices,
        )
        adaptive_regularizer = AdaptiveRegularizer(reg_config, n_phi)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 3 - Adaptive Regularization")
        logger.info(f"  Mode: {regularization_mode}")
        logger.info(f"  Auto-tuned lambda: {adaptive_regularizer.lambda_value:.2f}")
        logger.info(f"  Target CV: {target_cv} ({target_cv * 100:.0f}% variation)")
        logger.info(f"  Max CV: {max_cv}")
        logger.info(f"  Group indices: {adaptive_regularizer.group_indices}")
        logger.info("=" * 60)

        # Update group variance settings to use adaptive regularizer's lambda
        # This ensures NLSQ's built-in regularization is consistent
        enable_group_variance_regularization = True
        group_variance_lambda = adaptive_regularizer.lambda_value

    # Layer 4: Gradient Collapse Monitor Configuration
    gradient_monitor_enabled = gradient_monitoring_config.get("enable", True)
    gradient_monitor = None
    if gradient_monitor_enabled and per_angle_scaling:
        # Compute mode-aware parameter count
        # n_per_angle depends on per-angle mode: fixed_constant, auto_averaged, fourier, or individual
        if use_fixed_scaling:
            # fixed_constant: 0 per-angle params (scaling is fixed)
            n_per_angle = 0
        elif use_averaged_scaling:
            # auto_averaged: 2 per-angle params (1 contrast + 1 offset)
            n_per_angle = 2
        elif fourier_reparameterizer is not None:
            # Fourier mode: n_coeffs Fourier coefficients
            n_per_angle = fourier_reparameterizer.n_coeffs
        else:
            # Independent mode: 2 * n_phi per-angle params
            n_per_angle = 2 * n_phi
        # n_physical defined unconditionally above
        # Use numpy arrays for indices (JAX compatibility)
        per_angle_indices = np.arange(n_per_angle, dtype=np.intp)
        physical_indices = np.arange(
            n_per_angle, n_per_angle + n_physical, dtype=np.intp
        )

        # Compute gamma_dot_t0 index for watch_parameters
        # In laminar_flow, physical params are [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        # gamma_dot_t0 is at physical_indices[3] = n_per_angle + 3
        gamma_dot_t0_idx = n_per_angle + 3  # Index of gamma_dot_t0 in full param vector

        monitor_config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=float(
                gradient_monitoring_config.get("ratio_threshold", 0.01)
            ),
            consecutive_triggers=gradient_monitoring_config.get(
                "consecutive_triggers", 5
            ),
            response_mode=gradient_monitoring_config.get("response", "hierarchical"),
            # NEW (Dec 2025): Watch gamma_dot_t0 specifically for gradient collapse
            # This detects when shear parameter gradient vanishes during L-BFGS warmup
            watch_parameters=[gamma_dot_t0_idx],
            watch_threshold=float(
                gradient_monitoring_config.get("watch_threshold", 1e-8)
            ),
        )
        gradient_monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=physical_indices,
            per_angle_indices=per_angle_indices,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 4 - Gradient Collapse Monitor")
        logger.info(f"  Enabled: {gradient_monitor_enabled}")
        logger.info(f"  Ratio threshold: {monitor_config.ratio_threshold}")
        logger.info(f"  Consecutive triggers: {monitor_config.consecutive_triggers}")
        logger.info(f"  Response mode: {monitor_config.response_mode}")
        logger.info("=" * 60)

    # Layer 5: Shear-Sensitivity Weighting (v2.9.1)
    # Prevents gradient cancellation for shear parameters by emphasizing
    # shear-sensitive angles (parallel/antiparallel to flow direction)
    shear_weighting_config = ad_config.get("shear_weighting", {})
    shear_weighting_enabled = shear_weighting_config.get("enable", True)
    shear_weighter: ShearSensitivityWeighting | None = None

    if is_laminar_flow and shear_weighting_enabled and n_phi > 3:
        # Get initial phi0 from config or use default
        initial_phi0 = shear_weighting_config.get("initial_phi0", None)
        if initial_phi0 is None:
            # Try to get from initial parameters
            initial_phi0 = float(initial_params[-1]) if len(initial_params) > 0 else 0.0

        sw_config = ShearWeightingConfig(
            enable=True,
            min_weight=float(shear_weighting_config.get("min_weight", 0.3)),
            alpha=float(shear_weighting_config.get("alpha", 1.0)),
            update_frequency=int(shear_weighting_config.get("update_frequency", 1)),
            initial_phi0=initial_phi0,
            normalize=shear_weighting_config.get("normalize", True),
        )
        shear_weighter = ShearSensitivityWeighting(
            phi_angles=phi_unique,
            n_physical=n_physical,
            phi0_index=6,  # phi0 is last of 7 physical params
            config=sw_config,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 5 - Shear-Sensitivity Weighting")
        logger.info(f"  Enabled: {shear_weighting_enabled}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info(f"  min_weight: {sw_config.min_weight:.2f}")
        logger.info(f"  alpha: {sw_config.alpha:.1f}")
        logger.info(f"  initial_phi0: {initial_phi0:.1f} deg")
        logger.info("=" * 60)

    # Store anti-degeneracy components for diagnostics
    anti_degeneracy_components = {
        "per_angle_mode": per_angle_mode_actual,
        "use_constant": use_constant,  # T031: Track constant mode status
        "use_fixed_scaling": use_fixed_scaling,  # v2.17.0: Track fixed scaling status
        "fourier_reparameterizer": fourier_reparameterizer,
        "hierarchical_optimizer": hierarchical_optimizer,
        "adaptive_regularizer": adaptive_regularizer,
        "gradient_monitor": gradient_monitor,
        "shear_weighter": shear_weighter,
    }
    # ===================================================================== #
    if enable_group_variance_regularization and group_variance_indices_raw is None:
        if is_laminar_flow and per_angle_scaling and n_phi > 3:
            # T031: Handle fixed scaling, constant, Fourier, and individual modes
            # Fixed scaling mode (v2.17.0): 0 per-angle params (all fixed)
            # Constant mode: 1 value per group (contrast/offset)
            # Fourier mode: n_coeffs_per_param values per group
            # Individual mode: n_phi values per group
            if use_fixed_scaling:
                # No per-angle params to regularize - skip group variance
                n_per_group = 0
                group_variance_indices = []
                logger.info(
                    "  Fixed scaling mode: skipping group variance regularization "
                    "(no per-angle params)"
                )
            elif use_constant:
                n_per_group = 1
            elif fourier_reparameterizer is not None:
                n_per_group = fourier_reparameterizer.n_coeffs_per_param
            else:
                n_per_group = n_phi

            # Only compute group indices if not using fixed scaling
            if not use_fixed_scaling:
                # Per-angle contrast: params[0:n_per_group]
                # Per-angle offset: params[n_per_group:2*n_per_group]
                group_variance_indices = [
                    (0, n_per_group),
                    (n_per_group, 2 * n_per_group),
                ]
                logger.info(
                    f"  Auto-computed group_variance_indices for {n_phi} angles: "
                    f"{group_variance_indices} (mode: {per_angle_mode_actual})"
                )
        else:
            group_variance_indices = None
            if enable_group_variance_regularization:
                logger.warning(
                    "Group variance regularization enabled but no indices provided. "
                    "Auto-computation requires laminar_flow mode with per_angle_scaling "
                    f"and n_phi > 3. (is_laminar_flow={is_laminar_flow}, "
                    f"per_angle_scaling={per_angle_scaling}, n_phi={n_phi})"
                )
    else:
        # Convert raw indices to list of tuples if provided
        if group_variance_indices_raw is not None:
            group_variance_indices = [tuple(idx) for idx in group_variance_indices_raw]
        else:
            group_variance_indices = None

    logger.info("Hybrid streaming config:")
    logger.info(f"  Normalization: {normalization_strategy}")
    logger.info(f"  Warmup iterations: {warmup_iterations}")
    logger.info(f"  Max warmup iterations: {max_warmup_iterations}")
    logger.info(f"  Learning rate: {warmup_learning_rate}")
    if use_learning_rate_schedule:
        logger.info(
            f"  LR schedule: warmup={lr_schedule_warmup_steps}, "
            f"decay={lr_schedule_decay_steps}, end={lr_schedule_end_value}"
        )
    logger.info(f"  Gauss-Newton iterations: {gauss_newton_max_iterations}")
    logger.info(f"  Gauss-Newton tolerance: {gauss_newton_tol}")
    logger.info(f"  Chunk size: {chunk_size:,}")
    logger.info("  4-Layer Defense Strategy (NLSQ 0.3.6):")
    logger.info(f"    L1 Warm Start Detection: {enable_warm_start_detection}")
    logger.info(f"    L2 Adaptive LR: {enable_adaptive_warmup_lr}")
    logger.info(f"    L3 Cost Guard: {enable_cost_guard}")
    logger.info(f"    L4 Step Clipping: {enable_step_clipping}")
    if enable_group_variance_regularization:
        logger.info("  Group Variance Regularization (NLSQ 0.3.8):")
        logger.info(f"    Enabled: {enable_group_variance_regularization}")
        logger.info(f"    Lambda: {group_variance_lambda}")
        logger.info(f"    Indices: {group_variance_indices}")

    # Prepare residual weighting for NLSQ optimizer (Layer 5 of Anti-Degeneracy)
    # Homodyne computes shear-sensitivity weights and passes them to NLSQ
    # as generic residual weights - NLSQ doesn't need to know about XPCS physics
    enable_residual_weighting = shear_weighter is not None
    residual_weights_list = None
    if enable_residual_weighting:
        # Compute shear-sensitivity weights in xpcsjax, pass to NLSQ as generic weights
        assert shear_weighter is not None  # guarded by enable_residual_weighting
        residual_weights_list = shear_weighter.get_weights().tolist()
        logger.info("  Residual Weighting (Shear-Sensitivity):")
        logger.info(f"    Enabled: {enable_residual_weighting}")
        logger.info(f"    n_weights: {len(residual_weights_list)}")
        logger.info(
            f"    Weight range: [{min(residual_weights_list):.3f}, "
            f"{max(residual_weights_list):.3f}]"
        )

    # Create HybridStreamingConfig with 4-layer defense
    optimizer_config = HybridStreamingConfig(
        normalize=normalize,
        normalization_strategy=normalization_strategy,
        warmup_iterations=warmup_iterations,
        max_warmup_iterations=max_warmup_iterations,
        warmup_learning_rate=warmup_learning_rate,
        gauss_newton_max_iterations=gauss_newton_max_iterations,
        gauss_newton_tol=gauss_newton_tol,
        chunk_size=chunk_size,
        trust_region_initial=trust_region_initial,
        regularization_factor=regularization_factor,
        enable_checkpoints=enable_checkpoints,
        checkpoint_frequency=checkpoint_frequency,
        validate_numerics=validate_numerics,
        use_learning_rate_schedule=use_learning_rate_schedule,
        lr_schedule_warmup_steps=lr_schedule_warmup_steps,
        lr_schedule_decay_steps=lr_schedule_decay_steps,
        lr_schedule_end_value=lr_schedule_end_value,
        # 4-Layer Defense Strategy
        enable_warm_start_detection=enable_warm_start_detection,
        warm_start_threshold=warm_start_threshold,
        enable_adaptive_warmup_lr=enable_adaptive_warmup_lr,
        warmup_lr_refinement=warmup_lr_refinement,
        warmup_lr_careful=warmup_lr_careful,
        enable_cost_guard=enable_cost_guard,
        cost_increase_tolerance=cost_increase_tolerance,
        enable_step_clipping=enable_step_clipping,
        max_warmup_step_size=max_warmup_step_size,
        # Group Variance Regularization (NLSQ 0.3.8)
        enable_group_variance_regularization=enable_group_variance_regularization,
        group_variance_lambda=group_variance_lambda,
        group_variance_indices=group_variance_indices,
        # Residual Weighting (NLSQ 0.4.x)
        # Homodyne computes shear-sensitivity weights and passes them as generic
        # residual weights - NLSQ just does weighted least squares
        enable_residual_weighting=enable_residual_weighting,
        residual_weights=residual_weights_list,
        verbose=config_dict.get("verbose", 1),
        log_frequency=config_dict.get("log_frequency", 1),
    )

    # Initialize optimizer
    optimizer = AdaptiveHybridStreamingOptimizer(optimizer_config)

    # Extract global metadata from stratified data
    if hasattr(stratified_data, "chunks") and len(stratified_data.chunks) > 0:
        first_chunk = stratified_data.chunks[0]
        q = first_chunk.q
        L = first_chunk.L
        dt = first_chunk.dt
    else:
        q = stratified_data.q
        L = stratified_data.L
        dt = stratified_data.dt

    logger.debug(f"Global metadata: q={q}, L={L}, dt={dt}")

    # Extract unique values for theory computation
    all_phi = []
    all_t1 = []
    all_t2 = []
    if hasattr(stratified_data, "chunks"):
        for chunk in stratified_data.chunks:
            all_phi.extend(chunk.phi.tolist())
            all_t1.extend(chunk.t1.tolist())
            all_t2.extend(chunk.t2.tolist())
    else:
        all_phi = stratified_data.phi_flat.tolist()
        all_t1 = stratified_data.t1_flat.tolist()
        all_t2 = stratified_data.t2_flat.tolist()

    phi_unique = np.array(sorted(set(all_phi)))
    t1_unique = np.array(sorted(set(all_t1)))
    n_phi = len(phi_unique)

    logger.info(f"Unique values: {n_phi} phi, {len(t1_unique)} t1")

    # Import physics utilities
    from xpcsjax.core.physics_utils import (
        PI,
        calculate_diffusion_coefficient,
        calculate_shear_rate,
        safe_sinc,
        trapezoid_cumsum,
    )

    # Pre-compute physics factors
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    # Convert to JAX arrays
    phi_unique_jax = jnp.asarray(phi_unique)
    t1_unique_jax = jnp.asarray(t1_unique)

    # Create model function
    is_laminar_flow = "gamma_dot_t0" in physical_param_names

    # T042: Compute n_per_angle for model function based on mode
    # In fixed scaling mode: 0 (all params are physical)
    # In constant mode (fallback): 1 contrast + 1 offset = 2
    # In individual mode: n_phi contrast + n_phi offset = 2*n_phi
    # In Fourier mode: n_coeffs contrast + n_coeffs offset = 2*n_coeffs
    if use_fixed_scaling:
        # Fixed scaling: all params are physical, no per-angle params in vector
        n_per_angle = 0
    elif use_constant:
        n_per_angle = 2
    elif fourier_reparameterizer is not None:
        n_per_angle = fourier_reparameterizer.n_coeffs
    else:
        n_per_angle = 2 * n_phi

    @jax.jit
    def model_fn_pointwise(
        x_batch: jnp.ndarray, *params_tuple: jnp.ndarray
    ) -> jnp.ndarray:
        """Point-wise model function for hybrid streaming optimizer."""
        # Handle both single points (1D) and batches (2D)
        # The optimizer may call with single points during Jacobian computation
        x_batch_2d = jnp.atleast_2d(x_batch)

        params_all = jnp.stack(params_tuple)

        # Extract indices from x_batch (now guaranteed 2D)
        phi_idx = x_batch_2d[:, 0].astype(jnp.int32)
        t1_idx = x_batch_2d[:, 1].astype(jnp.int32)
        t2_idx = x_batch_2d[:, 2].astype(jnp.int32)

        # T042: Extract scaling and physical parameters based on mode
        # Fixed scaling mode (v2.17.0): use pre-computed fixed arrays, all params are physical
        # Constant mode (fallback): params[0]=contrast, params[1]=offset, params[2:]=physical
        # Individual mode: params[:n_phi]=contrast, params[n_phi:2*n_phi]=offset, params[2*n_phi:]=physical
        if use_fixed_scaling:
            # Use pre-computed fixed per-angle scaling from quantiles
            # All params in params_all are physical
            contrast_all = fixed_contrast_jax
            offset_all = fixed_offset_jax
            physical_params = params_all
        elif use_constant:
            # Single contrast and offset shared across all angles
            contrast_all = jnp.full(n_phi, params_all[0])
            offset_all = jnp.full(n_phi, params_all[1])
            physical_params = params_all[2:]
        else:
            contrast_all = params_all[:n_phi]
            offset_all = params_all[n_phi : 2 * n_phi]
            physical_params = params_all[2 * n_phi :]

        # Extract physical parameters
        D0 = physical_params[0]
        alpha = physical_params[1]
        D_offset = physical_params[2]

        # Compute diffusion
        D_t = calculate_diffusion_coefficient(t1_unique_jax, D0, alpha, D_offset)
        D_cumsum = trapezoid_cumsum(D_t)
        D_diff = D_cumsum[t1_idx] - D_cumsum[t2_idx]
        # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
        D_integral_batch = jnp.sqrt(D_diff**2 + 1e-12)

        log_g1_diff = -wavevector_q_squared_half_dt * D_integral_batch
        log_g1_diff_bounded = jnp.clip(log_g1_diff, -700.0, 0.0)
        g1_diffusion = jnp.exp(log_g1_diff_bounded)

        if is_laminar_flow:
            # Shear parameters
            gamma_dot_0 = physical_params[3]
            beta = physical_params[4]
            gamma_dot_offset = physical_params[5]
            phi0 = physical_params[6]

            # Compute shear
            gamma_t = calculate_shear_rate(
                t1_unique_jax, gamma_dot_0, beta, gamma_dot_offset
            )
            gamma_cumsum = trapezoid_cumsum(gamma_t)
            gamma_diff = gamma_cumsum[t1_idx] - gamma_cumsum[t2_idx]
            # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
            gamma_integral_batch = jnp.sqrt(gamma_diff**2 + 1e-12)

            # Shear contribution with angle dependence
            # Formula: g₁_shear = [sinc(Φ)]² where Φ = sinc_prefactor * cos(φ₀-φ) * ∫γ̇
            phi_values = phi_unique_jax[phi_idx]
            angle_diff = jnp.deg2rad(phi0 - phi_values)  # Match physics: cos(φ₀-φ)
            cos_phi = jnp.cos(angle_diff)

            sinc_arg = sinc_prefactor * gamma_integral_batch * cos_phi
            sinc_val = safe_sinc(sinc_arg)
            g1_shear = sinc_val**2  # CRITICAL: g1_shear = sinc²(Φ)

            g1_total = g1_diffusion * g1_shear
            # P0-3: Use jnp.where (gradient-safe) instead of jnp.clip.
            # log-space clip above guarantees g1 ≤ 1.0; lower floor prevents log(0).
            epsilon = 1e-10
            g1 = jnp.where(g1_total > epsilon, g1_total, epsilon)
        else:
            epsilon = 1e-10
            g1 = jnp.where(g1_diffusion > epsilon, g1_diffusion, epsilon)

        # Compute g2 with per-angle scaling
        assert contrast_all is not None  # set in all branches above
        assert offset_all is not None  # set in all branches above
        contrast = contrast_all[phi_idx]
        offset = offset_all[phi_idx]
        g2_theory = offset + contrast * g1**2
        # P0-3: Removed jnp.clip(g2, 0.5, 2.5) — kills gradients at boundaries.
        # Bounds enforced via parameter bounds in optimizer, not g2 clipping.
        g2 = g2_theory

        # Squeeze output to match input dimensionality
        # Returns 0D scalar for single point, 1D array for batch
        return jnp.asarray(g2.squeeze())

    # Prepare data
    logger.info("Preparing hybrid streaming data...")
    prep_start = time.perf_counter()

    if hasattr(stratified_data, "chunks"):
        all_phi_data = np.concatenate([c.phi for c in stratified_data.chunks])
        all_t1_data = np.concatenate([c.t1 for c in stratified_data.chunks])
        all_t2_data = np.concatenate([c.t2 for c in stratified_data.chunks])
        y_data = np.concatenate([c.g2 for c in stratified_data.chunks])
    else:
        all_phi_data = stratified_data.phi_flat
        all_t1_data = stratified_data.t1_flat
        all_t2_data = stratified_data.t2_flat
        y_data = stratified_data.g2_flat

    # Convert to indices (vectorized).
    # NOTE: Both t1 and t2 index into t1_unique because XPCS correlation
    # matrices use a shared time grid (t1_unique == t2_unique).
    def _bin_to_grid(values: np.ndarray, grid: np.ndarray, axis_name: str) -> np.ndarray:
        """searchsorted + boundary clip, warning on out-of-grid points.

        An unguarded clip silently routes data lying outside the fitted grid to
        the boundary bin, mis-associating it with the wrong (phi, t1, t2) cell —
        a data-integrity violation. We clip (to stay in-bounds) but surface how
        many points were affected so misaligned data/config is not silent.
        """
        raw = np.searchsorted(grid, values)
        n_oob = int(np.sum(raw >= len(grid)))
        if n_oob > 0:
            logger.warning(
                "%d data point(s) lie beyond the %s grid; clipped to the boundary "
                "bin. Check data/config grid alignment.",
                n_oob,
                axis_name,
            )
        return np.clip(raw, 0, len(grid) - 1)

    phi_idx_arr = _bin_to_grid(all_phi_data, phi_unique, "phi")
    t1_idx_arr = _bin_to_grid(all_t1_data, t1_unique, "t1")
    t2_idx_arr = _bin_to_grid(all_t2_data, t1_unique, "t2")

    # B2: Store grid indices as int32 (not float64).  Grid indices are
    # non-negative integers; int32 covers any realistic grid (max ~2.1B).
    # The float64 cast wasted ~276 MB at 23M pts and the consumption sites
    # (model_fn_pointwise lines 1258-1260, loss_fn phi_indices_jax line 1712)
    # cast back to int32 anyway.  JAX_ENABLE_X64 stays on — this is indices,
    # not physical data.
    x_data = np.column_stack(
        [phi_idx_arr.astype(np.int32), t1_idx_arr.astype(np.int32), t2_idx_arr.astype(np.int32)]
    )
    y_data = np.asarray(y_data, dtype=np.float64)

    # =====================================================================
    # Diagonal Handling (v2.14.2+)
    # =====================================================================
    # Hybrid streaming uses point-wise theory computation (no 2D grid), so
    # apply_diagonal_correction() cannot be applied to theory.
    #
    # Instead, diagonal points are FILTERED OUT from the data entirely:
    # - Data: Already has diagonal correction applied at load time
    # - Theory: Never computes diagonal values (filtered points excluded)
    # - Residual: Diagonal points excluded from loss (equivalent to mask=0)
    #
    # This is architecturally equivalent to correction + masking used in
    # Stratified LS and Out-of-Core methods. The result is the same:
    # diagonal points contribute ZERO to the optimization objective.
    # =====================================================================
    n_points_before = len(y_data)
    non_diagonal_mask = t1_idx_arr != t2_idx_arr
    x_data = x_data[non_diagonal_mask]
    y_data = y_data[non_diagonal_mask]
    n_diagonal_removed = n_points_before - len(y_data)

    prep_time = time.perf_counter() - prep_start
    logger.info(f"Data preparation completed in {prep_time:.2f}s")
    logger.info(f"  Dataset size: {len(y_data):,} points")
    logger.info(
        f"  Diagonal points removed: {n_diagonal_removed:,} "
        f"({100 * n_diagonal_removed / n_points_before:.1f}%)"
    )

    # NOTE (Dec 2025): Data is already pre-shuffled at stratification stage
    # in _apply_stratification_if_needed(). No additional shuffle needed here.
    # The pre-shuffle prevents L-BFGS warmup from seeing angle-sequential data,
    # which would cause local minimum traps (gamma_dot_t0 -> 0).

    # =====================================================================
    # Anti-Degeneracy Defense System v2.9.0 - EXECUTION INTEGRATION
    # =====================================================================
    # Transform parameters and execute appropriate optimization path
    use_hierarchical = (
        hierarchical_optimizer is not None
        and anti_degeneracy_components.get("hierarchical_optimizer") is not None
        and ad_config.get("enable", True)
    )
    use_fourier = (
        fourier_reparameterizer is not None
        and anti_degeneracy_components.get("fourier_reparameterizer") is not None
        and ad_config.get("enable", True)
    )

    # Track params for fitting
    fit_initial_params = initial_params.copy()
    fit_bounds = bounds

    # T034-T038: Constant mode parameter transformation
    # v2.17.0: When use_fixed_scaling=True, use physical params only (fixed contrast/offset from quantiles)
    # Fallback: Transform per-angle params (2*n_phi) to constant (2) by taking means
    if use_fixed_scaling:
        # FIXED SCALING MODE (v2.17.0): Use quantile-derived fixed per-angle scaling
        # Parameters are physical-only, contrast/offset are NOT in the param vector
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Fixed Per-Angle Scaling (v2.17.0)")
        physical_params = initial_params[2 * n_phi :]

        # New parameter layout: [physical_params] only
        fit_initial_params = physical_params

        logger.info(f"  Original params: {len(initial_params)}")
        logger.info(
            f"  Fixed scaling params: {len(fit_initial_params)} (physical only)"
        )
        logger.info(f"  Per-angle reduction: {2 * n_phi} -> 0 (using fixed arrays)")

        # Transform bounds to physical only
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            fit_bounds = (lower_bounds[2 * n_phi :], upper_bounds[2 * n_phi :])
            logger.info(
                f"  Bounds reduced to physical only: {len(fit_bounds[0])} params"
            )
        logger.info("=" * 60)
    elif use_averaged_scaling:
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Auto Averaged Scaling Mode")
        # Transform per-angle params to single values (means) for optimization
        per_angle_params = initial_params[: 2 * n_phi]
        physical_params = initial_params[2 * n_phi :]

        # Split per-angle into contrast and offset groups
        contrast_per_angle = per_angle_params[:n_phi]
        offset_per_angle = per_angle_params[n_phi : 2 * n_phi]

        # Use quantile-based averaged values if computed, else take means
        if averaged_contrast_init is not None and averaged_offset_init is not None:
            contrast_mean = averaged_contrast_init
            offset_mean = averaged_offset_init
            logger.info("  Using quantile-based averaged initial values (OPTIMIZED)")
        else:
            contrast_mean = np.nanmean(contrast_per_angle)
            offset_mean = np.nanmean(offset_per_angle)
            logger.info("  Using parameter-based averaged initial values (OPTIMIZED)")

        # New parameter layout: [contrast_const, offset_const, physical_params]
        fit_initial_params = np.concatenate(
            [[contrast_mean], [offset_mean], physical_params]
        )

        logger.info(f"  Original params: {len(initial_params)}")
        logger.info(f"  Constant params: {len(fit_initial_params)}")
        logger.info(f"  Per-angle reduction: {2 * n_phi} -> 2")
        logger.info(f"  Contrast mean: {contrast_mean:.6f}")
        logger.info(f"  Offset mean: {offset_mean:.6f}")

        # T039: Transform bounds for constant mode
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            # For constant mode, use the bounds of the first per-angle param
            # (all per-angle bounds are typically the same)
            fit_lower = np.concatenate(
                [
                    [lower_bounds[0]],
                    [lower_bounds[n_phi]],
                    lower_bounds[2 * n_phi :],
                ]
            )
            fit_upper = np.concatenate(
                [
                    [upper_bounds[0]],
                    [upper_bounds[n_phi]],
                    upper_bounds[2 * n_phi :],
                ]
            )
            fit_bounds = (fit_lower, fit_upper)
        logger.info("=" * 60)

    # Layer 1: Fourier reparameterization of initial parameters
    elif use_fourier:
        assert fourier_reparameterizer is not None  # guarded by use_fourier
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Fourier Reparameterization")
        # Transform per-angle params to Fourier coefficients
        per_angle_params = initial_params[: 2 * n_phi]
        physical_params = initial_params[2 * n_phi :]

        # Split per-angle into contrast and offset groups
        contrast_per_angle = per_angle_params[:n_phi]
        offset_per_angle = per_angle_params[n_phi : 2 * n_phi]

        # Transform to Fourier coefficients
        contrast_coeffs = fourier_reparameterizer.to_fourier(contrast_per_angle)
        offset_coeffs = fourier_reparameterizer.to_fourier(offset_per_angle)

        # New parameter layout: [contrast_coeffs, offset_coeffs, physical_params]
        fit_initial_params = np.concatenate(
            [contrast_coeffs, offset_coeffs, physical_params]
        )

        logger.info(f"  Original params: {len(initial_params)}")
        logger.info(f"  Fourier params: {len(fit_initial_params)}")
        logger.info(
            f"  Per-angle reduction: {2 * n_phi} -> {len(contrast_coeffs) + len(offset_coeffs)}"
        )

        # Transform bounds for Fourier space
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            # Per-angle bounds are typically (0,1) for contrast, (0.5, 1.5) for offset
            # Fourier coefficients can have wider bounds since they combine linearly
            # Use n_coeffs_per_param (e.g., 5 for order=2), NOT n_coeffs (total=10)
            n_half = fourier_reparameterizer.n_coeffs_per_param

            # Fourier coefficient bounds: a0 keeps the mean, others can be ±range
            contrast_lower = np.concatenate(
                [
                    [lower_bounds[0]],  # a0 (mean) lower bound
                    np.full(n_half - 1, -1.0),  # Other coeffs can be negative
                ]
            )
            contrast_upper = np.concatenate(
                [
                    [upper_bounds[0]],  # a0 (mean) upper bound
                    np.full(n_half - 1, 1.0),  # Other coeffs bounded
                ]
            )
            offset_lower = np.concatenate(
                [
                    [lower_bounds[n_phi]],  # a0 (mean) lower bound
                    np.full(n_half - 1, -0.5),  # Other coeffs
                ]
            )
            offset_upper = np.concatenate(
                [
                    [upper_bounds[n_phi]],  # a0 (mean) upper bound
                    np.full(n_half - 1, 0.5),  # Other coeffs
                ]
            )

            fit_lower = np.concatenate(
                [contrast_lower, offset_lower, lower_bounds[2 * n_phi :]]
            )
            fit_upper = np.concatenate(
                [contrast_upper, offset_upper, upper_bounds[2 * n_phi :]]
            )
            fit_bounds = (fit_lower, fit_upper)
        logger.info("=" * 60)

    # =====================================================================
    # Anti-Degeneracy Defense: Create Fourier-wrapped model function
    # =====================================================================
    # When using Fourier mode, wrap model_fn to convert Fourier coeffs -> per-angle
    if use_fourier:
        assert fourier_reparameterizer is not None  # guarded by use_fourier
        n_coeffs_per_param = fourier_reparameterizer.n_coeffs_per_param
        _fourier_basis_matrix = fourier_reparameterizer._basis_matrix

        @jax.jit
        def model_fn_fourier(
            x_batch: jnp.ndarray, *params_tuple: jnp.ndarray
        ) -> jnp.ndarray:
            """Model function with Fourier coefficient inputs."""
            # Handle both single points (1D) and batches (2D)
            x_batch_2d = jnp.atleast_2d(x_batch)
            params_all = jnp.stack(params_tuple)

            # Extract Fourier coefficients and physical params
            # Layout: [contrast_coeffs, offset_coeffs, physical_params]
            n_coeffs = n_coeffs_per_param  # captured from outer scope
            contrast_coeffs = params_all[:n_coeffs]
            offset_coeffs = params_all[n_coeffs : 2 * n_coeffs]
            physical_params = params_all[2 * n_coeffs :]

            # Convert Fourier coefficients to per-angle values
            # Uses precomputed basis matrix: values = B @ coeffs
            basis_matrix = jnp.asarray(_fourier_basis_matrix)
            contrast_all = basis_matrix @ contrast_coeffs
            offset_all = basis_matrix @ offset_coeffs

            # Extract indices from x_batch (now guaranteed 2D)
            phi_idx = x_batch_2d[:, 0].astype(jnp.int32)
            t1_idx = x_batch_2d[:, 1].astype(jnp.int32)
            t2_idx = x_batch_2d[:, 2].astype(jnp.int32)

            # Extract physical parameters
            D0 = physical_params[0]
            alpha = physical_params[1]
            D_offset = physical_params[2]

            # Compute diffusion
            D_t = calculate_diffusion_coefficient(t1_unique_jax, D0, alpha, D_offset)
            D_cumsum = trapezoid_cumsum(D_t)
            D_diff = D_cumsum[t1_idx] - D_cumsum[t2_idx]
            # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
            D_integral_batch = jnp.sqrt(D_diff**2 + 1e-12)

            log_g1_diff = -wavevector_q_squared_half_dt * D_integral_batch
            log_g1_diff_bounded = jnp.clip(log_g1_diff, -700.0, 0.0)
            g1_diffusion = jnp.exp(log_g1_diff_bounded)

            if is_laminar_flow:
                # Shear parameters
                gamma_dot_0 = physical_params[3]
                beta = physical_params[4]
                gamma_dot_offset = physical_params[5]
                phi0 = physical_params[6]

                # Compute shear
                gamma_t = calculate_shear_rate(
                    t1_unique_jax, gamma_dot_0, beta, gamma_dot_offset
                )
                gamma_cumsum = trapezoid_cumsum(gamma_t)
                gamma_diff = gamma_cumsum[t1_idx] - gamma_cumsum[t2_idx]
                # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
                gamma_integral_batch = jnp.sqrt(gamma_diff**2 + 1e-12)

                # Shear contribution with angle dependence
                phi_values = phi_unique_jax[phi_idx]
                angle_diff = jnp.deg2rad(phi0 - phi_values)
                cos_phi = jnp.cos(angle_diff)

                sinc_arg = sinc_prefactor * gamma_integral_batch * cos_phi
                sinc_val = safe_sinc(sinc_arg)
                g1_shear = sinc_val**2

                g1_total = g1_diffusion * g1_shear
                # P0-3: Use jnp.where (gradient-safe) instead of jnp.clip.
                # log-space clip above guarantees g1 ≤ 1.0; lower floor prevents log(0).
                epsilon = 1e-10
                g1 = jnp.where(g1_total > epsilon, g1_total, epsilon)
            else:
                epsilon = 1e-10
                g1 = jnp.where(g1_diffusion > epsilon, g1_diffusion, epsilon)

            # Compute g2 with per-angle scaling (from Fourier-derived values)
            contrast = contrast_all[phi_idx]
            offset = offset_all[phi_idx]
            g2_theory = offset + contrast * g1**2
            # P0-3: Removed jnp.clip(g2, 0.5, 2.5) — kills gradients at boundaries.
            # Bounds enforced via parameter bounds in optimizer, not g2 clipping.
            g2 = g2_theory

            return jnp.asarray(g2.squeeze())

        # Use Fourier model function for optimization
        active_model_fn = model_fn_fourier
        logger.info("  Using Fourier-wrapped model function")
    else:
        # Use standard per-angle model function
        active_model_fn = model_fn_pointwise

    # Run hybrid optimization
    logger.info("Starting hybrid optimization (L-BFGS + Gauss-Newton)...")
    opt_start = time.perf_counter()

    # Layer 2: Hierarchical optimization path
    # Can be combined with Fourier mode (hierarchical operates on Fourier params)
    result: dict[str, Any]
    if use_hierarchical:
        # Use hierarchical two-stage optimization
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Hierarchical Two-Stage Optimization")

        # Pre-extract phi indices for shear weighting (x_data[:, 0] contains phi indices)
        phi_indices_jax = jnp.asarray(x_data[:, 0])  # already int32 (B2)
        shear_weighter_local = cast(
            ShearSensitivityWeighting | None,
            anti_degeneracy_components.get("shear_weighter"),
        )

        def loss_fn(params: Any) -> Any:
            """Loss function for hierarchical optimizer.

            CRITICAL: Must use jnp (JAX) operations, NOT np (NumPy).
            Using np.mean breaks the JAX autodiff computation graph,
            resulting in zero gradients for all parameters.

            Layer 5: Shear-sensitivity weighting is applied here to prevent
            gradient cancellation for shear parameters (gamma_dot_t0, phi0).
            """
            # Convert params to JAX array if needed for tracing
            params_jax = jnp.asarray(params)
            pred = active_model_fn(x_data, *params_jax)

            # Convert y_data to JAX for proper gradient flow
            y_data_jax = jnp.asarray(y_data)
            residuals = y_data_jax - pred

            # Layer 5: Apply shear-sensitivity weighting if enabled
            # This emphasizes angles parallel/antiparallel to flow direction,
            # preventing gradient cancellation for shear parameters
            if shear_weighter_local is not None:
                # Use shear-weighted loss instead of uniform MSE
                weighted_loss = shear_weighter_local.apply_weights_to_loss(
                    residuals, phi_indices_jax
                )
            else:
                # CRITICAL: Use jnp.mean, NOT np.mean!
                # np.mean breaks JAX autodiff and causes zero gradients
                weighted_loss = jnp.mean(residuals**2) * len(y_data)

            # Add adaptive regularization if enabled
            if adaptive_regularizer is not None:
                # Use JAX-compatible method for autodiff compatibility
                # Note: weighted_loss already includes the normalization
                mse_for_reg = weighted_loss / len(y_data)
                reg_term = adaptive_regularizer.compute_regularization_jax(
                    params_jax, mse_for_reg, len(y_data)
                )
                return weighted_loss + reg_term
            return weighted_loss

        def grad_fn(params: Any) -> Any:
            """Gradient function with optional monitoring."""
            # Use JAX autodiff for gradient computation
            grad = jax.grad(lambda p: loss_fn(p))(params)

            # Layer 4: Gradient monitoring
            if gradient_monitor is not None:
                gradient_monitor.check(
                    grad, iteration_counter[0], params, loss_fn(params)
                )
                iteration_counter[0] += 1

            return grad

        iteration_counter = [0]  # Mutable counter for gradient monitor

        # Layer 5: Create callback for shear weight updates
        # Updates weights based on current phi0 estimate at start of each outer iteration
        def shear_weight_update_callback(params: np.ndarray, outer_iter: int) -> None:
            """Update shear-sensitivity weights based on current phi0."""
            if shear_weighter_local is not None:
                shear_weighter_local.update_phi0(params, outer_iter)

        assert hierarchical_optimizer is not None  # guarded by use_hierarchical
        assert fit_bounds is not None  # hierarchical requires bounds
        hier_result = hierarchical_optimizer.fit(
            loss_fn=loss_fn,
            grad_fn=grad_fn,
            p0=fit_initial_params,
            bounds=fit_bounds,
            outer_iteration_callback=shear_weight_update_callback,
        )

        # Compute covariance from Hessian of loss function (BUG-15 fix)
        # Gauss-Newton approximation: H ≈ 2 * J^T J for least-squares loss
        # So pcov = s² * inv(J^T J) ≈ 2 * s² * inv(H)
        n_hier_data = len(y_data)
        n_hier_params = len(hier_result.x)
        s2_hier = hier_result.fun / max(n_hier_data - n_hier_params, 1)
        try:
            popt_jax = jnp.asarray(hier_result.x)
            H = np.asarray(jax.hessian(loss_fn)(popt_jax))
        except Exception as e:
            logger.warning(
                f"Could not compute Hessian: {e}. Using identity placeholder."
            )
            H = None

        covariance_is_placeholder = False
        if H is not None:
            try:
                pcov_hier = 2.0 * s2_hier * np.linalg.inv(H)
                logger.info(
                    f"Hierarchical covariance from Hessian: s^2={s2_hier:.6e} "
                    f"(n_data={n_hier_data}, n_params={n_hier_params})"
                )
            except np.linalg.LinAlgError:
                logger.warning(
                    "Singular Hessian in hierarchical path, using pseudo-inverse"
                )
                pcov_hier = 2.0 * s2_hier * np.linalg.pinv(H)
        else:
            # H-5: an identity covariance is fabricated, not measured. Reported
            # uncertainties (±1.0 for every parameter) are meaningless; flag it
            # explicitly so downstream consumers do not treat them as real.
            logger.error(
                "Hessian computation failed in hierarchical path; covariance is an "
                "identity placeholder — reported uncertainties are NOT meaningful."
            )
            pcov_hier = np.eye(n_hier_params)
            covariance_is_placeholder = True

        # Convert HierarchicalResult to standard format
        result = {
            "x": hier_result.x,
            "pcov": pcov_hier,
            "success": hier_result.success,
            "message": hier_result.message,
            "function_evaluations": hier_result.n_outer_iterations * 150,  # Estimate
            "covariance_is_placeholder": covariance_is_placeholder,
            "streaming_diagnostics": {
                "phase_iterations": {
                    "phase1": 0,
                    "phase2": hier_result.n_outer_iterations,
                },
                "warmup_diagnostics": {},
                "gauss_newton_diagnostics": {
                    "final_cost": hier_result.fun,
                },
                "hierarchical_history": hier_result.history,
                "covariance_is_placeholder": covariance_is_placeholder,
            },
        }
        logger.info(f"  Hierarchical result: success={hier_result.success}")
        logger.info(f"  Outer iterations: {hier_result.n_outer_iterations}")
        logger.info(f"  Final loss: {hier_result.fun:.6e}")
        logger.info("=" * 60)
    else:
        # Standard hybrid streaming optimization path
        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=active_model_fn,
            p0=fit_initial_params,
            bounds=fit_bounds,
            verbose=1,
        )

    opt_time = time.perf_counter() - opt_start
    total_time = time.perf_counter() - start_time

    # Extract diagnostics from NLSQ result structure
    # NLSQ uses nested dicts: streaming_diagnostics -> phase_iterations/warmup_diagnostics
    diagnostics = result.get("streaming_diagnostics", {})
    phase_iterations = diagnostics.get("phase_iterations", {})
    warmup_diag = diagnostics.get("warmup_diagnostics", {})
    gn_diag = diagnostics.get("gauss_newton_diagnostics", {})

    lbfgs_epochs = phase_iterations.get("phase1", 0)
    gn_iterations = phase_iterations.get("phase2", 0)
    final_lbfgs_loss = warmup_diag.get("final_loss", float("inf"))
    final_gn_cost = gn_diag.get("final_cost", float("inf"))

    logger.info("=" * 80)
    logger.info("HYBRID STREAMING OPTIMIZATION COMPLETE")
    logger.info(f"  Success: {result.get('success', False)}")
    logger.info(f"  L-BFGS final loss: {final_lbfgs_loss:.6e}")
    logger.info(f"  GN final cost: {final_gn_cost:.6e}")
    logger.info(f"  L-BFGS epochs: {lbfgs_epochs}")
    logger.info(f"  GN iterations: {gn_iterations}")
    logger.info(f"  Optimization time: {opt_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info("=" * 80)

    # Extract results
    popt = np.asarray(result["x"])

    # =====================================================================
    # Anti-Degeneracy Defense System v2.9.0 - INVERSE TRANSFORMATION
    # =====================================================================
    # Transform Fourier coefficients back to per-angle parameters
    if use_fourier:
        assert fourier_reparameterizer is not None  # guarded by use_fourier
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Inverse Fourier Transform")
        # Use n_coeffs_per_param (e.g., 5 for order=2), NOT n_coeffs (total=10)
        # Layout: [contrast_coeffs (5), offset_coeffs (5), physical (7)]
        n_half = fourier_reparameterizer.n_coeffs_per_param

        # Extract Fourier coefficients and physical params from optimized result
        fourier_contrast_coeffs = popt[:n_half]
        fourier_offset_coeffs = popt[n_half : 2 * n_half]
        physical_params_opt = popt[2 * n_half :]

        # Transform back to per-angle parameters
        contrast_per_angle_opt = fourier_reparameterizer.from_fourier(
            fourier_contrast_coeffs
        )
        offset_per_angle_opt = fourier_reparameterizer.from_fourier(
            fourier_offset_coeffs
        )

        # Reconstruct full parameter vector in original layout
        popt = np.concatenate(
            [contrast_per_angle_opt, offset_per_angle_opt, physical_params_opt]
        )

        logger.info(f"  Fourier params: {2 * n_half + len(physical_params_opt)}")
        logger.info(f"  Restored per-angle params: {len(popt)}")

        # Transform covariance from Fourier space to per-angle space
        # J_fourier = d(per_angle)/d(fourier_coeffs)
        # pcov_per_angle = J_full @ pcov_fourier @ J_full.T
        pcov_fourier = result.get("pcov", None)
        n_fourier_total = 2 * n_half + len(physical_params_opt)

        if (
            pcov_fourier is not None
            and pcov_fourier.shape[0] == n_fourier_total
            and pcov_fourier.shape[1] == n_fourier_total
        ):
            # Get Jacobian for per-angle transformation
            # This is the Fourier basis matrix that maps coefficients to per-angle values
            jacobian_per_angle = fourier_reparameterizer.get_jacobian_transform()
            # jacobian_per_angle shape: (2 * n_phi, n_coeffs_fourier)
            # where n_coeffs_fourier = 2 * n_half

            # Build full Jacobian for complete parameter space transformation
            # Layout: [n_phi contrast, n_phi offset, n_physical]
            # Fourier layout: [n_half contrast_coeffs, n_half offset_coeffs, n_physical]
            n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
            n_physical = len(physical_params_opt)
            n_total_restored = n_per_angle_total + n_physical

            J_full = np.zeros((n_total_restored, n_fourier_total))
            # Block for per-angle params: use Fourier Jacobian
            J_full[:n_per_angle_total, : 2 * n_half] = jacobian_per_angle
            # Block for physical params: identity (pass-through)
            J_full[n_per_angle_total:, 2 * n_half :] = np.eye(n_physical)

            # Transform covariance: pcov_full = J @ pcov_fourier @ J.T
            try:
                pcov_transformed = J_full @ pcov_fourier @ J_full.T
                # Store for later use (override the result dict lookup)
                result["pcov_transformed"] = pcov_transformed
                logger.info("  Covariance transformed from Fourier to per-angle space")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning(
                    f"  Covariance transformation failed: {e}. Using identity fallback."
                )
                result["pcov_transformed"] = None
        else:
            pcov_shape = pcov_fourier.shape if pcov_fourier is not None else None
            logger.warning(
                f"  Fourier covariance unavailable or wrong shape (got {pcov_shape}, "
                f"expected ({n_fourier_total}, {n_fourier_total})). "
                "Using identity fallback."
            )
            result["pcov_transformed"] = None

        logger.info("=" * 60)

    # v2.17.0: Fixed scaling mode inverse transformation
    # Expand physical-only params back to per-angle format using fixed scaling arrays
    elif use_fixed_scaling:
        assert (
            fixed_contrast_per_angle is not None
        )  # set when use_fixed_scaling is True
        assert fixed_offset_per_angle is not None  # set when use_fixed_scaling is True
        logger.info("=" * 60)
        logger.info(
            "ANTI-DEGENERACY EXECUTION: Inverse Fixed Scaling Transform (v2.17.0)"
        )
        # Layout: [physical_params] - popt contains ONLY physical parameters
        physical_params_opt = popt

        # Use the pre-computed fixed per-angle scaling from quantiles
        contrast_per_angle_opt = fixed_contrast_per_angle
        offset_per_angle_opt = fixed_offset_per_angle

        # Reconstruct full parameter vector in original layout
        popt = np.concatenate(
            [contrast_per_angle_opt, offset_per_angle_opt, physical_params_opt]
        )

        logger.info(f"  Physical params: {len(physical_params_opt)}")
        logger.info(f"  Fixed per-angle scaling restored: {len(popt)} total params")
        logger.info(
            f"  Contrast (fixed): mean={np.nanmean(contrast_per_angle_opt):.4f}, "
            f"range=[{np.nanmin(contrast_per_angle_opt):.4f}, {np.nanmax(contrast_per_angle_opt):.4f}]"
        )
        logger.info(
            f"  Offset (fixed): mean={np.nanmean(offset_per_angle_opt):.4f}, "
            f"range=[{np.nanmin(offset_per_angle_opt):.4f}, {np.nanmax(offset_per_angle_opt):.4f}]"
        )

        # Transform covariance from physical-only space to full space
        # For fixed scaling mode, the Jacobian is simpler:
        # Per-angle params are fixed (variance = 0), physical params have identity
        # J[i, j] = 0 for per-angle params (i < 2*n_phi)
        # J[2*n_phi+i, i] = 1 for physical params (identity)
        pcov_physical = result.get("pcov", None)
        n_physical = len(physical_params_opt)

        if (
            pcov_physical is not None
            and pcov_physical.shape[0] == n_physical
            and pcov_physical.shape[1] == n_physical
        ):
            n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
            n_total_restored = n_per_angle_total + n_physical

            # Build full covariance matrix
            # Per-angle params have zero covariance (they're fixed)
            # Physical params have the original covariance
            try:
                pcov_full = np.zeros((n_total_restored, n_total_restored))
                # Physical params covariance block
                pcov_full[2 * n_phi :, 2 * n_phi :] = pcov_physical
                result["pcov_transformed"] = pcov_full
                logger.info(
                    "  Covariance expanded: per-angle=0 (fixed), physical=preserved"
                )
            except (
                ValueError,
                RuntimeError,
                MemoryError,
                np.linalg.LinAlgError,
            ) as e:
                logger.warning(
                    f"  Covariance expansion failed: {e}. Using identity fallback."
                )
                result["pcov_transformed"] = None
        else:
            pcov_shape = pcov_physical.shape if pcov_physical is not None else None
            logger.warning(
                f"  Physical covariance unavailable or wrong shape (got {pcov_shape}, "
                f"expected ({n_physical}, {n_physical})). "
                "Using identity fallback."
            )
            result["pcov_transformed"] = None

        logger.info("=" * 60)

    # T046-T049: Auto averaged mode inverse transformation
    # Expand averaged parameters back to per-angle format for backward compatibility
    elif use_averaged_scaling:
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Inverse Auto Averaged Transform")
        # Layout: [contrast_const, offset_const, physical_params]
        from xpcsjax.optimization.nlsq.data_prep import (
            expand_per_angle_parameters,
        )

        contrast_const = popt[0]
        offset_const = popt[1]
        n_physical_opt = len(popt) - 2
        expanded = expand_per_angle_parameters(
            popt,
            None,
            n_phi,
            n_physical_opt,
        )
        popt = expanded.params

        logger.info(f"  Constant params: 2 + {n_physical_opt} physical")
        logger.info(f"  Restored per-angle params: {len(popt)}")
        logger.info(f"  Contrast (uniform): {contrast_const:.6f}")
        logger.info(f"  Offset (uniform): {offset_const:.6f}")

        # Transform covariance from constant space to per-angle space
        # For constant mode, the Jacobian is simpler: broadcasting matrix
        # J[i, 0] = 1 for i in 0..n_phi-1 (contrast params)
        # J[n_phi+i, 1] = 1 for i in 0..n_phi-1 (offset params)
        # J[2*n_phi+i, 2+i] = 1 for physical params (identity)
        pcov_constant = result.get("pcov", None)
        n_constant_total = 2 + n_physical_opt

        if (
            pcov_constant is not None
            and pcov_constant.shape[0] == n_constant_total
            and pcov_constant.shape[1] == n_constant_total
        ):
            n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
            n_physical = n_physical_opt
            n_total_restored = n_per_angle_total + n_physical

            # Build Jacobian for constant → per-angle transformation
            J_full = np.zeros((n_total_restored, n_constant_total))
            # Contrast broadcast: d(contrast_per_angle[i])/d(contrast_const) = 1
            J_full[:n_phi, 0] = 1.0
            # Offset broadcast: d(offset_per_angle[i])/d(offset_const) = 1
            J_full[n_phi : 2 * n_phi, 1] = 1.0
            # Physical params: identity (pass-through)
            J_full[2 * n_phi :, 2:] = np.eye(n_physical)

            # Transform covariance: pcov_full = J @ pcov_constant @ J.T
            try:
                pcov_transformed = J_full @ pcov_constant @ J_full.T
                result["pcov_transformed"] = pcov_transformed
                logger.info("  Covariance transformed from constant to per-angle space")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning(
                    f"  Covariance transformation failed: {e}. Using identity fallback."
                )
                result["pcov_transformed"] = None
        else:
            pcov_shape = pcov_constant.shape if pcov_constant is not None else None
            logger.warning(
                f"  Constant covariance unavailable or wrong shape (got {pcov_shape}, "
                f"expected ({n_constant_total}, {n_constant_total})). "
                "Using identity fallback."
            )
            result["pcov_transformed"] = None

        logger.info("=" * 60)

    # Log gradient monitor summary if available
    if gradient_monitor is not None:
        gradient_monitor.log_summary()
        if gradient_monitor.collapse_detected:
            logger.warning("=" * 60)
            logger.warning("GRADIENT COLLAPSE WAS DETECTED DURING OPTIMIZATION")
            logger.warning(
                f"  Collapse events: {len(gradient_monitor.collapse_events)}"
            )
            for event in gradient_monitor.collapse_events:
                logger.warning(
                    f"    Iteration {event.iteration}: ratio={event.ratio:.6f}"
                )
            logger.warning("=" * 60)

    # Get covariance (properly transformed from normalized space)
    # Priority: 1) pcov_transformed (from Fourier space), 2) pcov, 3) identity fallback
    pcov = result.get("pcov_transformed", None)
    if pcov is None:
        pcov = result.get("pcov", None)
    if pcov is None or pcov.shape[0] != len(popt):
        logger.debug(
            f"Covariance size mismatch or unavailable: expected ({len(popt)}, {len(popt)}), "
            f"got {pcov.shape if pcov is not None else None}. Using identity fallback."
        )
        pcov = np.eye(len(popt))

    # Enforce bounds on final parameters
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        popt = np.clip(popt, lower_bounds, upper_bounds)

    # Check for parameters stuck at bounds with zero/near-zero uncertainty
    # This indicates the optimizer could not move these parameters away from bounds
    bound_stuck_warning = None
    if bounds is not None and is_laminar_flow:
        perr = safe_uncertainties_from_pcov(pcov, len(popt))
        param_statuses = _classify_parameter_status(
            popt, lower_bounds, upper_bounds, atol=1e-6
        )

        # Map indices to physical parameter names for laminar_flow mode
        # Layout: [n_phi contrasts] + [n_phi offsets] + [7 physical params]
        physical_indices_list = list(range(2 * n_phi, len(popt)))
        physical_param_names_local = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

        bound_stuck_params = []
        for i, idx in enumerate(physical_indices_list):
            if idx < len(param_statuses) and idx < len(popt):
                status = param_statuses[idx]
                uncertainty = perr[idx] if idx < len(perr) else 0.0
                if status != "active" and (uncertainty == 0.0 or uncertainty < 1e-15):
                    param_name = (
                        physical_param_names_local[i]
                        if i < len(physical_param_names_local)
                        else f"param[{idx}]"
                    )
                    bound_stuck_params.append(
                        (param_name, status, popt[idx], uncertainty)
                    )

        if bound_stuck_params:
            logger.warning("=" * 80)
            logger.warning("PARAMETER BOUNDS WARNING")
            logger.warning(
                "The following parameters are stuck at bounds with zero uncertainty:"
            )
            for param_name, status, value, unc in bound_stuck_params:
                logger.warning(
                    f"  {param_name}: {value:.6e} ({status}, uncertainty={unc:.2e})"
                )
            logger.warning("")
            logger.warning("This may indicate:")
            logger.warning(
                "  1. The optimizer cannot find gradient information for these parameters"
            )
            logger.warning("  2. The initial guess was already at or near the bounds")
            logger.warning(
                "  3. The model is insensitive to these parameters with this data coverage"
            )
            logger.warning("")
            logger.warning("RECOMMENDED ACTIONS:")
            logger.warning(
                "  - Enable phi_filtering to use only angles near 0 and 90 deg for laminar flow"
            )
            logger.warning(
                "  - Use multi-start optimization to explore multiple parameter basins"
            )
            logger.warning(
                "  - Check if gamma_dot_t0 ~ 0 means shear contribution is missing"
            )
            logger.warning("=" * 80)

            # Store for info dict
            bound_stuck_warning = {
                "parameters_at_bounds": [
                    {
                        "name": name,
                        "status": status,
                        "value": float(val),
                        "uncertainty": float(unc),
                    }
                    for name, status, val, unc in bound_stuck_params
                ]
            }

    # Build info dict
    info: dict[str, Any] = {
        "success": result.get("success", False),
        "message": result.get("message", "Hybrid streaming optimization completed"),
        "nfev": result.get("function_evaluations", 0),
        "nit": lbfgs_epochs + gn_iterations,
        "final_loss": final_gn_cost
        if final_gn_cost != float("inf")
        else final_lbfgs_loss,
        "lbfgs_epochs": lbfgs_epochs,
        "gauss_newton_iterations": gn_iterations,
        "optimization_time": opt_time,
        "total_time": total_time,
        "method": "adaptive_hybrid_streaming",
        "hybrid_streaming_diagnostics": diagnostics,
    }

    # Add anti-degeneracy defense diagnostics
    info["anti_degeneracy"] = {
        "version": "2.18.0",
        "per_angle_mode": anti_degeneracy_components["per_angle_mode"],
        "use_constant": anti_degeneracy_components.get("use_constant", False),
        "use_fixed_scaling": use_fixed_scaling,
        "fourier_enabled": fourier_reparameterizer is not None,
        "hierarchical_enabled": hierarchical_optimizer is not None,
        "adaptive_regularization_enabled": adaptive_regularizer is not None,
        "gradient_monitor_enabled": gradient_monitor is not None,
        "shear_weighting_enabled": shear_weighter is not None,
    }
    if fourier_reparameterizer is not None:
        info["anti_degeneracy"]["fourier"] = {
            "order": fourier_order,
            "n_coeffs": fourier_reparameterizer.n_coeffs,
            "param_reduction": f"{2 * n_phi} -> {fourier_reparameterizer.n_coeffs}",
        }
    # T048: Add constant mode diagnostics
    if use_fixed_scaling:
        # v2.18.0: Fixed scaling mode - per-angle values are fixed, not optimized
        assert (
            fixed_contrast_per_angle is not None
        )  # set when use_fixed_scaling is True
        assert fixed_offset_per_angle is not None  # set when use_fixed_scaling is True
        info["anti_degeneracy"]["fixed_scaling"] = {
            "param_reduction": f"{2 * n_phi} -> 0 (physical only)",
            "method": "quantile_estimation",
            "contrast_mean": float(np.nanmean(fixed_contrast_per_angle)),
            "contrast_range": [
                float(np.nanmin(fixed_contrast_per_angle)),
                float(np.nanmax(fixed_contrast_per_angle)),
            ],
            "offset_mean": float(np.nanmean(fixed_offset_per_angle)),
            "offset_range": [
                float(np.nanmin(fixed_offset_per_angle)),
                float(np.nanmax(fixed_offset_per_angle)),
            ],
        }
    elif use_averaged_scaling:
        # v2.18.0: Auto averaged mode - averaged values are OPTIMIZED
        info["anti_degeneracy"]["auto_averaged"] = {
            "param_reduction": f"{2 * n_phi} -> 2 (averaged scaling)",
            "method": "quantile_estimation_averaged",
            # After inverse transform, popt[0] is first contrast (uniform)
            "contrast_optimized": float(popt[0]) if len(popt) > 0 else None,
            "offset_optimized": float(popt[n_phi]) if len(popt) > n_phi else None,
        }
    if hierarchical_optimizer is not None:
        info["anti_degeneracy"]["hierarchical"] = (
            hierarchical_optimizer.get_diagnostics()
        )
    if adaptive_regularizer is not None:
        info["anti_degeneracy"]["regularization"] = (
            adaptive_regularizer.get_diagnostics()
        )
    if gradient_monitor is not None:
        info["anti_degeneracy"]["gradient_monitor"] = gradient_monitor.get_diagnostics()
    if shear_weighter is not None:
        info["anti_degeneracy"]["shear_weighting"] = shear_weighter.get_diagnostics()

    # Add bounds warning info if detected
    if bound_stuck_warning is not None:
        info["bound_stuck_warning"] = bound_stuck_warning

    # Check for shear collapse: gamma_dot_t0 essentially zero
    if is_laminar_flow and len(popt) > 2 * n_phi + 3:
        gamma_dot_t0_idx = 2 * n_phi + 3
        gamma_dot_t0_value = popt[gamma_dot_t0_idx]
        # Check if shear rate is effectively zero (< 1e-5 s^-1)
        if abs(gamma_dot_t0_value) < 1e-5:
            logger.warning("=" * 80)
            logger.warning("SHEAR COLLAPSE WARNING")
            logger.warning(
                f"gamma_dot_t0 = {gamma_dot_t0_value:.2e} s^-1 is effectively zero"
            )
            logger.warning("")
            logger.warning("This means the shear contribution to g1 is negligible.")
            logger.warning(
                "The model has effectively collapsed to static_isotropic mode."
            )
            logger.warning("")
            logger.warning("POSSIBLE CAUSES:")
            logger.warning("  1. Per-angle contrast/offset absorbed the shear signal")
            logger.warning(
                "  2. Inconsistent initialization of per-angle vs physical params"
            )
            logger.warning(
                "  3. Physical parameters at bounds with weak gradient signal"
            )
            logger.warning("  4. The data may genuinely have no measurable shear")
            logger.warning("")
            logger.warning("RECOMMENDED ACTIONS:")
            logger.warning(
                "  - Enable multi-start optimization to explore parameter basins"
            )
            logger.warning(
                "  - Check reduced chi-squared: if worse than expected, re-run optimization"
            )
            logger.warning(
                "  - Verify per-angle contrast/offset are not varying excessively"
            )
            logger.warning(
                "  - Consider static_isotropic mode if shear is truly absent"
            )
            logger.warning("=" * 80)
            info["shear_collapse_warning"] = {
                "gamma_dot_t0": float(gamma_dot_t0_value),
                "threshold": 1e-5,
                "message": "Shear contribution effectively zero",
            }

    return popt, pcov, info


def estimate_memory_for_stratified_ls(
    n_points: int,
    n_params: int,
    n_chunks: int,
) -> float:
    """Estimate peak memory usage for stratified least-squares optimization.

    The main memory consumers are:
    1. Padded arrays: n_chunks × max_chunk_size × 5 arrays × 8 bytes
    2. Dense Jacobian: n_points × n_params × 8 bytes
    3. JAX autodiff intermediates: ~3× Jacobian size for backprop
    4. JAX compilation cache: ~5-10 GB

    Args:
        n_points: Total number of data points
        n_params: Number of parameters
        n_chunks: Number of stratified chunks

    Returns:
        Estimated peak memory in bytes
    """
    bytes_per_float = 8

    # Padded arrays (5 arrays: phi, t1, t2, g2, mask)
    max_chunk_size = (n_points + n_chunks - 1) // n_chunks
    padded_arrays = n_chunks * max_chunk_size * 5 * bytes_per_float

    # Dense Jacobian
    jacobian = n_points * n_params * bytes_per_float

    # JAX autodiff intermediates (keep all grids for backprop)
    # This is the main memory killer - originally estimated at 3× Jacobian
    # but empirical testing shows 5× is more accurate for large datasets
    # (C020 dataset: estimated 44.9 GB at 3×, actual ~60 GB at 96% pressure)
    autodiff_intermediates = jacobian * 5

    # JAX compilation cache
    jax_cache = 5 * 1e9  # ~5 GB

    total = padded_arrays + jacobian + autodiff_intermediates + jax_cache

    return total


def should_use_streaming(
    n_points: int,
    n_params: int,
    n_chunks: int,
    memory_threshold_gb: float | None = None,
    memory_fraction: float | None = None,
) -> tuple[bool, float, str]:
    """Determine if streaming optimizer should be used based on memory estimate.

    Uses adaptive memory thresholding (v2.7.0+) to automatically compute
    an appropriate threshold based on total system memory.

    Args:
        n_points: Total number of data points
        n_params: Number of parameters
        n_chunks: Number of stratified chunks
        memory_threshold_gb: Memory threshold in GB above which to use streaming.
            If None (default), computes adaptive threshold as 75% of total memory.
        memory_fraction: Fraction of total memory for adaptive threshold (0.1-0.9).
            Only used if memory_threshold_gb is None.

    Returns:
        (use_streaming, estimated_gb, reason) tuple
    """
    try:
        import psutil
    except ImportError:
        from xpcsjax.optimization.nlsq.memory import detect_total_system_memory

        total_bytes = detect_total_system_memory()
        if total_bytes is not None:
            total_gb = total_bytes / (1024**3)
            return (
                False,
                0.0,
                f"psutil not available; system has {total_gb:.1f} GB",
            )
        return (False, 0.0, "psutil not available; system memory unknown")

    # Compute adaptive threshold if not explicitly provided
    if memory_threshold_gb is None:
        memory_threshold_gb, threshold_info = get_adaptive_memory_threshold(
            memory_fraction=memory_fraction
        )
        _memory_logger.debug(
            f"_should_use_streaming using adaptive threshold: "
            f"{memory_threshold_gb:.1f} GB ({threshold_info})"
        )

    # Get available system memory
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    # Estimate memory for stratified LS
    estimated_bytes = estimate_memory_for_stratified_ls(n_points, n_params, n_chunks)
    estimated_gb = estimated_bytes / (1024**3)

    # Decision logic
    # Use streaming if:
    # 1. Estimated memory exceeds threshold, OR
    # 2. Estimated memory exceeds 85% of available memory
    #
    # Note: Increased from 70% to 85% because non-streaming Levenberg-Marquardt
    # is more accurate than streaming optimization. The 85% threshold allows
    # more datasets to use the preferred non-streaming path.
    use_streaming = False
    reason = ""

    if estimated_gb > memory_threshold_gb:
        use_streaming = True
        reason = (
            f"Estimated memory ({estimated_gb:.1f} GB) exceeds "
            f"threshold ({memory_threshold_gb:.1f} GB)"
        )
    elif estimated_gb > available_gb * 0.85:
        use_streaming = True
        reason = (
            f"Estimated memory ({estimated_gb:.1f} GB) exceeds "
            f"85% of available memory ({available_gb:.1f} GB available)"
        )
    else:
        reason = (
            f"Estimated memory ({estimated_gb:.1f} GB) within limits "
            f"(threshold={memory_threshold_gb:.1f} GB, "
            f"available={available_gb:.1f} GB)"
        )

    return use_streaming, estimated_gb, reason
