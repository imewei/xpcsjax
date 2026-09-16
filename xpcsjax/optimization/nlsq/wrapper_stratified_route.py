"""Shared stratified-least-squares (>=1M-point) fit route for :class:`NLSQWrapper.fit`.

``NLSQWrapper.fit`` activates this route when angle-stratified chunking has
produced a flat dataset of at least 1,000,000 points with per-angle scaling
enabled (the "double-chunking" case NLSQ's own ``curve_fit``/``curve_fit_large``
cannot handle directly). This ~640-line block used to live inline inside
``NLSQWrapper.fit``: it re-validates bounds, expands per-angle parameters,
re-runs strategy selection (recreating a second HYBRID_STREAMING/OUT_OF_CORE
decision from the stratified, per-angle-expanded point/parameter counts --
distinct from the initial, pre-stratification decision earlier in
``fit()``), tries hybrid streaming, falls back to stratified least-squares,
and assembles the :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.

D1 of the 2026-09-15 codebase review (deferred by A4/bf8f35a, which extracted
the sibling out-of-core route into :mod:`wrapper_out_of_core_route` first).
This module is the single owner of that route now; ``wrapper.py`` funnels
through :func:`run_stratified_ls_route`.

Returns ``None`` to signal "fall through to the standard in-memory path" --
the one case where the original inline block did not return: a
``(ValueError, RuntimeError, OSError)`` from the stratified least-squares
solve itself (a ``MemoryError`` still propagates -- the dense fallback needs
STRICTLY MORE memory than the stratified route that just OOM'd, so masking it
as recoverable only guarantees a second, worse OOM).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.memory import (
    NLSQStrategy,
    get_adaptive_memory_threshold,
)
from xpcsjax.optimization.nlsq.result_helpers import _info_cov_placeholder
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.optimization.nlsq.strategies.chunking import StratificationDiagnostics
from xpcsjax.optimization.nlsq.strategies.residual import (
    create_stratified_residual_function,
)

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.parameter_utils import ResolvedPhysicalParameters
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

__all__ = ["run_stratified_ls_route"]


def run_stratified_ls_route(
    wrapper: NLSQWrapper,
    *,
    stratified_data: Any,
    data: Any,
    per_angle_scaling: bool,
    analysis_mode: AnalysisMode,
    config: Any,
    initial_params: np.ndarray | None,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    resolved_physical: ResolvedPhysicalParameters | None,
    logger: Any,
    start_time: float,
    stratification_diagnostics: StratificationDiagnostics | None,
) -> OptimizationResult | None:
    """Run the >=1M-point stratified-least-squares route.

    Parameters mirror what ``NLSQWrapper.fit`` has in hand at the point it
    used to inline this block -- see the module docstring for why this now
    lives in one place. Returns ``None`` when the caller should fall through
    to the standard in-memory path (see module docstring); every other
    outcome either returns an :class:`OptimizationResult` or raises.
    """
    import time

    # Deferred import, resolved at call time on purpose: tests force the
    # streaming tier by monkeypatching these flags on the wrapper module, and
    # _routing_effective_n_params is the wrapper's static-pin owner. (The
    # pure result helpers come from result_helpers at module level.)
    from xpcsjax.optimization.nlsq.wrapper import (
        HYBRID_STREAMING_AVAILABLE,
        STREAMING_AVAILABLE,
        _routing_effective_n_params,
    )

    logger.info("=" * 80)
    logger.info("STRATIFIED LEAST-SQUARES PATH ACTIVATED")
    logger.info("Solving double-chunking problem with NLSQ's least_squares()")
    logger.info("=" * 80)

    # Validate initial parameters
    if initial_params is None:
        raise ValueError("initial_params must be provided")
    validated_params = wrapper._validate_initial_params(initial_params, bounds)

    # Convert bounds
    nlsq_bounds = wrapper._convert_bounds(bounds)

    # Get physical parameter names for this analysis mode
    physical_param_names = wrapper._get_physical_param_names(analysis_mode)
    logger.info(f"Physical parameters for {analysis_mode}: {physical_param_names}")

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
        logger.info(f"  Input parameters: {len(validated_params)} (expected: {n_physical + 2})")

        # Validate input parameter count
        expected_input = n_physical + 2  # Physical params + single contrast + single offset
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

    logger.info(f"Parameter validation passed: {len(validated_params)} parameters")

    # Step: Re-run unified strategy selection with EFFECTIVE parameter count
    # (anti-degeneracy pre-check)
    #
    # The expanded param count (e.g. 53 for 23 angles individual) may be much
    # larger than the effective count after anti-degeneracy mode selection
    # (e.g. 9 for averaged). Using the expanded count for memory estimation
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

    n_angles_check = len(np.unique(stratified_data.phi_flat))
    if per_angle_scaling and config is not None and hasattr(config, "config"):
        nlsq_cfg = (config.config.get("optimization") or {}).get("nlsq") or {}
        ad_cfg = nlsq_cfg.get("anti_degeneracy", {})
        # Static-pinned resolver owner (spec Seam 1): same numeric outcome
        # as the former inline auto/constant ladder for laminar, but static
        # fits always route on the dense individual count so a static
        # auto/averaged/constant config never under-estimates Jacobian
        # memory and mis-routes a large dataset.
        effective_n_params = _routing_effective_n_params(
            analysis_mode,
            ad_cfg,
            n_phi=n_angles_check,
            n_physical=n_physical,
            actual_n_params=actual_n_params,
        )
        if effective_n_params < actual_n_params:
            logger.info(
                f"Anti-Degeneracy pre-check: effective params "
                f"{effective_n_params} (expanded: {actual_n_params})"
            )

    # Resolved through the wrapper module at call time, not the module-level
    # import above: tests (and any caller) that monkeypatch
    # ``xpcsjax.optimization.nlsq.wrapper.select_nlsq_strategy`` to force a
    # tier must still steer this re-check, exactly as they did when the
    # block lived inside ``NLSQWrapper.fit``. Function-local import: wrapper
    # imports this module lazily, so there is no cycle at import time.
    from xpcsjax.optimization.nlsq import wrapper as _wrapper_mod

    strategy_recheck = _wrapper_mod.select_nlsq_strategy(n_total_points, effective_n_params)

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

        from xpcsjax.optimization.nlsq.wrapper_out_of_core_route import (
            run_out_of_core_route,
        )

        return run_out_of_core_route(
            stratified_data=stratified_data,
            data=data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=validated_params,
            bounds=nlsq_bounds,
            logger=logger,
            config=config,
            analysis_mode=analysis_mode,
            n_points=n_total_points,
            n_phi=n_angles_check,
            resolved_physical=resolved_physical,
            start_time=start_time,
            strategy_reason=strategy_recheck.reason,
            recovery_tag="out_of_core_recheck_delegation",
            fast_mode_override=wrapper.fast_mode,
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

    # Compute adaptive memory threshold
    # Default: 75% of total system memory instead of fixed 16 GB
    memory_fraction: float | None = None  # Will use default or env var
    memory_threshold_gb: float | None = None  # Will be computed adaptively

    if config is not None and hasattr(config, "config"):
        strat_config = (config.config.get("optimization") or {}).get("stratification", {})
        target_chunk_size = strat_config.get("target_chunk_size", 100_000)

        # Extract streaming configuration
        nlsq_config = (config.config.get("optimization") or {}).get("nlsq") or {}
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
        logger.debug(f"Using explicit memory threshold from config: {memory_threshold_gb:.1f} GB")

    # Check for hybrid streaming mode (preferred for large datasets)
    if hybrid_streaming_config is not None:
        use_hybrid_streaming = hybrid_streaming_config.get("enable", False)

    # Check for forced streaming mode from config
    # Also set from strategy_recheck if it returned HYBRID_STREAMING
    if config is not None and hasattr(config, "config"):
        nlsq_config = (config.config.get("optimization") or {}).get("nlsq") or {}
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
            # Extract anti-degeneracy config for defense system
            anti_degeneracy_config = nlsq_config.get("anti_degeneracy", {})
            try:
                popt, pcov, info = wrapper._fit_with_stratified_hybrid_streaming(
                    stratified_data=stratified_data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=logger,
                    hybrid_config=hybrid_streaming_config,
                    anti_degeneracy_config=anti_degeneracy_config,
                    resolved_physical=resolved_physical,
                )

                # Compute final residuals for result creation
                chunked_data = wrapper._create_stratified_chunks(stratified_data, target_chunk_size)
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
                # In averaged mode, popt has compressed length (e.g. 9),
                # but the true model DOF is 2*n_phi + n_physical (e.g. 53).
                # A fixed physical parameter must not consume a DOF either --
                # subtract the fixed count from n_physical before computing
                # the constrained-mode formula, and override the
                # individual-mode None fallback (which would otherwise
                # default to len(popt), overcounting by the fixed count
                # since popt is restored to full length by this point).
                _hs_n_fixed_physical = (
                    0
                    if resolved_physical is None
                    else n_physical - int(resolved_physical.free_mask.sum())
                )
                _hs_n_params_effective: int | None = None
                if per_angle_scaling and anti_degeneracy_config:
                    from xpcsjax.optimization.nlsq.per_angle_mode import (
                        effective_constrained_dof as _eff_dof,
                    )
                    from xpcsjax.optimization.nlsq.per_angle_mode import (
                        resolve_per_angle_mode_static_pinned as _resolve_pam_pinned,
                    )

                    _hs_ad_mode = anti_degeneracy_config.get("per_angle_mode", "auto")
                    _hs_thresh = anti_degeneracy_config.get("constant_scaling_threshold", 3)
                    # Resolve through the static pin so a static fit's DOF reflects
                    # the dense individual vector (2*n_phi + n_physical), not the
                    # inert config token. EXPLICIT averaged still gets the expanded
                    # constrained DOF on laminar (Codex Finding 2).
                    _hs_n_params_effective = _eff_dof(
                        _resolve_pam_pinned(
                            _hs_ad_mode,
                            n_angles_check,
                            _hs_thresh,
                            is_laminar_flow=(analysis_mode == AnalysisMode.LAMINAR_FLOW),
                        ),
                        n_phi=n_angles_check,
                        n_physical=n_physical - _hs_n_fixed_physical,
                    )
                if _hs_n_params_effective is None and _hs_n_fixed_physical > 0:
                    _hs_n_params_effective = len(popt) - _hs_n_fixed_physical

                # Create result
                result = wrapper._create_fit_result(
                    popt=popt,
                    pcov=pcov,
                    residuals=final_residuals,
                    n_data=n_data,
                    iterations=info.get("nit", 0),
                    execution_time=execution_time,
                    convergence_status=("converged" if info.get("success", False) else "failed"),
                    convergence_reason=info.get("convergence_reason"),
                    solver_status=info.get("status"),
                    recovery_actions=["hybrid_streaming_optimizer_method"],
                    streaming_diagnostics=info.get("hybrid_streaming_diagnostics"),
                    stratification_diagnostics=stratification_diagnostics,
                    diagnostics_payload=None,
                    n_params_effective=_hs_n_params_effective,
                    anti_degeneracy_info=info.get("anti_degeneracy"),
                    covariance_is_placeholder=_info_cov_placeholder(info),
                )

                # A fixed physical parameter's true covariance diagonal is
                # exactly 0 -- `fit_with_stratified_hybrid_streaming`
                # already restores it that way. But `_create_fit_result`'s
                # `_safe_uncertainties_from_pcov` floors ANY near-zero
                # diagonal entry as a numerical-safety net for genuinely
                # singular/ill-conditioned solves; it cannot distinguish
                # "singular" from "deliberately fixed". Force the reported
                # uncertainty back to exactly 0.0 at every FIXED physical
                # position, mirroring `_post_process_results`'s equivalent
                # re-zero for the plain/out-of-core/stratified-LS tiers.
                from xpcsjax.optimization.nlsq.parameter_utils import (
                    zero_fixed_uncertainties,
                )

                result.uncertainties = zero_fixed_uncertainties(
                    result.uncertainties, resolved_physical
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
            # AdaptiveHybridStreamingOptimizer not available
            logger.error(
                "Streaming mode requested but AdaptiveHybridStreamingOptimizer "
                "not available. Upgrade NLSQ. "
                "Falling back to stratified least-squares."
            )
            # Fall through to stratified least-squares

    # Extract NLSQ config dict for tolerance propagation and anti-degeneracy
    nlsq_config_dict = None
    anti_degeneracy_config = None
    if config is not None and hasattr(config, "config"):
        nlsq_config_dict = (config.config.get("optimization") or {}).get("nlsq") or {}
        anti_degeneracy_config = nlsq_config_dict.get("anti_degeneracy", {})
        if anti_degeneracy_config:
            logger.info(
                f"Anti-Degeneracy config loaded: per_angle_mode="
                f"{anti_degeneracy_config.get('per_angle_mode', 'auto')}"
            )

    # Call stratified least_squares optimization
    try:
        popt, pcov, info = wrapper._fit_with_stratified_least_squares(
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
            resolved_physical=resolved_physical,
        )

        # Compute final residuals for result creation
        # We need to recreate the residual function to compute final residuals
        chunked_data = wrapper._create_stratified_chunks(stratified_data, target_chunk_size)
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
        # In averaged mode, popt has compressed length (e.g. 9),
        # but the true model DOF is 2*n_phi + n_physical (e.g. 53).
        _sls_n_params_effective: int | None = None
        if per_angle_scaling:
            from xpcsjax.optimization.nlsq.per_angle_mode import (
                effective_constrained_dof as _eff_dof,
            )
            from xpcsjax.optimization.nlsq.per_angle_mode import (
                resolve_per_angle_mode_static_pinned as _resolve_pam_pinned,
            )

            # An empty/absent anti_degeneracy block still activates averaged
            # scaling downstream (stratified_ls gates on `is not None`), so the
            # DOF must come from the resolved default mode, not len(popt).
            _sls_ad_cfg = anti_degeneracy_config or {}
            _sls_ad_mode = _sls_ad_cfg.get("per_angle_mode", "auto")
            _sls_thresh = _sls_ad_cfg.get("constant_scaling_threshold", 3)
            # Resolve through the static pin so a static fit's DOF reflects the
            # dense individual vector (2*n_phi + n_physical), not the inert config
            # token. EXPLICIT averaged still gets the expanded constrained DOF on
            # laminar (Codex Finding 2).
            _sls_n_params_effective = _eff_dof(
                _resolve_pam_pinned(
                    _sls_ad_mode,
                    n_angles_check,
                    _sls_thresh,
                    is_laminar_flow=(analysis_mode == AnalysisMode.LAMINAR_FLOW),
                ),
                n_phi=n_angles_check,
                n_physical=n_physical,
            )

        # Create result
        result = wrapper._create_fit_result(
            popt=popt,
            pcov=pcov,
            residuals=final_residuals,
            n_data=n_data,
            iterations=info.get("nit", 0),
            execution_time=execution_time,
            convergence_status=("converged" if info.get("success", False) else "failed"),
            convergence_reason=info.get("convergence_reason"),
            solver_status=info.get("status"),
            recovery_actions=["stratified_least_squares_method"],
            streaming_diagnostics=None,
            stratification_diagnostics=stratification_diagnostics,
            diagnostics_payload=None,
            n_params_effective=_sls_n_params_effective,
            anti_degeneracy_info=info.get("anti_degeneracy"),
            covariance_is_placeholder=_info_cov_placeholder(info),
        )

        # A fixed physical parameter's true covariance diagonal is
        # exactly 0 -- `fit_with_stratified_least_squares` already
        # restores it that way. But `_create_fit_result`'s
        # `_safe_uncertainties_from_pcov` floors ANY near-zero
        # diagonal entry as a numerical-safety net for genuinely
        # singular/ill-conditioned solves; it cannot distinguish
        # "singular" from "deliberately fixed". Force the reported
        # uncertainty back to exactly 0.0 at every FIXED physical
        # position, mirroring `_post_process_results`'s equivalent
        # re-zero for the plain/out-of-core tiers.
        from xpcsjax.optimization.nlsq.parameter_utils import (
            zero_fixed_uncertainties,
        )

        result.uncertainties = zero_fixed_uncertainties(result.uncertainties, resolved_physical)

        logger.info("=" * 80)
        logger.info("STRATIFIED LEAST-SQUARES COMPLETE")
        logger.info(
            f"Final chi2: {result.chi_squared:.4e}, Reduced chi2: {result.reduced_chi_squared:.4f}"
        )
        logger.info("=" * 80)

        return result

    except MemoryError:
        # The dense in-memory curve_fit_large fallback below needs
        # STRICTLY MORE memory than the stratified route that just
        # OOM'd (it re-materializes the full, unchunked residual/
        # Jacobian) -- falling through here only guarantees a second,
        # worse OOM. Propagate instead of masking it as recoverable.
        raise
    except (ValueError, RuntimeError, OSError) as e:
        logger.error(
            f"Stratified least_squares failed: {e}\n"
            f"Falling back to standard curve_fit_large path..."
        )
        # Fall through to standard optimization path below
        return None
