"""Fallback chain logic for NLSQ optimization strategy selection.

Extracted from wrapper.py to reduce file size and improve maintainability.

This module provides:
- OptimizationStrategy enum for strategy selection
- Strategy info retrieval for logging/diagnostics
- NLSQ result normalization across different return formats
- Fallback strategy chain (STREAMING -> CHUNKED -> LARGE -> STANDARD)
- Optimization execution with automatic fallback
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Local optimization strategy enum for internal use.

    Note: This replaces the deprecated selection.py OptimizationStrategy.
    For new code, use NLSQStrategy from memory.py instead.
    """

    STANDARD = "standard"
    LARGE = "large"
    CHUNKED = "chunked"
    STREAMING = "streaming"


def _get_strategy_info(strategy: OptimizationStrategy) -> dict:
    """Get information about a strategy for logging/diagnostics."""
    info = {
        OptimizationStrategy.STANDARD: {
            "name": "Standard",
            "supports_progress": False,
        },
        OptimizationStrategy.LARGE: {
            "name": "Large",
            "supports_progress": True,
        },
        OptimizationStrategy.CHUNKED: {
            "name": "Chunked",
            "supports_progress": True,
        },
        OptimizationStrategy.STREAMING: {
            "name": "Streaming",
            "supports_progress": True,
        },
    }
    return info.get(strategy, {"name": "Unknown", "supports_progress": False})


def get_fallback_strategy(
    current_strategy: OptimizationStrategy,
) -> OptimizationStrategy | None:
    """Get fallback strategy when current strategy fails.

    Implements the degradation chain
    ``STREAMING -> CHUNKED -> LARGE -> STANDARD -> None``.

    Parameters
    ----------
    current_strategy : OptimizationStrategy
        Strategy that failed.

    Returns
    -------
    OptimizationStrategy | None
        Next strategy to try, or None if no fallback is available.
    """
    fallback_chain = {
        OptimizationStrategy.STREAMING: OptimizationStrategy.CHUNKED,
        OptimizationStrategy.CHUNKED: OptimizationStrategy.LARGE,
        OptimizationStrategy.LARGE: OptimizationStrategy.STANDARD,
        OptimizationStrategy.STANDARD: None,  # No further fallback
    }
    return fallback_chain.get(current_strategy)


def handle_nlsq_result(
    result: Any,
    strategy: OptimizationStrategy,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalize NLSQ return values to consistent format.

    NLSQ v0.1.5 has inconsistent return types across different functions:
    - curve_fit: Returns tuple (popt, pcov) OR CurveFitResult object
    - curve_fit_large: Returns tuple (popt, pcov) OR OptimizeResult object
    - StreamingOptimizer.fit: Returns dict with 'x', 'pcov', 'streaming_diagnostics'

    This function normalizes all these formats to a consistent
    ``(popt, pcov, info)`` tuple.

    Parameters
    ----------
    result : Any
        Return value from an NLSQ optimization call.
    strategy : OptimizationStrategy
        Optimization strategy used (for logging/diagnostics).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        ``(popt, pcov, info)`` where ``popt`` is the optimized parameters,
        ``pcov`` is the covariance matrix (identity if missing), and ``info``
        is a dict of additional information (empty if not available).

    Raises
    ------
    AttributeError
        If an object result has neither an ``x`` nor a ``popt`` attribute.
    TypeError
        If the result format is unrecognized or a tuple has an unexpected
        length.
    """
    _logger = get_logger(__name__)

    # Case 1: Dict (from StreamingOptimizer)
    if isinstance(result, dict):
        popt = np.asarray(result.get("x", result.get("popt")))
        pcov = np.asarray(result.get("pcov", np.eye(len(popt))))  # Identity if missing
        info = {
            "streaming_diagnostics": result.get("streaming_diagnostics", {}),
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "best_loss": result.get("best_loss", None),
            "final_epoch": result.get("final_epoch", None),
        }
        _logger.debug(f"Normalized StreamingOptimizer dict result (strategy: {strategy.value})")
        return popt, pcov, info

    # Case 2: Tuple with 2 or 3 elements
    if isinstance(result, tuple):
        if len(result) == 2:
            # (popt, pcov) - most common case
            popt, pcov = result
            info = {}
            _logger.debug(f"Normalized (popt, pcov) tuple (strategy: {strategy.value})")
        elif len(result) == 3:
            # (popt, pcov, info) - from curve_fit with full_output=True
            popt, pcov, info = result
            # Ensure info is a dict
            if not isinstance(info, dict):
                _logger.warning(f"Info object is not a dict: {type(info)}. Converting to dict.")
                info = {"raw_info": info}
            _logger.debug(f"Normalized (popt, pcov, info) tuple (strategy: {strategy.value})")
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(result)}. "
                f"Expected 2 (popt, pcov) or 3 (popt, pcov, info). "
                f"Got: {result}"
            )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
    if hasattr(result, "x") or hasattr(result, "popt"):
        # Extract popt
        popt_raw = getattr(result, "x", getattr(result, "popt", None))
        if popt_raw is None:
            raise AttributeError(
                f"Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(result)}"
            )
        popt = np.asarray(popt_raw)

        # Extract pcov
        pcov_raw = getattr(result, "pcov", None)
        if pcov_raw is None:
            # No covariance available, create identity matrix
            _logger.warning("No pcov attribute in result object. Using identity matrix.")
            pcov = np.eye(len(popt))
        else:
            pcov = np.asarray(pcov_raw)

        # Extract info dict
        info = {}
        # Common attributes to extract
        for attr in [
            "message",
            "success",
            "nfev",
            "njev",
            "fun",
            "jac",
            "optimality",
        ]:
            if hasattr(result, attr):
                info[attr] = getattr(result, attr)

        # Check for 'info' attribute (some objects nest additional info)
        if hasattr(result, "info") and isinstance(result.info, dict):
            info.update(result.info)

        _logger.debug(
            f"Normalized object result (type: {type(result).__name__}, strategy: {strategy.value})"
        )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 4: Unrecognized format
    raise TypeError(
        f"Unrecognized NLSQ result format: {type(result)}. "
        f"Expected tuple, dict, or object with 'x'/'popt' attributes. "
        f"Available attributes: {dir(result) if hasattr(result, '__dict__') else 'N/A'}"
    )


def execute_optimization_with_fallback(
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
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    enable_recovery: bool,
    execute_with_recovery_fn: Callable,
    fit_with_hybrid_streaming_fn: Callable,
    streaming_available: bool,
    curve_fit_fn: Callable,
    curve_fit_large_fn: Callable,
    fast_mode: bool = False,
    callback: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[str], str]:
    """Execute optimization with automatic strategy fallback.

    Tries the selected strategy first, then degrades to simpler strategies via
    :func:`get_fallback_strategy` until one succeeds or all are exhausted.

    Parameters
    ----------
    strategy : OptimizationStrategy
        Initial strategy to attempt.
    wrapped_residual_fn : Callable
        Residual function passed to the NLSQ solvers.
    xdata, ydata : np.ndarray
        Independent and dependent data arrays.
    validated_params : np.ndarray
        Validated initial parameter guess.
    nlsq_bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds as ``(lower, upper)``, or None.
    loss_name : str
        Loss function name.
    x_scale_value : float | str
        Parameter scaling value (or ``"jac"``-style string) for NLSQ.
    config : Any
        NLSQ configuration object.
    start_time : float
        Optimization start time (for elapsed-time reporting on failure).
    log : logging.Logger | logging.LoggerAdapter
        Logger instance.
    enable_recovery : bool
        Whether to route through ``execute_with_recovery_fn``.
    execute_with_recovery_fn : Callable
        Recovery-enabled execution function.
    fit_with_hybrid_streaming_fn : Callable
        Hybrid-streaming fit function (used for the STREAMING strategy).
    streaming_available : bool
        Whether the streaming backend is available.
    curve_fit_fn : Callable
        NLSQ standard ``curve_fit`` callable.
    curve_fit_large_fn : Callable
        NLSQ large-dataset ``curve_fit`` callable.
    fast_mode : bool, optional
        Reserved fast-mode flag.
    callback : Callable | None, optional
        Per-iteration L4 monitor callback (strictly observational).

    Returns
    -------
    tuple
        ``(popt, pcov, info, recovery_actions, convergence_status)``.

    Raises
    ------
    RuntimeError
        If every strategy in the fallback chain fails.
    """
    current_strategy = strategy
    strategy_attempts: list[OptimizationStrategy] = []

    while current_strategy is not None:
        try:
            strategy_info = _get_strategy_info(current_strategy)
            log.info(f"Attempting optimization with {current_strategy.value} strategy...")

            if current_strategy == OptimizationStrategy.STREAMING and streaming_available:
                log.info("Using NLSQ AdaptiveHybridStreamingOptimizer for large datasets...")

                popt, pcov, info = fit_with_hybrid_streaming_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=log,
                    nlsq_config=config,
                )
                recovery_actions = info.get("recovery_actions", [])
                convergence_status = "converged" if info.get("success", False) else "partial"

            elif enable_recovery:
                popt, pcov, info, recovery_actions, convergence_status = execute_with_recovery_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    strategy=current_strategy,
                    logger=log,
                    loss_name=loss_name,
                    x_scale_value=x_scale_value,
                    callback=callback,
                )
            else:
                use_large = current_strategy != OptimizationStrategy.STANDARD

                if use_large:
                    result_tuple = curve_fit_large_fn(
                        wrapped_residual_fn,
                        xdata,
                        ydata,
                        p0=validated_params.tolist(),
                        bounds=nlsq_bounds if nlsq_bounds is not None else (-np.inf, np.inf),
                        loss=loss_name,
                        x_scale=x_scale_value,
                        gtol=1e-6,
                        ftol=1e-6,
                        max_nfev=5000,
                        verbose=2,
                        full_output=True,
                        show_progress=strategy_info["supports_progress"],
                        stability="auto",
                        rescale_data=False,
                    )
                    popt, pcov, info = result_tuple
                else:
                    from xpcsjax.optimization.nlsq.gradient_monitor import (
                        _get_debug_curvefit_callback,
                    )

                    _std_kwargs: dict = dict(
                        bounds=nlsq_bounds,
                        loss=loss_name,
                        x_scale=x_scale_value,
                        gtol=1e-6,
                        ftol=1e-6,
                        max_nfev=5000,
                        verbose=0,
                        stability="auto",
                        rescale_data=False,
                    )
                    if callback is not None and "callback" not in _std_kwargs:
                        # Real L4 per-iteration monitor callback (strictly
                        # observational); precedence over the Task-0 debug seam.
                        _std_kwargs["callback"] = callback
                    _dbg_cb = _get_debug_curvefit_callback()
                    if _dbg_cb is not None and "callback" not in _std_kwargs:
                        _std_kwargs["callback"] = _dbg_cb
                    popt, pcov = curve_fit_fn(
                        wrapped_residual_fn,
                        xdata,
                        ydata,
                        p0=validated_params.tolist(),
                        **_std_kwargs,
                    )
                    info = {}

                log.info("NLSQ Result Analysis:")
                log.info(f"  p0 (initial):  {validated_params}")
                log.info(f"  popt (fitted): {popt}")
                log.info(f"  bounds lower:  {nlsq_bounds[0] if nlsq_bounds else 'None'}")
                log.info(f"  bounds upper:  {nlsq_bounds[1] if nlsq_bounds else 'None'}")
                log.info(f"  pcov diagonal: {np.diag(pcov)}")

                params_unchanged = np.allclose(popt, validated_params, rtol=1e-10, atol=1e-14)
                uncertainties_zero = np.any(np.abs(np.diag(pcov)) < 1e-15)

                if params_unchanged:
                    log.warning(
                        "Optimization failure: Parameters unchanged from initial guess!\n"
                        "   This suggests curve_fit returned immediately without optimizing.\n"
                        "   Possible causes: (1) Already at optimum, (2) Singular Jacobian, (3) Bounds too tight"
                    )

                if uncertainties_zero:
                    zero_unc_indices = np.where(np.abs(np.diag(pcov)) < 1e-15)[0]
                    log.warning(
                        f"Degenerate covariance: Zero uncertainties for parameters at indices {zero_unc_indices}\n"
                        f"   pcov diagonal: {np.diag(pcov)}\n"
                        f"   This indicates singular/ill-conditioned Jacobian matrix.\n"
                        f"   Affected parameters may not have been optimized properly."
                    )

                recovery_actions = []
                convergence_status = "converged"

            if strategy_attempts:
                recovery_actions.append(f"strategy_fallback_to_{current_strategy.value}")
                log.info(
                    f"Successfully optimized with fallback strategy: {current_strategy.value}\n"
                    f"  Previous attempts: {[s.value for s in strategy_attempts]}"
                )
            break

        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            OSError,
            MemoryError,
        ) as e:
            strategy_attempts.append(current_strategy)

            fallback_strategy = get_fallback_strategy(current_strategy)

            if fallback_strategy is not None:
                log.warning(
                    f"Strategy {current_strategy.value} failed: {str(e)[:100]}\n"
                    f"  Attempting fallback to {fallback_strategy.value} strategy..."
                )
                current_strategy = fallback_strategy
            else:
                execution_time = time.time() - start_time
                log.error(
                    f"All strategies failed after {execution_time:.2f}s\n"
                    f"  Attempted: {[s.value for s in strategy_attempts]}\n"
                    f"  Final error: {e}"
                )

                if isinstance(e, RuntimeError) and (
                    "Recovery actions" in str(e) or "Suggestions" in str(e)
                ):
                    raise
                else:
                    raise RuntimeError(
                        f"Optimization failed with all strategies: {[s.value for s in strategy_attempts]}"
                    ) from e

    return popt, pcov, info, recovery_actions, convergence_status
