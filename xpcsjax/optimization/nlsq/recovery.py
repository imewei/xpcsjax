"""Error recovery and diagnostics for NLSQ optimization.

Extracted from wrapper.py to reduce file size and improve maintainability.

This module provides:
- Safe uncertainty extraction from covariance matrices
- Automatic error recovery with retry strategies (T022-T024)
- Error diagnosis with actionable recovery guidance
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from xpcsjax.optimization.nlsq.fallback_chain import OptimizationStrategy
from xpcsjax.optimization.nlsq.results import FunctionEvaluationCounter
from xpcsjax.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


def safe_uncertainties_from_pcov(pcov: np.ndarray, n_params: int) -> np.ndarray:
    """Extract uncertainties with diagonal regularization for singular pcov."""
    if pcov.shape[0] != n_params:
        return np.zeros(n_params)
    diag = np.diag(pcov)
    if np.any(diag < 1e-15):
        logger.warning(
            f"Singular covariance: {np.sum(diag < 1e-15)}/{n_params} near-zero entries. "
            "Applying regularization."
        )
        diag = np.diag(pcov + np.eye(n_params) * 1e-10)
    return np.asarray(np.sqrt(np.maximum(diag, 0.0)))


def execute_with_recovery(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    xdata: np.ndarray,
    ydata: np.ndarray,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    strategy: OptimizationStrategy,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    loss_name: str,
    x_scale_value: float | str | np.ndarray,
    handle_nlsq_result_fn: Callable,
    curve_fit_fn: Callable,
    curve_fit_large_fn: Callable,
    callback: Callable | None = None,
    convergence: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, list[str], str]:
    """Execute optimization with automatic error recovery (T022-T024).

    Implements intelligent retry strategies across up to three attempts:
    the first uses the original parameters, and subsequent attempts apply
    recovery actions chosen by :func:`diagnose_error` (e.g. parameter
    perturbation). Parameter stagnation or identity covariance triggers a
    perturbed retry. On exhaustion, a ``RuntimeError`` carries comprehensive
    diagnostics.

    Parameters
    ----------
    residual_fn : Callable
        Residual function.
    xdata, ydata : np.ndarray
        Data arrays.
    initial_params : np.ndarray
        Initial parameter guess.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds, or None.
    strategy : OptimizationStrategy
        Optimization strategy to use.
    log : logging.Logger | logging.LoggerAdapter
        Logger instance.
    loss_name : str
        Loss function name.
    x_scale_value : float | str | np.ndarray
        Scaling value for parameters.
    handle_nlsq_result_fn : Callable
        Function to normalize NLSQ results.
    curve_fit_fn : Callable
        Standard ``curve_fit`` function.
    curve_fit_large_fn : Callable
        Large-dataset ``curve_fit`` function.
    callback : Callable | None, optional
        Per-iteration L4 monitor callback (strictly observational).

    Returns
    -------
    tuple
        ``(popt, pcov, info, recovery_actions, convergence_status)``.

    Raises
    ------
    RuntimeError
        If optimization fails after all retry attempts or the error is
        diagnosed as unrecoverable.
    """
    recovery_actions = []
    max_retries = 3
    current_params = initial_params.copy()

    # Resolve solver tolerances from the caller's NLSQ config when supplied,
    # falling back to the recovery-path defaults only for keys left unset. This
    # makes the recovery executor honor user ftol/gtol/xtol/max_iterations instead
    # of always hard-coding 1e-6 / 5000.
    _conv = convergence or {}
    _tol_kwargs: dict[str, float] = {
        "gtol": float(_conv.get("gtol", 1e-6)),
        "ftol": float(_conv.get("ftol", 1e-6)),
        "max_nfev": int(_conv.get("max_nfev", 5000)),
    }
    if _conv.get("xtol") is not None:
        # xtol was not passed on the legacy path; only forward it when the config
        # explicitly sets it, so the default recovery solve stays unchanged.
        _tol_kwargs["xtol"] = float(_conv["xtol"])

    # Compute initial cost for optimization success tracking
    if hasattr(residual_fn, "n_total_points") or isinstance(residual_fn, FunctionEvaluationCounter):
        initial_residuals = residual_fn(initial_params)
    else:
        initial_residuals = residual_fn(xdata, *initial_params)
    initial_cost = np.sum(initial_residuals**2)

    # Determine if we should use large dataset functions
    use_large = strategy != OptimizationStrategy.STANDARD
    show_progress = strategy in [
        OptimizationStrategy.LARGE,
        OptimizationStrategy.CHUNKED,
        OptimizationStrategy.STREAMING,
    ]

    for attempt in range(max_retries):
        try:
            log.info(
                f"Optimization attempt {attempt + 1}/{max_retries} ({strategy.value} strategy)"
            )

            if use_large:
                log.debug("Using curve_fit_large with NLSQ automatic memory management")

                if isinstance(x_scale_value, (int, float)):
                    x_scale_large = np.abs(current_params) + 1e-3
                    log.info(
                        f"Replacing scalar x_scale={x_scale_value} with magnitude-based scaling"
                    )
                elif isinstance(x_scale_value, np.ndarray):
                    x_scale_large = x_scale_value
                else:
                    x_scale_large = np.abs(current_params) + 1e-3

                result = curve_fit_large_fn(
                    residual_fn,
                    xdata,
                    ydata,
                    p0=current_params.tolist(),
                    bounds=bounds,
                    loss=loss_name,
                    x_scale=x_scale_large,
                    verbose=2,
                    show_progress=show_progress,
                    stability="auto",
                    **_tol_kwargs,
                )
                popt, pcov, info = handle_nlsq_result_fn(result, OptimizationStrategy.LARGE)
                info["initial_cost"] = initial_cost
            else:
                x_scale_array = np.abs(current_params) + 1e-3

                n_show = min(8, len(current_params))
                log.info(
                    f"DEBUG: Bounds and scaling (showing first {n_show} of {len(current_params)} params):"
                )
                if bounds is not None:
                    lower, upper = bounds
                    for i in range(n_show):
                        log.info(
                            f"  param[{i}]: [{lower[i]:.6f}, {upper[i]:.6f}], "
                            f"initial={current_params[i]:.6f}, x_scale={x_scale_array[i]:.6e}"
                        )
                else:
                    log.info("  bounds=None (unbounded)")
                    for i in range(n_show):
                        log.info(
                            f"  param[{i}]: initial={current_params[i]:.6f}, "
                            f"x_scale={x_scale_array[i]:.6e}"
                        )

                from xpcsjax.optimization.nlsq.gradient_monitor import (
                    _get_debug_curvefit_callback,
                )

                _std_kwargs: dict = dict(
                    bounds=bounds,
                    loss=loss_name,
                    x_scale=x_scale_array,
                    verbose=2,
                    stability="auto",
                    rescale_data=False,
                    **_tol_kwargs,
                )
                if callback is not None and "callback" not in _std_kwargs:
                    # Real L4 per-iteration monitor callback (strictly
                    # observational); takes precedence over the Task-0 debug seam.
                    _std_kwargs["callback"] = callback
                _dbg_cb = _get_debug_curvefit_callback()
                if _dbg_cb is not None and "callback" not in _std_kwargs:
                    _std_kwargs["callback"] = _dbg_cb
                popt, pcov = curve_fit_fn(
                    residual_fn,
                    xdata,
                    ydata,
                    p0=current_params.tolist(),
                    **_std_kwargs,
                )
                info = {"initial_cost": initial_cost}

                log.info("=" * 80)
                log.info("NLSQ curve_fit RESULT DIAGNOSTICS")
                log.info("=" * 80)
                log.info(f"  Initial params (p0):  {current_params}")
                log.info(f"  Fitted params (popt): {popt}")
                log.info(f"  Params changed: {not np.allclose(popt, current_params, rtol=1e-10)}")
                log.info(f"  pcov shape: {pcov.shape}")
                log.info(f"  pcov diagonal (uncertainties^2): {np.diag(pcov)}")
                log.info(f"  pcov condition number: {np.linalg.cond(pcov):.2e}")

                zero_unc_mask = np.abs(np.diag(pcov)) < 1e-15
                if np.any(zero_unc_mask):
                    zero_indices = np.where(zero_unc_mask)[0]
                    log.warning(
                        f"ZERO UNCERTAINTIES detected for parameters at indices: {zero_indices}"
                    )
                    log.warning("   This indicates singular/ill-conditioned Jacobian matrix!")
                    log.warning("   Affected parameters were likely NOT optimized by NLSQ.")
                log.info("=" * 80)

            # Validate result
            params_unchanged = np.allclose(popt, current_params, rtol=1e-10)
            identity_covariance = np.allclose(pcov, np.eye(len(popt)), rtol=1e-10)

            if params_unchanged or identity_covariance:
                log.warning(
                    f"Potential optimization failure detected:\n"
                    f"  Parameters unchanged: {params_unchanged}\n"
                    f"  Identity covariance: {identity_covariance}\n"
                    f"  This may indicate NLSQ streaming bug or failed optimization"
                )

                if attempt < max_retries - 1:
                    recovery_actions.append("detected_parameter_stagnation")
                    log.info("Retrying with perturbed parameters...")
                    _rng = np.random.default_rng(seed=42 + attempt)
                    # Use an absolute-value scale with a unit floor so parameters
                    # that are exactly zero still receive a non-zero perturbation
                    # (a purely multiplicative kick cannot escape a zero value).
                    perturb_scale = np.where(
                        np.abs(current_params) > 1e-12, np.abs(current_params), 1.0
                    )
                    perturbation = (
                        0.05 * perturb_scale * _rng.uniform(-1, 1, size=len(current_params))
                    )
                    current_params = current_params + perturbation
                    if bounds is not None:
                        current_params = np.clip(current_params, bounds[0], bounds[1])
                    continue
                else:
                    log.error(
                        "Optimization returned unchanged parameters after all retries. "
                        "This may indicate a bug in NLSQ or an intractable problem."
                    )
                    # Honest status: a detected stagnation on the final attempt is
                    # a failure, not a convergence. Returning "converged" here would
                    # silently mislabel an unoptimized fit as successful.
                    recovery_actions.append("stagnation_after_all_retries")
                    return popt, pcov, info, recovery_actions, "failed"

            # Success!
            convergence_status = "converged" if attempt == 0 else "converged_with_recovery"
            log.info(f"Optimization converged on attempt {attempt + 1}")
            return popt, pcov, info, recovery_actions, convergence_status

        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            OSError,
            MemoryError,
        ) as e:
            log_exception(
                log,
                e,
                context={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "strategy": strategy.value,
                    "n_params": len(current_params),
                    "params_summary": f"[{current_params[0]:.4g}, ..., {current_params[-1]:.4g}]",
                },
                level=logging.WARNING,
            )

            diagnostic = diagnose_error(
                error=e,
                params=current_params,
                bounds=bounds,
                attempt=attempt,
            )

            log.warning(
                f"Attempt {attempt + 1} failed: {diagnostic['error_type']}",
            )
            log.info(f"Diagnostic: {diagnostic['message']}")

            recovery_strategy = diagnostic["recovery_strategy"]
            if recovery_strategy.get("action") == "no_recovery_available":
                error_msg = (
                    f"Optimization failed: {diagnostic['error_type']} (unrecoverable)\n"
                    f"Diagnostic: {diagnostic['message']}\n"
                    f"Suggestions:\n"
                )
                for suggestion in diagnostic["suggestions"]:
                    error_msg += f"  - {suggestion}\n"

                log.error(error_msg)
                raise RuntimeError(error_msg) from e

            if attempt < max_retries - 1:
                recovery_actions.append(recovery_strategy["action"])
                params_before = current_params.copy()

                log.info(f"Applying recovery: {recovery_strategy['action']}")

                current_params = recovery_strategy["new_params"]

                log.info(
                    f"Recovery parameter adjustment:\n"
                    f"  Before: [{params_before[0]:.4g}, ..., {params_before[-1]:.4g}]\n"
                    f"  After:  [{current_params[0]:.4g}, ..., {current_params[-1]:.4g}]\n"
                    f"  Max change: {np.max(np.abs(current_params - params_before)):.4g}"
                )

                continue
            else:
                error_msg = (
                    f"Optimization failed after {max_retries} attempts.\n"
                    f"Recovery actions attempted: {recovery_actions}\n"
                    f"Final diagnostic: {diagnostic['message']}\n"
                    f"Suggestions:\n"
                )
                for suggestion in diagnostic["suggestions"]:
                    error_msg += f"  - {suggestion}\n"

                log.error(error_msg)
                raise RuntimeError(error_msg) from e

    # Unreachable: loop always returns or raises, but mypy needs this
    raise RuntimeError("Optimization failed: exhausted all retry attempts")


def diagnose_error(
    error: Exception,
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    attempt: int,
) -> dict[str, Any]:
    """Diagnose an optimization error and propose a recovery strategy.

    Classifies the error (out-of-memory, convergence failure, bounds
    violation, ill-conditioned Jacobian, numerical instability, or unknown)
    and returns actionable suggestions plus a concrete recovery action.

    Parameters
    ----------
    error : Exception
        Exception raised during optimization.
    params : np.ndarray
        Current parameter values.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds, or None.
    attempt : int
        Current attempt number (0-indexed).

    Returns
    -------
    dict[str, Any]
        Diagnostic dictionary with ``error_type``, ``message``,
        ``suggestions``, and a ``recovery_strategy`` (which may carry
        ``new_params`` or signal ``no_recovery_available``).
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    diagnostic: dict[str, Any] = {
        "error_type": error_type,
        "message": str(error),
        "suggestions": [],
        "recovery_strategy": {},
    }

    if "resource_exhausted" in error_str or "out of memory" in error_str:
        diagnostic["error_type"] = "out_of_memory"
        diagnostic["suggestions"] = [
            "Dataset too large for available CPU memory",
            "IMMEDIATE FIX: Reduce dataset size:",
            "  - Enable phi angle filtering in config (reduce angles from 23 to 8-12)",
            "  - Reduce time points via subsampling (1001x1001 -> 200x200)",
            "  - Use smaller time window in config (frames: 1000-2000 -> 1000-1500)",
            "ALTERNATIVE: Increase system memory or use machine with more RAM",
            "NOTE: curve_fit_large() is disabled - residual function not chunk-aware",
        ]
        diagnostic["recovery_strategy"] = {
            "action": "no_recovery_available",
            "reason": "Memory exhaustion requires data reduction",
            "suggested_actions": [
                "enable_angle_filtering",
                "reduce_time_points",
                "increase_system_memory",
            ],
        }

    elif "convergence" in error_str or "max" in error_str or "iteration" in error_str:
        diagnostic["error_type"] = "convergence_failure"
        diagnostic["suggestions"] = [
            "Try different initial parameters",
            "Relax convergence tolerance",
            "Check if data quality is sufficient",
            "Verify parameter bounds are reasonable",
        ]

        if attempt == 0:
            perturbation = np.random.default_rng(seed=42).standard_normal(params.shape) * 0.1
            new_params = params * (1.0 + perturbation)
            if bounds is not None:
                new_params = np.clip(new_params, bounds[0], bounds[1])
            diagnostic["recovery_strategy"] = {
                "action": "perturb_initial_parameters_10pct",
                "new_params": new_params,
            }
        else:
            perturbation = np.random.default_rng(seed=123).standard_normal(params.shape) * 0.2
            new_params = params * (1.0 + perturbation)
            if bounds is not None:
                new_params = np.clip(new_params, bounds[0], bounds[1])
            diagnostic["recovery_strategy"] = {
                "action": "perturb_initial_parameters_20pct",
                "new_params": new_params,
            }

    elif "bound" in error_str or "constraint" in error_str:
        diagnostic["error_type"] = "bounds_violation"
        diagnostic["suggestions"] = [
            "Check that lower bounds < upper bounds",
            "Verify bounds are physically reasonable",
            "Consider expanding bounds if parameters consistently hit limits",
        ]
        if bounds is not None:
            lower, upper = bounds
            range_width = upper - lower
            new_params = lower + 0.5 * range_width
            diagnostic["recovery_strategy"] = {
                "action": "reset_to_bounds_center",
                "new_params": new_params,
            }
        else:
            new_params = params * 0.9
            diagnostic["recovery_strategy"] = {
                "action": "scale_parameters_0.9x",
                "new_params": new_params,
            }

    elif "singular" in error_str or "condition" in error_str or "rank" in error_str:
        diagnostic["error_type"] = "ill_conditioned_jacobian"
        diagnostic["suggestions"] = [
            "Data may be insufficient to constrain all parameters",
            "Consider fixing some parameters",
            "Check for parameter correlation",
            "Verify data quality and noise levels",
        ]
        new_params = params * 0.1
        if bounds is not None:
            new_params = np.clip(new_params, bounds[0], bounds[1])
        diagnostic["recovery_strategy"] = {
            "action": "scale_parameters_0.1x_for_conditioning",
            "new_params": new_params,
        }

    elif "nan" in error_str or "inf" in error_str:
        diagnostic["error_type"] = "numerical_instability"
        diagnostic["suggestions"] = [
            "Check for extreme parameter values",
            "Verify data contains no NaN/Inf values",
            "Consider parameter rescaling",
            "Check residual function implementation",
        ]
        if bounds is not None:
            lower, upper = bounds
            new_params = np.sqrt(np.abs(lower * upper))
            new_params = np.clip(new_params, lower, upper)
        else:
            new_params = np.ones_like(params) * 0.5
        diagnostic["recovery_strategy"] = {
            "action": "reset_to_geometric_mean_of_bounds",
            "new_params": new_params,
        }

    else:
        diagnostic["error_type"] = "unknown_error"
        diagnostic["suggestions"] = [
            f"Unexpected error: {error_type}",
            "Check data format and residual function",
            "Verify NLSQ package installation",
            "Consult error message for details",
        ]
        perturbation = np.random.default_rng(seed=99 + attempt).standard_normal(params.shape) * 0.05
        new_params = params * (1.0 + perturbation)
        if bounds is not None:
            new_params = np.clip(new_params, bounds[0], bounds[1])
        diagnostic["recovery_strategy"] = {
            "action": "generic_perturbation_5pct",
            "new_params": new_params,
        }

    return diagnostic
