"""Build NLSQResult from raw optimizer output.

Centralizes result construction so that every strategy produces
consistent NLSQResult objects with covariance, uncertainties,
reduced chi-squared, and metadata.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_data_prep import compute_degrees_of_freedom
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

logger = get_logger(__name__)


def build_result_from_scipy(
    opt_result: OptimizeResult,
    parameter_names: list[str],
    n_data: int,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct NLSQResult from nlsq.CurveFit (JAX-native trust-region) output.

    Args:
        opt_result: Raw scipy OptimizeResult
        parameter_names: Names for each fitted parameter
        n_data: Number of data points (for reduced chi²)
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata to attach

    Returns:
        Populated NLSQResult
    """
    params = np.asarray(opt_result.x, dtype=np.float64)
    n_params = len(params)

    # Covariance from Jacobian: cov ≈ (J^T J)^{-1} * s²
    covariance = None
    uncertainties = None
    jacobian = getattr(opt_result, "jac", None)

    if jacobian is not None:
        covariance = _compute_covariance(jacobian, opt_result.fun, n_data, n_params)
        if covariance is not None:
            uncertainties = np.sqrt(np.diag(np.abs(covariance)))

    # Reduced chi-squared
    residuals = np.asarray(opt_result.fun, dtype=np.float64)
    cost = float(np.sum(residuals**2))
    dof = compute_degrees_of_freedom(n_data, n_params)
    reduced_chi2 = cost / dof

    # Map scipy status to success
    success = (
        opt_result.status > 0 if hasattr(opt_result, "status") else opt_result.success
    )
    message = getattr(opt_result, "message", str(opt_result.get("message", "")))

    result = NLSQResult(
        parameters=params,
        parameter_names=parameter_names,
        success=bool(success),
        message=str(message),
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=getattr(opt_result, "nit", 0),
        n_function_evals=getattr(opt_result, "nfev", 0),
        convergence_reason=_status_to_reason(getattr(opt_result, "status", -1)),
        residuals=residuals,
        jacobian=jacobian,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )
    logger.debug(
        "Built result: success=%s, n_iter=%d, n_fev=%d, chi2=%.4f, status=%d",
        bool(success),
        getattr(opt_result, "nit", 0),
        getattr(opt_result, "nfev", 0),
        reduced_chi2,
        getattr(opt_result, "status", -1),
    )
    return result


def build_result_from_arrays(
    parameters: np.ndarray,
    parameter_names: list[str],
    residuals: np.ndarray,
    n_data: int,
    success: bool = True,
    message: str = "",
    jacobian: np.ndarray | None = None,
    n_iterations: int = 0,
    n_function_evals: int = 0,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct NLSQResult from raw arrays (for non-scipy backends).

    Args:
        parameters: Fitted parameter values
        parameter_names: Names in order
        residuals: Residual vector
        n_data: Number of data points
        success: Whether optimization converged
        message: Status message
        jacobian: Optional Jacobian at solution
        n_iterations: Number of iterations
        n_function_evals: Number of function evaluations
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata

    Returns:
        Populated NLSQResult
    """
    params = np.asarray(parameters, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    n_params = len(params)

    covariance = None
    uncertainties = None
    if jacobian is not None:
        covariance = _compute_covariance(jacobian, residuals, n_data, n_params)
        if covariance is not None:
            uncertainties = np.sqrt(np.diag(np.abs(covariance)))

    cost = float(np.sum(residuals**2))
    dof = compute_degrees_of_freedom(n_data, n_params)
    reduced_chi2 = cost / dof

    return NLSQResult(
        parameters=params,
        parameter_names=parameter_names,
        success=success,
        message=message,
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=n_iterations,
        n_function_evals=n_function_evals,
        convergence_reason=message,
        residuals=residuals,
        jacobian=jacobian,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )


def build_result_from_nlsq(
    nlsq_result: Any,
    parameter_names: list[str],
    n_data: int,
    wall_time: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Normalize any NLSQ package return format to NLSQResult.

    Handles 4 return formats:
    - dict with 'x'/'popt', 'pcov' keys (AdaptiveHybridStreamingOptimizer)
    - (popt, pcov) tuple (curve_fit)
    - (popt, pcov, info) tuple (curve_fit with full_output)
    - object with .x/.popt, .pcov attributes (CurveFit result)

    Args:
        nlsq_result: Raw return value from an NLSQ optimization call
        parameter_names: Names for each fitted parameter
        n_data: Number of data points (for reduced chi-squared)
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata to attach

    Returns:
        Populated NLSQResult

    Raises:
        TypeError: If result format is unrecognized
    """
    merged_meta: dict[str, Any] = dict(metadata) if metadata else {}
    popt: np.ndarray
    pcov: np.ndarray | None
    residuals: np.ndarray | None = None

    # Case 1: Dict (from StreamingOptimizer)
    if isinstance(nlsq_result, dict):
        popt_raw = nlsq_result.get("x", nlsq_result.get("popt"))
        if popt_raw is None:
            raise TypeError(
                "Dict result has neither 'x' nor 'popt' key. "
                f"Available keys: {list(nlsq_result.keys())}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)
        pcov_raw = nlsq_result.get("pcov")
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

        # Extract residuals if present
        fun_raw = nlsq_result.get("fun")
        if fun_raw is not None:
            residuals = np.asarray(fun_raw, dtype=np.float64)

        # Merge dict info into metadata.
        # Include nfev/nit/njev so that nlsq CurveFitResult (OptimizeResult
        # subclass, which is a dict) exposes iteration counts correctly.
        for key in (
            "streaming_diagnostics",
            "success",
            "message",
            "best_loss",
            "final_epoch",
            "nfev",
            "nit",
            "njev",
        ):
            val = nlsq_result.get(key)
            if val is not None:
                merged_meta[key] = val

        logger.debug("Normalized StreamingOptimizer dict result")

    # Case 2: Tuple with 2 or 3 elements
    elif isinstance(nlsq_result, tuple):
        if len(nlsq_result) == 2:
            popt_raw, pcov_raw = nlsq_result
            logger.debug("Normalized (popt, pcov) tuple")
        elif len(nlsq_result) == 3:
            popt_raw, pcov_raw, info = nlsq_result
            if isinstance(info, dict):
                merged_meta.update(info)
            else:
                logger.warning(
                    "Info object is not a dict: %s. Wrapping as raw_info.",
                    type(info),
                )
                merged_meta["raw_info"] = info
            logger.debug("Normalized (popt, pcov, info) tuple")
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(nlsq_result)}. "
                "Expected 2 (popt, pcov) or 3 (popt, pcov, info)."
            )
        popt = np.asarray(popt_raw, dtype=np.float64)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

    # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
    elif hasattr(nlsq_result, "x") or hasattr(nlsq_result, "popt"):
        popt_raw = getattr(nlsq_result, "x", getattr(nlsq_result, "popt", None))
        if popt_raw is None:
            raise TypeError(
                "Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(nlsq_result)}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)

        pcov_raw = getattr(nlsq_result, "pcov", None)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None
        if pcov_raw is None:
            logger.warning("No pcov attribute in result object")

        # Extract residuals if present
        fun_raw = getattr(nlsq_result, "fun", None)
        if fun_raw is not None:
            residuals = np.asarray(fun_raw, dtype=np.float64)

        # Extract common attributes into metadata
        for attr in ("message", "success", "nfev", "nit", "njev", "optimality"):
            if hasattr(nlsq_result, attr):
                merged_meta[attr] = getattr(nlsq_result, attr)

        if hasattr(nlsq_result, "info") and isinstance(nlsq_result.info, dict):
            merged_meta.update(nlsq_result.info)

        logger.debug("Normalized object result (type: %s)", type(nlsq_result).__name__)

    # Case 4: Unrecognized format
    else:
        raise TypeError(
            f"Unrecognized NLSQ result format: {type(nlsq_result)}. "
            "Expected tuple, dict, or object with 'x'/'popt' attributes."
        )

    # --- Build NLSQResult ---
    n_params = len(popt)

    # Uncertainties from covariance diagonal
    uncertainties: np.ndarray | None = None
    if pcov is not None:
        uncertainties = np.sqrt(np.diag(np.abs(pcov)))

    # Cost and reduced chi-squared from residuals (if available)
    final_cost: float | None = None
    reduced_chi2: float | None = None
    if residuals is not None:
        final_cost = float(np.sum(residuals**2))
        dof = compute_degrees_of_freedom(n_data, n_params)
        reduced_chi2 = final_cost / dof

    return NLSQResult(
        parameters=popt,
        parameter_names=parameter_names,
        success=bool(merged_meta.get("success", True)),
        message=str(merged_meta.get("message", "")),
        uncertainties=uncertainties,
        covariance=pcov,
        final_cost=final_cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=int(merged_meta.get("nit", 0)),
        n_function_evals=int(merged_meta.get("nfev", 0)),
        convergence_reason=str(merged_meta.get("message", "")),
        residuals=residuals,
        wall_time_seconds=wall_time,
        metadata=merged_meta,
    )


def build_failed_result(
    parameter_names: list[str],
    message: str,
    initial_params: np.ndarray | None = None,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct a failed NLSQResult.

    Args:
        parameter_names: Names for parameters
        message: Failure description
        initial_params: Initial guess (returned as "best" params)
        wall_time: Wall-clock time before failure
        metadata: Additional metadata

    Returns:
        NLSQResult with success=False
    """
    logger.warning("NLSQ failed: %s", message)
    params = (
        initial_params if initial_params is not None else np.zeros(len(parameter_names))
    )
    return NLSQResult(
        parameters=np.asarray(params, dtype=np.float64),
        parameter_names=parameter_names,
        success=False,
        message=message,
        convergence_reason=message,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )


class TimedContext:
    """Context manager for timing optimizer calls.

    Usage::

        timer = TimedContext()
        with timer:
            result = optimizer.run(...)
        print(f"Took {timer.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> TimedContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def _compute_covariance(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    n_data: int,
    n_params: int,
) -> np.ndarray | None:
    """Compute parameter covariance from Jacobian.

    Uses the Gauss-Newton approximation:
        cov = s² * (J^T J)^{-1}
    where s² = sum(residuals²) / (n_data - n_params).

    Args:
        jacobian: Jacobian matrix at solution, shape (n_residuals, n_params)
        residuals: Residual vector at solution
        n_data: Number of independent data points
        n_params: Number of parameters

    Returns:
        Covariance matrix of shape (n_params, n_params), or None on failure
    """
    try:
        jac = np.asarray(jacobian, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64)

        # J^T J
        jtj = jac.T @ jac

        # Regularize if near-singular
        cond = np.linalg.cond(jtj)
        if cond > 1e14:
            logger.warning(
                "J^T J condition number %.2e; adding Tikhonov regularization", cond
            )
            jtj += 1e-10 * np.eye(n_params)

        jtj_inv = np.linalg.inv(jtj)

        # Variance estimate
        dof = max(n_data - n_params, 1)
        s2 = float(np.sum(res**2)) / dof

        return s2 * jtj_inv

    except np.linalg.LinAlgError:
        logger.warning("Failed to compute covariance: singular J^T J")
        return None


def _status_to_reason(status: int) -> str:
    """Map scipy least_squares status codes to human-readable reasons."""
    reasons = {
        -1: "Improper input parameters",
        0: "Maximum function evaluations reached",
        1: "gtol convergence (gradient sufficiently small)",
        2: "xtol convergence (parameter change sufficiently small)",
        3: "ftol convergence (cost change sufficiently small)",
        4: "Both xtol and ftol convergence",
    }
    return reasons.get(status, f"Unknown status: {status}")


def build_hybrid_streaming_result(
    *,
    model: Any,
    popt: np.ndarray,
    pcov: np.ndarray,
    info: dict[str, Any],
    phi_angles: np.ndarray,
    per_angle_mode: str = "hybrid_streaming",
    scaling_source: str = "quantile",
    chi2_per_angle: np.ndarray | None = None,
    stratification_diagnostics: Any | None = None,
) -> Any:
    """Build an OptimizationResult from heterodyne hybrid-streaming optimizer output.

    Mirrors the tail of ``_fit_joint_averaged_multi_phi`` in ``heterodyne_core.py``.
    The result carries the full ``nlsq_diagnostics`` schema expected by downstream
    consumers including the ``shear_weighting="not_applicable_heterodyne"`` marker
    (set by ``_build_heterodyne_diagnostics``).

    Parameters
    ----------
    model :
        HeterodyneModel; provides ``param_manager``.
    popt :
        Fitted varying-parameter vector, shape (n_varying,).
    pcov :
        Parameter covariance, shape (n_varying, n_varying).
    info :
        Raw optimizer info dict (at least ``nit``/``success`` and optionally
        ``hybrid_streaming_diagnostics``).
    phi_angles :
        Array of phi angles (degrees) used in the fit, shape (n_phi,).

    Returns
    -------
    OptimizationResult
        Populated result with ``nlsq_diagnostics`` and proper schema.
    """
    from xpcsjax.optimization.nlsq.heterodyne_core import _build_heterodyne_diagnostics
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.optimization.nlsq.validation import classify_quality_flag

    popt = np.asarray(popt, dtype=np.float64)
    pcov = np.asarray(pcov, dtype=np.float64)
    n = len(popt)
    n_phi = len(np.asarray(phi_angles))

    # ------------------------------------------------------------------
    # Uncertainties from covariance diagonal
    # ------------------------------------------------------------------
    uncertainties = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

    # ------------------------------------------------------------------
    # chi2 placeholder — the hybrid-streaming optimizer does not decompose
    # per-angle chi2 during Phase 2-A; fill with NaN so downstream can detect.
    # The stratified-LS driver passes a real ``chi2_per_angle`` (computed from
    # the final residual grouped by phi_idx) so the SSR-conservation invariant
    # ``chi2_per_angle.sum() == chi_squared`` holds for that path.
    # ------------------------------------------------------------------
    if chi2_per_angle is None:
        chi2_per_angle = np.full(n_phi, np.nan, dtype=np.float64)
    else:
        chi2_per_angle = np.asarray(chi2_per_angle, dtype=np.float64)

    # ------------------------------------------------------------------
    # Build nlsq_diagnostics via the canonical heterodyne helper
    # ------------------------------------------------------------------
    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode=per_angle_mode,
        chi2_per_angle=chi2_per_angle,
        scaling_source=scaling_source,
        fourier_basis_dim=None,
        parameter_names=list(model.param_manager.varying_names),
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=int(n_phi),
        n_iterations=int(info.get("nit", 0)),
        success=bool(info.get("success", True)),
    )

    # ------------------------------------------------------------------
    # Attach hybrid-streaming-specific diagnostics block
    # ------------------------------------------------------------------
    diagnostics["hybrid_streaming"] = info.get(
        "hybrid_streaming_diagnostics",
        {k: info[k] for k in ("nit", "success") if k in info},
    )

    # ------------------------------------------------------------------
    # Convergence status and quality
    # ------------------------------------------------------------------
    success = bool(info.get("success", True))
    convergence_status: str = "converged" if success else "failed"
    quality_flag = classify_quality_flag(reduced_chi2=1.0) if success else "poor"

    # Coerce to valid ConvergenceStatus literal
    if convergence_status not in ("converged", "max_iter", "failed", "partial"):
        convergence_status = "failed"

    # ------------------------------------------------------------------
    # SSR: not directly available from streaming result — use 0.0 placeholder
    # ------------------------------------------------------------------
    ssr = float(info.get("cost", 0.0)) * 2.0  # optimizer cost = 0.5 * SSR
    # Finding 3: dof = n_data - n_params.  n_data_points is threaded from the
    # wrapper via info so this is always correct when the hybrid path ran.
    n_data = int(info.get("n_data_points", 0))
    n_dof = max(1, n_data - n)
    reduced_chi2 = ssr / n_dof if ssr > 0 else 0.0

    return OptimizationResult(
        parameters=popt,
        uncertainties=uncertainties,
        covariance=pcov,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,  # type: ignore[arg-type]
        iterations=int(info.get("nit", 0)),
        execution_time=float(info.get("wall_time", 0.0)),
        device_info={"backend": "cpu", "adapter": "AdaptiveHybridStreamingOptimizer"},
        recovery_actions=[],
        quality_flag=quality_flag,  # type: ignore[arg-type]
        streaming_diagnostics=None,
        stratification_diagnostics=stratification_diagnostics,
        nlsq_diagnostics=diagnostics,
    )
