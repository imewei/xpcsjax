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
    """Construct an NLSQResult from ``nlsq.CurveFit`` (JAX-native trust-region) output.

    Covariance is estimated from the Jacobian when present, and the reduced
    chi-squared is computed against the supplied data-point count.

    Parameters
    ----------
    opt_result
        Raw scipy-style ``OptimizeResult`` returned by the solver.
    parameter_names
        Names for each fitted parameter, in order.
    n_data
        Number of data points (used for the reduced chi-squared).
    wall_time
        Wall-clock time in seconds, or ``None`` if not measured.
    metadata
        Additional metadata to attach to the result.

    Returns
    -------
    NLSQResult
        The populated result.
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
    success = opt_result.status > 0 if hasattr(opt_result, "status") else opt_result.success
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
    """Construct an NLSQResult from raw arrays (for non-scipy backends).

    Parameters
    ----------
    parameters
        Fitted parameter values.
    parameter_names
        Names in order.
    residuals
        Residual vector at the solution.
    n_data
        Number of data points (used for the reduced chi-squared).
    success
        Whether the optimization converged.
    message
        Status message.
    jacobian
        Optional Jacobian at the solution, used to estimate covariance.
    n_iterations
        Number of solver iterations.
    n_function_evals
        Number of residual-function evaluations.
    wall_time
        Wall-clock time in seconds, or ``None`` if not measured.
    metadata
        Additional metadata to attach to the result.

    Returns
    -------
    NLSQResult
        The populated result.
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
    """Normalize any NLSQ-package return format into an NLSQResult.

    Handles the four return shapes the NLSQ backends may produce:

    - a dict with ``'x'``/``'popt'`` and ``'pcov'`` keys
      (``AdaptiveHybridStreamingOptimizer``);
    - a ``(popt, pcov)`` tuple (``curve_fit``);
    - a ``(popt, pcov, info)`` tuple (``curve_fit`` with ``full_output``);
    - an object exposing ``.x``/``.popt`` and ``.pcov`` attributes
      (``CurveFit`` result).

    Parameters
    ----------
    nlsq_result
        Raw return value from an NLSQ optimization call.
    parameter_names
        Names for each fitted parameter, in order.
    n_data
        Number of data points (used for the reduced chi-squared).
    wall_time
        Wall-clock time in seconds.
    metadata
        Additional metadata to attach to the result.

    Returns
    -------
    NLSQResult
        The populated result.

    Raises
    ------
    TypeError
        If the result format is not one of the four recognised shapes.
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

    Parameters
    ----------
    parameter_names
        Names for the parameters.
    message
        Description of the failure.
    initial_params
        Initial guess, returned as the "best" parameters; defaults to zeros
        when ``None``.
    wall_time
        Wall-clock time elapsed before the failure, or ``None``.
    metadata
        Additional metadata to attach to the result.

    Returns
    -------
    NLSQResult
        A result with ``success=False``.
    """
    logger.warning("NLSQ failed: %s", message)
    params = initial_params if initial_params is not None else np.zeros(len(parameter_names))
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
        """Start the timer and return the context manager."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the timer, recording the elapsed seconds on :attr:`elapsed`."""
        self.elapsed = time.perf_counter() - self._start


def _compute_covariance(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    n_data: int,
    n_params: int,
) -> np.ndarray | None:
    r"""Compute the parameter covariance from the Jacobian.

    Uses the Gauss-Newton approximation
    :math:`\mathrm{cov} = s^2 (J^T J)^{-1}` with
    :math:`s^2 = \sum r^2 / (n_\mathrm{data} - n_\mathrm{params})`. A
    near-singular :math:`J^T J` is stabilised with a small Tikhonov term.

    Parameters
    ----------
    jacobian
        Jacobian matrix at the solution, shape ``(n_residuals, n_params)``.
    residuals
        Residual vector at the solution.
    n_data
        Number of independent data points.
    n_params
        Number of parameters.

    Returns
    -------
    numpy.ndarray or None
        Covariance matrix of shape ``(n_params, n_params)``, or ``None`` if the
        linear solve fails.
    """
    try:
        jac = np.asarray(jacobian, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64)

        # J^T J
        jtj = jac.T @ jac

        # Regularize if near-singular
        cond = np.linalg.cond(jtj)
        if cond > 1e14:
            logger.warning("J^T J condition number %.2e; adding Tikhonov regularization", cond)
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
    parameter_names: list[str] | None = None,
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
    # ``parameter_names`` defaults to physics-only ``varying_names`` (length
    # n_physics), preserving the existing hybrid-streaming caller. The
    # stratified-LS driver passes the FULL joint name list ([physics | scaling])
    # so the diagnostics names align with ``popt`` length (Fix 4).
    diag_param_names = (
        list(parameter_names)
        if parameter_names is not None
        else list(model.param_manager.varying_names)
    )
    # Propagate the REAL anti-degeneracy activation block computed by the
    # streaming fit (``info["anti_degeneracy"]``) into ``nlsq_diagnostics`` so the
    # public OptimizationResult surfaces the SAME activation keys as the in-memory
    # heterodyne paths. Without threading these flags through, the assembler would
    # default L2/L3/L4 to inactive and the streaming result would under-report what
    # actually ran. ``shear_weighting`` stays the canonical heterodyne
    # ``not_applicable_heterodyne`` marker (set inside the helper); we forward only
    # the activation flags + the real per-angle mode.
    ad_block = info.get("anti_degeneracy") or {}
    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode=ad_block.get("per_angle_mode", per_angle_mode),
        chi2_per_angle=chi2_per_angle,
        scaling_source=scaling_source,
        fourier_basis_dim=None,
        parameter_names=diag_param_names,
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=int(n_phi),
        n_iterations=int(info.get("nit", 0)),
        success=bool(info.get("success", True)),
        hierarchical_active=bool(ad_block.get("hierarchical_active", False)),
        regularization_active=bool(ad_block.get("regularization_active", False)),
        gradient_monitor=ad_block.get("gradient_monitor"),
    )

    # Surface controller_diagnostics when the stratified-LS path captured the
    # AntiDegeneracyController (mirrors laminar's strategies/stratified_ls.py
    # ``anti_degeneracy_info["controller_diagnostics"] = ad_controller.get_diagnostics()``
    # which wrapper.py then threads into the public nlsq_diagnostics dict).
    # Present only when ad_controller was successfully constructed; absent
    # otherwise (best-effort contract preserved).
    _cd = ad_block.get("controller_diagnostics")
    if _cd is not None:
        diagnostics["controller_diagnostics"] = _cd

    # ------------------------------------------------------------------
    # Attach hybrid-streaming-specific diagnostics block
    # ------------------------------------------------------------------
    diagnostics["hybrid_streaming"] = info.get(
        "hybrid_streaming_diagnostics",
        {k: info[k] for k in ("nit", "success") if k in info},
    )

    # ------------------------------------------------------------------
    # SSR + noise-normalized reduced chi^2
    # ------------------------------------------------------------------
    ssr = float(info.get("cost", 0.0)) * 2.0  # optimizer cost = 0.5 * SSR
    # Finding 3: dof = n_data - n_params.  n_data_points is threaded from the
    # wrapper via info so this is always correct when the hybrid path ran.
    n_data = int(info.get("n_data_points", 0))
    n_dof = max(1, n_data - n)
    # Noise-normalized reduced chi^2 (targets ~1.0 for a good fit), mirroring the
    # in-memory averaged/fourier joint paths (heterodyne_core: noise_normalized_
    # reduced_chi2). The driver threads an estimated far-lag photon-noise variance
    # via ``info['sigma2_noise']``; dividing SSR by it restores the conventional
    # chi^2_red scale. Without it, raw SSR/dof collapses to MSE << 1 on normalized
    # C2 data (C2 ~ 1, residuals ~ 5%), which is not an interpretable goodness-of-
    # fit (this produced the bogus chi^2_red = 0.0024 on the stratified-LS path).
    # Falls back to plain MSE when the noise estimate is absent/degenerate,
    # matching ``noise_normalized_reduced_chi2``.
    sigma2_noise = float(info.get("sigma2_noise", 0.0))
    if ssr <= 0:
        reduced_chi2 = 0.0
    elif sigma2_noise > 1e-12:
        reduced_chi2 = ssr / (sigma2_noise * n_dof)
    else:
        reduced_chi2 = ssr / n_dof

    # ------------------------------------------------------------------
    # Convergence status and quality
    # ------------------------------------------------------------------
    success = bool(info.get("success", True))
    # Distinguish "budget exhausted" from "diverged". A trust-region solve that
    # hits SciPy's ``max_nfev`` (status 0) reports ``success=False`` even though it
    # may have landed on a perfectly usable point — common on near-degenerate
    # problems where ``gtol`` is never certified (C044 two_component: reduced
    # chi^2 ~= 0.68 yet status was FAILED). The driver threads the SciPy
    # termination reason via ``info['convergence_reason']`` so we can report
    # ``max_iter`` (honest: not certified-converged, but graded on its real chi^2)
    # instead of a blanket ``failed`` / ``poor``. Callers that do not thread a
    # reason keep the previous converged/failed behavior unchanged.
    reason = str(info.get("convergence_reason", ""))
    hit_max_nfev = (
        not success
        and reason == "Maximum function evaluations reached"
        and np.isfinite(reduced_chi2)
    )
    if success:
        convergence_status = "converged"
    elif hit_max_nfev:
        convergence_status = "max_iter"
    else:
        convergence_status = "failed"
    # Classify from the REAL noise-normalized reduced chi^2 (parity with the
    # in-memory joint paths) rather than a hardcoded 1.0 — a converged-but-poor
    # fit must not advertise "good". A genuinely failed solve is forced to "poor";
    # a max_iter solve is graded on its actual reduced chi^2 (it produced a fit).
    if convergence_status in ("converged", "max_iter"):
        quality_flag = classify_quality_flag(reduced_chi2=reduced_chi2)
    else:
        quality_flag = "poor"

    # Coerce to valid ConvergenceStatus literal
    if convergence_status not in ("converged", "max_iter", "failed", "partial"):
        convergence_status = "failed"

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
