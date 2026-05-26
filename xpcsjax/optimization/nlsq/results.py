"""NLSQ optimization result classes.

This module extracts result dataclasses from nlsq_wrapper.py
to reduce file size and improve maintainability.

Extracted from nlsq_wrapper.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.strategies.chunking import StratificationDiagnostics

# Closed sets for optimization status fields. Kept as Literals so existing string
# comparisons (e.g. ``convergence_status == "converged"``) keep working at runtime
# while static checkers gain exhaustiveness over the allowed values.
ConvergenceStatus = Literal["converged", "max_iter", "failed", "partial"]
QualityFlag = Literal["good", "marginal", "poor", "unknown"]


@dataclass
class FunctionEvaluationCounter:
    """Wraps a callable and counts invocations.

    Useful for tracking the number of function evaluations during optimization.
    """

    fn: Callable[..., Any]
    count: int = 0

    def __call__(self, *args, **kwargs):
        """Call the wrapped function and increment count."""
        self.count += 1
        return self.fn(*args, **kwargs)


@dataclass
class OptimizationResult:
    """Complete optimization result with fit quality metrics and diagnostics.

    Attributes
    ----------
    parameters : np.ndarray
        Converged parameter values.
    uncertainties : np.ndarray
        Standard deviations from covariance matrix diagonal.
    covariance : np.ndarray
        Full parameter covariance matrix.
    chi_squared : float
        Sum of squared residuals.
    reduced_chi_squared : float
        chi_squared / (n_data - n_params).
    convergence_status : str
        'converged', 'max_iter', or 'failed'.
    iterations : int
        Number of optimization iterations.
    execution_time : float
        Wall-clock execution time in seconds.
    device_info : dict[str, Any]
        Device used for computation (CPU details).
    recovery_actions : list[str]
        List of error recovery actions taken.
    quality_flag : str
        'good', 'marginal', or 'poor'.
    streaming_diagnostics : dict[str, Any] | None
        Enhanced diagnostics for streaming optimization.
    stratification_diagnostics : StratificationDiagnostics | None
        Diagnostics for angle-stratified chunking.
    nlsq_diagnostics : dict[str, Any] | None
        Additional NLSQ-specific diagnostics.
    """

    parameters: np.ndarray
    uncertainties: np.ndarray
    covariance: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    convergence_status: ConvergenceStatus
    iterations: int
    execution_time: float
    device_info: dict[str, Any]
    recovery_actions: list[str] = field(default_factory=list)
    quality_flag: QualityFlag = "good"
    streaming_diagnostics: dict[str, Any] | None = None
    stratification_diagnostics: StratificationDiagnostics | None = None
    nlsq_diagnostics: dict[str, Any] | None = None
    sigma_is_default: bool = False

    def __post_init__(self) -> None:
        """Enforce result invariants so illegal states cannot be constructed.

        H-2: the central result type previously trusted every one of its ~15
        construction sites to assemble a coherent object, so a result could claim
        ``convergence_status='converged'`` with empty or non-finite parameters, or
        carry a covariance whose shape disagreed with the parameter vector
        (silently producing wrong error bars). These checks make that
        unrepresentable while staying tolerant of legitimate failed/partial
        results (empty parameters with a non-converged status) and of placeholder
        arrays passed on degraded paths.
        """
        params = np.asarray(self.parameters)
        n = int(params.size)

        if self.convergence_status == "converged":
            if n == 0:
                raise ValueError(
                    "OptimizationResult: convergence_status='converged' requires "
                    "non-empty parameters (got an empty array)."
                )
            if not np.all(np.isfinite(params)):
                n_bad = int(np.sum(~np.isfinite(params)))
                raise ValueError(
                    "OptimizationResult: convergence_status='converged' but "
                    f"{n_bad} of {n} parameter(s) are non-finite (NaN/Inf)."
                )

        if self.uncertainties is not None:
            unc = np.asarray(self.uncertainties)
            if unc.ndim == 1 and unc.size and unc.size != n:
                raise ValueError(
                    f"OptimizationResult: uncertainties length {unc.size} does not "
                    f"match number of parameters {n}."
                )

        if self.covariance is not None:
            cov = np.asarray(self.covariance)
            if cov.ndim == 2 and cov.size and cov.shape != (n, n):
                raise ValueError(
                    f"OptimizationResult: covariance shape {cov.shape} does not "
                    f"match ({n}, {n}) for the parameter vector."
                )

    @property
    def success(self) -> bool:
        """Return True if optimization converged (backward compatibility)."""
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Return descriptive message about optimization outcome."""
        if self.convergence_status == "converged":
            return f"Optimization converged successfully. chi2={self.chi_squared:.6f}"
        elif self.convergence_status == "max_iter":
            return "Optimization stopped: maximum iterations reached"
        else:
            return f"Optimization failed: {self.convergence_status}"


# =============================================================================
# T010: FallbackInfo dataclass for adapter-to-wrapper fallback tracking
# =============================================================================
@dataclass
class FallbackInfo:
    """Tracks fallback from NLSQAdapter to NLSQWrapper.

    Included in OptimizationResult.device_info when fallback occurs.

    Attributes:
        fallback_occurred: True if fallback was triggered
        adapter_used: "NLSQAdapter" or "NLSQWrapper"
        adapter_error: Error message if adapter failed (None if succeeded)
        wrapper_error: Error message if wrapper also failed (None otherwise)

    States:

    * NLSQAdapter + fallback_occurred=False + adapter_error=None: Adapter succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="...": Fallback succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="..." + wrapper_error="...": Both failed
    """

    fallback_occurred: bool
    adapter_used: str  # "NLSQAdapter" or "NLSQWrapper"
    adapter_error: str | None = None
    wrapper_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for inclusion in device_info."""
        return {
            "fallback_occurred": self.fallback_occurred,
            "adapter_used": self.adapter_used,
            "adapter_error": self.adapter_error,
            "wrapper_error": self.wrapper_error,
        }


@dataclass
class UseSequentialOptimization:
    """Marker indicating sequential per-angle optimization should be used.

    This is returned by _apply_stratification_if_needed when conditions require
    sequential per-angle optimization as a fallback strategy.

    Attributes
    ----------
    data : Any
        Original XPCS data object.
    reason : str
        Why sequential optimization is needed.
    """

    data: Any
    reason: str
