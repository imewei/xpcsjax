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
    # Length of the leading physics block in ``parameters`` (physics-first
    # ``[physics | scaling]`` layout). Enables the typed physics/scaling
    # accessors; ``None`` when the split is unknown (e.g. homodyne paths that
    # do not carry a scaling tail).
    n_physics: int | None = None

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

        # A 'good' fit must have a finite objective. This blocks the
        # data-integrity failure mode where a non-finite reduced chi-squared
        # (e.g. NaN residuals from corrupt input) surfaces as a confidently
        # "good" scientific result.
        if self.quality_flag == "good" and not np.isfinite(self.reduced_chi_squared):
            raise ValueError(
                "OptimizationResult: quality_flag='good' requires a finite "
                f"reduced_chi_squared (got {self.reduced_chi_squared})."
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

    @property
    def physics_parameters(self) -> np.ndarray:
        """Leading physics block of :attr:`parameters` (requires ``n_physics``).

        Makes the physics-first ``[physics | scaling]`` layout explicit so
        callers stop slicing by hardcoded offsets — the root of past scaling
        mis-application bugs.
        """
        if self.n_physics is None:
            raise ValueError(
                "physics_parameters requires n_physics to be set on the result "
                "(the physics/scaling split point is unknown)."
            )
        return np.asarray(self.parameters)[: self.n_physics]

    @property
    def scaling_parameters(self) -> np.ndarray:
        """Trailing scaling block of :attr:`parameters` (requires ``n_physics``)."""
        if self.n_physics is None:
            raise ValueError(
                "scaling_parameters requires n_physics to be set on the result "
                "(the physics/scaling split point is unknown)."
            )
        return np.asarray(self.parameters)[self.n_physics :]

    @property
    def global_escape(self) -> str | None:
        """Global-escape tag from diagnostics, or ``None``.

        Typed accessor for ``nlsq_diagnostics['global_escape']`` so a misspelled
        key cannot silently read as "no escape".
        """
        if not self.nlsq_diagnostics:
            return None
        tag = self.nlsq_diagnostics.get("global_escape")
        return str(tag) if tag is not None else None


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
