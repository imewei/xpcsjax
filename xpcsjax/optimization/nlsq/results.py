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
    """Callable wrapper that counts how many times it is invoked.

    Used to track the number of model/residual evaluations during a fit.

    Attributes
    ----------
    fn : Callable[..., Any]
        The wrapped callable; forwarded to verbatim on each invocation.
    count : int
        Running count of invocations, incremented before each forwarded call.
    """

    fn: Callable[..., Any]
    count: int = 0

    def __call__(self, *args, **kwargs):
        """Increment :attr:`count` and forward the call to :attr:`fn`."""
        self.count += 1
        return self.fn(*args, **kwargs)


@dataclass
class OptimizationResult:
    """Complete NLSQ fit result with parameters, quality metrics, and diagnostics.

    The single return type of :func:`xpcsjax.optimization.nlsq.fit_nlsq` for both
    the homodyne (``laminar_flow`` / ``static_*``) and heterodyne
    (``two_component``) physics models. The dataclass enforces its own
    invariants in :meth:`__post_init__`: a ``convergence_status='converged'``
    result must have non-empty, finite ``parameters``; a ``quality_flag='good'``
    result must have a finite ``reduced_chi_squared``; and ``uncertainties`` /
    ``covariance`` shapes must agree with the parameter count. Illegal
    combinations raise :class:`ValueError` at construction.

    Parameters follow the physics-first ``[physics | scaling]`` layout. When the
    split point is known it is recorded in :attr:`n_physics`, which enables the
    :attr:`physics_parameters` and :attr:`scaling_parameters` accessors; prefer
    those over hardcoded slicing.

    Attributes
    ----------
    parameters : numpy.ndarray
        Converged parameter values in physics-first ``[physics | scaling]``
        order. Empty on a ``failed`` / ``partial`` result.
    uncertainties : numpy.ndarray
        One-sigma standard deviations from the covariance-matrix diagonal.
        ``NaN`` on a global-escape result (no covariance solve was performed;
        see Notes).
    covariance : numpy.ndarray
        Full ``(n_params, n_params)`` parameter covariance matrix. All-``NaN``
        on a global-escape result. For heterodyne ``individual`` mode the
        off-diagonal blocks are zero **by construction** (sequential per-angle
        fits with held-fixed scaling), flagged via
        ``nlsq_diagnostics['covariance_structure'] == 'block_diagonal_sequential'``.
    chi_squared : float
        Sum of squared residuals at the optimum.
    reduced_chi_squared : float
        ``chi_squared / (n_data - n_params)``.
    convergence_status : {'converged', 'max_iter', 'failed', 'partial'}
        Terminal optimizer state. ``'partial'`` denotes a degraded path that
        produced usable parameters without a full convergence guarantee.
    iterations : int
        Number of optimizer iterations. ``0`` on a global-escape result, where
        the kept vector came from CMA-ES / multistart rather than a
        trust-region solve (see Notes).
    execution_time : float
        Wall-clock fit time in seconds.
    device_info : dict[str, Any]
        Compute device description (v0.1 is CPU-only).
    recovery_actions : list[str], optional
        Error-recovery / fallback actions taken during the fit. Empty by
        default.
    quality_flag : {'good', 'marginal', 'poor', 'unknown'}, optional
        Heuristic fit-quality label derived from ``reduced_chi_squared``.
        Defaults to ``'good'``.
    streaming_diagnostics : dict[str, Any] | None, optional
        Per-chunk diagnostics from the hybrid-streaming path; ``None`` for
        in-memory fits.
    stratification_diagnostics : StratificationDiagnostics | None, optional
        Angle-stratified chunking diagnostics; ``None`` when stratification did
        not run.
    nlsq_diagnostics : dict[str, Any] | None, optional
        Path-specific NLSQ diagnostics. Carries the symmetric anti-degeneracy
        activation keys (``hierarchical_active``, ``regularization_active``,
        ``shear_weighting``, ``gradient_monitor``) and, on a global escape, the
        ``global_escape`` tag. For heterodyne this is also where mode-specific
        per-angle data lives (``chi2_per_angle``, ``parameter_names``,
        ``contrast_per_angle`` / ``offset_per_angle``,
        ``covariance_structure``).
    sigma_is_default : bool, optional
        ``True`` when the fit used the default unit ``sigma`` (no per-point
        uncertainty was supplied in ``data``), in which case
        ``reduced_chi_squared`` is sigma-normalized rather than an absolute
        goodness-of-fit. Defaults to ``False``.
    n_physics : int | None, optional
        Length of the leading physics block in :attr:`parameters`. Enables the
        :attr:`physics_parameters` / :attr:`scaling_parameters` accessors;
        ``None`` when the physics/scaling split is unknown (e.g. homodyne paths
        with no scaling tail).

    Notes
    -----
    **Global-escape contract.** When a heterodyne joint CMA-ES or multistart
    escape produces the kept vector, the result is tagged
    ``nlsq_diagnostics['global_escape']`` (read it via :attr:`global_escape`)
    and, *by construction*, carries all-``NaN`` covariance / uncertainties and
    ``iterations == 0`` — no covariance solve is run on the kept vector. This is
    an intentional, documented invariant, not a missing computation.

    See Also
    --------
    xpcsjax.optimization.nlsq.fit_nlsq : Single-entry fit returning this type.

    Examples
    --------
    >>> from xpcsjax import fit_nlsq
    >>> result = fit_nlsq(data, "config.yaml")  # doctest: +SKIP
    >>> result.success  # doctest: +SKIP
    True
    >>> result.reduced_chi_squared  # doctest: +SKIP
    1.07
    >>> result.parameters  # doctest: +SKIP
    array([...])
    >>> # Physics/scaling split (when n_physics is known, e.g. heterodyne):
    >>> result.physics_parameters  # doctest: +SKIP
    array([...])
    >>> # Detect a global-escape result (NaN covariance, iterations == 0):
    >>> result.global_escape  # doctest: +SKIP
    'cmaes'
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
        """Whether the fit converged (``convergence_status == 'converged'``).

        Provided for backward compatibility with SciPy-style result objects.
        """
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Human-readable one-line summary of the optimization outcome."""
        if self.convergence_status == "converged":
            return f"Optimization converged successfully. chi2={self.chi_squared:.6f}"
        elif self.convergence_status == "max_iter":
            return "Optimization stopped: maximum iterations reached"
        else:
            return f"Optimization failed: {self.convergence_status}"

    @property
    def physics_parameters(self) -> np.ndarray:
        """Leading physics block of :attr:`parameters` (requires :attr:`n_physics`).

        Makes the physics-first ``[physics | scaling]`` layout explicit so
        callers stop slicing by hardcoded offsets — the root of past scaling
        mis-application bugs.

        Raises
        ------
        ValueError
            If :attr:`n_physics` is ``None`` (the split point is unknown).
        """
        if self.n_physics is None:
            raise ValueError(
                "physics_parameters requires n_physics to be set on the result "
                "(the physics/scaling split point is unknown)."
            )
        return np.asarray(self.parameters)[: self.n_physics]

    @property
    def scaling_parameters(self) -> np.ndarray:
        """Trailing scaling block of :attr:`parameters` (requires :attr:`n_physics`).

        Raises
        ------
        ValueError
            If :attr:`n_physics` is ``None`` (the split point is unknown).
        """
        if self.n_physics is None:
            raise ValueError(
                "scaling_parameters requires n_physics to be set on the result "
                "(the physics/scaling split point is unknown)."
            )
        return np.asarray(self.parameters)[self.n_physics :]

    @property
    def global_escape(self) -> str | None:
        """Global-escape tag from :attr:`nlsq_diagnostics`, or ``None``.

        Typed accessor for ``nlsq_diagnostics['global_escape']`` so a misspelled
        key cannot silently read as "no escape". A non-``None`` value (e.g.
        ``'cmaes'`` / ``'multistart'``) marks a result whose kept vector came
        from a global optimizer; such results carry ``NaN`` covariance and
        ``iterations == 0`` by construction (see the class Notes).
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
