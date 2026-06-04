"""Validation utilities for NLSQ optimization.

Consolidates three modules from the homodyne port — ``fit_quality``,
``input_validator``, and ``result_validator`` — into a single module so the
public API is reachable from ``xpcsjax.optimization.nlsq.validation``.

The fit-quality classifier (``classify_fit_quality``) was previously inlined
inside ``heterodyne_core.py``; it lives here now and is the single source of
truth for quality band labels (``good`` / ``acceptable`` / ``poor`` /
``unknown``). The bands match homodyne's :class:`FitQualityConfig` defaults
(``<=2`` good, ``<=5`` acceptable, ``>5`` poor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.results import QualityFlag

logger = get_logger(__name__)


# =============================================================================
# Fit quality validation
# =============================================================================


@dataclass
class FitQualityConfig:
    """Configuration for fit quality validation.

    Attributes
    ----------
    enable : bool
        Whether to enable quality validation. Default: True.
    reduced_chi_squared_threshold : float
        Warn if reduced chi-squared exceeds this. Default: 10.0.
    chi2_good_threshold : float
        Reduced chi-squared at or below which fit is classified as "good".
        Default: 2.0.
    chi2_acceptable_threshold : float
        Reduced chi-squared at or below which fit is classified as
        "acceptable". Default: 5.0.
    min_parameter_significance : float
        Minimum parameter/uncertainty ratio for significance. Default: 2.0.
    max_condition_number : float
        Maximum covariance matrix condition number. Default: 1e12.
    warn_on_max_restarts : bool
        Warn if CMA-ES reached max_restarts. Default: True.
    warn_on_bounds_hit : bool
        Warn if physical parameters hit bounds. Default: True.
    warn_on_convergence_failure : bool
        Warn if convergence_status indicates failure. Default: True.
    bounds_tolerance : float
        Tolerance for "at bounds" detection. Default: 1e-9.
    """

    enable: bool = True
    reduced_chi_squared_threshold: float = 10.0
    chi2_good_threshold: float = 2.0
    chi2_acceptable_threshold: float = 5.0
    min_parameter_significance: float = 2.0
    max_condition_number: float = 1e12
    warn_on_max_restarts: bool = True
    warn_on_bounds_hit: bool = True
    warn_on_convergence_failure: bool = True
    bounds_tolerance: float = 1e-9

    @classmethod
    def from_validation_config(cls, validation_config: dict[str, Any] | None) -> FitQualityConfig:
        """Create FitQualityConfig from an NLSQValidationConfig dict.

        Parameters
        ----------
        validation_config : dict or None
            Dictionary with keys from NLSQValidationConfig TypedDict.
            If None, returns defaults.

        Returns
        -------
        FitQualityConfig
            Configuration with values from the dict, falling back to defaults.
        """
        if validation_config is None:
            return cls()
        return cls(
            chi2_good_threshold=validation_config.get("chi2_good_threshold", 2.0),
            chi2_acceptable_threshold=validation_config.get("chi2_acceptable_threshold", 5.0),
            min_parameter_significance=validation_config.get("min_parameter_significance", 2.0),
            max_condition_number=validation_config.get("max_condition_number", 1e12),
        )


@dataclass
class FitQualityReport:
    """Report from fit quality validation.

    Attributes
    ----------
    passed : bool
        True if no warnings were generated.
    warnings : list[str]
        List of warning messages.
    checks_performed : dict[str, bool]
        Which checks were performed and their pass/fail status.
    """

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    checks_performed: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving in results."""
        return {
            "quality_validation_passed": self.passed,
            "quality_warnings": self.warnings,
            "quality_checks": self.checks_performed,
        }


def classify_fit_quality(
    reduced_chi2: float | None,
    config: FitQualityConfig | None = None,
) -> str:
    """Classify reduced chi-squared into a quality band label.

    Bands (using ``<=``, not ``<``):

    - ``reduced_chi2 <= chi2_good_threshold`` → ``"good"`` (default: <= 2)
    - ``reduced_chi2 <= chi2_acceptable_threshold`` → ``"acceptable"``
      (default: <= 5)
    - otherwise → ``"poor"``
    - ``None`` or non-finite → ``"unknown"``

    Parameters
    ----------
    reduced_chi2 : float or None
        Reduced chi-squared value.
    config : FitQualityConfig, optional
        Configuration providing the band thresholds. Defaults to
        :class:`FitQualityConfig` defaults.

    Returns
    -------
    str
        One of ``"good"``, ``"acceptable"``, ``"poor"``, ``"unknown"``.
    """
    if reduced_chi2 is None or not np.isfinite(reduced_chi2):
        return "unknown"
    if config is None:
        config = FitQualityConfig()
    if reduced_chi2 <= config.chi2_good_threshold:
        return "good"
    if reduced_chi2 <= config.chi2_acceptable_threshold:
        return "acceptable"
    return "poor"


# The fit-quality band vocabulary (``classify_fit_quality``) uses "acceptable",
# but ``OptimizationResult.quality_flag`` (a ``QualityFlag`` Literal) uses
# "marginal". This bridge maps band → flag so result objects never carry the
# out-of-contract "acceptable" value.
_BAND_TO_QUALITY_FLAG: dict[str, QualityFlag] = {
    "good": "good",
    "acceptable": "marginal",
    "poor": "poor",
    "unknown": "unknown",
}


def classify_quality_flag(
    reduced_chi2: float | None,
    config: FitQualityConfig | None = None,
) -> QualityFlag:
    """Classify reduced chi-squared into an ``OptimizationResult`` ``QualityFlag``.

    Wraps :func:`classify_fit_quality` and maps its band label into the
    ``QualityFlag`` vocabulary (``"acceptable"`` → ``"marginal"``). Use this
    when populating ``OptimizationResult.quality_flag``; use
    :func:`classify_fit_quality` directly only when the band vocabulary itself
    is wanted.

    Parameters
    ----------
    reduced_chi2 : float or None
        Reduced chi-squared value.
    config : FitQualityConfig, optional
        Band thresholds. Defaults to :class:`FitQualityConfig` defaults.

    Returns
    -------
    QualityFlag
        One of ``"good"``, ``"marginal"``, ``"poor"``, ``"unknown"``.
    """
    return _BAND_TO_QUALITY_FLAG.get(classify_fit_quality(reduced_chi2, config), "unknown")


def _classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    atol: float = 1e-9,
) -> list[str]:
    """Classify each parameter's status relative to bounds."""
    statuses = []
    for val, lb, ub in zip(values, lower, upper, strict=False):
        if abs(val - lb) < atol * (1.0 + abs(lb)):
            statuses.append("at_lower_bound")
        elif abs(val - ub) < atol * (1.0 + abs(ub)):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def _is_physical_param(label: str) -> bool:
    """Return True if the label is for a physical (non per-angle-scaling) parameter."""
    from xpcsjax.config.parameter_registry import ParameterRegistry

    scaling = ParameterRegistry().scaling_names
    return not any(
        label.startswith(f"{s}_") or label.startswith(f"{s}[") or label == s for s in scaling
    )


def validate_fit_quality(
    result: Any,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    config: FitQualityConfig | None = None,
    param_labels: list[str] | None = None,
) -> FitQualityReport:
    """Validate fit quality and log warnings.

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds (lower, upper) for bounds checking.
    config : FitQualityConfig | None
        Validation configuration. Uses defaults if None.
    param_labels : list[str] | None
        Parameter labels for identifying physical vs scaling params.

    Returns
    -------
    FitQualityReport
        Validation report with warnings and check results.
    """
    if config is None:
        config = FitQualityConfig()

    if not config.enable:
        return FitQualityReport(passed=True, checks_performed={"enabled": False})

    report = FitQualityReport()

    # Check 1: Reduced chi-squared threshold
    reduced_chi_squared = getattr(result, "reduced_chi_squared", None)
    if reduced_chi_squared is not None:
        passed = reduced_chi_squared <= config.reduced_chi_squared_threshold
        report.checks_performed["reduced_chi_squared"] = passed

        if not passed:
            sigma_is_default = getattr(result, "sigma_is_default", False)
            if sigma_is_default:
                warning = (
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}), but sigma was not provided "
                    f"(using default 0.01). Chi-squared is not physically meaningful "
                    f"without experimental uncertainties. Fit quality should be assessed "
                    f"by inspecting residuals directly."
                )
            else:
                warning = (
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}). Consider reviewing fit quality."
                )
            report.warnings.append(warning)
            logger.warning(f"[FitQuality] {warning}")
            report.passed = False

        # Classify fit quality using configurable thresholds
        band = classify_fit_quality(reduced_chi_squared, config)
        if band == "good":
            report.checks_performed["chi2_quality"] = True
            logger.info(
                "[FitQuality] Chi-squared quality: good (%.4g <= %.4g)",
                reduced_chi_squared,
                config.chi2_good_threshold,
            )
        elif band == "acceptable":
            report.checks_performed["chi2_quality"] = True
            logger.info(
                "[FitQuality] Chi-squared quality: acceptable (%.4g <= %.4g)",
                reduced_chi_squared,
                config.chi2_acceptable_threshold,
            )
        else:
            report.checks_performed["chi2_quality"] = False
            logger.warning(
                "[FitQuality] Chi-squared quality: poor (%.4g > %.4g)",
                reduced_chi_squared,
                config.chi2_acceptable_threshold,
            )

    # Check 1b: Parameter significance (parameter / uncertainty ratio)
    params = getattr(result, "parameters", None)
    uncertainties = getattr(result, "uncertainties", None)
    if params is not None and uncertainties is not None:
        try:
            params_arr = np.asarray(params, dtype=np.float64)
            uncert_arr = np.asarray(uncertainties, dtype=np.float64)
            if params_arr.shape == uncert_arr.shape and len(params_arr) > 0:
                finite_mask = np.isfinite(uncert_arr) & (uncert_arr > 0) & np.isfinite(params_arr)
                if np.any(finite_mask):
                    significance = np.abs(params_arr[finite_mask]) / uncert_arr[finite_mask]
                    insignificant = significance < config.min_parameter_significance
                    if np.any(insignificant):
                        n_insig = int(np.sum(insignificant))
                        report.checks_performed["parameter_significance"] = False
                        warning = (
                            f"{n_insig} parameter(s) below significance threshold "
                            f"(|param/uncertainty| < {config.min_parameter_significance}). "
                            f"These parameters may be poorly constrained."
                        )
                        report.warnings.append(warning)
                        logger.warning(f"[FitQuality] {warning}")
                    else:
                        report.checks_performed["parameter_significance"] = True
        except (TypeError, ValueError):
            pass

    # Check 1c: Covariance matrix condition number
    pcov = getattr(result, "covariance", None)
    if pcov is None:
        pcov = getattr(result, "pcov", None)
    if pcov is not None:
        try:
            pcov_arr = np.asarray(pcov, dtype=np.float64)
            if pcov_arr.ndim == 2 and pcov_arr.shape[0] == pcov_arr.shape[1]:
                cond = np.linalg.cond(pcov_arr)
                if np.isfinite(cond):
                    if cond > config.max_condition_number:
                        report.checks_performed["condition_number"] = False
                        warning = (
                            f"Covariance matrix condition number ({cond:.2e}) exceeds "
                            f"threshold ({config.max_condition_number:.2e}). "
                            f"Parameters may be highly correlated or poorly determined."
                        )
                        report.warnings.append(warning)
                        logger.warning(f"[FitQuality] {warning}")
                        report.passed = False
                    else:
                        report.checks_performed["condition_number"] = True
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass

    # Check 2: CMA-ES max_restarts convergence
    if config.warn_on_max_restarts:
        device_info = getattr(result, "device_info", {}) or {}
        convergence_reason = device_info.get("convergence_reason", "")

        if convergence_reason == "max_restarts":
            report.checks_performed["cmaes_convergence"] = False
            warning = (
                "CMA-ES reached maximum restarts without convergence. "
                "Consider increasing max_restarts or adjusting sigma."
            )
            report.warnings.append(warning)
            logger.warning(f"[FitQuality] {warning}")
            report.passed = False
        elif convergence_reason:
            report.checks_performed["cmaes_convergence"] = True

    # Check 3: Physical parameters at bounds
    if config.warn_on_bounds_hit and bounds is not None:
        params = getattr(result, "parameters", None)
        if params is not None and len(params) > 0:
            lower, upper = bounds
            if len(params) == len(lower) == len(upper):
                statuses = _classify_parameter_status(params, lower, upper, config.bounds_tolerance)

                at_bounds = []
                for i, status in enumerate(statuses):
                    if status in ("at_lower_bound", "at_upper_bound"):
                        label = (
                            param_labels[i]
                            if param_labels and i < len(param_labels)
                            else f"param[{i}]"
                        )

                        if not _is_physical_param(label):
                            continue

                        at_bounds.append((label, status))

                report.checks_performed["physical_bounds"] = len(at_bounds) == 0

                if at_bounds:
                    params_str = ", ".join(f"{label} ({status})" for label, status in at_bounds)
                    warning = (
                        f"Physical parameters at bounds: {params_str}. "
                        "Consider expanding bounds or reviewing initial parameters."
                    )
                    report.warnings.append(warning)
                    logger.warning(f"[FitQuality] {warning}")
                    report.passed = False

    # Check 4: Convergence status
    if config.warn_on_convergence_failure:
        status = getattr(result, "convergence_status", "")
        failed_statuses = {"max_iter", "failed", "diverged", "max_iterations"}

        if status:
            passed = status.lower() not in failed_statuses
            report.checks_performed["convergence_status"] = passed

            if not passed:
                warning = f"Optimization did not converge successfully (status: {status})."
                report.warnings.append(warning)
                logger.warning(f"[FitQuality] {warning}")
                report.passed = False

    # Log summary
    if report.passed:
        logger.info("[FitQuality] All quality checks passed")
    else:
        logger.warning(f"[FitQuality] {len(report.warnings)} quality warning(s) generated")

    return report


# =============================================================================
# Input validation
# =============================================================================


class InputValidator:
    """Validator for NLSQ optimization input data."""

    def __init__(self, strict_mode: bool = True):
        """Initialize InputValidator.

        Parameters
        ----------
        strict_mode : bool, optional
            If True, raise errors on validation failures.
            If False, log warnings but continue.
        """
        self.strict_mode = strict_mode
        self._validation_errors: list[str] = []

    def validate_all(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> bool:
        """Validate all input data."""
        self._validation_errors = []

        if not validate_array_dimensions(xdata, ydata):
            self._validation_errors.append(
                f"Array dimension mismatch: xdata.shape[0]={len(xdata)}, "
                f"ydata.shape[0]={len(ydata)}"
            )

        if not validate_no_nan_inf(xdata, "xdata"):
            self._validation_errors.append("xdata contains NaN or Inf values")
        if not validate_no_nan_inf(ydata, "ydata"):
            self._validation_errors.append("ydata contains NaN or Inf values")
        if not validate_no_nan_inf(initial_params, "initial_params"):
            self._validation_errors.append("initial_params contains NaN or Inf values")

        if bounds is not None:
            if not validate_bounds_consistency(bounds, initial_params):
                self._validation_errors.append("Bounds are inconsistent with initial parameters")

        if not _validate_initial_params_within_bounds(initial_params, bounds):
            self._validation_errors.append("Initial parameters outside bounds")

        if self._validation_errors:
            if self.strict_mode:
                raise ValueError(f"Input validation failed: {'; '.join(self._validation_errors)}")
            for error in self._validation_errors:
                logger.warning(f"Input validation warning: {error}")
            return False

        return True

    @property
    def validation_errors(self) -> list[str]:
        """Get list of validation errors from last validate_all() call."""
        return self._validation_errors.copy()


def validate_array_dimensions(xdata: np.ndarray, ydata: np.ndarray) -> bool:
    """Validate that xdata and ydata have compatible dimensions."""
    if len(xdata) == 0:
        logger.warning("xdata is empty")
        return False

    if len(ydata) == 0:
        logger.warning("ydata is empty")
        return False

    if len(xdata) != len(ydata):
        logger.warning(f"Array length mismatch: xdata={len(xdata)}, ydata={len(ydata)}")
        return False

    return True


def validate_no_nan_inf(
    arr: np.ndarray,
    name: str,
    iteration: int | None = None,
    context: dict[str, Any] | None = None,
) -> bool:
    """Validate that array contains no NaN or Inf values."""
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))

        nan_indices = np.where(np.isnan(arr))[0][:10]
        inf_indices = np.where(np.isinf(arr))[0][:10]

        context_str = ""
        if iteration is not None:
            context_str += f" [iteration={iteration}]"
        if context:
            context_str += f" [context={context}]"

        logger.warning(
            f"{name} contains numerical issues{context_str}:\n"
            f"  NaN count: {nan_count}, first indices: {nan_indices.tolist()}\n"
            f"  Inf count: {inf_count}, first indices: {inf_indices.tolist()}\n"
            f"  Array shape: {arr.shape}, dtype: {arr.dtype}\n"
            f"  Array range: [{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]"
        )
        return False
    return True


def validate_bounds_consistency(
    bounds: tuple[np.ndarray, np.ndarray],
    initial_params: np.ndarray,
) -> bool:
    """Validate that bounds are consistent."""
    lower, upper = bounds

    if len(lower) != len(initial_params):
        logger.warning(f"Lower bounds length {len(lower)} != params length {len(initial_params)}")
        return False
    if len(upper) != len(initial_params):
        logger.warning(f"Upper bounds length {len(upper)} != params length {len(initial_params)}")
        return False

    if not np.all(lower <= upper):
        violations = np.where(lower > upper)[0]
        logger.warning(f"Lower > upper at indices: {violations}")
        return False

    return True


def _validate_initial_params_within_bounds(
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    """Validate that initial parameters are within bounds.

    Renamed from homodyne's ``validate_initial_params`` to avoid a name
    clash with the existing public function of the same name in
    :mod:`xpcsjax.optimization.nlsq.data_prep`, which validates against
    bounds with a different signature during data preparation.
    """
    if bounds is None:
        return True

    lower, upper = bounds

    below_lower = initial_params < lower
    above_upper = initial_params > upper

    if np.any(below_lower):
        indices = np.where(below_lower)[0]
        logger.warning(f"Params below lower bound at indices: {indices}")
        return False

    if np.any(above_upper):
        indices = np.where(above_upper)[0]
        logger.warning(f"Params above upper bound at indices: {indices}")
        return False

    return True


# =============================================================================
# Result validation
# =============================================================================


class ResultValidator:
    """Validator for NLSQ optimization results."""

    def __init__(self, strict_mode: bool = False):
        """Initialize ResultValidator.

        Parameters
        ----------
        strict_mode : bool, optional
            If True, raise errors on validation failures.
            If False, log warnings but continue.
        """
        self.strict_mode = strict_mode
        self._validation_warnings: list[str] = []

    def validate_all(
        self,
        params: np.ndarray,
        covariance: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        chi_squared: float | None = None,
    ) -> bool:
        """Validate all result components."""
        self._validation_warnings = []

        if not validate_optimized_params(params, bounds):
            self._validation_warnings.append("Optimized parameters outside bounds")

        if covariance is not None:
            if not validate_covariance(covariance, len(params)):
                self._validation_warnings.append("Covariance matrix invalid")

        if chi_squared is not None:
            if not validate_result_consistency(params, chi_squared):
                self._validation_warnings.append("Result consistency check failed")

        if self._validation_warnings:
            if self.strict_mode:
                raise ValueError(
                    f"Result validation failed: {'; '.join(self._validation_warnings)}"
                )
            for warning in self._validation_warnings:
                logger.warning(f"Result validation warning: {warning}")
            return False

        return True

    @property
    def validation_warnings(self) -> list[str]:
        """Get list of validation warnings from last validate_all() call."""
        return self._validation_warnings.copy()


def validate_optimized_params(
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    tolerance: float = 1e-10,
) -> bool:
    """Validate that optimized parameters are finite and within bounds."""
    if not np.all(np.isfinite(params)):
        nan_count = np.sum(np.isnan(params))
        inf_count = np.sum(np.isinf(params))
        logger.warning(f"Optimized params contain {nan_count} NaN, {inf_count} Inf")
        return False

    if bounds is None:
        return True

    lower, upper = bounds

    below_lower = params < (lower - tolerance)
    above_upper = params > (upper + tolerance)

    if np.any(below_lower) or np.any(above_upper):
        violations = []
        if np.any(below_lower):
            indices = np.where(below_lower)[0]
            violations.append(f"below lower at {indices.tolist()}")
        if np.any(above_upper):
            indices = np.where(above_upper)[0]
            violations.append(f"above upper at {indices.tolist()}")
        logger.warning(f"Params outside bounds: {', '.join(violations)}")
        return False

    return True


def validate_covariance(covariance: np.ndarray, n_params: int) -> bool:
    """Validate covariance matrix properties."""
    if covariance.shape != (n_params, n_params):
        logger.warning(f"Covariance shape {covariance.shape} != expected ({n_params}, {n_params})")
        return False

    if not np.all(np.isfinite(covariance)):
        nan_count = np.sum(np.isnan(covariance))
        inf_count = np.sum(np.isinf(covariance))
        logger.warning(f"Covariance contains {nan_count} NaN, {inf_count} Inf")
        return False

    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        max_diff = np.nanmax(np.abs(covariance - covariance.T))
        logger.warning(f"Covariance not symmetric, max diff={max_diff:.2e}")
        return False

    diag = np.diag(covariance)
    if np.any(diag < 0):
        neg_indices = np.where(diag < 0)[0]
        logger.warning(f"Covariance has negative diagonal at indices: {neg_indices.tolist()}")
        return False

    return True


def validate_result_consistency(
    params: np.ndarray,
    chi_squared: float,
) -> bool:
    """Validate consistency of optimization result.

    Checks that the chi-squared value is finite, non-negative, and within a
    plausible numeric range, and that ``params`` is non-empty and finite so
    a downstream caller cannot pair a "good" chi-squared with degenerate
    parameters.
    """
    if params.size == 0 or not np.all(np.isfinite(params)):
        logger.warning(
            "Result params are empty or contain non-finite entries; "
            "chi-squared cannot be consistent with them."
        )
        return False

    if not np.isfinite(chi_squared):
        logger.warning(f"Chi-squared is not finite: {chi_squared}")
        return False

    if chi_squared < 0:
        logger.warning(f"Chi-squared is negative: {chi_squared}")
        return False

    if chi_squared < 1e-15:
        logger.warning(f"Chi-squared suspiciously low: {chi_squared:.2e}")

    if chi_squared > 1e10:
        logger.warning(f"Chi-squared very high: {chi_squared:.2e}")

    return True


__all__ = [
    # Fit quality
    "FitQualityConfig",
    "FitQualityReport",
    "classify_fit_quality",
    "classify_quality_flag",
    "validate_fit_quality",
    # Input validation
    "InputValidator",
    "validate_array_dimensions",
    "validate_bounds_consistency",
    "validate_no_nan_inf",
    # Result validation
    "ResultValidator",
    "validate_covariance",
    "validate_optimized_params",
    "validate_result_consistency",
]
