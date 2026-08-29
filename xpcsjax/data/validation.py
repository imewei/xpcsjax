"""Data validation for XPCS datasets.

Comprehensive data quality validation and physics consistency checks for XPCS
data. Integrates with v2 physics constants and provides detailed validation
reports. Enhanced with incremental and stage-based validation for quality
control integration.

This module provides:

- Physics-based validation using v2 ``PhysicsConstants``.
- Data quality and integrity checks.
- Correlation matrix validation.
- Statistical consistency checks.
- Integration with the YAML configuration system.
- Incremental validation with caching for performance.
- Stage-based validation for the data processing pipeline.
- Selective validation for specific data components.

Notes
-----
Validation levels:

- ``"basic"``: Essential data integrity checks.
- ``"full"``: Comprehensive physics and statistical validation.
- ``"custom"``: User-configurable validation rules.
- ``"incremental"``: Optimized validation using cached results.

Enhanced features:

- Incremental validation with intelligent caching.
- Stage-aware validation for different processing phases.
- Selective validation of data subsets.
- Performance-optimized validation with early termination.
- Integration with ``DataQualityController`` for comprehensive quality control.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

# JAX integration
try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore

# V2 integration
try:
    from xpcsjax.core.physics import PhysicsConstants

    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False
    PhysicsConstants = None  # type: ignore

import logging

try:
    from xpcsjax.utils.logging import get_logger, log_exception

    HAS_V2_LOGGING = True
except ImportError:
    HAS_V2_LOGGING = False

    # Fallback shim. The real ``xpcsjax.utils.logging.get_logger`` has a
    # broader contract (optional name, optional context, may return a
    # LoggerAdapter); this fallback only needs to feed module-level ``logger``
    # and never sees the context kwarg. ``# type: ignore[misc]`` acknowledges
    # the signature delta with the try-branch import.
    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)

    def log_exception(  # type: ignore[misc]
        logger: logging.Logger,
        exc: BaseException,
        context: dict[str, Any] | None = None,
        level: int = logging.ERROR,
        include_traceback: bool = True,
    ) -> None:
        """Fallback: emit the exception via the stdlib logger.

        Mirrors the real ``xpcsjax.utils.logging.log_exception`` contract
        closely enough for the ERROR-level data-integrity path; observational
        only, never re-raises.
        """
        suffix = f" (context: {context})" if context else ""
        logger.log(level, f"{type(exc).__name__}: {exc}{suffix}", exc_info=include_traceback)


logger = get_logger(__name__)


# Closed value sets for ValidationIssue. add_issue() triages on `severity`, so a
# typo (e.g. "err") would silently land an error in the info bucket without
# flipping is_valid. Pinning these as Literal makes such a typo a type error.
SeverityLevel = Literal["error", "warning", "info"]
IssueCategory = Literal[
    "physics",
    "data_quality",
    "statistics",
    "format",
    "validation",
    "completeness",
    "consistency",
    "preprocessing",
]


class ValidationLevel(Enum):
    """Validation level enumeration."""

    NONE = "none"
    BASIC = "basic"
    FULL = "full"
    CUSTOM = "custom"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    severity: SeverityLevel
    category: IssueCategory
    message: str
    parameter: str | None = None
    value: Any | None = None
    recommendation: str | None = None


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""

    is_valid: bool
    validation_level: str
    total_issues: int
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)

    # Statistics
    data_statistics: dict[str, Any] = field(default_factory=dict)
    physics_checks: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the report."""
        if issue.severity == "error":
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == "warning":
            self.warnings.append(issue)
        else:
            self.info.append(issue)

        self.total_issues += 1

    def get_summary(self) -> dict[str, Any]:
        """Get summary of validation results."""
        return {
            "is_valid": self.is_valid,
            "validation_level": self.validation_level,
            "total_issues": self.total_issues,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len(self.info),
            "quality_score": self.quality_score,
            "has_physics_validation": bool(self.physics_checks),
            "has_statistics": bool(self.data_statistics),
        }


def validate_xpcs_data(
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
    validation_level: str = "basic",
) -> DataQualityReport:
    """Run comprehensive XPCS data validation.

    Parameters
    ----------
    data
        XPCS data dictionary with keys ``wavevector_q_list``,
        ``phi_angles_list``, ``t1``, ``t2``, and ``c2_exp``.
    config
        Optional configuration dictionary.
    validation_level
        Validation level; one of ``"basic"``, ``"full"``, or ``"none"``.

    Returns
    -------
    DataQualityReport
        Comprehensive data quality report.
    """
    logger.info(f"Starting XPCS data validation (level: {validation_level})")

    report = DataQualityReport(
        is_valid=True,
        validation_level=validation_level,
        total_issues=0,
    )

    if validation_level == "none":
        logger.info("Validation disabled - skipping all checks")
        return report

    try:
        # Basic validation
        _validate_data_structure(data, report)
        _validate_data_integrity(data, report)
        _validate_array_shapes(data, report)

        if validation_level == "full":
            # Comprehensive validation. Physics validation expects a config
            # dict — pass an empty one if the caller didn't supply config,
            # so downstream "missing key" checks treat it as "no constraints
            # to enforce" rather than crashing on a None deref.
            _validate_physics_parameters(data, config or {}, report)
            _validate_correlation_matrices(data, report)
            _validate_statistical_properties(data, report)
            _compute_data_statistics(data, report)

        # Compute overall quality score
        report.quality_score = _compute_quality_score(report)

        logger.info(
            f"Validation completed: {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings, quality_score={report.quality_score:.2f}",
        )

    except (ValueError, TypeError, KeyError) as e:
        # Narrow catch: data-shape/format errors become a graceful report, but
        # programming errors (AttributeError, AssertionError, ...) propagate
        # rather than being silently masked as a passing validation.
        logger.error(f"Validation failed with data error: {e}")
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="validation",
                message=f"Validation process failed: {str(e)}",
                recommendation="Check data format and try again",
            ),
        )

    return report


def _validate_data_structure(data: dict[str, Any], report: DataQualityReport) -> None:
    """Validate basic data structure and required keys."""
    required_keys = ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]

    for key in required_keys:
        if key not in data:
            report.add_issue(
                ValidationIssue(
                    severity="error",
                    category="format",
                    message=f"Missing required data key: {key}",
                    parameter=key,
                    recommendation="Check data loading process",
                ),
            )


def _validate_data_integrity(data: dict[str, Any], report: DataQualityReport) -> None:
    """Validate data integrity (finite values, reasonable ranges)."""
    for key, value in data.items():
        if isinstance(value, (np.ndarray, list)) or (HAS_JAX and hasattr(value, "shape")):
            # Convert to numpy for validation
            arr = np.asarray(value)

            # Check for non-finite values. wavevector_q_list tolerates NaN
            # (legitimate bad-pixel masking, one entry per (q, phi) pair, per
            # xpcs_loader.py::_validate_loaded_arrays) but still rejects inf.
            if key == "wavevector_q_list":
                has_bad = np.isinf(arr).any()
                bad_count = np.sum(np.isinf(arr))
            else:
                has_bad = not np.all(np.isfinite(arr))
                bad_count = np.sum(~np.isfinite(arr))
            if has_bad:
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        category="data_quality",
                        message=f"Non-finite values found in {key}: {bad_count} values",
                        parameter=key,
                        value=bad_count,
                        recommendation="Check data preprocessing and file integrity",
                    ),
                )

            # Check for reasonable value ranges based on parameter type
            if key == "c2_exp":
                if np.any(arr < 0):
                    negative_count = np.sum(arr < 0)
                    report.add_issue(
                        ValidationIssue(
                            severity="warning",
                            category="data_quality",
                            message=f"Negative correlation values found: {negative_count} values",
                            parameter=key,
                            value=negative_count,
                            recommendation="Check correlation calculation and baseline correction",
                        ),
                    )

            elif key in ["wavevector_q_list"]:
                if np.any(arr <= 0):
                    non_positive_count = np.sum(arr <= 0)
                    report.add_issue(
                        ValidationIssue(
                            severity="error",
                            category="physics",
                            message=f"Non-positive q-values found: {non_positive_count} values",
                            parameter=key,
                            value=non_positive_count,
                            recommendation="Q-values must be positive",
                        ),
                    )

            elif key in ["t1", "t2"]:
                if np.any(arr < 0):
                    negative_count = np.sum(arr < 0)
                    report.add_issue(
                        ValidationIssue(
                            severity="error",
                            category="physics",
                            message=f"Negative time values found in {key}: {negative_count} values",
                            parameter=key,
                            value=negative_count,
                            recommendation="Time values must be non-negative",
                        ),
                    )


def _validate_array_shapes(data: dict[str, Any], report: DataQualityReport) -> None:
    """Validate array shape consistency."""
    try:
        np.asarray(data.get("wavevector_q_list", []))
        np.asarray(data.get("phi_angles_list", []))
        t1 = np.asarray(data.get("t1", []))
        t2 = np.asarray(data.get("t2", []))
        c2_exp = np.asarray(data.get("c2_exp", []))

        # Check time array consistency
        if t1.shape != t2.shape:
            report.add_issue(
                ValidationIssue(
                    severity="error",
                    category="format",
                    message=f"t1 and t2 have inconsistent shapes: {t1.shape} vs {t2.shape}",
                    recommendation="Time arrays must have same shape",
                ),
            )

        # Check correlation matrix dimensions
        if c2_exp.ndim >= 3:
            n_matrices, matrix_size1, matrix_size2 = c2_exp.shape[-3:]

            if matrix_size1 != matrix_size2:
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        category="format",
                        message=f"Correlation matrices not square: {matrix_size1} x {matrix_size2}",
                        recommendation="Correlation matrices must be square",
                    ),
                )

            if matrix_size1 != len(t1):
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="format",
                        message=f"Matrix size {matrix_size1} doesn't match time array length {len(t1)}",
                        recommendation="Matrix dimensions should match time array length",
                    ),
                )

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "_validate_array_shapes"},
            level=logging.ERROR,
        )
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="validation",
                message=f"validation crashed: {exc}",
            ),
        )


def _validate_physics_parameters(
    data: dict[str, Any],
    config: dict[str, Any],
    report: DataQualityReport,
) -> None:
    """Validate physics parameters against known constraints."""
    if not HAS_PHYSICS:
        report.add_issue(
            ValidationIssue(
                severity="info",
                category="physics",
                message="Physics validation unavailable - v2 physics module not found",
                recommendation="Install v2 physics module for enhanced validation",
            ),
        )
        return

    try:
        q_values = np.asarray(data.get("wavevector_q_list", []))

        # Validate q-range against physics constants.
        # Use nanmin/nanmax: q_values from HDF5 may contain NaN for bad pixels.
        if len(q_values) > 0:
            q_min, q_max = np.nanmin(q_values), np.nanmax(q_values)

            if q_min < PhysicsConstants.Q_MIN_TYPICAL:
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="physics",
                        message=f"Q-values below typical range: min={q_min:.2e}, typical_min={PhysicsConstants.Q_MIN_TYPICAL:.2e}",
                        parameter="wavevector_q_list",
                        value=q_min,
                        recommendation="Check experimental setup and detector geometry",
                    ),
                )

            if q_max > PhysicsConstants.Q_MAX_TYPICAL:
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="physics",
                        message=f"Q-values above typical range: max={q_max:.2e}, typical_max={PhysicsConstants.Q_MAX_TYPICAL:.2e}",
                        parameter="wavevector_q_list",
                        value=q_max,
                        recommendation="Check experimental setup and resolution limits",
                    ),
                )

        # Validate time parameters from config
        if config:
            analyzer_params = config.get("analyzer_parameters", {})
            dt = analyzer_params.get("dt")

            if dt is not None:
                if dt < PhysicsConstants.TIME_MIN_XPCS:
                    report.add_issue(
                        ValidationIssue(
                            severity="warning",
                            category="physics",
                            message=f"Time step dt={dt}s below typical XPCS minimum: {PhysicsConstants.TIME_MIN_XPCS}s",
                            parameter="dt",
                            value=dt,
                            recommendation="Check time resolution and detector capabilities",
                        ),
                    )

                if dt > PhysicsConstants.TIME_MAX_XPCS:
                    report.add_issue(
                        ValidationIssue(
                            severity="info",
                            category="physics",
                            message=f"Time step dt={dt}s above typical XPCS range: {PhysicsConstants.TIME_MAX_XPCS}s",
                            parameter="dt",
                            value=dt,
                        ),
                    )

        if len(q_values) > 0:
            # Reuse q_min/q_max already computed above (avoids redundant nanmin/nanmax).
            _q_min = float(q_min)
            _q_max = float(q_max)
            report.physics_checks = {
                "q_range_valid": PhysicsConstants.Q_MIN_TYPICAL
                <= _q_min
                <= _q_max
                <= PhysicsConstants.Q_MAX_TYPICAL,
                "q_min": _q_min,
                "q_max": _q_max,
            }
        else:
            report.physics_checks = {
                "q_range_valid": False,
                "q_min": None,
                "q_max": None,
            }

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "_validate_physics_parameters"},
            level=logging.ERROR,
        )
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="physics",
                message=f"validation crashed: {exc}",
            ),
        )


def _validate_correlation_matrices(
    data: dict[str, Any],
    report: DataQualityReport,
) -> None:
    """Validate correlation matrix properties."""
    try:
        c2_exp = np.asarray(data.get("c2_exp", []))

        if c2_exp.size == 0:
            return

        # Check correlation matrix properties
        for i, matrix in enumerate(c2_exp):
            if matrix.ndim != 2:
                continue

            # Check symmetry
            if not np.allclose(matrix, matrix.T, atol=1e-10):
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="data_quality",
                        message=f"Correlation matrix {i} not symmetric",
                        parameter="c2_exp",
                        recommendation="Check matrix reconstruction process",
                    ),
                )

            # Check the near-zero-lag correlation against the Siegert ceiling
            # g2(0) = 1 + beta in [1.0, 2.0] (beta in [0, 1]). The EXACT tau=0
            # main diagonal (matrix[k, k]) is the self-correlation / shot-noise
            # spike — for raw two-time XPCS it routinely sits at ~2.4 (single
            # pixels far higher) and is excluded from analysis (frame-0 / diagonal
            # exclusion). Evaluating it here would flag every angle of valid data.
            # Use the finite MEDIAN over the whole first superdiagonal (the
            # smallest non-zero lag), where the Siegert relation actually holds;
            # this still catches genuinely over-normalized data (lag-1 g2 > 2.0)
            # while a single boundary artifact in an otherwise-clean angle cannot
            # by itself trip the warning. Mirrors data/filtering_utils.py.
            if matrix.shape[1] > 1:
                superdiag = np.diagonal(matrix, offset=1)
                finite_superdiag = superdiag[np.isfinite(superdiag)]
                near_zero_lag_correlation = (
                    float(np.median(finite_superdiag)) if finite_superdiag.size > 0 else 0.0
                )
                has_lag1 = True
            else:
                # No lag-1 (first superdiagonal) exists for a <=1-frame matrix,
                # so there is nothing to check against the Siegert ceiling.
                # diagonal[0] would be the excluded tau=0 self-correlation
                # spike itself -- exactly the value this check exists to avoid.
                near_zero_lag_correlation = 0.0
                has_lag1 = False

            if has_lag1 and not (0.5 <= near_zero_lag_correlation <= 2.0):
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="data_quality",
                        message=(
                            f"Unusual near-zero-lag correlation in matrix {i}: "
                            f"{near_zero_lag_correlation:.3f}"
                        ),
                        parameter="c2_exp",
                        value=near_zero_lag_correlation,
                        recommendation="Check normalization and baseline correction",
                    ),
                )

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "_validate_correlation_matrices"},
            level=logging.ERROR,
        )
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="validation",
                message=f"validation crashed: {exc}",
            ),
        )


def _validate_statistical_properties(
    data: dict[str, Any],
    report: DataQualityReport,
) -> None:
    """Validate statistical properties of the data."""
    try:
        c2_exp = np.asarray(data.get("c2_exp", []))

        if c2_exp.size == 0:
            return

        # Check for reasonable statistical properties
        mean_correlation = np.nanmean(c2_exp)
        std_correlation = np.nanstd(c2_exp)

        if mean_correlation < 0.5 or mean_correlation > 2.0:
            report.add_issue(
                ValidationIssue(
                    severity="warning",
                    category="statistics",
                    message=f"Unusual mean correlation value: {mean_correlation:.3f}",
                    value=mean_correlation,
                    recommendation="Check data normalization",
                ),
            )

        # Check for excessive noise
        if std_correlation > mean_correlation:
            report.add_issue(
                ValidationIssue(
                    severity="info",
                    category="statistics",
                    message=f"High correlation variability: std={std_correlation:.3f}, mean={mean_correlation:.3f}",
                    recommendation="Data may be noisy - consider preprocessing",
                ),
            )

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "_validate_statistical_properties"},
            level=logging.ERROR,
        )
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="validation",
                message=f"validation crashed: {exc}",
            ),
        )


def _compute_data_statistics(data: dict[str, Any], report: DataQualityReport) -> None:
    """Compute comprehensive data statistics."""
    try:
        stats = {}

        for key, value in data.items():
            if isinstance(value, (np.ndarray, list)) or (HAS_JAX and hasattr(value, "shape")):
                arr = np.asarray(value)

                stats[key] = {
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "mean": float(np.nanmean(arr)) if arr.size > 0 else 0.0,
                    "std": float(np.nanstd(arr)) if arr.size > 0 else 0.0,
                    "min": float(np.nanmin(arr)) if arr.size > 0 else 0.0,
                    "max": float(np.nanmax(arr)) if arr.size > 0 else 0.0,
                    "finite_fraction": (
                        float(np.sum(np.isfinite(arr)) / arr.size) if arr.size > 0 else 0.0
                    ),
                    # sum/first/last are used by _identify_changed_components
                    # as a change-detection fingerprint. Must be stored here or
                    # incremental validation always falls back to full re-validation.
                    # Use nan-safe reductions so NaN-containing arrays still produce
                    # stable, finite fingerprints (np.sum/std propagate NaN →
                    # np.nan != np.nan is True → cache miss on every call).
                    "sum": float(np.nansum(arr)) if arr.size > 0 else 0.0,
                    "first": (
                        float(arr.flat[0]) if arr.size > 0 and np.isfinite(arr.flat[0]) else 0.0
                    ),
                    "last": (
                        float(arr.flat[-1]) if arr.size > 0 and np.isfinite(arr.flat[-1]) else 0.0
                    ),
                }

        report.data_statistics = stats

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "_compute_data_statistics"},
            level=logging.ERROR,
        )
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="validation",
                message=f"validation crashed: {exc}",
            ),
        )


def _compute_quality_score(report: DataQualityReport) -> float:
    """Compute overall data quality score (0.0 to 1.0).

    Factors:
    - Errors significantly reduce score
    - Warnings moderately reduce score
    - Data integrity issues affect score
    - Physics validation results contribute
    """
    base_score = 1.0

    # Error penalties
    error_penalty = len(report.errors) * 0.2
    warning_penalty = len(report.warnings) * 0.05

    # Data integrity bonus/penalty
    integrity_bonus = 0.0
    if report.data_statistics:
        # Bonus for having complete statistics
        integrity_bonus += 0.1

        # Penalty for non-finite data
        for _key, stats in report.data_statistics.items():
            finite_fraction = stats.get("finite_fraction", 0.0)
            if finite_fraction < 1.0:
                integrity_bonus -= (1.0 - finite_fraction) * 0.1

    # Physics validation bonus
    physics_bonus = 0.0
    if report.physics_checks:
        if report.physics_checks.get("q_range_valid", False):
            physics_bonus += 0.1

    final_score = base_score - error_penalty - warning_penalty + integrity_bonus + physics_bonus

    return max(0.0, min(1.0, final_score))


# Export main functions including enhanced features
__all__ = [
    "validate_xpcs_data",
    "DataQualityReport",
    "ValidationIssue",
    "ValidationLevel",
]
