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

import hashlib
import time
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

            # Check for non-finite values
            if not np.all(np.isfinite(arr)):
                non_finite_count = np.sum(~np.isfinite(arr))
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        category="data_quality",
                        message=f"Non-finite values found in {key}: {non_finite_count} values",
                        parameter=key,
                        value=non_finite_count,
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

            # Check diagonal values (should be around 1.0 at t=0)
            diagonal = np.diag(matrix)
            t0_correlation = diagonal[0] if len(diagonal) > 0 else 0

            if not (0.5 <= t0_correlation <= 2.0):
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="data_quality",
                        message=f"Unusual t=0 correlation value in matrix {i}: {t0_correlation:.3f}",
                        parameter="c2_exp",
                        value=t0_correlation,
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


# Enhanced validation features for quality control integration


@dataclass
class IncrementalValidationCache:
    """Cache for incremental validation results."""

    data_hash: str
    validation_level: str
    report: DataQualityReport
    timestamp: float
    # Reserved for future incremental validation -- not currently used
    component_hashes: dict[str, str] = field(default_factory=dict)

    def is_valid_for_data(
        self,
        data: dict[str, Any],
        validation_level: str,
        max_age: float = 3600,
    ) -> bool:
        """Check if cached result is valid for given data."""
        # Check validation level
        if self.validation_level != validation_level:
            return False

        # Check age
        if time.time() - self.timestamp > max_age:
            return False

        # Check data hash
        current_hash = _compute_data_hash(data)
        return current_hash == self.data_hash


# Global cache for incremental validation
_validation_cache: dict[str, IncrementalValidationCache] = {}


def validate_xpcs_data_incremental(
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
    validation_level: str = "basic",
    previous_report: DataQualityReport | None = None,
    force_revalidate: bool = False,
) -> DataQualityReport:
    """Run XPCS validation with incremental caching and stage awareness.

    Parameters
    ----------
    data
        XPCS data dictionary.
    config
        Optional configuration dictionary.
    validation_level
        Validation level; one of ``"basic"``, ``"full"``, ``"incremental"``,
        or ``"none"``.
    previous_report
        Previous validation report used for incremental comparison.
    force_revalidate
        Force full revalidation, ignoring any cached result.

    Returns
    -------
    DataQualityReport
        Comprehensive data quality report with incremental optimization.
    """
    logger.info(
        f"Starting incremental XPCS data validation (level: {validation_level})",
    )

    if validation_level == "none":
        return DataQualityReport(
            is_valid=True,
            validation_level=validation_level,
            total_issues=0,
            quality_score=1.0,
        )

    # Check cache for incremental validation
    if not force_revalidate and validation_level == "incremental":
        cache_key = _generate_cache_key(data, validation_level)
        cached_result = _check_validation_cache(cache_key, data, validation_level)
        if cached_result:
            logger.debug("Using cached validation result")
            return cached_result

    # Perform full or incremental validation
    start_time = time.time()

    if validation_level == "incremental" and previous_report:
        # Pass an empty dict when no config supplied — incremental validation
        # only reads the keys it specifically needs, treating missing keys as
        # "no constraint" rather than tripping on a None deref.
        report = _perform_incremental_validation(data, config or {}, previous_report)
    else:
        # Use existing validation function
        report = validate_xpcs_data(data, config, validation_level)

    # Cache result if appropriate
    if validation_level in ["incremental", "full"]:
        _cache_validation_result(data, validation_level, report)

    processing_time = time.time() - start_time
    logger.info(
        f"Incremental validation completed in {processing_time:.3f}s: "
        f"{len(report.errors)} errors, {len(report.warnings)} warnings",
    )

    return report


def validate_data_component(
    data: dict[str, Any],
    component_name: str,
    validation_level: str = "basic",
    config: dict[str, Any] | None = None,
) -> DataQualityReport:
    """Validate a specific component of XPCS data for selective validation.

    Parameters
    ----------
    data
        Complete XPCS data dictionary.
    component_name
        Specific component to validate (e.g. ``'c2_exp'``,
        ``'wavevector_q_list'``).
    validation_level
        Validation level.
    config
        Optional configuration dictionary.

    Returns
    -------
    DataQualityReport
        Data quality report focused on the specified component.
    """
    logger.debug(f"Validating data component: {component_name}")

    report = DataQualityReport(
        is_valid=True,
        validation_level=f"{validation_level}_component",
        total_issues=0,
    )

    if component_name not in data:
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="format",
                message=f"Required component '{component_name}' not found",
                parameter=component_name,
                recommendation="Check data loading and component naming",
            ),
        )
        return report

    # Component-specific validation
    component_data = {component_name: data[component_name]}

    # Add related components if needed
    if component_name == "c2_exp":
        if "t1" in data:
            component_data["t1"] = data["t1"]
        if "t2" in data:
            component_data["t2"] = data["t2"]

    # Validate using existing functions
    if component_name in ["wavevector_q_list", "phi_angles_list"]:
        _validate_array_component(component_data, component_name, report, config)
    elif component_name == "c2_exp":
        _validate_correlation_component(component_data, report)
    elif component_name in ["t1", "t2"]:
        _validate_time_component(component_data, component_name, report)

    report.quality_score = _compute_quality_score(report)
    return report


def _perform_incremental_validation(
    data: dict[str, Any],
    config: dict[str, Any],
    previous_report: DataQualityReport,
) -> DataQualityReport:
    """Perform optimized incremental validation using previous results."""
    logger.debug("Performing incremental validation")

    # Start with previous report as base
    report = DataQualityReport(
        is_valid=previous_report.is_valid,
        validation_level="incremental",
        total_issues=0,
        data_statistics=previous_report.data_statistics.copy(),
        physics_checks=previous_report.physics_checks.copy(),
    )

    # Identify what has changed since last validation
    changed_components = _identify_changed_components(data, previous_report)

    if not changed_components:
        logger.debug("No data changes detected - using cached results")
        # Copy issues from previous report
        report.errors = previous_report.errors.copy()
        report.warnings = previous_report.warnings.copy()
        report.info = previous_report.info.copy()
        report.total_issues = previous_report.total_issues
        report.quality_score = previous_report.quality_score
        return report

    logger.debug(f"Re-validating changed components: {changed_components}")

    # Re-validate only changed components
    for component in changed_components:
        component_report = validate_data_component(data, component, "basic", config)

        # Merge component validation results
        report.errors.extend(component_report.errors)
        report.warnings.extend(component_report.warnings)
        report.info.extend(component_report.info)
        report.total_issues += component_report.total_issues

        if not component_report.is_valid:
            report.is_valid = False

    # Re-compute overall quality score
    report.quality_score = _compute_quality_score(report)

    return report


def _validate_array_component(
    data: dict[str, Any],
    component_name: str,
    report: DataQualityReport,
    config: dict[str, Any] | None = None,
) -> None:
    """Validate array components (q_list, phi_list)."""
    value = data[component_name]

    # Basic array validation
    if isinstance(value, (np.ndarray, list)) or (HAS_JAX and hasattr(value, "shape")):
        arr = np.asarray(value)

        # Check for non-finite values
        if not np.all(np.isfinite(arr)):
            non_finite_count = np.sum(~np.isfinite(arr))
            report.add_issue(
                ValidationIssue(
                    severity="error",
                    category="data_quality",
                    message=f"Non-finite values in {component_name}: {non_finite_count}",
                    parameter=component_name,
                    recommendation="Check data preprocessing",
                ),
            )

        # Component-specific checks
        if component_name == "wavevector_q_list":
            if np.any(arr <= 0):
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        category="physics",
                        message="Non-positive q-values found",
                        parameter=component_name,
                        recommendation="Q-values must be positive",
                    ),
                )

        elif component_name == "phi_angles_list":
            # Check phi angle range
            if np.any(arr < -360) or np.any(arr > 360):
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="data_quality",
                        message="Phi angles outside typical range [-360, 360]",
                        parameter=component_name,
                        recommendation="Check angle units and calibration",
                    ),
                )


def _validate_correlation_component(
    data: dict[str, Any],
    report: DataQualityReport,
) -> None:
    """Validate correlation matrix component."""
    c2_exp = data.get("c2_exp")
    if c2_exp is None:
        return

    arr = np.asarray(c2_exp)

    # Basic correlation validation
    if not np.all(np.isfinite(arr)):
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="data_quality",
                message="Non-finite values in correlation data",
                parameter="c2_exp",
                recommendation="Check correlation calculation",
            ),
        )

    # Check correlation value range
    if arr.size > 0:
        mean_val = np.nanmean(arr)
        if not (0.1 <= mean_val <= 10.0):
            report.add_issue(
                ValidationIssue(
                    severity="warning",
                    category="data_quality",
                    message=f"Unusual correlation values: mean={mean_val:.3f}",
                    parameter="c2_exp",
                    recommendation="Check normalization and baseline",
                ),
            )


def _validate_time_component(
    data: dict[str, Any],
    component_name: str,
    report: DataQualityReport,
) -> None:
    """Validate time array components."""
    value = data[component_name]
    arr = np.asarray(value)

    # Check for negative time values
    if np.any(arr < 0):
        negative_count = np.sum(arr < 0)
        report.add_issue(
            ValidationIssue(
                severity="error",
                category="physics",
                message=f"Negative time values in {component_name}: {negative_count}",
                parameter=component_name,
                recommendation="Time values must be non-negative",
            ),
        )


def _compute_data_hash(data: dict[str, Any]) -> str:
    """Compute hash of data for caching."""
    hash_data = []

    for key in sorted(data.keys()):
        value = data[key]
        if hasattr(value, "shape"):
            # Use shape and checksum for arrays
            arr = np.asarray(value)
            if arr.size > 0:
                # More robust hash: shape + sum + std + first/last/middle values.
                # Use nan-safe reductions: np.sum/std return NaN for arrays with
                # non-finite values, making np.nan != np.nan always True → every
                # hash comparison differs → persistent cache miss. np.nansum/nanstd
                # produce stable finite fingerprints for NaN-containing arrays.
                _first = arr.flat[0] if np.isfinite(arr.flat[0]) else 0.0
                _last = arr.flat[-1] if np.isfinite(arr.flat[-1]) else 0.0
                # Content-sensitive checksum over up to 64 evenly-spaced elements.
                # Without it, two distinct datasets sharing shape + summary stats
                # (sum/std/first/last) collide to the same cache key, and the second
                # dataset silently receives the first's cached validation report.
                n_sample = int(min(arr.size, 64))
                sample_idx = np.linspace(0, arr.size - 1, n_sample).astype(np.int64)
                sample = np.nan_to_num(arr.flat[sample_idx].astype(np.float64))
                sample_checksum = hashlib.sha256(sample.tobytes()).hexdigest()[:16]
                fingerprint = (
                    f"{key}:{arr.shape}:{arr.dtype}:{np.nansum(arr):.15g}:"
                    f"{np.nanstd(arr):.15g}:{float(_first):.15g}:"
                    f"{float(_last):.15g}:{sample_checksum}"
                )
                hash_data.append(fingerprint)
            else:
                hash_data.append(f"{key}:{arr.shape}:empty")
        else:
            # Non-array values use string hash
            hash_data.append(f"{key}:{hash(str(value))}")

    combined_str = "|".join(hash_data)
    return hashlib.sha256(combined_str.encode()).hexdigest()


def _generate_cache_key(data: dict[str, Any], validation_level: str) -> str:
    """Generate cache key for validation results."""
    data_hash = _compute_data_hash(data)
    return f"{validation_level}:{data_hash}"


def _check_validation_cache(
    cache_key: str,
    data: dict[str, Any],
    validation_level: str,
) -> DataQualityReport | None:
    """Check validation cache for existing results."""
    if cache_key in _validation_cache:
        cache_entry = _validation_cache[cache_key]
        if cache_entry.is_valid_for_data(data, validation_level):
            return cache_entry.report
    return None


def _cache_validation_result(
    data: dict[str, Any],
    validation_level: str,
    report: DataQualityReport,
) -> None:
    """Cache validation result for future use."""
    data_hash = _compute_data_hash(data)
    cache_key = f"{validation_level}:{data_hash}"

    cache_entry = IncrementalValidationCache(
        data_hash=data_hash,
        validation_level=validation_level,
        report=report,
        timestamp=time.time(),
    )

    _validation_cache[cache_key] = cache_entry

    # Limit cache size
    if len(_validation_cache) > 100:
        # Remove oldest entries by timestamp
        oldest_keys = sorted(
            _validation_cache.keys(),
            key=lambda k: _validation_cache[k].timestamp,
        )[:10]
        for key in oldest_keys:
            del _validation_cache[key]


def _identify_changed_components(
    data: dict[str, Any],
    previous_report: DataQualityReport,
) -> list[str]:
    """Identify which data components have changed since last validation."""
    # For now, simple implementation - in production could use more sophisticated change detection
    if not hasattr(previous_report, "data_statistics") or not previous_report.data_statistics:
        return list(data.keys())  # First time validation

    changed = []

    for key, value in data.items():
        if key in ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]:
            if hasattr(value, "shape"):
                arr = np.asarray(value)
                # Richer change detection: shape + sum + std + first + last
                # Matches the fingerprint stored by _compute_data_statistics.
                if arr.size > 0:
                    _first = float(arr.flat[0]) if np.isfinite(arr.flat[0]) else 0.0
                    _last = float(arr.flat[-1]) if np.isfinite(arr.flat[-1]) else 0.0
                    current_signature = (
                        f"{arr.shape}:{np.nansum(arr):.15g}:{np.nanstd(arr):.15g}"
                        f":{_first:.15g}:{_last:.15g}"
                    )
                else:
                    current_signature = f"{arr.shape}:0"

                if key in previous_report.data_statistics:
                    previous_stats = previous_report.data_statistics[key]
                    if isinstance(previous_stats, dict):
                        prev_shape = previous_stats.get("shape", "")
                        prev_sum = float(previous_stats.get("sum", 0))
                        prev_std = float(previous_stats.get("std", 0))
                        prev_first = float(previous_stats.get("first", 0))
                        prev_last = float(previous_stats.get("last", 0))
                        previous_signature = (
                            f"{prev_shape}:{prev_sum:.15g}:{prev_std:.15g}"
                            f":{prev_first:.15g}:{prev_last:.15g}"
                        )

                        if current_signature != previous_signature:
                            changed.append(key)
                    else:
                        changed.append(key)  # No previous statistics
                else:
                    changed.append(key)  # New component

    return changed


def clear_validation_cache() -> None:
    """Clear the validation cache."""
    _validation_cache.clear()
    logger.debug("Validation cache cleared")


def get_cache_stats() -> dict[str, Any]:
    """Get validation cache statistics."""
    if not _validation_cache:
        return {"cache_size": 0, "message": "Cache is empty"}

    cache_ages = [time.time() - entry.timestamp for entry in _validation_cache.values()]

    return {
        "cache_size": len(_validation_cache),
        "oldest_entry_age": max(cache_ages),
        "newest_entry_age": min(cache_ages),
        "average_age": np.mean(cache_ages),
    }


# Export main functions including enhanced features
__all__ = [
    "validate_xpcs_data",
    "validate_xpcs_data_incremental",
    "validate_data_component",
    "DataQualityReport",
    "ValidationIssue",
    "ValidationLevel",
    "IncrementalValidationCache",
    "clear_validation_cache",
    "get_cache_stats",
]
