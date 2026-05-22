"""Data Quality Controller for Homodyne
=======================================

Comprehensive data quality control system that integrates validation throughout
the data loading workflow. Provides real-time quality assessment, auto-repair
capabilities, and progressive quality control for XPCS data.

Architecture: Raw Data → Basic Validation → Filtering → Filter Validation →
             Preprocessing → Transform Validation → Final Validation → Quality Report

Key Features:
- Real-time quality assessment integration at each loading stage
- Progressive quality control system with configurable thresholds
- Auto-repair and enhancement capabilities
- Quality-based recommendations for processing settings
- Adaptive processing with fallback strategies
- Comprehensive quality metrics dashboard
- Exportable quality assessment reports

Integration Points:
- Extends existing validation.py for incremental checking
- Integrates with filtering_utils.py from Subagent 1
- Integrates with preprocessing.py from Subagent 2
- YAML configuration system for quality control parameters
- v2 logging system for comprehensive quality reporting

Quality Control Pipeline:
1. Stage 1 - Raw Data: Basic format and integrity validation
2. Stage 2 - Filtered Data: Validate filtering didn't remove too much data
3. Stage 3 - Preprocessed Data: Validate transformations preserved physics
4. Stage 4 - Final Data: Comprehensive quality assessment for analysis readiness

Authors: Homodyne Development Team
Institution: Argonne National Laboratory
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Core dependencies
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# JAX integration with fallback
try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jnp = np  # type: ignore[misc]
    HAS_JAX = False

# V2 system integration
try:
    from xpcsjax.utils.logging import get_logger, log_performance

    HAS_V2_LOGGING = True
except ImportError:
    import logging

    HAS_V2_LOGGING = False

    def get_logger(name):  # type: ignore[no-untyped-def,misc]
        return logging.getLogger(name)

    def log_performance(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator


# V2 validation system integration
try:
    from xpcsjax.data.validation import (
        DataQualityReport,
        ValidationIssue,
        ValidationLevel,
        validate_xpcs_data,
    )

    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    DataQualityReport = None  # type: ignore[assignment,misc]
    ValidationIssue = None  # type: ignore[assignment,misc]
    ValidationLevel = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


class QualityControlStage(Enum):
    """Quality control stage enumeration."""

    RAW_DATA = "raw_data"
    FILTERED_DATA = "filtered_data"
    PREPROCESSED_DATA = "preprocessed_data"
    FINAL_DATA = "final_data"


class QualityLevel(Enum):
    """Quality assessment levels."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class RepairStrategy(Enum):
    """Auto-repair strategy levels."""

    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class QualityMetrics:
    """Comprehensive data quality metrics."""

    overall_score: float = 0.0  # 0-100 quality score

    # Basic data integrity metrics
    finite_fraction: float = 0.0
    shape_consistency: bool = True
    data_range_valid: bool = True

    # Physics-based metrics
    correlation_validity: float = 0.0
    time_consistency: bool = True
    q_range_validity: float = 0.0

    # Statistical metrics
    signal_to_noise: float = 0.0
    correlation_decay: float = 0.0
    symmetry_score: float = 0.0

    # Processing stage metrics
    filtering_efficiency: float = 0.0
    preprocessing_success: bool = True
    transformation_fidelity: float = 0.0

    # Auto-repair metrics
    issues_detected: int = 0
    issues_repaired: int = 0
    repair_success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "overall_score": self.overall_score,
            "data_integrity": {
                "finite_fraction": self.finite_fraction,
                "shape_consistency": self.shape_consistency,
                "data_range_valid": self.data_range_valid,
            },
            "physics_validation": {
                "correlation_validity": self.correlation_validity,
                "time_consistency": self.time_consistency,
                "q_range_validity": self.q_range_validity,
            },
            "statistical_analysis": {
                "signal_to_noise": self.signal_to_noise,
                "correlation_decay": self.correlation_decay,
                "symmetry_score": self.symmetry_score,
            },
            "processing_metrics": {
                "filtering_efficiency": self.filtering_efficiency,
                "preprocessing_success": self.preprocessing_success,
                "transformation_fidelity": self.transformation_fidelity,
            },
            "auto_repair": {
                "issues_detected": self.issues_detected,
                "issues_repaired": self.issues_repaired,
                "repair_success_rate": self.repair_success_rate,
            },
        }


@dataclass
class QualityControlResult:
    """Result of quality control assessment."""

    stage: QualityControlStage
    passed: bool
    metrics: QualityMetrics
    issues: list[ValidationIssue] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    processing_time: float = 0.0

    # Data tracking
    data_shape_before: tuple | None = None
    data_shape_after: tuple | None = None
    data_modified: bool = False

    def get_summary(self) -> dict[str, Any]:
        """Get concise summary of quality control result."""
        return {
            "stage": self.stage.value,
            "passed": self.passed,
            "overall_score": self.metrics.overall_score,
            "issues_count": len(self.issues),
            "repairs_count": len(self.repairs_applied),
            "recommendations_count": len(self.recommendations),
            "processing_time": self.processing_time,
            "data_modified": self.data_modified,
        }


@dataclass
class QualityControlConfig:
    """Configuration for quality control system."""

    enabled: bool = True
    validation_level: str = "standard"  # none, basic, standard, comprehensive
    auto_repair: str = "conservative"  # disabled, conservative, aggressive

    # Quality thresholds (0-100 scale)
    pass_threshold: float = 50.0
    warn_threshold: float = 70.0
    excellent_threshold: float = 85.0

    # Stage-specific settings
    enable_raw_validation: bool = True
    enable_filtering_validation: bool = True
    enable_preprocessing_validation: bool = True
    enable_final_validation: bool = True

    # Auto-repair settings
    repair_nan_values: bool = True
    repair_infinite_values: bool = True
    repair_negative_correlations: bool = False  # Conservative default
    repair_scaling_issues: bool = True
    repair_format_inconsistencies: bool = True

    # Performance settings
    cache_validation_results: bool = True
    incremental_validation: bool = True
    parallel_validation: bool = False

    # Reporting settings
    generate_reports: bool = True
    export_detailed_reports: bool = False
    save_quality_history: bool = True

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "QualityControlConfig":
        """Create configuration from dictionary."""
        quality_config = config.get("quality_control", {})

        return cls(
            enabled=quality_config.get("enabled", True),
            validation_level=quality_config.get("validation_level", "standard"),
            auto_repair=quality_config.get("auto_repair", "conservative"),
            pass_threshold=quality_config.get("pass_threshold", 50.0),
            warn_threshold=quality_config.get("warn_threshold", 70.0),
            excellent_threshold=quality_config.get("excellent_threshold", 85.0),
            enable_raw_validation=quality_config.get("enable_raw_validation", True),
            enable_filtering_validation=quality_config.get(
                "enable_filtering_validation",
                True,
            ),
            enable_preprocessing_validation=quality_config.get(
                "enable_preprocessing_validation",
                True,
            ),
            enable_final_validation=quality_config.get("enable_final_validation", True),
            repair_nan_values=quality_config.get("repair_nan_values", True),
            repair_infinite_values=quality_config.get("repair_infinite_values", True),
            repair_negative_correlations=quality_config.get(
                "repair_negative_correlations",
                False,
            ),
            repair_scaling_issues=quality_config.get("repair_scaling_issues", True),
            repair_format_inconsistencies=quality_config.get(
                "repair_format_inconsistencies",
                True,
            ),
            cache_validation_results=quality_config.get(
                "cache_validation_results",
                True,
            ),
            incremental_validation=quality_config.get("incremental_validation", True),
            parallel_validation=quality_config.get("parallel_validation", False),
            generate_reports=quality_config.get("reporting", {}).get(
                "generate_reports",
                quality_config.get("generate_reports", True),
            ),
            export_detailed_reports=quality_config.get("reporting", {}).get(
                "export_detailed_reports",
                quality_config.get("export_detailed_reports", False),
            ),
            save_quality_history=quality_config.get("reporting", {}).get(
                "save_quality_history",
                quality_config.get("save_quality_history", True),
            ),
        )


class DataQualityController:
    """Main quality control orchestrator for XPCS data loading pipeline.

    Provides comprehensive data quality control with progressive validation,
    auto-repair capabilities, and integration with existing filtering and
    preprocessing systems.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize data quality controller.

        Args:
            config: Full configuration dictionary including quality_control section
        """
        self.config = config
        self.quality_config = QualityControlConfig.from_config_dict(config)

        # Initialize validation cache
        self._validation_cache: dict[str, QualityControlResult] = {}
        self._quality_history: list[QualityControlResult] = []

        # Performance tracking
        self._stage_timings: dict[str, float] = {}

        logger.info(
            f"DataQualityController initialized with validation_level='{self.quality_config.validation_level}', "
            f"auto_repair='{self.quality_config.auto_repair}'",
        )

    @log_performance(threshold=0.1)
    def validate_data_stage(
        self,
        data: dict[str, Any],
        stage: QualityControlStage,
        previous_result: QualityControlResult | None = None,
    ) -> QualityControlResult:
        """Validate data at specific pipeline stage with progressive quality control.

        Args:
            data: Data dictionary to validate
            stage: Current pipeline stage
            previous_result: Previous stage validation result for comparison

        Returns:
            Quality control result with metrics, issues, and recommendations
        """
        if not self.quality_config.enabled:
            logger.debug("Quality control disabled - creating minimal result")
            return self._create_minimal_result(stage, data)

        start_time = time.time()
        logger.info(f"Starting quality validation for stage: {stage.value}")

        # Check if validation is enabled for this stage
        if not self._is_stage_enabled(stage):
            logger.debug(f"Validation disabled for stage {stage.value}")
            return self._create_minimal_result(stage, data)

        # Check cache if incremental validation is enabled
        if self.quality_config.incremental_validation and previous_result:
            cached_result = self._check_incremental_cache(data, stage, previous_result)
            if cached_result:
                logger.debug(f"Using cached validation result for stage {stage.value}")
                return cached_result

        # Initialize result
        result = QualityControlResult(
            stage=stage,
            passed=True,
            metrics=QualityMetrics(),
            data_shape_before=self._get_data_shape(data),
        )

        try:
            # Stage-specific validation
            if stage == QualityControlStage.RAW_DATA:
                self._validate_raw_data(data, result)
            elif stage == QualityControlStage.FILTERED_DATA:
                self._validate_filtered_data(data, result, previous_result)
            elif stage == QualityControlStage.PREPROCESSED_DATA:
                self._validate_preprocessed_data(data, result, previous_result)
            elif stage == QualityControlStage.FINAL_DATA:
                self._validate_final_data(data, result, previous_result)

            # Apply auto-repair if enabled and issues found
            if self.quality_config.auto_repair != "disabled" and result.issues:
                data_modified = self._apply_auto_repair(data, result)
                if data_modified:
                    result.data_modified = True
                    result.data_shape_after = self._get_data_shape(data)
                    # Re-validate after repair
                    self._revalidate_after_repair(data, result)

            # Compute overall quality score
            result.metrics.overall_score = self._compute_overall_quality_score(result)

            # Determine pass/fail status
            result.passed = (
                result.metrics.overall_score >= self.quality_config.pass_threshold
            )

            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)

            # Record processing time
            result.processing_time = time.time() - start_time

            # Cache result if enabled
            if self.quality_config.cache_validation_results:
                self._cache_result(data, result)

            # Add to quality history
            if self.quality_config.save_quality_history:
                self._quality_history.append(result)

            logger.info(
                f"Quality validation completed for {stage.value}: "
                f"score={result.metrics.overall_score:.1f}, "
                f"passed={result.passed}, "
                f"issues={len(result.issues)}, "
                f"repairs={len(result.repairs_applied)}",
            )

            return result

        except Exception as e:
            logger.error(f"Quality validation failed for stage {stage.value}: {e}")
            result.passed = False
            result.issues.append(
                ValidationIssue(
                    severity="error",
                    category="validation",
                    message=f"Quality validation failed: {str(e)}",
                    recommendation="Check data format and configuration",
                ),
            )
            result.processing_time = time.time() - start_time
            return result

    def _is_stage_enabled(self, stage: QualityControlStage) -> bool:
        """Check if validation is enabled for specific stage."""
        if stage == QualityControlStage.RAW_DATA:
            return self.quality_config.enable_raw_validation
        elif stage == QualityControlStage.FILTERED_DATA:
            return self.quality_config.enable_filtering_validation
        elif stage == QualityControlStage.PREPROCESSED_DATA:
            return self.quality_config.enable_preprocessing_validation
        elif stage == QualityControlStage.FINAL_DATA:
            return self.quality_config.enable_final_validation
        return True  # type: ignore[unreachable]

    def _create_minimal_result(
        self,
        stage: QualityControlStage,
        data: dict[str, Any],
    ) -> QualityControlResult:
        """Create minimal validation result when validation is disabled."""
        return QualityControlResult(
            stage=stage,
            passed=True,
            metrics=QualityMetrics(overall_score=100.0),
            data_shape_before=self._get_data_shape(data),
        )

    def _get_data_shape(self, data: dict[str, Any]) -> tuple[Any, ...]:
        """Get shape summary of data for tracking changes."""
        try:
            c2_exp = data.get("c2_exp", [])
            if hasattr(c2_exp, "shape"):
                return c2_exp.shape  # type: ignore[no-any-return]
            elif isinstance(c2_exp, (list, tuple)) and len(c2_exp) > 0:
                return (len(c2_exp), getattr(c2_exp[0], "shape", "unknown"))
            return ("unknown",)
        except (AttributeError, TypeError, IndexError):
            return ("unknown",)

    @log_performance(threshold=0.05)
    def _validate_raw_data(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Validate raw data integrity and basic format."""
        logger.debug("Validating raw data stage")

        # Use existing validation system if available
        if HAS_VALIDATION:
            validation_level = (
                "basic"
                if self.quality_config.validation_level in ["none", "basic"]
                else "full"
            )
            validation_report = validate_xpcs_data(data, self.config, validation_level)

            # Convert validation report to our format
            result.issues.extend(validation_report.errors)
            result.issues.extend(validation_report.warnings)
            result.issues.extend(validation_report.info)

            # Extract metrics from validation report
            if validation_report.data_statistics:
                self._extract_metrics_from_validation(validation_report, result.metrics)
        else:
            # Fallback basic validation
            self._basic_raw_data_validation(data, result)

        logger.debug(
            f"Raw data validation completed: {len(result.issues)} issues found",
        )

    def _validate_filtered_data(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
        previous_result: QualityControlResult | None,
    ) -> None:
        """Validate that filtering didn't remove too much data or corrupt quality."""
        logger.debug("Validating filtered data stage")

        if previous_result and previous_result.data_shape_before:
            # Compare data sizes
            current_shape = self._get_data_shape(data)
            previous_shape = previous_result.data_shape_before

            if isinstance(current_shape, tuple) and isinstance(previous_shape, tuple):
                if len(current_shape) > 0 and len(previous_shape) > 0:
                    try:
                        current_size = (
                            current_shape[0] if isinstance(current_shape[0], int) else 1
                        )
                        previous_size = (
                            previous_shape[0]
                            if isinstance(previous_shape[0], int)
                            else 1
                        )

                        if previous_size > 0:
                            retention_fraction = current_size / previous_size
                            result.metrics.filtering_efficiency = (
                                retention_fraction * 100
                            )

                            if retention_fraction < 0.1:  # Less than 10% data retained
                                result.issues.append(
                                    ValidationIssue(
                                        severity="warning",
                                        category="data_quality",
                                        message=f"Filtering removed {(1 - retention_fraction) * 100:.1f}% of data",
                                        recommendation="Check filtering criteria - may be too restrictive",
                                    ),
                                )
                            elif retention_fraction > 0.95:  # More than 95% retained
                                result.issues.append(
                                    ValidationIssue(
                                        severity="info",
                                        category="data_quality",
                                        message=f"Filtering retained {retention_fraction * 100:.1f}% of data",
                                        recommendation="Filtering may not be necessary with current settings",
                                    ),
                                )
                    except (AttributeError, TypeError, IndexError):
                        logger.warning(
                            "Could not compare data sizes before/after filtering",
                        )

        # Validate filtered data quality
        self._basic_data_quality_checks(data, result)

        logger.debug(
            f"Filtered data validation completed: filtering_efficiency={result.metrics.filtering_efficiency:.1f}%",
        )

    def _validate_preprocessed_data(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
        previous_result: QualityControlResult | None,
    ) -> None:
        """Validate that preprocessing preserved physics and improved quality."""
        logger.debug("Validating preprocessed data stage")

        # Check that preprocessing didn't introduce artifacts
        c2_exp = data.get("c2_exp", [])
        if hasattr(c2_exp, "shape") or isinstance(c2_exp, (list, tuple)):
            # Check for processing artifacts
            result.metrics.preprocessing_success = self._check_preprocessing_artifacts(
                c2_exp,
            )

            if not result.metrics.preprocessing_success:
                result.issues.append(
                    ValidationIssue(
                        severity="error",
                        category="preprocessing",
                        message="Preprocessing introduced artifacts or corrupted data",
                        recommendation="Review preprocessing settings and validate input data",
                    ),
                )

        # Compare with previous stage if available
        if previous_result:
            result.metrics.transformation_fidelity = (
                self._compute_transformation_fidelity(data, previous_result)
            )

            if result.metrics.transformation_fidelity < 0.8:  # Less than 80% fidelity
                result.issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="preprocessing",
                        message=f"Preprocessing fidelity low: {result.metrics.transformation_fidelity:.2f}",
                        recommendation="Check preprocessing parameters for excessive modification",
                    ),
                )

        # Advanced quality checks
        self._advanced_data_quality_checks(data, result)

        logger.debug(
            f"Preprocessed data validation completed: fidelity={result.metrics.transformation_fidelity:.2f}",
        )

    def _validate_final_data(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
        previous_result: QualityControlResult | None,
    ) -> None:
        """Comprehensive validation for analysis-ready data."""
        logger.debug("Validating final data stage")

        # Comprehensive data quality assessment
        self._comprehensive_quality_assessment(data, result)

        # Physics validation if enabled
        validation_level = self.quality_config.validation_level
        if validation_level in ["standard", "comprehensive"] and HAS_VALIDATION:
            validation_report = validate_xpcs_data(data, self.config, "full")

            # Merge validation results
            result.issues.extend(
                [
                    issue
                    for issue in validation_report.errors + validation_report.warnings
                    if not any(
                        existing.message == issue.message for existing in result.issues
                    )
                ],
            )

            # Update metrics with physics validation
            if validation_report.physics_checks:
                result.metrics.q_range_validity = (
                    100.0
                    if validation_report.physics_checks.get("q_range_valid", False)
                    else 50.0
                )

        # Analysis readiness check
        readiness_score = self._assess_analysis_readiness(data, result)
        result.metrics.overall_score = max(
            result.metrics.overall_score,
            readiness_score,
        )

        logger.debug(
            f"Final data validation completed: readiness_score={readiness_score:.1f}",
        )

    def _basic_raw_data_validation(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Basic data validation fallback when full validation system unavailable."""
        required_keys = ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]

        for key in required_keys:
            if key not in data:
                result.issues.append(
                    ValidationIssue(
                        severity="error",
                        category="format",
                        message=f"Missing required data key: {key}",
                        recommendation="Check data loading process",
                    ),
                )

        # Basic data integrity
        for key, value in data.items():
            if hasattr(value, "shape") or isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value)
                    finite_fraction = (
                        np.sum(np.isfinite(arr)) / arr.size if arr.size > 0 else 0.0
                    )
                    result.metrics.finite_fraction = max(
                        result.metrics.finite_fraction,
                        finite_fraction,
                    )

                    if finite_fraction < 0.95:
                        result.issues.append(
                            ValidationIssue(
                                severity=(
                                    "warning" if finite_fraction > 0.8 else "error"
                                ),
                                category="data_quality",
                                message=f"Non-finite values in {key}: {(1 - finite_fraction) * 100:.1f}%",
                                recommendation="Check data preprocessing and source quality",
                            ),
                        )
                except (AttributeError, TypeError, IndexError):
                    pass

    def _basic_data_quality_checks(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Basic data quality checks for correlation matrices."""
        c2_exp = data.get("c2_exp", [])

        if hasattr(c2_exp, "shape") or isinstance(c2_exp, (list, tuple)):
            try:
                arr = np.asarray(c2_exp)

                # Check correlation validity
                if arr.size > 0:
                    # Check for reasonable correlation values
                    mean_val = np.nanmean(arr)
                    if 0.5 <= mean_val <= 3.0:
                        result.metrics.correlation_validity = 85.0
                    elif 0.1 <= mean_val <= 5.0:
                        result.metrics.correlation_validity = 60.0
                    else:
                        result.metrics.correlation_validity = 30.0
                        result.issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="data_quality",
                                message=f"Unusual correlation values: mean={mean_val:.3f}",
                                recommendation="Check data normalization and calibration",
                            ),
                        )

                    # Signal-to-noise estimation
                    std_val = np.nanstd(arr)
                    if mean_val > 0:
                        snr = mean_val / std_val if std_val > 0 else 100.0
                        result.metrics.signal_to_noise = min(
                            snr * 10,
                            100.0,
                        )  # Scale to 0-100
            except (AttributeError, TypeError, IndexError):
                logger.warning("Could not perform basic data quality checks")

    def _advanced_data_quality_checks(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Advanced quality checks including symmetry and decay analysis."""
        c2_exp = data.get("c2_exp", [])

        try:
            if isinstance(c2_exp, (list, tuple, np.ndarray)) and len(c2_exp) > 0:
                matrices = [np.asarray(matrix) for matrix in c2_exp]

                # Symmetry analysis for correlation matrices
                symmetry_scores = []
                for matrix in matrices:
                    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                        symmetry_error = np.nanmean(np.abs(matrix - matrix.T))
                        max_val = np.nanmax(np.abs(matrix))
                        if max_val > 0:
                            symmetry_score = max(
                                0,
                                100 * (1 - symmetry_error / max_val),
                            )
                            symmetry_scores.append(symmetry_score)

                if symmetry_scores:
                    result.metrics.symmetry_score = np.mean(symmetry_scores)

                    if result.metrics.symmetry_score < 80:
                        result.issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="data_quality",
                                message=f"Poor matrix symmetry: {result.metrics.symmetry_score:.1f}%",
                                recommendation="Check correlation matrix reconstruction",
                            ),
                        )

                # Correlation decay analysis
                decay_rates = []
                for matrix in matrices:
                    if matrix.ndim == 2:
                        diag = np.diag(matrix)
                        if len(diag) > 10 and diag[0] > 0:
                            decay_rate = (
                                diag[0] - diag[min(10, len(diag) - 1)]
                            ) / diag[0]
                            if 0 <= decay_rate <= 1:
                                decay_rates.append(decay_rate)

                if decay_rates:
                    result.metrics.correlation_decay = np.mean(decay_rates) * 100
        except (AttributeError, TypeError, IndexError):
            logger.warning("Could not perform advanced data quality checks")

    def _comprehensive_quality_assessment(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Comprehensive quality assessment for final data."""
        # Combine all previous checks
        self._basic_data_quality_checks(data, result)
        self._advanced_data_quality_checks(data, result)

        # Additional comprehensive checks
        try:
            # Data completeness check
            required_keys = [
                "wavevector_q_list",
                "phi_angles_list",
                "t1",
                "t2",
                "c2_exp",
            ]
            completeness = (
                sum(1 for key in required_keys if key in data)
                / len(required_keys)
                * 100
            )

            if completeness < 100:
                result.issues.append(
                    ValidationIssue(
                        severity="error",
                        category="completeness",
                        message=f"Data completeness: {completeness:.0f}%",
                        recommendation="Ensure all required data components are present",
                    ),
                )

            # Consistency checks
            self._check_data_consistency(data, result)

        except Exception as e:
            logger.warning(f"Comprehensive quality assessment failed: {e}")

    def _check_data_consistency(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Check consistency between different data components."""
        try:
            c2_exp = data.get("c2_exp", [])
            t1 = data.get("t1", [])
            t2 = data.get("t2", [])

            if (
                hasattr(c2_exp, "shape")
                and hasattr(t1, "shape")
                and hasattr(t2, "shape")
            ):
                c2_shape = c2_exp.shape
                t1_shape = t1.shape
                t2_shape = t2.shape

                # Check time-correlation consistency
                if len(c2_shape) >= 2:
                    matrix_size = c2_shape[-1]
                    if len(t1_shape) > 0 and t1_shape[-1] != matrix_size:
                        result.issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="consistency",
                                message=f"Time array size {t1_shape[-1]} doesn't match matrix size {matrix_size}",
                                recommendation="Check time array generation",
                            ),
                        )

                result.metrics.time_consistency = t1_shape == t2_shape and (
                    len(c2_shape) == 0 or t1_shape[-1] == c2_shape[-1]
                )
        except (AttributeError, TypeError, IndexError):
            logger.warning("Could not perform data consistency checks")

    def _check_preprocessing_artifacts(self, c2_exp: Any) -> bool:
        """Check for preprocessing artifacts."""
        try:
            arr = np.asarray(c2_exp)

            # Check for NaN/Inf introduction
            if not np.all(np.isfinite(arr)):
                return False

            # Check for unrealistic value ranges
            if arr.size > 0:
                min_val, max_val = np.nanmin(arr), np.nanmax(arr)
                if min_val < -10 or max_val > 100:  # Unrealistic correlation values
                    return False

            return True
        except (AttributeError, TypeError, IndexError):
            return False

    def _compute_transformation_fidelity(
        self,
        current_data: dict[str, Any],
        previous_result: QualityControlResult,
    ) -> float:
        """Compute fidelity of data transformation."""
        try:
            # Simple fidelity measure based on data statistics preservation
            current_c2 = current_data.get("c2_exp", [])
            if hasattr(current_c2, "shape") and hasattr(current_c2, "size"):
                current_arr = np.asarray(current_c2)

                # Use previous quality score as baseline
                if previous_result.metrics.overall_score > 0:
                    # Higher fidelity if current quality is maintained
                    current_finite_fraction = (
                        np.sum(np.isfinite(current_arr)) / current_arr.size
                        if current_arr.size > 0
                        else 0.0
                    )

                    # Simple fidelity measure
                    return min(
                        1.0,
                        current_finite_fraction
                        / max(0.1, previous_result.metrics.finite_fraction),
                    )

            return 0.8  # Default reasonable fidelity
        except (AttributeError, TypeError, IndexError):
            return 0.5  # Conservative default

    def _assess_analysis_readiness(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> float:
        """Assess overall readiness for analysis."""
        readiness_factors = []

        # Data completeness (weight: 0.3)
        required_keys = ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]
        completeness = sum(1 for key in required_keys if key in data) / len(
            required_keys,
        )
        readiness_factors.append((completeness * 100, 0.3))

        # Data quality (weight: 0.4)
        quality_score = (
            result.metrics.finite_fraction * 100
            + result.metrics.correlation_validity
            + result.metrics.signal_to_noise
        ) / 3
        readiness_factors.append((quality_score, 0.4))

        # Physics validity (weight: 0.2)
        physics_score = (
            result.metrics.q_range_validity
            + (100 if result.metrics.time_consistency else 0)
        ) / 2
        readiness_factors.append((physics_score, 0.2))

        # Processing success (weight: 0.1)
        processing_score = (
            (100 if result.metrics.preprocessing_success else 0)
            + result.metrics.filtering_efficiency
        ) / 2
        readiness_factors.append((processing_score, 0.1))

        # Weighted average
        total_score = sum(score * weight for score, weight in readiness_factors)
        total_weight = sum(weight for _, weight in readiness_factors)

        return total_score / total_weight if total_weight > 0 else 0.0

    @log_performance(threshold=0.1)
    def _apply_auto_repair(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> bool:
        """Apply automatic data repair based on detected issues."""
        if self.quality_config.auto_repair == "disabled":
            return False

        logger.info(
            f"Applying auto-repair with strategy: {self.quality_config.auto_repair}",
        )
        data_modified = False
        repairs_applied: list[str] = []

        try:
            # Repair NaN values
            if self.quality_config.repair_nan_values:
                modified = self._repair_nan_values(data, repairs_applied)
                data_modified = data_modified or modified

            # Repair infinite values
            if self.quality_config.repair_infinite_values:
                modified = self._repair_infinite_values(data, repairs_applied)
                data_modified = data_modified or modified

            # Repair negative correlations (conservative)
            if (
                self.quality_config.repair_negative_correlations
                and self.quality_config.auto_repair == "aggressive"
            ):
                modified = self._repair_negative_correlations(data, repairs_applied)
                data_modified = data_modified or modified

            # Repair scaling issues — aggressive mode only.
            # The heuristic (mean > 100 → ÷100) would silently corrupt raw-count
            # matrices where values > 100 are physically valid.
            if (
                self.quality_config.repair_scaling_issues
                and self.quality_config.auto_repair == "aggressive"
            ):
                modified = self._repair_scaling_issues(data, repairs_applied)
                data_modified = data_modified or modified

            # Update repair metrics
            result.repairs_applied = repairs_applied
            result.metrics.issues_detected = len(
                [
                    issue
                    for issue in result.issues
                    if issue.severity in ["error", "warning"]
                ],
            )
            result.metrics.issues_repaired = len(repairs_applied)

            if result.metrics.issues_detected > 0:
                result.metrics.repair_success_rate = (
                    result.metrics.issues_repaired
                    / result.metrics.issues_detected
                    * 100
                )

            if repairs_applied:
                logger.info(
                    f"Auto-repair completed: {len(repairs_applied)} repairs applied",
                )

        except Exception as e:
            logger.error(f"Auto-repair failed: {e}")

        return data_modified

    def _repair_nan_values(
        self,
        data: dict[str, Any],
        repairs_applied: list[str],
    ) -> bool:
        """Repair NaN values in data."""
        data_modified = False

        for key, value in data.items():
            if hasattr(value, "shape") or isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value)
                    nan_mask = ~np.isfinite(arr)

                    if np.any(nan_mask):
                        if key == "c2_exp" and arr.ndim >= 2:
                            # For correlation matrices, interpolate from neighbors
                            for i in range(len(arr)):
                                matrix = arr[i] if arr.ndim > 2 else arr
                                if np.any(~np.isfinite(matrix)):
                                    # Simple interpolation from finite neighbors
                                    finite_values = matrix[np.isfinite(matrix)]
                                    if len(finite_values) > 0:
                                        replacement_value = np.median(finite_values)
                                        matrix[~np.isfinite(matrix)] = replacement_value
                                        data_modified = True
                            # Write repaired array back; np.asarray() of a list/JAX
                            # array creates a detached copy, so the loop above mutated
                            # `arr` only — we must propagate it back to `data`.
                            data[key] = arr
                        else:
                            # For other arrays, use median replacement
                            finite_values = arr[np.isfinite(arr)]
                            if len(finite_values) > 0:
                                replacement_value = np.median(finite_values)
                                arr[nan_mask] = replacement_value
                                data[key] = arr
                                data_modified = True

                        if data_modified:
                            repairs_applied.append(f"Repaired NaN values in {key}")
                except (AttributeError, TypeError, IndexError):
                    pass

        return data_modified

    def _repair_infinite_values(
        self,
        data: dict[str, Any],
        repairs_applied: list[str],
    ) -> bool:
        """Repair infinite values in data."""
        data_modified = False

        for key, value in data.items():
            if hasattr(value, "shape") or isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value)
                    inf_mask = np.isinf(arr)

                    if np.any(inf_mask):
                        finite_values = arr[np.isfinite(arr)]
                        if len(finite_values) > 0:
                            # Replace with max/min of finite values
                            pos_inf_mask = np.isposinf(arr)
                            neg_inf_mask = np.isneginf(arr)

                            if np.any(pos_inf_mask):
                                arr[pos_inf_mask] = np.max(finite_values)
                            if np.any(neg_inf_mask):
                                arr[neg_inf_mask] = np.min(finite_values)

                            data[key] = arr
                            data_modified = True
                            repairs_applied.append(f"Repaired infinite values in {key}")
                except (AttributeError, TypeError, IndexError):
                    pass

        return data_modified

    def _repair_negative_correlations(
        self,
        data: dict[str, Any],
        repairs_applied: list[str],
    ) -> bool:
        """Repair negative correlation values (aggressive mode only)."""
        data_modified = False

        c2_exp = data.get("c2_exp")
        if c2_exp is not None:
            try:
                arr = np.asarray(c2_exp)
                negative_mask = arr < 0

                if np.any(negative_mask):
                    # Simple approach: set negatives to small positive value
                    arr[negative_mask] = 1e-6
                    data["c2_exp"] = arr
                    data_modified = True
                    repairs_applied.append("Repaired negative correlation values")
            except (AttributeError, TypeError, IndexError):
                pass

        return data_modified

    def _repair_scaling_issues(
        self,
        data: dict[str, Any],
        repairs_applied: list[str],
    ) -> bool:
        """Repair obvious scaling issues in data."""
        data_modified = False

        # Check correlation values for unrealistic scales
        c2_exp = data.get("c2_exp")
        if c2_exp is not None:
            try:
                arr = np.asarray(c2_exp)
                if arr.size > 0:
                    mean_val = np.nanmean(arr)

                    # If correlations are way off scale, apply simple rescaling
                    if mean_val > 100:  # Likely scaled by 100x
                        arr = arr / 100
                        data["c2_exp"] = arr
                        data_modified = True
                        repairs_applied.append("Applied correlation rescaling (÷100)")
                    elif mean_val > 10:  # Likely scaled by 10x
                        arr = arr / 10
                        data["c2_exp"] = arr
                        data_modified = True
                        repairs_applied.append("Applied correlation rescaling (÷10)")
                    elif mean_val < 0.01 and mean_val > 0:  # Likely under-scaled
                        arr = arr * 10
                        data["c2_exp"] = arr
                        data_modified = True
                        repairs_applied.append("Applied correlation rescaling (×10)")
            except (AttributeError, TypeError, IndexError):
                pass

        return data_modified

    def _revalidate_after_repair(
        self,
        data: dict[str, Any],
        result: QualityControlResult,
    ) -> None:
        """Re-validate data after applying repairs."""
        logger.debug("Re-validating data after auto-repair")

        # Update metrics after repair
        self._basic_data_quality_checks(data, result)

        # Remove issues that were fixed
        remaining_issues = []
        for issue in result.issues:
            if not self._issue_was_repaired(issue, result.repairs_applied):
                remaining_issues.append(issue)

        result.issues = remaining_issues

    def _issue_was_repaired(
        self,
        issue: ValidationIssue,
        repairs_applied: list[str],
    ) -> bool:
        """Check if an issue was addressed by repairs."""
        issue_keywords = {
            "non-finite": ["NaN", "infinite"],
            "negative": ["negative"],
            "scaling": ["rescaling", "scaling"],
        }

        for repair in repairs_applied:
            for _issue_type, keywords in issue_keywords.items():
                if any(keyword.lower() in repair.lower() for keyword in keywords):
                    if any(
                        keyword.lower() in issue.message.lower() for keyword in keywords
                    ):
                        return True

        return False

    def _compute_overall_quality_score(self, result: QualityControlResult) -> float:
        """Compute overall quality score from individual metrics."""
        # Weighted scoring system
        score_components = [
            (result.metrics.finite_fraction * 100, 0.2),  # Data integrity
            (result.metrics.correlation_validity, 0.25),  # Correlation validity
            (min(result.metrics.signal_to_noise, 100), 0.2),  # Signal quality
            (result.metrics.symmetry_score, 0.15),  # Matrix symmetry
            (result.metrics.q_range_validity, 0.1),  # Physics validity
            (100 if result.metrics.time_consistency else 50, 0.1),  # Time consistency
        ]

        # Apply issue penalties
        error_penalty = (
            len([issue for issue in result.issues if issue.severity == "error"]) * 10
        )
        warning_penalty = (
            len([issue for issue in result.issues if issue.severity == "warning"]) * 5
        )

        # Weighted score
        weighted_score = sum(score * weight for score, weight in score_components)
        total_weight = sum(weight for _, weight in score_components)
        base_score = weighted_score / total_weight if total_weight > 0 else 0

        # Apply penalties and repair bonus
        final_score = base_score - error_penalty - warning_penalty

        # Bonus for successful repairs
        if result.metrics.repair_success_rate > 0:
            repair_bonus = result.metrics.repair_success_rate * 0.1
            final_score += repair_bonus

        return max(0.0, min(100.0, final_score))

    def _generate_recommendations(self, result: QualityControlResult) -> list[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []

        # Score-based recommendations
        if result.metrics.overall_score < self.quality_config.pass_threshold:
            recommendations.append(
                "Data quality below acceptable threshold - review preprocessing settings",
            )

        if result.metrics.finite_fraction < 0.95:
            recommendations.append(
                "Consider additional data cleaning to remove non-finite values",
            )

        if result.metrics.correlation_validity < 70:
            recommendations.append(
                "Check correlation calculation and normalization procedures",
            )

        if result.metrics.signal_to_noise < 30:
            recommendations.append(
                "Consider noise reduction preprocessing or longer acquisition times",
            )

        if result.metrics.symmetry_score < 80:
            recommendations.append("Review correlation matrix reconstruction method")

        if result.metrics.filtering_efficiency < 50:
            recommendations.append("Filtering may be too restrictive - review criteria")

        if result.metrics.filtering_efficiency > 95:
            recommendations.append(
                "Filtering criteria may be too permissive - consider tightening",
            )

        # Stage-specific recommendations
        if result.stage == QualityControlStage.FINAL_DATA:
            if result.metrics.overall_score >= self.quality_config.excellent_threshold:
                recommendations.append("Excellent data quality - ready for analysis")
            elif result.metrics.overall_score >= self.quality_config.warn_threshold:
                recommendations.append("Good data quality - proceed with analysis")
            elif result.metrics.overall_score >= self.quality_config.pass_threshold:
                recommendations.append(
                    "Acceptable data quality - consider additional preprocessing",
                )

        return recommendations

    def _check_incremental_cache(
        self,
        data: dict[str, Any],
        stage: QualityControlStage,
        previous_result: QualityControlResult,
    ) -> QualityControlResult | None:
        """Check if incremental validation can use cached results."""
        # Simple cache key based on data shape and stage
        data_shape = self._get_data_shape(data)
        cache_key = f"{stage.value}_{hash(str(data_shape))}"

        if cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            # Check if cache is still valid (simple heuristic)
            if cached_result.data_shape_before == data_shape:
                return cached_result

        return None

    def _cache_result(self, data: dict[str, Any], result: QualityControlResult) -> None:
        """Cache validation result for future incremental validation."""
        data_shape = self._get_data_shape(data)
        cache_key = f"{result.stage.value}_{hash(str(data_shape))}"
        self._validation_cache[cache_key] = result

        # Limit cache size
        if len(self._validation_cache) > 100:
            # Remove oldest entries
            keys_to_remove = list(self._validation_cache.keys())[:20]
            for key in keys_to_remove:
                del self._validation_cache[key]

    def _extract_metrics_from_validation(
        self,
        validation_report: Any,
        metrics: QualityMetrics,
    ) -> None:
        """Extract metrics from existing validation system report."""
        if hasattr(validation_report, "data_statistics"):
            stats = validation_report.data_statistics
            for _key, stat in stats.items():
                if isinstance(stat, dict) and "finite_fraction" in stat:
                    metrics.finite_fraction = max(
                        metrics.finite_fraction,
                        stat["finite_fraction"],
                    )

        if hasattr(validation_report, "quality_score"):
            # Use existing quality score as baseline
            existing_score = (
                validation_report.quality_score * 100
            )  # Convert to 0-100 scale
            metrics.overall_score = max(metrics.overall_score, existing_score)

    @log_performance(threshold=0.05)
    def generate_quality_report(
        self,
        results: list[QualityControlResult],
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive quality assessment report.

        Args:
            results: List of quality control results from all stages
            output_path: Optional path to save report

        Returns:
            Comprehensive quality report dictionary
        """
        logger.info("Generating comprehensive quality assessment report")

        report: dict[str, Any] = {
            "metadata": {
                "report_timestamp": time.time(),
                "report_version": "1.0.0",
                "quality_controller_config": {
                    "validation_level": self.quality_config.validation_level,
                    "auto_repair": self.quality_config.auto_repair,
                    "thresholds": {
                        "pass": self.quality_config.pass_threshold,
                        "warn": self.quality_config.warn_threshold,
                        "excellent": self.quality_config.excellent_threshold,
                    },
                },
            },
            "overall_summary": self._generate_overall_summary(results),
            "stage_results": {},
            "quality_evolution": self._analyze_quality_evolution(results),
            "recommendations": self._generate_final_recommendations(results),
            "detailed_metrics": {},
        }

        # Stage-specific results
        stage_results_dict: dict[str, Any] = {}
        for result in results:
            stage_name = result.stage.value
            stage_results_dict[stage_name] = {
                "summary": result.get_summary(),
                "metrics": result.metrics.to_dict(),
                "issues": [
                    {
                        "severity": issue.severity,
                        "category": issue.category,
                        "message": issue.message,
                        "recommendation": issue.recommendation,
                    }
                    for issue in result.issues
                ],
                "repairs_applied": result.repairs_applied,
                "recommendations": result.recommendations,
            }

            report["detailed_metrics"][stage_name] = result.metrics.to_dict()

        report["stage_results"] = stage_results_dict

        # Save report if path provided
        if output_path and self.quality_config.export_detailed_reports:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Quality report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save quality report: {e}")

        return report

    def _generate_overall_summary(
        self,
        results: list[QualityControlResult],
    ) -> dict[str, Any]:
        """Generate overall quality summary from all stages."""
        if not results:
            return {"status": "no_data", "overall_score": 0.0}

        final_result = results[-1] if results else None

        # Overall status determination
        if final_result:
            if (
                final_result.metrics.overall_score
                >= self.quality_config.excellent_threshold
            ):
                status = "excellent"
            elif (
                final_result.metrics.overall_score >= self.quality_config.warn_threshold
            ):
                status = "good"
            elif (
                final_result.metrics.overall_score >= self.quality_config.pass_threshold
            ):
                status = "acceptable"
            else:
                status = "poor"
        else:
            status = "unknown"

        # Aggregate statistics
        total_issues = sum(len(result.issues) for result in results)
        total_repairs = sum(len(result.repairs_applied) for result in results)
        avg_processing_time = np.mean([result.processing_time for result in results])

        return {
            "status": status,
            "overall_score": (
                final_result.metrics.overall_score if final_result else 0.0
            ),
            "total_stages_processed": len(results),
            "total_issues_found": total_issues,
            "total_repairs_applied": total_repairs,
            "average_processing_time": avg_processing_time,
            "data_modified": any(result.data_modified for result in results),
            "all_stages_passed": all(result.passed for result in results),
        }

    def _analyze_quality_evolution(
        self,
        results: list[QualityControlResult],
    ) -> dict[str, Any]:
        """Analyze how quality evolved through the processing pipeline."""
        if len(results) < 2:
            return {"evolution": "insufficient_data"}

        scores = [result.metrics.overall_score for result in results]
        stages = [result.stage.value for result in results]

        evolution_analysis: dict[str, Any] = {
            "score_progression": dict(zip(stages, scores, strict=False)),
            "quality_trend": "improving" if scores[-1] > scores[0] else "declining",
            "max_improvement": max(scores) - min(scores),
            "final_vs_initial": scores[-1] - scores[0],
        }

        # Identify quality bottlenecks
        bottlenecks_list: list[dict[str, Any]] = []
        for _i, result in enumerate(results):
            if result.metrics.overall_score < self.quality_config.pass_threshold:
                bottlenecks_list.append(
                    {
                        "stage": result.stage.value,
                        "score": result.metrics.overall_score,
                        "main_issues": [issue.category for issue in result.issues[:3]],
                    },
                )

        evolution_analysis["bottlenecks"] = bottlenecks_list

        return evolution_analysis

    def _generate_final_recommendations(
        self,
        results: list[QualityControlResult],
    ) -> list[str]:
        """Generate final recommendations based on all stage results."""
        recommendations = []

        if not results:
            return ["No quality control results available"]

        final_result = results[-1]

        # Overall quality recommendations
        if (
            final_result.metrics.overall_score
            >= self.quality_config.excellent_threshold
        ):
            recommendations.append(
                "Excellent data quality achieved - ready for analysis",
            )
        elif final_result.metrics.overall_score >= self.quality_config.warn_threshold:
            recommendations.append("Good data quality - proceed with confidence")
        elif final_result.metrics.overall_score >= self.quality_config.pass_threshold:
            recommendations.append(
                "Acceptable data quality - monitor results carefully",
            )
        else:
            recommendations.append(
                "Poor data quality - consider reprocessing or different parameters",
            )

        # Process improvement recommendations
        total_repairs = sum(len(result.repairs_applied) for result in results)
        if total_repairs > 0:
            recommendations.append(
                f"ℹ {total_repairs} automatic repairs were applied - review source data quality",
            )

        # Stage-specific recommendations
        bottleneck_stages = [
            result
            for result in results
            if result.metrics.overall_score < self.quality_config.pass_threshold
        ]
        if bottleneck_stages:
            stage_names = [result.stage.value for result in bottleneck_stages]
            recommendations.append(
                f"Quality issues in stages: {', '.join(stage_names)}",
            )

        return recommendations

    def get_quality_history(self) -> list[dict[str, Any]]:
        """Get quality control history for analysis."""
        return [result.get_summary() for result in self._quality_history]

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
        logger.debug("Validation cache cleared")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self._quality_history:
            return {"message": "No quality control history available"}

        processing_times = [result.processing_time for result in self._quality_history]

        return {
            "total_validations": len(self._quality_history),
            "average_processing_time": np.mean(processing_times),
            "max_processing_time": np.max(processing_times),
            "min_processing_time": np.min(processing_times),
            "cache_size": len(self._validation_cache),
            "stages_processed": len(
                {result.stage.value for result in self._quality_history},
            ),
        }


# Quality control utility functions for easy integration
def create_quality_controller(config: dict[str, Any]) -> DataQualityController:
    """Convenience function to create quality controller from config."""
    return DataQualityController(config)


def validate_data_with_quality_control(
    data: dict[str, Any],
    config: dict[str, Any],
    stage: QualityControlStage = QualityControlStage.FINAL_DATA,
) -> QualityControlResult:
    """Convenience function for single-stage quality validation."""
    controller = DataQualityController(config)
    return controller.validate_data_stage(data, stage)


# Export main classes and functions
__all__ = [
    "DataQualityController",
    "QualityControlStage",
    "QualityLevel",
    "RepairStrategy",
    "QualityMetrics",
    "QualityControlResult",
    "QualityControlConfig",
    "create_quality_controller",
    "validate_data_with_quality_control",
]
