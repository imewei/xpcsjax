r"""Advanced data preprocessing pipeline for the homodyne data layer.

Intelligent data transformation system that builds on config-based filtering to
provide sophisticated preprocessing capabilities for XPCS correlation data.

The pipeline architecture is Load -> Filter -> Transform -> Normalize ->
Validate.

Key Features
------------
- Multi-stage configurable preprocessing pipeline.
- Enhanced diagonal correction with statistical methods.
- Multiple normalization approaches (baseline, statistical, physics-based).
- Noise reduction algorithms (median filtering, gaussian smoothing).
- Data standardization across APS vs APS-U formats.
- Outlier detection and treatment.
- Complete transformation audit trail and reproducibility.
- JAX-accelerated performance with numpy fallback.
- Memory-efficient chunked processing for large datasets.

Notes
-----
Pipeline stages:

1. ``load_raw``: Load raw data (handled by ``xpcs_loader.py``).
2. ``apply_filtering``: Use config-based filtering from ``filtering_utils.py``.
3. ``correct_diagonal``: Enhanced diagonal correction methods.
4. ``normalize_data``: Multiple normalization strategies.
5. ``reduce_noise``: Optional denoising algorithms.
6. ``standardize_format``: Ensure consistent data formats.
7. ``validate_output``: Final data integrity and physics validation.

Performance features:

- In-place operations to minimize memory copying.
- Chunked processing for large correlation matrices.
- Progress reporting for user feedback.
- Intelligent caching of intermediate results.
- JAX JIT compilation for hot paths.
"""

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
    from xpcsjax.core.jax_backend import jax_available

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax_available = False

# Scipy for advanced algorithms
try:
    from scipy import signal, stats
    from scipy.ndimage import gaussian_filter, median_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    ndimage = None
    signal = None
    stats = None
    median_filter = None
    gaussian_filter = None

# V2 logging integration
try:
    from xpcsjax.utils.logging import get_logger, log_calls, log_performance, log_phase

    HAS_V2_LOGGING = True
except ImportError:
    import logging
    from contextlib import contextmanager

    HAS_V2_LOGGING = False

    def get_logger(name):  # type: ignore[no-untyped-def,misc]
        return logging.getLogger(name)

    def log_performance(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator

    def log_calls(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator

    @contextmanager
    def log_phase(name, **kwargs):  # type: ignore[no-untyped-def,misc]
        """Fallback log_phase for environments without v2 logging."""
        yield type("PhaseContext", (), {"duration": 0.0, "memory_peak_gb": None})()


# Diagonal correction from unified module
try:
    from xpcsjax.core.diagonal_correction import apply_diagonal_correction

    HAS_DIAGONAL_CORRECTION = True
except ImportError:
    HAS_DIAGONAL_CORRECTION = False
    apply_diagonal_correction = None  # type: ignore[assignment]


logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing operations fail."""


class PreprocessingConfigurationError(Exception):
    """Raised when preprocessing configuration is invalid."""


class PreprocessingStage(Enum):
    """Enumeration of preprocessing pipeline stages."""

    LOAD_RAW = "load_raw"
    APPLY_FILTERING = "apply_filtering"
    CORRECT_DIAGONAL = "correct_diagonal"
    NORMALIZE_DATA = "normalize_data"
    REDUCE_NOISE = "reduce_noise"
    STANDARDIZE_FORMAT = "standardize_format"
    VALIDATE_OUTPUT = "validate_output"


class NormalizationMethod(Enum):
    """Enumeration of normalization methods."""

    BASELINE = "baseline"  # Normalize by t=0 value
    STATISTICAL = "statistical"  # Z-score normalization
    PHYSICS_BASED = "physics_based"  # Physics-constrained normalization
    MINMAX = "minmax"  # Min-max scaling
    ROBUST = "robust"  # Robust scaling using percentiles


class NoiseReductionMethod(Enum):
    """Enumeration of noise reduction methods."""

    NONE = "none"
    MEDIAN = "median"  # Median filtering
    GAUSSIAN = "gaussian"  # Gaussian smoothing
    WIENER = "wiener"  # Wiener filtering (requires scipy)
    SAVGOL = "savgol"  # Savitzky-Golay filtering


@dataclass
class TransformationRecord:
    """Record of a single transformation applied to data."""

    stage: PreprocessingStage
    method: str
    parameters: dict[str, Any]
    timestamp: float
    duration: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    memory_usage: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingProvenance:
    """Complete provenance record for preprocessing pipeline."""

    pipeline_id: str
    config_hash: str
    transformations: list[TransformationRecord] = field(default_factory=list)
    total_duration: float = 0.0
    peak_memory_usage: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance record to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "config_hash": self.config_hash,
            "transformations": [
                {
                    "stage": t.stage.value,
                    "method": t.method,
                    "parameters": t.parameters,
                    "timestamp": t.timestamp,
                    "duration": t.duration,
                    "input_shape": t.input_shape,
                    "output_shape": t.output_shape,
                    "memory_usage": t.memory_usage,
                    "warnings": t.warnings,
                }
                for t in self.transformations
            ],
            "total_duration": self.total_duration,
            "peak_memory_usage": self.peak_memory_usage,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline execution."""

    data: dict[str, Any]
    provenance: PreprocessingProvenance
    success: bool = True
    stage_results: dict[PreprocessingStage, bool] = field(default_factory=dict)


class PreprocessingPipeline:
    """Advanced data preprocessing pipeline with configurable transformation stages.

    Provides a flexible, high-performance preprocessing system that transforms
    raw XPCS correlation data through multiple configurable stages while maintaining
    complete audit trails and reproducibility.

    Features:
    - Configurable pipeline stages with individual enable/disable
    - Multiple algorithms for each transformation type
    - JAX acceleration with numpy fallback
    - Memory-efficient chunked processing
    - Complete transformation tracking and provenance
    - Error handling with graceful fallbacks
    - Progress reporting for long operations
    """

    @log_calls(include_args=False)
    def __init__(self, config: dict[str, Any]):
        """Initialize the preprocessing pipeline with configuration.

        Parameters
        ----------
        config
            Configuration dictionary containing preprocessing settings.

        Raises
        ------
        PreprocessingConfigurationError
            If the configuration is invalid.
        """
        self.config = config
        self.preprocessing_config = config.get("preprocessing", {})

        # Validate configuration
        self._validate_configuration()

        # Initialize pipeline settings
        self.enabled_stages = self._get_enabled_stages()
        self.cache_intermediates = self.preprocessing_config.get(
            "cache_intermediates",
            False,
        )
        self.progress_reporting = self.preprocessing_config.get(
            "progress_reporting",
            True,
        )
        self.chunk_size = self.preprocessing_config.get("chunk_size", None)

        # Initialize caching
        self.intermediate_cache: dict[str, Any] = {}

        # Generate pipeline ID for this instance
        self.pipeline_id = self._generate_pipeline_id()

        logger.info(
            f"Preprocessing pipeline initialized with {len(self.enabled_stages)} enabled stages",
        )
        logger.debug(f"Pipeline ID: {self.pipeline_id}")
        logger.debug(
            f"Enabled stages: {[stage.value for stage in self.enabled_stages]}",
        )

    def _validate_configuration(self) -> None:
        """Validate preprocessing configuration parameters."""
        if not isinstance(self.preprocessing_config, dict):
            raise PreprocessingConfigurationError(
                "preprocessing configuration must be a dictionary",
            )

        # Validate stage configurations
        stages_config = self.preprocessing_config.get("stages", {})
        for stage_name, stage_config in stages_config.items():
            if stage_name not in [s.value for s in PreprocessingStage]:
                logger.warning(f"Unknown preprocessing stage: {stage_name}")

            if not isinstance(stage_config, dict):
                raise PreprocessingConfigurationError(
                    f"Stage configuration for {stage_name} must be a dictionary",
                )

        # Validate normalization method
        norm_method = (
            self.preprocessing_config.get("stages", {})
            .get("normalize_data", {})
            .get("method", "baseline")
        )
        if norm_method not in [m.value for m in NormalizationMethod]:
            raise PreprocessingConfigurationError(
                f"Unknown normalization method: {norm_method}",
            )

        # Validate noise reduction method
        noise_method = (
            self.preprocessing_config.get("stages", {})
            .get("reduce_noise", {})
            .get("method", "none")
        )
        if noise_method not in [m.value for m in NoiseReductionMethod]:
            raise PreprocessingConfigurationError(
                f"Unknown noise reduction method: {noise_method}",
            )

        # Check required dependencies
        if noise_method in ["wiener", "savgol"] and not HAS_SCIPY:
            logger.warning(
                f"Noise reduction method '{noise_method}' requires scipy - falling back to 'none'",
            )

    def _get_enabled_stages(self) -> list[PreprocessingStage]:
        """Get list of enabled preprocessing stages based on configuration."""
        stages_config = self.preprocessing_config.get("stages", {})
        enabled_stages = []

        # Check each stage
        for stage in PreprocessingStage:
            stage_config = stages_config.get(stage.value, {})

            # Default to enabled for most stages, disabled for optional ones
            default_enabled = stage not in [PreprocessingStage.REDUCE_NOISE]

            if stage_config.get("enabled", default_enabled):
                enabled_stages.append(stage)

        return enabled_stages

    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID based on configuration hash."""
        config_str = json.dumps(self.preprocessing_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        timestamp = str(int(time.time()))
        return f"preprocess_{config_hash}_{timestamp}"

    @log_performance(threshold=1.0)
    def process(self, data: dict[str, Any]) -> PreprocessingResult:
        """Execute the full preprocessing pipeline on input data.

        Parameters
        ----------
        data
            Input data dictionary from the XPCS loader.

        Returns
        -------
        PreprocessingResult
            Result holding the processed data and its provenance.
        """
        start_time = time.time()

        # Initialize provenance tracking
        config_hash = hashlib.sha256(
            json.dumps(self.preprocessing_config, sort_keys=True).encode(),
        ).hexdigest()

        provenance = PreprocessingProvenance(
            pipeline_id=self.pipeline_id,
            config_hash=config_hash,
        )

        logger.info(f"Starting preprocessing pipeline {self.pipeline_id}")

        try:
            # Process data through each enabled stage
            processed_data = data.copy()
            stage_results = {}

            for i, stage in enumerate(self.enabled_stages):
                if self.progress_reporting:
                    logger.info(
                        f"Processing stage {i + 1}/{len(self.enabled_stages)}: {stage.value}",
                    )

                try:
                    # T036: Add log_phase to preprocessing stages
                    with log_phase(
                        f"preprocess_{stage.value}",
                        logger=logger,
                        track_memory=True,
                    ) as phase:
                        processed_data, transform_record = self._execute_stage(
                            stage,
                            processed_data,
                        )
                    stage_results[stage] = True
                    transform_record.duration = phase.duration
                    transform_record.memory_usage = phase.memory_peak_gb
                    provenance.transformations.append(transform_record)

                except Exception as e:
                    logger.error(f"Stage {stage.value} failed: {e}")
                    stage_results[stage] = False
                    provenance.errors.append(f"Stage {stage.value} failed: {str(e)}")

                    # Check if we should continue or abort
                    if self.preprocessing_config.get("abort_on_error", False):
                        raise PreprocessingError(
                            f"Pipeline aborted at stage {stage.value}: {e}",
                        ) from e
                    else:
                        logger.warning(
                            "Continuing pipeline after stage '%s' failure "
                            "(abort_on_error=False): the data passed downstream has "
                            "NOT had this operation applied. PreprocessingResult."
                            "success may still be True; inspect stage_results to see "
                            "which operations were skipped.",
                            stage.value,
                        )

            # Calculate final metrics
            provenance.total_duration = time.time() - start_time

            # Pipeline fails if any critical data-integrity stage ran and failed.
            # Both CORRECT_DIAGONAL and VALIDATE_OUTPUT gate downstream fit validity:
            # silently continuing past their failure (the default abort_on_error=False
            # path) would feed uncorrected/invalid data into the optimizer with a
            # success=True report. Otherwise succeed if any stage ran, or empty pipeline.
            critical_stages = (
                PreprocessingStage.CORRECT_DIAGONAL,
                PreprocessingStage.VALIDATE_OUTPUT,
            )
            critical_failed = any(stage_results.get(s) is False for s in critical_stages)
            if critical_failed:
                success = False
            else:
                success = (len(stage_results) == 0) or any(stage_results.values())

            logger.info(
                f"Preprocessing pipeline completed in {provenance.total_duration:.2f}s",
            )
            logger.info(
                f"Successful stages: {sum(stage_results.values())}/{len(stage_results)}",
            )

            return PreprocessingResult(
                data=processed_data,
                provenance=provenance,
                success=success,
                stage_results=stage_results,
            )

        except Exception as e:
            provenance.total_duration = time.time() - start_time
            provenance.errors.append(f"Pipeline failed: {str(e)}")
            logger.error(f"Preprocessing pipeline failed: {e}")

            return PreprocessingResult(
                data=data,  # Return original data on failure
                provenance=provenance,
                success=False,
                stage_results={},
            )

    def _execute_stage(
        self,
        stage: PreprocessingStage,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], TransformationRecord]:
        """Execute a single preprocessing stage."""
        stage_start = time.time()
        input_shape = self._get_data_shape(data)

        # Get stage configuration
        stage_config = self.preprocessing_config.get("stages", {}).get(stage.value, {})

        # Execute stage based on type
        if stage == PreprocessingStage.CORRECT_DIAGONAL:
            processed_data = self._correct_diagonal_enhanced(data, stage_config)
            method = stage_config.get("method", "statistical")

        elif stage == PreprocessingStage.NORMALIZE_DATA:
            processed_data = self._normalize_data(data, stage_config)
            method = stage_config.get("method", "baseline")

        elif stage == PreprocessingStage.REDUCE_NOISE:
            processed_data = self._reduce_noise(data, stage_config)
            method = stage_config.get("method", "none")

        elif stage == PreprocessingStage.STANDARDIZE_FORMAT:
            processed_data = self._standardize_format(data, stage_config)
            method = "format_standardization"

        elif stage == PreprocessingStage.VALIDATE_OUTPUT:
            processed_data = self._validate_output(data, stage_config)
            method = "integrity_validation"

        else:
            # Stage not implemented, pass through
            processed_data = data
            method = "passthrough"
            logger.warning(
                f"Stage {stage.value} not implemented - passing through data",
            )

        # Create transformation record
        duration = time.time() - stage_start
        output_shape = self._get_data_shape(processed_data)

        transform_record = TransformationRecord(
            stage=stage,
            method=method,
            parameters=stage_config,
            timestamp=stage_start,
            duration=duration,
            input_shape=input_shape,
            output_shape=output_shape,
        )

        logger.debug(f"Stage {stage.value} completed in {duration:.3f}s")

        return processed_data, transform_record

    def _get_data_shape(self, data: dict[str, Any]) -> tuple[int, ...]:
        """Get representative shape of correlation data for tracking."""
        if "c2_exp" in data:
            return tuple(data["c2_exp"].shape)
        return (0,)

    @log_performance(threshold=0.5)
    def _correct_diagonal_enhanced(
        self,
        data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Enhanced diagonal correction with multiple statistical methods.

        Uses unified diagonal_correction module with fallback
        to local implementations for backward compatibility.

        Goes beyond basic diagonal correction to provide statistical methods
        for improving correlation matrix quality.
        """
        method = config.get("method", "statistical")
        c2_exp = data["c2_exp"]

        logger.debug(f"Applying enhanced diagonal correction: {method}")

        corrected_data = {
            k: (np.array(v) if hasattr(v, "shape") else copy.deepcopy(v)) for k, v in data.items()
        }

        # Use unified module if available
        if HAS_DIAGONAL_CORRECTION and apply_diagonal_correction is not None:
            extra_kwargs = {
                k: v for k, v in config.items() if k not in ("method", "enabled", "backend")
            }
            for i in range(len(c2_exp)):
                corrected_data["c2_exp"][i] = apply_diagonal_correction(
                    c2_exp[i],
                    method=method,
                    backend="numpy",
                    **extra_kwargs,
                )
        else:
            # Fallback to local implementations
            if method == "basic":
                for i in range(len(c2_exp)):
                    corrected_data["c2_exp"][i] = self._basic_diagonal_correction(c2_exp[i])
            elif method == "statistical":
                for i in range(len(c2_exp)):
                    corrected_data["c2_exp"][i] = self._statistical_diagonal_correction(
                        c2_exp[i],
                        config,
                    )
            elif method == "interpolation":
                for i in range(len(c2_exp)):
                    corrected_data["c2_exp"][i] = self._interpolation_diagonal_correction(
                        c2_exp[i],
                        config,
                    )
            else:
                logger.warning(
                    f"Unknown diagonal correction method: {method}, using statistical",
                )
                return self._correct_diagonal_enhanced(
                    data,
                    {**config, "method": "statistical"},
                )

        return corrected_data

    def _basic_diagonal_correction(self, c2_mat: np.ndarray) -> np.ndarray:
        """Apply the basic diagonal correction as implemented in xpcs_loader.py.

        .. deprecated::
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction`
            with method="basic" instead.
        """
        size = c2_mat.shape[0]
        side_band = c2_mat[(np.arange(size - 1), np.arange(1, size))]
        diag_val = np.zeros(size)
        diag_val[:-1] += side_band
        diag_val[1:] += side_band
        norm = np.ones(size)
        norm[1:-1] = 2
        c2_mat = c2_mat.copy()
        c2_mat[np.diag_indices(size)] = diag_val / norm
        return c2_mat

    def _statistical_diagonal_correction(
        self,
        c2_mat: np.ndarray,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Statistical diagonal correction using robust estimators.

        .. deprecated::
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction`
            with method="statistical" instead.
        """
        c2_corrected = c2_mat.copy()
        size = c2_mat.shape[0]

        # Parameters
        window_size = config.get("window_size", 3)
        estimator = config.get(
            "estimator",
            "median",
        )  # 'mean', 'median', 'trimmed_mean'

        for i in range(size):
            # Get window of off-diagonal neighbors
            neighbors = []

            # Collect neighboring off-diagonal values
            for offset in range(1, min(window_size + 1, size)):
                if i - offset >= 0:
                    neighbors.append(c2_mat[i - offset, i])
                    neighbors.append(c2_mat[i, i - offset])
                if i + offset < size:
                    neighbors.append(c2_mat[i + offset, i])
                    neighbors.append(c2_mat[i, i + offset])

            if neighbors:
                if HAS_SCIPY:
                    neighbors_arr = np.array(neighbors)
                else:
                    neighbors_arr = neighbors  # type: ignore[assignment]

                # Apply statistical estimator (NaN-safe: off-diagonal c2 values
                # can be NaN for failed measurement points)
                if estimator == "median":
                    c2_corrected[i, i] = np.nanmedian(neighbors_arr)
                elif estimator == "mean":
                    c2_corrected[i, i] = np.nanmean(neighbors_arr)
                elif estimator == "trimmed_mean":
                    trim_fraction = config.get("trim_fraction", 0.2)
                    if HAS_SCIPY:
                        # Filter NaN before trim_mean (scipy has no NaN-safe variant)
                        finite_neighbors = neighbors_arr[np.isfinite(neighbors_arr)]
                        if finite_neighbors.size > 0:
                            c2_corrected[i, i] = stats.trim_mean(finite_neighbors, trim_fraction)
                        else:
                            c2_corrected[i, i] = np.nan
                    else:
                        # Fallback to median
                        c2_corrected[i, i] = np.nanmedian(neighbors_arr)
                else:
                    logger.warning(f"Unknown estimator: {estimator}, using median")
                    c2_corrected[i, i] = np.nanmedian(neighbors_arr)

        return c2_corrected

    def _interpolation_diagonal_correction(
        self,
        c2_mat: np.ndarray,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Interpolation-based diagonal correction.

        .. deprecated::
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction`
            with method="interpolation" instead.
        """
        c2_corrected = c2_mat.copy()
        size = c2_mat.shape[0]

        method = config.get("interpolation_method", "linear")  # 'linear', 'cubic'

        for i in range(size):
            # Get off-diagonal values for interpolation
            if i > 0 and i < size - 1:
                # Use neighboring off-diagonal values for interpolation
                y_points = [c2_mat[i - 1, i], c2_mat[i + 1, i]]

                # Simple linear interpolation (NaN-safe: neighbors may be NaN)
                if method == "linear":
                    c2_corrected[i, i] = np.nanmean(y_points)
                elif method == "cubic" and len(y_points) >= 2:
                    # For cubic, need more points - fall back to linear
                    c2_corrected[i, i] = np.nanmean(y_points)
            elif i == 0:
                # Use next off-diagonal value
                c2_corrected[i, i] = c2_mat[0, 1]
            elif i == size - 1:
                # Use previous off-diagonal value
                c2_corrected[i, i] = c2_mat[size - 2, size - 1]

        return c2_corrected

    @log_performance(threshold=0.3)
    def _normalize_data(
        self,
        data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply normalization to correlation data using multiple methods."""
        method = NormalizationMethod(config.get("method", "baseline"))
        c2_exp = data["c2_exp"]

        logger.debug(f"Applying normalization: {method.value}")

        # Deep-copy non-ndarray values (lists, dicts, etc.) so that downstream
        # mutations of e.g. "filters_applied" lists do not corrupt the original dict.
        # hasattr(v, 'shape') catches both numpy and JAX arrays; deepcopy of a JAX
        # array returns the same immutable object, so subsequent item-assignment fails.
        normalized_data = {
            k: (np.array(v) if hasattr(v, "shape") else copy.deepcopy(v)) for k, v in data.items()
        }

        if method == NormalizationMethod.BASELINE:
            # Normalize by t=0 value (diagonal)
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                baseline = c2_matrix[0, 0]  # t=0 correlation
                if abs(baseline) > np.finfo(float).eps:
                    normalized_data["c2_exp"][i] = c2_matrix / baseline
                else:
                    logger.warning(
                        f"Near-zero baseline value ({baseline:.3e}) at matrix {i}, "
                        "skipping normalization",
                    )

        elif method == NormalizationMethod.STATISTICAL:
            # Z-score normalization
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                mean_val = np.nanmean(c2_matrix)
                std_val = np.nanstd(c2_matrix)
                # Use epsilon guard instead of exact float equality: std_val near
                # zero (subnormal) would pass != 0 but cause overflow on division.
                if abs(std_val) > np.finfo(np.float64).eps:
                    normalized_data["c2_exp"][i] = (c2_matrix - mean_val) / std_val
                else:
                    logger.warning(
                        f"Zero standard deviation at matrix {i}, skipping normalization",
                    )

        elif method == NormalizationMethod.MINMAX:
            # Min-max scaling [0, 1]
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                min_val = np.nanmin(c2_matrix)
                max_val = np.nanmax(c2_matrix)
                if max_val - min_val > np.finfo(float).eps * max(abs(max_val), 1.0):
                    normalized_data["c2_exp"][i] = (c2_matrix - min_val) / (max_val - min_val)
                else:
                    logger.warning(
                        f"Constant values at matrix {i}, skipping normalization",
                    )

        elif method == NormalizationMethod.ROBUST:
            # Robust scaling using percentiles
            percentile_range = config.get("percentile_range", [25, 75])
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                q25, q75 = np.nanpercentile(c2_matrix, percentile_range)
                if q75 - q25 > np.finfo(float).eps * max(abs(q75), 1.0):
                    median_val = np.nanmedian(c2_matrix)
                    normalized_data["c2_exp"][i] = (c2_matrix - median_val) / (q75 - q25)
                else:
                    logger.warning(
                        f"No variance in percentile range at matrix {i}, skipping normalization",
                    )

        elif method == NormalizationMethod.PHYSICS_BASED:
            # Physics-constrained normalization
            # Ensure correlation function properties are preserved
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]

                # Normalize by maximum value but ensure t=0 correlation >= 1
                max_val = np.nanmax(c2_matrix)
                if max_val > 0:
                    normalized_matrix = c2_matrix / max_val

                    # Ensure physics constraints
                    t0_correlation = normalized_matrix[0, 0]
                    if t0_correlation < 1.0:
                        # Rescale to ensure t=0 correlation = 1
                        normalized_data["c2_exp"][i] = normalized_matrix / t0_correlation
                    else:
                        normalized_data["c2_exp"][i] = normalized_matrix
                else:
                    logger.warning(
                        f"Zero maximum value at matrix {i}, skipping normalization",
                    )

        return normalized_data

    @log_performance(threshold=0.4)
    def _reduce_noise(
        self,
        data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply noise reduction algorithms to correlation data."""
        method = NoiseReductionMethod(config.get("method", "none"))

        if method == NoiseReductionMethod.NONE:
            return data

        c2_exp = data["c2_exp"]
        logger.debug(f"Applying noise reduction: {method.value}")

        denoised_data = {
            k: (np.array(v) if hasattr(v, "shape") else copy.deepcopy(v)) for k, v in data.items()
        }

        if method == NoiseReductionMethod.MEDIAN:
            # Median filtering
            kernel_size = config.get("kernel_size", 3)
            if HAS_SCIPY:
                for i in range(len(c2_exp)):
                    denoised_data["c2_exp"][i] = median_filter(
                        c2_exp[i],
                        size=kernel_size,
                    )
            else:
                logger.warning("Scipy not available for median filtering, skipping")
                return data

        elif method == NoiseReductionMethod.GAUSSIAN:
            # Gaussian smoothing
            sigma = config.get("sigma", 1.0)
            if HAS_SCIPY:
                for i in range(len(c2_exp)):
                    denoised_data["c2_exp"][i] = gaussian_filter(c2_exp[i], sigma=sigma)
            else:
                logger.warning("Scipy not available for gaussian filtering, skipping")
                return data

        elif method == NoiseReductionMethod.WIENER:
            # Wiener filtering
            if HAS_SCIPY:
                noise_variance = config.get("noise_variance", None)
                for i in range(len(c2_exp)):
                    # Apply Wiener filter
                    denoised_data["c2_exp"][i] = signal.wiener(
                        c2_exp[i],
                        noise=noise_variance,
                    )
            else:
                logger.warning(
                    "Scipy not available for Wiener filtering, falling back to gaussian",
                )
                return self._reduce_noise(data, {**config, "method": "gaussian"})

        elif method == NoiseReductionMethod.SAVGOL:
            # Savitzky-Golay filtering
            if HAS_SCIPY:
                window_length = config.get("window_length", 5)
                polyorder = config.get("polyorder", 2)

                for i in range(len(c2_exp)):
                    c2_matrix = c2_exp[i]
                    # Apply along each row and column
                    filtered_matrix = c2_matrix.copy()

                    # Filter rows
                    for row in range(c2_matrix.shape[0]):
                        if c2_matrix.shape[1] > window_length:
                            filtered_matrix[row, :] = signal.savgol_filter(
                                c2_matrix[row, :],
                                window_length,
                                polyorder,
                            )

                    # Filter columns
                    for col in range(c2_matrix.shape[1]):
                        if c2_matrix.shape[0] > window_length:
                            filtered_matrix[:, col] = signal.savgol_filter(
                                filtered_matrix[:, col],
                                window_length,
                                polyorder,
                            )

                    denoised_data["c2_exp"][i] = filtered_matrix
            else:
                logger.warning(
                    "Scipy not available for Savitzky-Golay filtering, falling back to median",
                )
                return self._reduce_noise(data, {**config, "method": "median"})

        return denoised_data

    def _standardize_format(
        self,
        data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Standardize data format to ensure consistency across APS vs APS-U sources."""
        logger.debug("Standardizing data format")

        standardized_data = {
            k: (np.array(v) if hasattr(v, "shape") else copy.deepcopy(v)) for k, v in data.items()
        }

        # Ensure consistent data types
        if "wavevector_q_list" in data:
            standardized_data["wavevector_q_list"] = np.array(
                data["wavevector_q_list"],
                dtype=np.float64,
            )

        if "phi_angles_list" in data:
            standardized_data["phi_angles_list"] = np.array(
                data["phi_angles_list"],
                dtype=np.float64,
            )

        if "t1" in data:
            standardized_data["t1"] = np.array(data["t1"], dtype=np.float64)

        if "t2" in data:
            standardized_data["t2"] = np.array(data["t2"], dtype=np.float64)

        if "c2_exp" in data:
            standardized_data["c2_exp"] = np.array(data["c2_exp"], dtype=np.float64)

        # Ensure consistent array ordering and shapes
        c2_exp = standardized_data.get("c2_exp")
        if c2_exp is not None:
            # Ensure all correlation matrices are square
            for i, c2_matrix in enumerate(c2_exp):
                if c2_matrix.shape[0] != c2_matrix.shape[1]:
                    logger.warning(
                        f"Non-square correlation matrix at index {i}: {c2_matrix.shape}",
                    )
                    # Could implement automatic padding/truncation here if needed

        # Validate consistency between arrays
        q_list = standardized_data.get("wavevector_q_list")
        phi_list = standardized_data.get("phi_angles_list")

        if q_list is not None and phi_list is not None and c2_exp is not None:
            if len(q_list) != len(phi_list) or len(q_list) != len(c2_exp):
                logger.warning(
                    f"Inconsistent array lengths: q_list={len(q_list)}, "
                    f"phi_list={len(phi_list)}, c2_exp={len(c2_exp)}",
                )

        return standardized_data

    def _validate_output(
        self,
        data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate final data integrity and physics constraints."""
        logger.debug("Validating output data integrity")

        # Check for required keys
        required_keys = ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]
        for key in required_keys:
            if key not in data:
                raise PreprocessingError(f"Missing required data key: {key}")

        # Check for non-finite values (convert to ndarray first: np.isfinite on a
        # list-of-2D-arrays raises ValueError or produces an object array).
        for key in required_keys:
            values = data[key]
            if isinstance(values, (np.ndarray, list)) or hasattr(values, "shape"):
                arr = np.asarray(values)
                if np.any(~np.isfinite(arr)):
                    raise PreprocessingError(f"Non-finite values found in {key}")

        # Physics-based validation
        c2_exp = data["c2_exp"]

        # Check correlation matrix properties
        for i, c2_matrix in enumerate(c2_exp):
            # Check for negative correlation values (warning only)
            if np.any(c2_matrix < 0):
                logger.warning(f"Negative correlation values found in matrix {i}")

            # Check diagonal values (should generally be >= off-diagonal for t=0)
            diagonal = c2_matrix.diagonal()
            if len(diagonal) > 0:
                t0_corr = diagonal[0]
                if t0_corr <= 0:
                    logger.warning(
                        f"Non-positive t=0 correlation in matrix {i}: {t0_corr}",
                    )

        # Check time arrays
        t1, t2 = data["t1"], data["t2"]
        if len(t1) != len(t2):
            logger.warning(
                f"Time arrays have different lengths: t1={len(t1)}, t2={len(t2)}",
            )

        # Check for monotonicity in time arrays
        if len(t1) > 1:
            if not np.all(np.diff(t1) >= 0):
                logger.warning("t1 array is not monotonically increasing")

        logger.debug("Output validation completed")
        return data

    def save_provenance(
        self,
        provenance: PreprocessingProvenance,
        filepath: str | Path,
    ) -> None:
        """Save preprocessing provenance to file for reproducibility.

        Args:
            provenance: Provenance record to save
            filepath: Path where to save the provenance record
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(provenance.to_dict(), f, indent=2)

        logger.info(f"Preprocessing provenance saved to: {filepath}")

    def load_provenance(self, filepath: str | Path) -> PreprocessingProvenance:
        """Load preprocessing provenance from a file.

        Parameters
        ----------
        filepath
            Path to the provenance file.

        Returns
        -------
        PreprocessingProvenance
            The reconstructed provenance object.
        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct provenance object
        transformations = []
        for t_data in data.get("transformations", []):
            transformations.append(
                TransformationRecord(
                    stage=PreprocessingStage(t_data["stage"]),
                    method=t_data["method"],
                    parameters=t_data["parameters"],
                    timestamp=t_data["timestamp"],
                    duration=t_data["duration"],
                    input_shape=tuple(t_data["input_shape"]),
                    output_shape=tuple(t_data["output_shape"]),
                    memory_usage=t_data.get("memory_usage"),
                    warnings=t_data.get("warnings", []),
                ),
            )

        return PreprocessingProvenance(
            pipeline_id=data["pipeline_id"],
            config_hash=data["config_hash"],
            transformations=transformations,
            total_duration=data["total_duration"],
            peak_memory_usage=data["peak_memory_usage"],
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
        )


# Utility functions for easy integration


def create_default_preprocessing_config() -> dict[str, Any]:
    """Create the default preprocessing configuration.

    Returns
    -------
    dict
        Dictionary with default preprocessing settings.
    """
    return {
        "preprocessing": {
            "enabled": True,
            "cache_intermediates": False,
            "progress_reporting": True,
            "abort_on_error": False,
            "stages": {
                "correct_diagonal": {
                    "enabled": True,
                    "method": "statistical",
                    "window_size": 3,
                    "estimator": "median",
                },
                "normalize_data": {"enabled": True, "method": "baseline"},
                "reduce_noise": {
                    "enabled": False,
                    "method": "none",
                    "kernel_size": 3,
                    "sigma": 1.0,
                },
                "standardize_format": {"enabled": True},
                "validate_output": {"enabled": True},
            },
        },
    }


def preprocess_xpcs_data(
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> PreprocessingResult:
    """Preprocess XPCS data using a one-shot convenience wrapper.

    Builds a :class:`PreprocessingPipeline` and runs it over ``data``, using the
    default configuration when ``config`` is ``None``.

    Parameters
    ----------
    data
        Input data dictionary from the XPCS loader.
    config
        Optional preprocessing configuration; defaults from
        :func:`create_default_preprocessing_config` are used when ``None``.

    Returns
    -------
    PreprocessingResult
        Result holding the processed data and its provenance.

    Examples
    --------
    >>> result = preprocess_xpcs_data(data)
    >>> processed = result.data
    """
    if config is None:
        config = create_default_preprocessing_config()

    pipeline = PreprocessingPipeline(config)
    return pipeline.process(data)


# Export main classes and functions
__all__ = [
    "PreprocessingPipeline",
    "PreprocessingResult",
    "PreprocessingProvenance",
    "PreprocessingStage",
    "NormalizationMethod",
    "NoiseReductionMethod",
    "PreprocessingError",
    "PreprocessingConfigurationError",
    "create_default_preprocessing_config",
    "preprocess_xpcs_data",
]
