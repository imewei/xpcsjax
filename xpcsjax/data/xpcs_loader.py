"""XPCS Data Loader for Homodyne
================================

Enhanced XPCS data loader supporting both APS (old) and APS-U (new) HDF5 formats
with YAML-first configuration system, JAX compatibility, and modern architecture integration.

This module provides:
- YAML-first configuration with JSON support
- Smart NPZ caching to avoid reloading large HDF5 files
- Auto-detection of APS vs APS-U format
- Half-matrix reconstruction for correlation matrices
- Mandatory diagonal correction applied post-load
- JAX array output with numpy fallback
- Integration with v2 logging and physics validation

Key Features:
- Format Support: APS old format and APS-U new format
- Configuration: YAML primary, JSON via converter
- Caching: Intelligent NPZ caching with compression
- Output: JAX arrays when available, numpy fallback
- Validation: Optional physics-based data quality checks
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

# Handle optional dependencies with graceful fallback
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xpcsjax.data.dataset import XpcsDataset
else:
    NDArray = Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore[assignment]

# JAX integration
try:
    import jax.numpy as jnp

    from xpcsjax.core.jax_backend import jax_available

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax_available = False
    jnp = np  # type: ignore[misc]

# V2 system integration
try:
    from xpcsjax.utils.logging import (
        get_logger as _get_logger,
    )
    from xpcsjax.utils.logging import (
        log_calls as _log_calls,
    )
    from xpcsjax.utils.logging import (
        log_exception as _log_exception,
    )
    from xpcsjax.utils.logging import (
        log_performance as _log_performance,
    )
    from xpcsjax.utils.logging import (
        log_phase as _log_phase,
    )

    HAS_V2_LOGGING = True
    get_logger = _get_logger
    log_performance = _log_performance
    log_calls = _log_calls
    log_phase = _log_phase
    log_exception = _log_exception
except ImportError:
    # Fallback to standard logging if v2 logging not available
    import logging
    from collections.abc import Iterator
    from contextlib import contextmanager

    HAS_V2_LOGGING = False

    F = TypeVar("F", bound=Callable[..., Any])

    def get_logger(name: str | None = None, **kwargs: Any) -> logging.Logger:
        return logging.getLogger(name)

    def log_exception(  # type: ignore[misc]
        logger: Any,
        exc: BaseException,
        context: dict[str, Any] | None = None,
        level: int = logging.ERROR,
        include_traceback: bool = True,
    ) -> None:
        """Fallback log_exception when v2 logging is unavailable."""
        try:
            logger.log(level, "Exception: %r (context=%r)", exc, context)
        except Exception:
            pass

    def log_performance(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

    def log_calls(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

    @contextmanager
    def log_phase(name: str, **kwargs: Any) -> Iterator[Any]:  # type: ignore[misc]
        """Fallback log_phase for environments without v2 logging.

        The real ``log_phase`` takes ``(name, logger, level, track_memory,
        threshold)`` — this fallback only needs the name; ``**kwargs`` swallows
        the rest. ``# type: ignore[misc]`` acknowledges the signature delta
        with the try-branch import.
        """
        yield type("PhaseContext", (), {"duration": 0.0, "memory_peak_gb": None})()


# Physics validation integration
try:
    from xpcsjax.core.physics import (
        PhysicsConstants as _PhysicsConstants,
    )
    from xpcsjax.core.physics import (
        validate_experimental_setup as _validate_experimental_setup,
    )

    HAS_PHYSICS_VALIDATION = True
    PhysicsConstants = _PhysicsConstants
    validate_experimental_setup = _validate_experimental_setup
except ImportError:
    HAS_PHYSICS_VALIDATION = False
    PhysicsConstants = None  # type: ignore
    validate_experimental_setup = None  # type: ignore

# Diagonal correction from unified module
try:
    from xpcsjax.core.diagonal_correction import (
        apply_diagonal_correction_batch as _apply_diagonal_correction_batch,
    )

    HAS_DIAGONAL_CORRECTION = True
    apply_diagonal_correction_batch = _apply_diagonal_correction_batch
except ImportError:
    HAS_DIAGONAL_CORRECTION = False
    apply_diagonal_correction_batch = None  # type: ignore

# Performance engine integration
try:
    from xpcsjax.data.memory_manager import (
        AdvancedMemoryManager as _AdvancedMemoryManager,
    )
    from xpcsjax.data.optimization import (
        AdvancedDatasetOptimizer as _AdvancedDatasetOptimizer,
    )
    from xpcsjax.data.performance_engine import PerformanceEngine as _PerformanceEngine

    HAS_PERFORMANCE_ENGINE = True
    PerformanceEngine = _PerformanceEngine
    AdvancedMemoryManager = _AdvancedMemoryManager
    AdvancedDatasetOptimizer = _AdvancedDatasetOptimizer
except ImportError:
    HAS_PERFORMANCE_ENGINE = False
    PerformanceEngine = None  # type: ignore
    AdvancedMemoryManager = None  # type: ignore
    AdvancedDatasetOptimizer = None  # type: ignore

logger = get_logger(__name__)

# HDF5 chunk-cache tuning for the two production format loaders.
# APS correlation matrices are (n_t, n_t) float64.  At worst-case n_t=1000
# each matrix is 8 MB; we size the cache to hold ~12 matrices comfortably.
# rdcc_nslots must be a prime roughly 100× the number of cached chunks.
_HDF5_RDCC_N_MATRICES: int = 12
_HDF5_RDCC_MATRIX_BYTES: int = 1000 * 1000 * 8  # float64, n_t=1000 worst-case
_HDF5_RDCC_NBYTES: int = _HDF5_RDCC_N_MATRICES * _HDF5_RDCC_MATRIX_BYTES  # 96 MB
_HDF5_RDCC_NSLOTS: int = 6257   # prime; ≥ 100 × _HDF5_RDCC_N_MATRICES
_HDF5_RDCC_W0: float = 0.75     # prefer evicting chunks not likely to be re-read

# Regex to detect old str.format()-style placeholders: {var} or {var:.4f}
_OLD_FORMAT_RE = re.compile(r"\{(\w+)(?::[^}]*)?\}")


def _migrate_cache_template(template: str) -> str:
    """Auto-convert old {var} format templates to ${var} syntax.

    Returns the template unchanged if it already uses $ syntax.
    Logs a warning on first migration.
    """
    if "$" not in template and _OLD_FORMAT_RE.search(template):
        migrated = _OLD_FORMAT_RE.sub(r"${\1}", template)
        logger.warning(
            "Cache template uses deprecated {var} format; auto-migrated to ${var}. "
            "Update your YAML config: %r -> %r",
            template,
            migrated,
        )
        return migrated
    return template


class XPCSDataFormatError(Exception):
    """Raised when XPCS data format is not recognized or invalid."""


class XPCSDependencyError(Exception):
    """Raised when required dependencies are not available."""


class XPCSConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""


# Upper bound on the correlation-matrix time dimension. Real XPCS experiments
# run from hundreds to a few tens of thousands of frames; this generous cap
# exists only to stop a crafted/corrupt file from declaring an absurd dimension
# that triggers a multi-hundred-GB ``(n_sel, n_t, n_t)`` allocation (OOM/DoS)
# before any other validation runs.
MAX_CORRELATION_FRAMES = 100_000


def _check_frame_count(n_frames: int, *, source: str) -> None:
    """Validate a correlation-matrix time dimension before allocating on it.

    Raises ``XPCSDataFormatError`` if ``n_frames`` is non-positive or exceeds
    :data:`MAX_CORRELATION_FRAMES`. This guards the I/O boundary against an
    unbounded allocation driven by an untrusted file's declared shape.
    """
    if n_frames <= 0:
        raise XPCSDataFormatError(
            f"Invalid correlation frame count {n_frames} from {source!r} "
            "(must be positive)."
        )
    if n_frames > MAX_CORRELATION_FRAMES:
        raise XPCSDataFormatError(
            f"Correlation frame count {n_frames} from {source!r} exceeds the "
            f"{MAX_CORRELATION_FRAMES} cap; refusing to allocate "
            f"{n_frames}x{n_frames} matrices. Raise MAX_CORRELATION_FRAMES if "
            "this is a legitimately large experiment."
        )


# Directory/traversal/drive tokens that must never appear in a cache filename.
# Checked explicitly (not via ``os.sep``) so the guard is identical on POSIX and
# Windows: on Windows ``os.sep`` is ``\`` only, so ``/`` and ``C:`` drive/ADS
# specifiers would otherwise slip through.
_UNSAFE_FILENAME_TOKENS = ("/", "\\", "..", ":", "\x00")


def _assert_safe_cache_filename(name: str) -> None:
    """Reject a cache filename that is not a bare, in-directory file name.

    Raises ``ValueError`` if ``name`` contains a path separator (either
    platform's), a parent-directory traversal, a drive/ADS ``:`` specifier, or a
    null byte. Platform-independent by construction.
    """
    if any(tok in name for tok in _UNSAFE_FILENAME_TOKENS):
        raise ValueError(f"Unsafe cache filename from template: {name!r}")


def load_xpcs_config(config_path: str | Path) -> dict[str, Any]:
    """Load XPCS configuration from YAML or JSON file.

    Primary format: YAML
    JSON support: Automatically converted to YAML format

    Args:
        config_path: Path to YAML or JSON configuration file

    Returns:
        Configuration dictionary with YAML-style structure

    Raises:
        XPCSConfigurationError: If configuration format is unsupported or invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")

    try:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise XPCSDependencyError(
                    "PyYAML required for YAML configuration files",
                )

            # Native YAML loading
            with open(config_path, encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            logger.info(f"Loaded YAML configuration: {config_path}")
            return config

        elif config_path.suffix.lower() == ".json":
            # JSON loading with structure conversion
            with open(config_path, encoding="utf-8") as f:
                json_config: dict[str, Any] = json.load(f)

            logger.info(f"Loaded JSON configuration (converted to YAML): {config_path}")
            logger.info("Consider migrating to YAML format for better readability")

            # Convert JSON structure to YAML-style (for now, keep identical structure)
            # In future, can add more sophisticated conversion via existing converter
            return json_config

        else:
            raise XPCSConfigurationError(
                f"Unsupported configuration format: {config_path.suffix}. "
                f"Supported formats: .yaml, .yml, .json",
            )

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise XPCSConfigurationError(
            f"Failed to parse configuration file {config_path}: {e}",
        ) from e


class XPCSDataLoader:
    """Enhanced XPCS data loader for Homodyne.

    Supports both APS (old) and APS-U (new) formats with YAML-first configuration,
    intelligent caching, and JAX integration.

    Features:
    - YAML-first configuration with JSON support
    - Auto-detection of HDF5 format (APS vs APS-U)
    - Smart NPZ caching with compression
    - Half-matrix reconstruction for correlation matrices
    - Mandatory diagonal correction applied consistently
    - JAX array output when available
    - Integration with v2 physics validation
    """

    @log_calls(include_args=False)
    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        configure_logging: bool = True,
        generate_quality_reports: bool = False,  # Only generate reports when explicitly requested
    ):
        """Initialize XPCS data loader with YAML-first configuration.

        Args:
            config_path: Path to YAML or JSON configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            configure_logging: Whether to apply logging configuration from config
            generate_quality_reports: Whether to generate quality reports (default: False)

        Raises:
            XPCSDependencyError: If required dependencies are not available
            XPCSConfigurationError: If configuration is invalid
        """
        # Check for required dependencies
        self._check_dependencies()

        # Store whether to generate quality reports (only for --plot-experimental-data)
        self.generate_quality_reports = generate_quality_reports

        if config_path and config_dict:
            raise ValueError("Provide either config_path or config_dict, not both")

        if config_path:
            self.config = load_xpcs_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")

        # Transform flat structure to nested structure for backward compatibility
        self._normalize_config_structure()

        # Process v2 configuration enhancements
        self._process_v2_config_enhancements()

        # Extract main configuration sections
        self.exp_config = self.config.get("experimental_data", {})
        self.analyzer_config = self.config.get("analyzer_parameters", {})
        self.v2_config = self.config.get("v2_features", {})

        # Initialize performance optimization components
        self._init_performance_components()

        # Validate configuration
        self._validate_configuration()

        logger.info(
            f"XPCS data loader initialized with {len(self.config)} config sections",
        )

    def _check_dependencies(self) -> None:
        """Check for required dependencies and raise error if missing."""
        missing_deps = []

        if not HAS_NUMPY:
            missing_deps.append("numpy")
        if not HAS_H5PY:
            missing_deps.append("h5py")

        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}. "
            error_msg += "Please install them with: pip install " + " ".join(
                missing_deps,
            )
            logger.error(error_msg)
            raise XPCSDependencyError(error_msg)

    def _normalize_config_structure(self) -> None:
        """Transform flat config structure to nested structure for backward compatibility.

        Detects flat structure (config with data_file at root level) and transforms it to
        nested structure (config with experimental_data, analyzer_parameters sections).

        Flat structure example:
            {
                "data_file": "/path/to/file.h5",
                "analysis_mode": "static_isotropic",
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": -1,
            }

        Nested structure example:
            {
                "analysis_mode": "static_isotropic",
                "experimental_data": {
                    "data_folder_path": "/path/to",
                    "data_file_name": "file.h5",
                },
                "analyzer_parameters": {
                    "dt": 0.1,
                    "start_frame": 1,
                    "end_frame": -1,
                },
            }
        """
        # Check if already in nested structure (has experimental_data section)
        if "experimental_data" in self.config:
            return  # Already normalized

        # Check if in flat structure (has data_file at root)
        if "data_file" not in self.config:
            return  # Neither flat nor nested - let validation handle it

        # Transform flat to nested
        import os

        data_file = self.config.pop("data_file")
        data_folder_path = os.path.dirname(data_file) or "."
        data_file_name = os.path.basename(data_file)

        # Create experimental_data section
        self.config["experimental_data"] = {
            "data_folder_path": data_folder_path,
            "data_file_name": data_file_name,
        }

        # Move analyzer parameters to analyzer_parameters section with defaults
        analyzer_params = {
            "dt": 0.1,  # Default time step (seconds)
            "start_frame": 1,  # Default start frame
            "end_frame": -1,  # Default end frame (-1 means all frames)
        }
        self.config["analyzer_parameters"] = {}
        for param, default_value in analyzer_params.items():
            if param in self.config:
                self.config["analyzer_parameters"][param] = self.config.pop(param)
            else:
                # Provide default for backward compatibility
                self.config["analyzer_parameters"][param] = default_value

        # Move output parameters to output section if present
        output_params = ["output_directory"]
        if any(param in self.config for param in output_params):
            self.config["output"] = {}
            for param in output_params:
                if param in self.config:
                    self.config["output"][param] = self.config.pop(param)

        logger.debug("Transformed flat config structure to nested structure")

    def _process_v2_config_enhancements(self) -> None:
        """Process v2 configuration enhancements and set defaults."""
        if "v2_features" not in self.config:
            self.config["v2_features"] = {}

        v2_defaults = {
            "output_format": "auto",  # 'numpy', 'jax', 'auto'
            "validation_level": "basic",  # 'none', 'basic', 'full'
            "performance_optimization": True,
            "physics_validation": False,
            "cache_strategy": "intelligent",  # 'none', 'simple', 'intelligent'
        }

        for key, default_value in v2_defaults.items():
            if key not in self.config["v2_features"]:
                self.config["v2_features"][key] = default_value

        # Add performance optimization defaults
        performance_defaults = {
            "performance_engine_enabled": True,
            "memory_mapped_io": True,
            "advanced_chunking": True,
            "multi_level_caching": True,
            "background_prefetching": True,
            "memory_pressure_monitoring": True,
        }

        if "performance" not in self.config:
            self.config["performance"] = {}

        for key, default_value in performance_defaults.items():
            if key not in self.config["performance"]:
                self.config["performance"][key] = default_value

    def _init_performance_components(self) -> None:
        """Initialize performance optimization components."""
        self.performance_engine = None
        self.memory_manager = None
        self.advanced_optimizer = None

        # Check if performance optimization is enabled
        performance_config = self.config.get("performance", {})
        if not performance_config.get("performance_engine_enabled", True):
            logger.info("Performance engine disabled in configuration")
            return

        if not HAS_PERFORMANCE_ENGINE:
            logger.warning(
                "Performance engine not available - falling back to basic optimization",
            )
            return

        try:
            # Initialize performance engine
            if performance_config.get("performance_engine_enabled", True):
                self.performance_engine = PerformanceEngine(self.config)
                logger.info("Performance engine initialized")

            # Initialize memory manager
            if performance_config.get("memory_pressure_monitoring", True):
                self.memory_manager = AdvancedMemoryManager(self.config)
                logger.info("Advanced memory manager initialized")

            # Initialize advanced optimizer
            self.advanced_optimizer = AdvancedDatasetOptimizer(
                config=self.config,
                performance_engine=self.performance_engine,
                memory_manager=self.memory_manager,
            )
            logger.info("Advanced dataset optimizer initialized")

        except Exception as e:
            log_exception(
                logger,
                e,
                context={"operation": "init_performance_components"},
                level=logging.DEBUG,
            )
            logger.info("Falling back to basic optimization")
            self.performance_engine = None
            self.memory_manager = None
            self.advanced_optimizer = None

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        required_exp_data = ["data_folder_path", "data_file_name"]
        required_analyzer = ["dt", "start_frame", "end_frame"]

        for key in required_exp_data:
            if key not in self.exp_config:
                raise XPCSConfigurationError(
                    f"Missing required experimental_data parameter: {key}",
                )

        for key in required_analyzer:
            if key not in self.analyzer_config:
                raise XPCSConfigurationError(
                    f"Missing required analyzer_parameters parameter: {key}",
                )

        # Validate file existence
        data_file_path = os.path.join(
            self.exp_config["data_folder_path"],
            self.exp_config["data_file_name"],
        )
        if ".." in str(data_file_path) or "\x00" in str(data_file_path):
            raise ValueError(
                f"Path traversal detected in data file path: {data_file_path}"
            )

        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found: {data_file_path}")
            logger.info("File will be checked again during data loading")

    def _get_output_format(self) -> str:
        """Get output array format from configuration."""
        format_val: Any = self.v2_config.get("output_format", "auto")
        return str(format_val)

    def _should_perform_validation(self) -> dict[str, bool]:
        """Get validation settings from configuration."""
        validation_level = self.v2_config.get("validation_level", "basic")
        return {
            "physics_checks": self.v2_config.get("physics_validation", False)
            and HAS_PHYSICS_VALIDATION,
            "data_quality": validation_level != "none",
            "comprehensive": validation_level == "full",
        }

    def _convert_arrays_to_target_format(
        self,
        data: dict[str, NDArray],
    ) -> dict[str, Any]:
        """Convert arrays to target format based on configuration.

        Args:
            data: Dictionary with numpy arrays

        Returns:
            Dictionary with arrays in target format (JAX or numpy)
        """
        output_format = self._get_output_format()

        if output_format == "jax" and HAS_JAX and jax_available:
            logger.debug("Converting arrays to JAX format")
            return {
                k: jnp.asarray(np.ascontiguousarray(v), dtype=jnp.float64)
                if isinstance(v, np.ndarray)
                else v
                for k, v in data.items()
            }

        elif output_format == "auto" and HAS_JAX and jax_available:
            logger.debug("Auto-selecting JAX format (available)")
            return {
                k: jnp.asarray(np.ascontiguousarray(v), dtype=jnp.float64)
                if isinstance(v, np.ndarray)
                else v
                for k, v in data.items()
            }

        elif output_format == "auto":
            logger.debug("Auto-selecting numpy format (JAX not available)")

        return data  # Keep numpy format

    @log_performance(threshold=0.5)
    def load_experimental_data(self) -> dict[str, Any]:
        """Load experimental data with priority: cache NPZ → raw HDF → error.

        Returns:
            Dictionary containing:
            - wavevector_q_list: Array of q values
            - phi_angles_list: Array of phi angles
            - t1: Time array for first dimension
            - t2: Time array for second dimension
            - c2_exp: Experimental correlation data
        """
        # Construct file paths
        data_folder = self.exp_config.get("data_folder_path", "./")
        data_file = self.exp_config.get("data_file_name", "")
        cache_folder = self.exp_config.get("cache_file_path", data_folder)

        # Get frame parameters
        start_frame = self.analyzer_config.get("start_frame", 1)
        end_frame = self.analyzer_config.get("end_frame", 8000)

        # Construct cache filename (using string.Template for safety)
        cache_template = _migrate_cache_template(
            self.exp_config.get(
                "cache_filename_template",
                "cached_c2_frames_${start_frame}_${end_frame}.npz",
            )
        )

        # Get wavevector_q for cache filename (selective caching support)
        scattering_config = self.analyzer_config.get("scattering", {})
        wavevector_q = scattering_config.get("wavevector_q", 0.0054)

        tmpl = string.Template(cache_template)
        cache_filename = tmpl.safe_substitute(
            start_frame=start_frame,
            end_frame=end_frame,
            wavevector_q=f"{wavevector_q:.4f}",
        )
        _assert_safe_cache_filename(cache_filename)
        cache_path = os.path.join(cache_folder, cache_filename)
        # The filename check above does not cover ``cache_folder`` (config-
        # supplied): a crafted ``cache_file_path`` could traverse out of the
        # intended tree before the npz is written. Run the assembled path
        # through the shared validator (rejects ``..``/null-byte traversal;
        # absolute paths remain allowed since the user owns their own config).
        from xpcsjax.utils.path_validation import validate_save_path

        validate_save_path(
            cache_path, allowed_extensions=(".npz",), require_parent_exists=False
        )

        # If user provided a direct NPZ path, prefer it
        direct_path = os.path.join(data_folder, data_file) if data_file else ""
        if direct_path.endswith(".npz") and os.path.exists(direct_path):
            logger.info(f"Loading data from NPZ override: {direct_path}")
            data = self._load_from_cache(direct_path)

        # Otherwise, try cache then raw HDF
        elif (
            os.path.exists(cache_path)
            and self.v2_config.get("cache_strategy", "intelligent") != "none"
        ):
            logger.info(f"Loading cached data from: {cache_path}")
            data = self._load_from_cache(cache_path)
        else:
            # Load from raw HDF file
            hdf_path = os.path.join(data_folder, data_file)
            if not os.path.exists(hdf_path):
                raise FileNotFoundError(
                    f"Neither cache file {cache_path} nor HDF file {hdf_path} exists",
                )

            logger.info(f"Loading raw data from: {hdf_path}")
            data = self._load_from_hdf(hdf_path)

            # Save to cache if caching enabled
            if self.v2_config.get("cache_strategy", "intelligent") != "none":
                logger.info(f"Saving processed data to cache: {cache_path}")
                self._save_to_cache(data, cache_path)

            # Generate text files
            self._save_text_files(data)

        # Initialize quality control if enabled
        quality_controller = self._initialize_quality_control()
        quality_results = []

        # Stage 1: Raw data validation
        if quality_controller:
            raw_validation_result = quality_controller.validate_data_stage(
                data,
                quality_controller.QualityControlStage.RAW_DATA,
            )
            quality_results.append(raw_validation_result)

            # Apply auto-repair if data was modified
            if raw_validation_result.data_modified:
                logger.info("Raw data was modified by quality control auto-repair")

        # Apply filtering with quality control validation
        if quality_controller:
            filtered_validation_result = quality_controller.validate_data_stage(
                data,
                quality_controller.QualityControlStage.FILTERED_DATA,
                previous_result=quality_results[-1] if quality_results else None,
            )
            quality_results.append(filtered_validation_result)

        # Apply preprocessing pipeline if enabled with quality control
        data = self._apply_preprocessing_pipeline(
            data,
            quality_controller,
            quality_results,
        )

        # Convert to target array format (JAX or numpy)
        data = self._convert_arrays_to_target_format(data)

        # Apply mandatory diagonal correction (post-load for consistent behavior)
        # Uses unified diagonal_correction module (v2.14.2+)
        logger.debug("Applying mandatory diagonal correction to correlation matrices")
        if HAS_DIAGONAL_CORRECTION:
            data["c2_exp"] = apply_diagonal_correction_batch(data["c2_exp"])
        else:
            # Fallback to local implementation if unified module not available
            data["c2_exp"] = self._correct_diagonal_batch(data["c2_exp"])

        # Final quality control validation
        if quality_controller:
            final_validation_result = quality_controller.validate_data_stage(
                data,
                quality_controller.QualityControlStage.FINAL_DATA,
                previous_result=quality_results[-1] if quality_results else None,
            )
            quality_results.append(final_validation_result)

            # Generate quality report only when explicitly requested (--plot-experimental-data)
            # Do NOT generate reports during normal optimization runs
            if self.generate_quality_reports and self.v2_config.get(
                "quality_control",
                {},
            ).get("generate_reports", True):
                quality_report = quality_controller.generate_quality_report(
                    quality_results,
                    self._get_quality_report_path(),
                )
                logger.info(
                    f"Quality report generated with overall status: {quality_report['overall_summary']['status']}",
                )

        # Perform legacy validation if enabled
        validation_settings = self._should_perform_validation()
        if any(validation_settings.values()) and not quality_controller:
            self._validate_loaded_data(data, validation_settings)

        logger.info(
            f"Data loaded successfully - shapes: q{data['wavevector_q_list'].shape}, "
            f"phi{data['phi_angles_list'].shape}, c2{data['c2_exp'].shape}",
        )

        return data

    @log_performance(threshold=0.2)
    def _load_from_cache(self, cache_path: str) -> dict[str, Any]:
        """Load data from NPZ cache file with q-vector validation.

        Returns 1D time arrays for NLSQ (meshgrids generated on demand).
        Only supports new 1D array cache format. Old 2D caches must be regenerated.

        Cache files live in config-controlled paths, so this loader treats them
        as untrusted input: ``allow_pickle=False`` blocks object deserialization,
        metadata is read from a JSON-encoded scalar (``cache_metadata_json``),
        and legacy object-array ``cache_metadata`` is refused.
        """
        with np.load(cache_path, allow_pickle=False, mmap_mode="r") as data:
            if "cache_metadata_json" in data:
                metadata_text = str(np.asarray(data["cache_metadata_json"]).item())
                try:
                    metadata = json.loads(metadata_text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cache {cache_path} has malformed cache_metadata_json "
                        f"(not valid JSON): {exc}"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Cache {cache_path}: cache_metadata_json must encode a "
                        f"JSON object, got {type(metadata).__name__}"
                    )
                self._validate_cache_q_vector(metadata)
                logger.debug(f"Cache metadata validation passed: {metadata}")
            elif "cache_metadata" in data.files:
                # Legacy object-serialized metadata is a trust-boundary problem
                # (arbitrary code via deserialization from a config-controlled
                # path). Refuse it; the user must regenerate the cache.
                raise ValueError(
                    f"Cache {cache_path} uses the legacy 'cache_metadata' "
                    "object-array format, which xpcsjax refuses to deserialize "
                    "for safety. Delete the cache file and regenerate; the new "
                    "format stores metadata as JSON under "
                    "'cache_metadata_json'."
                )

            # Extract correlation data — np.array() copies from mmap before
            # the context manager closes the file (prevents dangling mmap views).
            # allow_pickle=False causes object-dtype arrays to raise here; we
            # surface that as a clearer error rather than letting numpy's
            # internal message leak.
            try:
                c2_exp = np.array(data["c2_exp"])
                t1 = np.array(data["t1"])
                t2 = np.array(data["t2"])
                wavevector_q_list = np.array(data["wavevector_q_list"])
                phi_angles_list = np.array(data["phi_angles_list"])
            except ValueError as exc:
                raise ValueError(
                    f"Cache {cache_path} contains an object-dtype array under "
                    "a data key, which is not allowed (allow_pickle=False). "
                    "Delete the cache file and regenerate."
                ) from exc

            # Reject old 2D meshgrid cache format
            if t1.ndim == 2 or t2.ndim == 2:
                raise ValueError(
                    f"Old 2D meshgrid cache format detected in {cache_path}. "
                    "Please delete the cache file and regenerate with current code. "
                    "New cache format uses 1D time arrays."
                )

            return {
                "wavevector_q_list": wavevector_q_list,
                "phi_angles_list": phi_angles_list,
                "t1": t1,  # 1D array: [0, dt, 2*dt, ...]
                "t2": t2,  # 1D array: [0, dt, 2*dt, ...]
                "c2_exp": c2_exp,
            }

    @log_performance(threshold=1.0)
    def _load_from_hdf(self, hdf_path: str) -> dict[str, Any]:
        """Load and process data from HDF5 file."""
        # T037: Add log_phase for data loading with memory tracking
        with log_phase("hdf5_data_loading", logger=logger, track_memory=True) as phase:
            # Detect format
            logger.debug("Starting HDF5 format detection")
            format_type = self._detect_format(hdf_path)
            logger.info(f"Detected format: {format_type}")

            # Load based on format
            if format_type == "aps_old":
                data = self._load_aps_old_format(hdf_path)
            elif format_type == "aps_u":
                data = self._load_aps_u_format(hdf_path)
            else:
                raise XPCSDataFormatError(f"Unsupported format: {format_type}")

        logger.info(
            f"HDF5 loading completed in {phase.duration:.2f}s, "
            f"peak memory: {phase.memory_peak_gb:.2f} GB"
            if phase.memory_peak_gb
            else f"HDF5 loading completed in {phase.duration:.2f}s"
        )
        return data

    @log_performance(threshold=0.1)
    def _detect_format(self, hdf_path: str) -> str:
        """Detect whether HDF5 file is APS old or APS-U new format.

        Returns:
            "aps_u" for APS-U format
            "aps_old" for APS old format
            "unknown" for unrecognized or empty files
        """
        with h5py.File(hdf_path, "r") as f:
            # Check for APS-U format keys
            if (
                "xpcs" in f
                and "qmap" in f["xpcs"]
                and "dynamic_v_list_dim0" in f["xpcs/qmap"]
                and "twotime" in f["xpcs"]
                and "correlation_map" in f["xpcs/twotime"]
            ):
                return "aps_u"

            # Check for APS old format keys
            elif (
                "xpcs" in f
                and "dqlist" in f["xpcs"]
                and "dphilist" in f["xpcs"]
                and "exchange" in f
                and "C2T_all" in f["exchange"]
            ):
                return "aps_old"

            else:
                # Log the top-level keys for debugging unrecognized formats
                top_keys = list(f.keys())
                logger.warning(
                    f"Unrecognized HDF5 format: top-level keys={top_keys}. "
                    "Expected APS-U (xpcs/twotime/correlation_map) or "
                    "APS old (xpcs/dqlist + exchange/C2T_all)."
                )
                return "unknown"

    @log_performance(threshold=0.8)
    def _load_aps_old_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS old format HDF5 file.

        Optimization (v2.9.1): Uses selective HDF5 reads when quality filtering
        is disabled. Instead of loading all matrices upfront, we:
        1. First determine which indices are needed based on q-selection
        2. Only load those specific matrices from HDF5

        This reduces I/O by up to 98% for typical datasets where only ~23 of
        ~1150 matrices are actually used.
        """
        with h5py.File(
            hdf_path,
            "r",
            rdcc_nbytes=_HDF5_RDCC_NBYTES,
            rdcc_nslots=_HDF5_RDCC_NSLOTS,
            rdcc_w0=_HDF5_RDCC_W0,
        ) as f:
            # Load q and phi lists (small metadata - always needed)
            dqlist = f["xpcs/dqlist"][0, :]  # Shape (1, N) -> (N,)
            dphilist = f["xpcs/dphilist"][0, :]  # Shape (1, N) -> (N,)

            # Load correlation data from exchange/C2T_all
            c2t_group = f["exchange/C2T_all"]
            # APS old format: keys are in HDF5 creation order, which IS the correct
            # positional order matching dqlist/dphilist indices. Do NOT sort — integer
            # keys like "1","2","10" sort lexicographically wrong. The APS-U path uses
            # sorted() because it has zero-padded keys (c2_00001, c2_00002, …).
            c2_keys = list(c2t_group.keys())
            if not c2_keys:
                raise ValueError(
                    f"APS old-format HDF5 file contains no correlation matrices "
                    f"in 'exchange/C2T_all': {hdf_path}"
                )

            # Check if quality-based filtering is enabled (requires loading all matrices)
            filtering_config = self.config.get("data_filtering", {})
            quality_filtering_enabled = filtering_config.get(
                "enabled", False
            ) and filtering_config.get("quality_filtering", {}).get("enabled", False)

            # Select optimal q-vector first (doesn't require matrices)
            logger.debug("Selecting optimal q-vector for caching")
            selected_q_idx = self._select_optimal_wavevector(dqlist)
            selected_q = dqlist[selected_q_idx]

            # Calculate q-vector tolerance as fraction of selected q-vector
            q_tolerance_fraction = self.config.get("q_tolerance_fraction", 0.1)
            q_tolerance = selected_q * q_tolerance_fraction
            q_matching_indices = np.where(np.abs(dqlist - selected_q) <= q_tolerance)[0]

            # If we still get too few phi angles, expand the search
            if len(q_matching_indices) < 5:
                # Sort by distance from selected q and take closest N entries
                q_distances = np.abs(dqlist - selected_q)
                closest_indices = np.argsort(q_distances)
                # Take up to 10 closest q-vectors to ensure good phi angle coverage
                n_desired = min(10, len(closest_indices))
                q_matching_indices_list = [int(i) for i in closest_indices[:n_desired]]
                q_matching_indices = np.array(q_matching_indices_list, dtype=int)
                logger.debug(
                    f"Expanded selection to {len(q_matching_indices)} closest q-vectors for better phi coverage",
                )

            logger.debug(
                f"Selected {len(q_matching_indices)} (q,phi) pairs with q-range: "
                f"{dqlist[q_matching_indices].min():.6f} - {dqlist[q_matching_indices].max():.6f} AA^-1",
            )

            if quality_filtering_enabled:
                # Two-pass optimization: metadata filter first, then load + quality filter
                # Pass 1: phi/q filtering without loading matrices (metadata only)
                logger.debug(
                    "Quality filtering enabled - running metadata-only pre-filter"
                )
                metadata_indices = self._get_selected_indices(
                    dqlist,
                    dphilist,
                    None,  # No matrices needed for phi-only filtering
                )

                # Narrow to candidates via q + phi intersection
                if metadata_indices is not None:
                    candidate_indices = np.intersect1d(
                        q_matching_indices, metadata_indices
                    )
                else:
                    candidate_indices = q_matching_indices

                logger.debug(
                    f"Pre-filter: {len(c2_keys)} total -> {len(candidate_indices)} candidates "
                    f"({len(candidate_indices) / len(c2_keys) * 100:.1f}% I/O reduction)"
                )

                # Pass 2: load only candidate matrices from HDF5
                candidate_matrices = []
                for idx in candidate_indices:
                    key = c2_keys[int(idx)]
                    c2_half = c2t_group[key][()]
                    c2_full = self._reconstruct_full_matrix(c2_half)
                    candidate_matrices.append(c2_full)

                # Apply quality filtering on the loaded subset
                quality_indices = self._get_selected_indices(
                    dqlist[candidate_indices],
                    dphilist[candidate_indices],
                    candidate_matrices,
                )

                # Map quality filter results back to original indices
                if quality_indices is not None:
                    final_indices = candidate_indices[quality_indices]
                    selected_c2_matrices = [
                        candidate_matrices[i] for i in quality_indices
                    ]
                    logger.debug(
                        f"After quality filtering: {len(candidate_indices)} -> {len(final_indices)} matrices",
                    )
                else:
                    final_indices = candidate_indices
                    selected_c2_matrices = candidate_matrices
            else:
                # OPTIMIZATION: No quality filtering - selective HDF5 reads
                # Only load the matrices we actually need (up to 98% I/O reduction)
                logger.debug("Applying phi-only filtering (no quality filtering)")
                selected_indices = self._get_selected_indices(
                    dqlist,
                    dphilist,
                    None,  # Don't pass matrices - not needed for phi-only filtering
                )

                # Apply additional phi filtering if enabled
                if selected_indices is not None:
                    final_indices = np.intersect1d(q_matching_indices, selected_indices)
                    logger.debug(
                        f"After phi filtering: {len(q_matching_indices)} -> {len(final_indices)} matrices",
                    )
                else:
                    final_indices = q_matching_indices
                    logger.debug(
                        f"No phi filtering - using all {len(final_indices)} (q,phi) pairs",
                    )

                # Selective load: only read the matrices we need.
                # C1 perf: pre-allocate a single C-order output buffer and write
                # each reconstructed matrix directly, eliminating the Python-list
                # accumulation + np.array() re-stack copy (~30-50% peak-RSS saving).
                logger.info(
                    f"Selective HDF5 read: loading {len(final_indices)} of {len(c2_keys)} matrices "
                    f"({len(final_indices) / len(c2_keys) * 100:.1f}% I/O)"
                )
                n_sel = len(final_indices)
                # Read first matrix to get the time-axis dimension without storing it.
                # Preserve the source dtype (do NOT force float64): _reconstruct_full_matrix
                # did the c2_half + c2_half.T arithmetic in the stored dtype, and parity is
                # bit-exact — upcasting here would change the reconstructed bits.
                _probe_half = c2t_group[c2_keys[int(final_indices[0])]][()]
                _n_t = _probe_half.shape[0]
                _check_frame_count(int(_n_t), source="HDF5 correlation dataset")
                c2_matrices_array = np.empty((n_sel, _n_t, _n_t), dtype=_probe_half.dtype, order="C")
                # Write the already-read probe matrix into slot 0 (exact same arithmetic
                # as _reconstruct_full_matrix: c2_half + c2_half.T, diagonal /= 2).
                c2_matrices_array[0] = _probe_half + _probe_half.T
                _diag_idx = np.diag_indices(_n_t)
                c2_matrices_array[0][_diag_idx] /= 2
                del _probe_half
                # Load remaining matrices directly into pre-allocated slots.
                for _out_i, idx in enumerate(final_indices[1:], start=1):
                    key = c2_keys[int(idx)]
                    _c2_half = c2t_group[key][()]
                    c2_matrices_array[_out_i] = _c2_half + _c2_half.T
                    c2_matrices_array[_out_i][_diag_idx] /= 2

            # Extract metadata for final indices
            filtered_dqlist = dqlist[final_indices]
            filtered_dphilist = dphilist[final_indices]
            # c2_matrices_array already built (pre-allocated in no-quality-filter path;
            # stacked from candidate_matrices list in the quality-filter path below)
            if quality_filtering_enabled:
                c2_matrices_array = np.array(selected_c2_matrices)

            # Apply frame slicing to selected q-vector data
            logger.debug(
                f"Applying frame slicing to selected q-vector data: shape {c2_matrices_array.shape}",
            )
            c2_exp = self._apply_frame_slicing_to_selected_q(c2_matrices_array)

            # Calculate 1D time array (meshgrids generated by NLSQ as needed)
            time_1d = self._calculate_time_arrays(c2_exp.shape[-1])

            return {
                "wavevector_q_list": filtered_dqlist,  # Selected q-vectors (may be multiple for APS old)
                "phi_angles_list": filtered_dphilist,  # Corresponding phi angles
                "t1": time_1d,  # 1D time array starting from 0: [0, dt, 2*dt, ...]
                "t2": time_1d.copy(),  # Independent copy (prevent aliasing mutation)
                "c2_exp": c2_exp,  # Shape: (n_selected_pairs, sliced_frames, sliced_frames)
            }

    @log_performance(threshold=0.8)
    def _load_aps_u_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS-U new format HDF5 file using processed_bins mapping."""
        with h5py.File(
            hdf_path,
            "r",
            rdcc_nbytes=_HDF5_RDCC_NBYTES,
            rdcc_nslots=_HDF5_RDCC_NSLOTS,
            rdcc_w0=_HDF5_RDCC_W0,
        ) as f:
            # Load the processed_bins mapping - this tells us which (q,phi) pairs have correlation data
            processed_bins = f["xpcs/twotime/processed_bins"][()]

            # Load the q and phi lists
            q_values = f["xpcs/qmap/dynamic_v_list_dim0"][()]  # All q values
            phi_values = f["xpcs/qmap/dynamic_v_list_dim1"][
                ()
            ]  # All phi values available

            n_q = len(q_values)
            n_phi = len(phi_values)

            logger.debug(f"APS-U format: {n_q} q-values, {n_phi} phi-values")
            logger.debug(f"Q range: {q_values.min():.6f} to {q_values.max():.6f} A^-1")
            logger.debug(f"Phi values: {phi_values}")
            logger.debug(
                f"Processed bins: {len(processed_bins)} correlation matrices available",
            )

            # The processed_bins represent which (q,phi) combinations have correlation data
            # We need to map these to actual (q,phi) pairs using the grid structure
            # For APS-U format: bin_idx = processed_bin - 1; q_idx = bin_idx // n_phi; phi_idx = bin_idx % n_phi
            qphi_pairs = []
            valid_bin_indices = []

            for i, processed_bin in enumerate(processed_bins):
                bin_idx = processed_bin - 1  # Convert to 0-based
                q_idx = bin_idx // n_phi
                phi_idx = bin_idx % n_phi

                # Check if indices are valid
                if 0 <= q_idx < n_q and 0 <= phi_idx < n_phi:
                    q_val = q_values[q_idx]
                    phi_val = phi_values[phi_idx]
                    qphi_pairs.append((q_val, phi_val))
                    valid_bin_indices.append(
                        i,
                    )  # Track which correlation matrix this corresponds to
                else:
                    logger.warning(
                        f"Invalid bin mapping: processed_bin={processed_bin}, q_idx={q_idx}, phi_idx={phi_idx}",
                    )

            if len(qphi_pairs) == 0:
                raise XPCSDataFormatError(
                    "No valid (q,phi) pairs found from processed_bins mapping",
                )

            # Convert to arrays for processing
            qphi_array = np.array(qphi_pairs)
            filtered_dqlist = qphi_array[:, 0]  # q values for valid pairs
            filtered_dphilist = qphi_array[:, 1]  # phi values for valid pairs

            logger.debug(
                f"Extracted {len(valid_bin_indices)} valid (q,phi) pairs from processed_bins",
            )

            # Load correlation matrices - only for the valid bins
            corr_group = f["xpcs/twotime/correlation_map"]
            c2_keys = sorted(
                corr_group.keys(),
            )  # Sort alphabetically (which works for c2_00001 format)

            logger.debug(
                f"Loading {len(valid_bin_indices)} correlation matrices corresponding to valid (q,phi) pairs",
            )
            c2_matrices_for_filtering = []

            # Load only the correlation matrices that correspond to valid (q,phi) pairs
            for bin_idx in valid_bin_indices:
                if bin_idx < len(c2_keys):
                    key = c2_keys[bin_idx]
                    c2_half = corr_group[key][()]  # Key is already a string
                    # Reconstruct full matrix from half matrix
                    c2_full = self._reconstruct_full_matrix(c2_half)
                    c2_matrices_for_filtering.append(c2_full)
                else:
                    logger.warning(
                        f"Matrix index {bin_idx} exceeds available matrices ({len(c2_keys)})",
                    )

            # Ensure we have consistent array sizes
            min_count = min(len(c2_matrices_for_filtering), len(filtered_dqlist))
            if len(c2_matrices_for_filtering) != len(filtered_dqlist):
                n_matrices = len(c2_matrices_for_filtering)
                n_pairs = len(filtered_dqlist)
                n_discarded = abs(n_matrices - n_pairs)
                logger.warning(
                    f"APS-U matrix/pair count mismatch: {n_matrices} matrices vs "
                    f"{n_pairs} (q,phi) pairs - truncating to {min_count} entries, "
                    f"discarding {n_discarded} unmatched {'matrices' if n_matrices > n_pairs else '(q,phi) pairs'}. "
                    "Check HDF5 file integrity."
                )
                c2_matrices_for_filtering = c2_matrices_for_filtering[:min_count]
                filtered_dqlist = filtered_dqlist[:min_count]
                filtered_dphilist = filtered_dphilist[:min_count]

            # Apply comprehensive data filtering
            logger.debug("Applying comprehensive data filtering")
            selected_indices = self._get_selected_indices(
                filtered_dqlist,
                filtered_dphilist,
                c2_matrices_for_filtering,
            )

            # Select optimal q-vector (closest match) from the filtered data
            selected_q_idx = self._select_optimal_wavevector(filtered_dqlist)
            selected_q = filtered_dqlist[selected_q_idx]

            logger.debug(
                f"Selected optimal q-vector: {selected_q:.6f} AA^-1 (index {selected_q_idx})",
            )

            # Find all (q,phi) pairs matching the selected q-vector
            q_matching_indices = np.where(np.abs(filtered_dqlist - selected_q) < 1e-10)[
                0
            ]
            logger.debug(
                f"Found {len(q_matching_indices)} (q,phi) pairs matching selected q-vector",
            )

            # If phi filtering was applied, intersect with q-vector selection
            if selected_indices is not None:
                # Keep only indices that match both q-vector selection AND phi filtering
                final_indices = np.intersect1d(q_matching_indices, selected_indices)
                logger.debug(
                    f"After intersecting with phi filtering: {len(final_indices)} pairs remain",
                )
            else:
                # No phi filtering, use all pairs for selected q-vector
                final_indices = q_matching_indices
                logger.debug(
                    f"No phi filtering applied - using all {len(final_indices)} pairs for selected q-vector",
                )

            # Extract data for selected indices
            if len(final_indices) == 0:
                logger.warning(
                    "No valid indices found, using first available entry as fallback",
                )
                final_indices = np.array([0], dtype=int)

            # Use final indices for both (q,phi) pairs and correlation matrices
            final_dqlist = filtered_dqlist[final_indices]
            final_dphilist = filtered_dphilist[final_indices]

            logger.debug(f"Final selection: {len(final_indices)} correlation matrices")

            # C1 perf: pre-allocate a single C-order output buffer and write each
            # selected matrix directly, eliminating the Python-list + np.array() re-stack
            # copy (~30-50% peak-RSS saving at 23M-point scale).
            _n_sel_u = len(final_indices)
            if _n_sel_u == 0:
                # Fallback already handled above (final_indices = [0]); guard here for safety.
                c2_matrices_array = np.empty((0,), dtype=np.float64)
            else:
                # Preserve source dtype (original was np.array(c2_matrices)); forcing
                # float64 here would change reconstructed bits vs the parity baseline.
                _first_mat = np.asarray(c2_matrices_for_filtering[int(final_indices[0])])
                _n_t_u = _first_mat.shape[0]
                _check_frame_count(int(_n_t_u), source="HDF5 correlation dataset")
                c2_matrices_array = np.empty(
                    (_n_sel_u, _n_t_u, _n_t_u), dtype=_first_mat.dtype, order="C"
                )
                c2_matrices_array[0] = _first_mat
                del _first_mat
                for _out_j, _sel_i in enumerate(final_indices[1:], start=1):
                    c2_matrices_array[_out_j] = np.asarray(
                        c2_matrices_for_filtering[int(_sel_i)]
                    )

            # Apply frame slicing to the selected q-vector data
            c2_exp = self._apply_frame_slicing_to_selected_q(c2_matrices_array)

            # Calculate 1D time array (meshgrids generated by NLSQ as needed)
            time_1d = self._calculate_time_arrays(c2_exp.shape[-1])

            return {
                "wavevector_q_list": final_dqlist,
                "phi_angles_list": final_dphilist,
                "t1": time_1d,  # 1D time array starting from 0: [0, dt, 2*dt, ...]
                "t2": time_1d.copy(),  # Independent copy (prevent aliasing mutation)
                "c2_exp": c2_exp,
            }

    def _reconstruct_full_matrix(self, c2_half: NDArray) -> NDArray:
        """Reconstruct full correlation matrix from half matrix (APS storage format).

        Based on pyXPCSViewer's approach:
        c2 = c2_half + c2_half.T
        c2[diag] /= 2

        Note: Diagonal correction is now applied post-load for consistent behavior.
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for matrix reconstruction")
        c2_full = c2_half + c2_half.T
        # Correct diagonal (was doubled in addition)
        diag_indices = np.diag_indices(c2_half.shape[0])
        c2_full[diag_indices] /= 2

        return c2_full  # type: ignore[no-any-return]

    def _correct_diagonal(self, c2_mat: NDArray) -> NDArray:
        """Apply diagonal correction to correlation matrix.

        .. deprecated:: 2.16.0
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction`
            instead. This method is kept for backward compatibility only.

        Based on pyXPCSViewer's correct_diagonal_c2 function.
        Handles both JAX and NumPy arrays.
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for diagonal correction")
        size = c2_mat.shape[0]
        side_band = c2_mat[(np.arange(size - 1), np.arange(1, size))]

        # Create diagonal values using the same array type as input
        if HAS_JAX and hasattr(c2_mat, "device"):  # JAX array
            diag_val = jnp.zeros(size, dtype=c2_mat.dtype)
            diag_val = diag_val.at[:-1].add(side_band)
            diag_val = diag_val.at[1:].add(side_band)
            norm = jnp.ones(size, dtype=c2_mat.dtype)
            norm = norm.at[1:-1].set(2)
            # Update diagonal using JAX immutable operations
            diag_indices = np.diag_indices(size)
            c2_corrected = c2_mat.at[diag_indices].set(diag_val / norm)  # type: ignore
            return c2_corrected  # type: ignore
        else:  # NumPy array
            diag_val = np.zeros(size)
            diag_val[:-1] += side_band
            diag_val[1:] += side_band
            norm = np.ones(size)
            norm[1:-1] = 2
            # Only copy if array is read-only (e.g., from mmap)
            if not c2_mat.flags.writeable:
                c2_corrected = c2_mat.copy()
            else:
                c2_corrected = c2_mat
            # Use fill_diagonal for efficient in-place update
            np.fill_diagonal(c2_corrected, diag_val / norm)
            return c2_corrected

    # Performance Optimization (Spec 006 - FR-006, FR-006a): Batch diagonal correction
    def _correct_diagonal_batch(self, c2_matrices: NDArray) -> NDArray:
        """Apply diagonal correction to all matrices in batch.

        .. deprecated:: 2.16.0
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction_batch`
            instead. This method is kept for backward compatibility only.

        Performance Optimization (Spec 006 - FR-006):
        Pre-allocates output array and uses direct assignment instead of
        list append pattern. Expected memory reduction: 30%.

        Args:
            c2_matrices: Correlation matrices, shape (n_phi, n_t1, n_t2)

        Returns:
            Corrected matrices with same shape
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for diagonal correction")
        n_phi = c2_matrices.shape[0]
        size = c2_matrices.shape[1]

        # FR-006: Pre-allocate output array (avoid list append)
        if HAS_JAX and hasattr(c2_matrices, "device"):
            # JAX path: use vmap for vectorized correction (FR-006a)
            return self._correct_diagonal_batch_jax(c2_matrices)  # type: ignore
        else:
            # NumPy path: pre-allocate and direct assignment
            c2_corrected = np.empty_like(c2_matrices)

            # Pre-compute normalization array (reused for all matrices)
            norm = np.ones(size)
            norm[1:-1] = 2

            # Pre-compute index arrays
            idx_upper = np.arange(size - 1)
            idx_lower = np.arange(1, size)
            diag_indices = np.diag_indices(size)

            for i in range(n_phi):
                c2_mat = c2_matrices[i]
                # Extract side band values
                side_band = c2_mat[(idx_upper, idx_lower)]

                # Compute diagonal values
                diag_val = np.zeros(size)
                diag_val[:-1] += side_band
                diag_val[1:] += side_band

                # Copy and apply correction (direct assignment)
                c2_corrected[i] = c2_mat.copy()
                c2_corrected[i][diag_indices] = diag_val / norm

            return c2_corrected

    def _correct_diagonal_batch_jax(self, c2_matrices: Any) -> Any:
        """Vectorized diagonal correction using JAX vmap.

        Performance Optimization (Spec 006 - FR-006a):
        Uses jax.vmap for parallel diagonal correction across all angles.
        Expected speedup: 2-4x for diagonal correction.

        Args:
            c2_matrices: JAX array of shape (n_phi, n_t1, n_t2)

        Returns:
            Corrected matrices with same shape
        """
        if not HAS_JAX:
            raise RuntimeError("JAX is required for JAX diagonal correction")
        import jax

        size = c2_matrices.shape[1]

        # Pre-compute normalization and indices once
        norm = jnp.ones(size)
        norm = norm.at[1:-1].set(2)
        idx_upper = jnp.arange(size - 1)
        idx_lower = jnp.arange(1, size)

        def correct_single(c2_mat: Any) -> Any:
            """Correct diagonal for a single matrix."""
            # Extract side band
            side_band = c2_mat[idx_upper, idx_lower]

            # Compute diagonal values
            diag_val = jnp.zeros(size, dtype=c2_mat.dtype)
            diag_val = diag_val.at[:-1].add(side_band)
            diag_val = diag_val.at[1:].add(side_band)

            # Apply correction
            diag_indices = jnp.diag_indices(size)
            return c2_mat.at[diag_indices].set(diag_val / norm)

        # Vectorize over all matrices
        correct_all = jax.vmap(correct_single)
        return correct_all(c2_matrices)

    def _get_selected_indices(
        self,
        dqlist: NDArray,
        dphilist: NDArray,
        correlation_matrices: list[NDArray] | None = None,
    ) -> NDArray | None:
        """Get indices for comprehensive data filtering based on configuration.

        Implements multi-criteria filtering including:
        - Q-range filtering based on wavevector values
        - Phi angle filtering (integrates with existing phi_filtering.py)
        - Quality-based filtering using correlation matrix properties
        - Frame-based filtering with configurable criteria
        - Combined filtering with AND/OR logic

        Args:
            dqlist: Array of q-values (wavevector magnitudes)
            dphilist: Array of phi angles in degrees
            correlation_matrices: Optional list of correlation matrices for quality filtering

        Returns:
            Array of selected indices, or None if no filtering is applied
        """
        try:
            # Import filtering utilities
            from xpcsjax.data.filtering_utils import DataFilteringError, XPCSDataFilter

            # Check if filtering is enabled
            filtering_config = self.config.get("data_filtering", {})
            if not filtering_config.get("enabled", False):
                logger.debug("Data filtering disabled in configuration")
                return None

            logger.info(
                f"Applying comprehensive data filtering to {len(dqlist)} data points",
            )

            # Initialize data filter
            data_filter = XPCSDataFilter(self.config)

            # Apply comprehensive filtering
            filtering_result = data_filter.apply_filtering(
                dqlist,
                dphilist,
                correlation_matrices,
            )

            # Log filtering statistics
            if filtering_result.filter_statistics:
                logger.info("Filtering statistics:")
                for filter_name, stats in filtering_result.filter_statistics.items():
                    if isinstance(stats, dict) and "selected_count" in stats:
                        logger.info(
                            f"  {filter_name}: {stats['selected_count']} selected "
                            f"({stats.get('selection_fraction', 0.0):.2%})",
                        )

            # Handle warnings and errors
            if filtering_result.warnings:
                for warning in filtering_result.warnings:
                    logger.warning(f"Data filtering warning: {warning}")

            if filtering_result.errors:
                for error in filtering_result.errors:
                    logger.error(f"Data filtering error: {error}")
                if not filtering_result.fallback_used:
                    raise DataFilteringError(
                        f"Data filtering failed: {filtering_result.errors}",
                    )

            # Log final result
            if filtering_result.selected_indices is not None:
                selected_count = len(filtering_result.selected_indices)
                total_count = len(dqlist)
                selection_fraction = (
                    selected_count / total_count if total_count > 0 else 0.0
                )

                logger.info(
                    f"Data filtering completed: {selected_count}/{total_count} "
                    f"data points selected ({selection_fraction:.2%})",
                )

                if filtering_result.fallback_used:
                    logger.warning("Filtering used fallback - all data points included")

                # Additional integration with phi filtering for compatibility
                selected_indices = self._integrate_with_phi_filtering(
                    filtering_result.selected_indices,
                    dphilist,
                    filtering_result,
                )

                return selected_indices
            else:
                logger.warning(
                    "No data filtering criteria matched - returning all angles. "
                    "Check filter configuration if this is unexpected."
                )
                return None

        except ImportError as e:
            logger.warning(
                f"Filtering utilities not available: {e}. Skipping data filtering.",
            )
            return None
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Data filtering failed: {e}")

            # Check if we should fallback or raise
            fallback_on_empty = filtering_config.get("fallback_on_empty", True)
            if fallback_on_empty:
                logger.warning("Falling back to no filtering due to error")
                return None
            else:
                raise XPCSDataFormatError(f"Data filtering failed: {e}") from e

    def _integrate_with_phi_filtering(
        self,
        selected_indices: NDArray,
        dphilist: NDArray,
        filtering_result: Any,
    ) -> NDArray:
        """Integrate with existing phi filtering system for backward compatibility.

        This method ensures that the new filtering system works well with
        existing phi angle filtering configurations and provides consistent results.
        """
        try:
            # Import existing phi filtering system
            from xpcsjax.data.phi_filtering import PhiAngleFilter

            # Check if phi filtering was already applied in the main filtering
            if "phi_range" in filtering_result.filters_applied:
                logger.debug("Phi filtering already applied in main filtering system")
                return selected_indices

            # Check for legacy phi filtering configuration
            optimization_config = self.config.get("optimization_config", {})
            angle_filtering = optimization_config.get("angle_filtering", {})

            if not angle_filtering.get("enabled", False):
                logger.debug("Legacy phi filtering not enabled")
                return selected_indices

            # Apply legacy phi filtering to already filtered data
            selected_phi_angles = dphilist[selected_indices]

            phi_filter = PhiAngleFilter(self.config)
            phi_indices, filtered_angles = phi_filter.filter_angles_for_optimization(
                selected_phi_angles,
            )

            # Map back to original indices
            final_selected_indices = selected_indices[phi_indices]

            logger.info(
                f"Legacy phi filtering applied: {len(final_selected_indices)} "
                f"out of {len(selected_indices)} filtered indices selected",
            )

            return final_selected_indices

        except ImportError:
            logger.debug(
                "Phi filtering system not available - using original selection",
            )
            return selected_indices
        except (TypeError, IndexError, KeyError) as e:
            logger.warning(
                f"Phi filtering integration failed: {e} - using original selection",
            )
            return selected_indices

    def _select_optimal_wavevector(self, dqlist: NDArray) -> int:
        """Select q-vector index closest to config value (no tolerance).

        Args:
            dqlist: Array of available q-vector values

        Returns:
            Index of selected q-vector in dqlist
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for wavevector selection")
        # Get target q-vector from configuration
        scattering_config = self.analyzer_config.get("scattering", {})
        config_q = scattering_config.get("wavevector_q", 0.0054)

        logger.debug(f"Target q-vector: {config_q:.6f} A^-1")

        # Find closest q-vector to target
        closest_idx = int(np.argmin(np.abs(dqlist - config_q)))
        selected_q = dqlist[closest_idx]
        deviation = abs(selected_q - config_q)

        logger.info(
            f"Selected closest q-vector: {selected_q:.6f} AA^-1 (target: {config_q:.6f} AA^-1, index: {closest_idx}, deviation: {deviation:.6f} AA^-1)",
        )

        return closest_idx

    def _apply_frame_slicing_to_selected_q(self, c2_matrices: NDArray) -> NDArray:
        """Apply frame slicing to already q-filtered correlation matrices.

        Args:
            c2_matrices: Correlation matrices for selected q-vector, shape (n_phi, full_frames, full_frames)

        Returns:
            Frame-sliced correlation matrices, shape (n_phi, sliced_frames, sliced_frames)
        """
        raw_start_frame = self.analyzer_config.get("start_frame", 1)
        if raw_start_frame < 1:
            logger.warning(f"start_frame={raw_start_frame} < 1, clamping to 1")
            raw_start_frame = 1
        start_frame = raw_start_frame - 1  # Convert to 0-based indexing
        end_frame = self.analyzer_config.get("end_frame", -1)
        if end_frame < 0:
            end_frame = c2_matrices.shape[-1]

        # Validate frame parameters
        max_frames = c2_matrices.shape[-1]
        if start_frame < 0:
            logger.warning(f"start_frame adjusted to 0 (was {start_frame + 1})")
            start_frame = 0
        if end_frame > max_frames:
            original_end_frame = end_frame
            end_frame = max_frames
            logger.warning(
                f"end_frame adjusted to {max_frames} (was {original_end_frame})"
            )

        # Apply frame slicing if needed
        if start_frame > 0 or end_frame < max_frames:
            c2_exp = c2_matrices[:, start_frame:end_frame, start_frame:end_frame]
            sliced_frames = end_frame - start_frame
            logger.debug(
                f"Applied frame slicing: [{start_frame}:{end_frame}] -> shape {c2_exp.shape}",
            )
            logger.debug(
                f"Frame reduction: {max_frames}x{max_frames} -> {sliced_frames}x{sliced_frames}",
            )
        else:
            c2_exp = c2_matrices
            logger.debug("No frame slicing needed - using full range")

        return c2_exp

    def _calculate_time_arrays(self, matrix_size: int) -> NDArray:
        """Calculate 1D time array for correlation analysis.

        Returns 1D array that is converted to 2D meshgrids by the NLSQ wrapper
        as needed.

        Time starts from 0 (frame 0 corresponds to t=0). The t=0 exclusion
        for D(t) singularity prevention is handled during analysis, not caching.

        Args:
            matrix_size: Number of time points (frames after slicing)

        Returns:
            1D time array: [0, dt, 2*dt, ..., (N-1)*dt]
        """
        dt = self.analyzer_config.get("dt", 1.0)

        # Create 1D time array starting from 0
        # Last point at index (N-1), not N
        time_max = dt * (matrix_size - 1)
        time_1d = np.linspace(0, time_max, matrix_size)

        return time_1d

    @log_performance(threshold=0.3)
    def _save_to_cache(self, data: dict[str, Any], cache_path: str) -> None:
        """Save processed data to NPZ cache file with q-vector metadata."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for cache saving")
        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Convert JAX arrays back to numpy for caching
        cache_data: dict[str, Any] = {}
        for key, value in data.items():
            if HAS_JAX and hasattr(value, "device"):  # JAX array
                cache_data[key] = np.array(value)
            else:
                cache_data[key] = value

        # Add cache metadata for q-vector validation
        scattering_config = self.analyzer_config.get("scattering", {})
        config_q = scattering_config.get("wavevector_q", 0.0054)

        # Calculate actual q-vector stats from cached data
        q_values = cache_data["wavevector_q_list"]
        # Use nan-safe variants: q_values from HDF5 may contain NaN for bad pixels.
        actual_q = float(np.nanmean(q_values)) if len(q_values) > 0 else config_q
        q_variance = float(np.nanstd(q_values)) if len(q_values) > 1 else 0.0

        cache_metadata = {
            "config_wavevector_q": float(config_q),
            "actual_wavevector_q": actual_q,
            "q_variance": q_variance,
            "q_count": len(q_values),
            "start_frame": self.analyzer_config.get("start_frame", 1),
            # Normalize end_frame=-1 sentinel to actual last frame
            "end_frame": (
                self.analyzer_config.get("end_frame")
                if self.analyzer_config.get("end_frame", -1) != -1
                else (
                    cache_data["c2_exp"].shape[-1]
                    + self.analyzer_config.get("start_frame", 1)
                    - 1
                )
            ),
            "phi_count": len(cache_data["phi_angles_list"]),
            "cache_version": "2.0",
            "selective_q_caching": True,
        }

        # Metadata is stored as a JSON-encoded scalar (not a Python dict via
        # object pickling) so the loader can read it with allow_pickle=False.
        cache_data["cache_metadata_json"] = np.asarray(json.dumps(cache_metadata))

        # Save with compression if specified
        if self.exp_config.get("cache_compression", True):
            np.savez_compressed(cache_path, **cache_data)
        else:
            np.savez(cache_path, **cache_data)

        # Log cache statistics
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        logger.info(f"Cache saved: {cache_path}")
        logger.info(
            f"Cache size: {file_size_mb:.2f} MB, Q-vectors: {cache_metadata['q_count']}, Phi angles: {cache_metadata['phi_count']}",
        )
        logger.debug(f"Q-vector: {actual_q:.6f} +/- {q_variance:.6f} A^-1")

    def _validate_cache_q_vector(self, cache_metadata: dict[str, Any]) -> None:
        """Validate that cached q-vector is compatible with current configuration."""
        scattering_config = self.analyzer_config.get("scattering", {})
        current_config_q = scattering_config.get("wavevector_q", 0.0054)
        cached_config_q = cache_metadata.get("config_wavevector_q", current_config_q)

        # Check if configuration q-vectors match (within floating point precision)
        if abs(current_config_q - cached_config_q) > 1e-8:
            logger.warning(
                f"Cache q-vector mismatch: current={current_config_q:.6f}, cached={cached_config_q:.6f} AA^-1",
            )

        # Check if cache uses selective q-caching (v2.0 feature)
        is_selective = cache_metadata.get("selective_q_caching", False)
        if not is_selective:
            logger.warning(
                "Loading legacy cache without selective q-vector optimization",
            )
        else:
            actual_q = cache_metadata.get("actual_wavevector_q", cached_config_q)
            q_variance = cache_metadata.get("q_variance", 0.0)
            logger.debug(
                f"Validated selective cache: q={actual_q:.6f} +/- {q_variance:.6f} AA^-1",
            )

    def _generate_cache_path(self) -> Path:
        """Generate cache file path based on current configuration."""
        # Get data folder and cache configuration
        data_folder = self.exp_config.get("data_folder_path", "./data/")
        cache_folder = self.exp_config.get("cache_file_path", data_folder)

        # Get frame parameters
        start_frame = self.analyzer_config.get("start_frame", 1)
        end_frame = self.analyzer_config.get("end_frame", 8000)

        # Get wavevector_q for cache filename
        scattering_config = self.analyzer_config.get("scattering", {})
        wavevector_q = scattering_config.get("wavevector_q", 0.0054)

        # Construct cache filename (using string.Template for safety)
        cache_template = _migrate_cache_template(
            self.exp_config.get(
                "cache_filename_template",
                "cached_c2_frames_${start_frame}_${end_frame}.npz",
            )
        )

        tmpl = string.Template(cache_template)
        cache_filename = tmpl.safe_substitute(
            start_frame=start_frame,
            end_frame=end_frame,
            wavevector_q=f"{wavevector_q:.4f}",
        )
        _assert_safe_cache_filename(cache_filename)

        return Path(str(cache_folder)) / cache_filename  # type: ignore[no-any-return]

    @log_performance(threshold=0.1)
    def _save_text_files(self, data: dict[str, Any]) -> None:
        """Save phi_angles and wavevector_q lists to text files."""
        # Get output directory
        phi_folder = self.exp_config.get("phi_angles_path", "./")
        data_folder = self.exp_config.get("data_folder_path", "./")

        # Convert JAX arrays to numpy for text file saving
        phi_angles = (
            np.array(data["phi_angles_list"]) if HAS_JAX else data["phi_angles_list"]
        )
        q_values = (
            np.array(data["wavevector_q_list"])
            if HAS_JAX
            else data["wavevector_q_list"]
        )

        # Save phi angles list
        phi_file = os.path.join(phi_folder, "phi_angles_list.txt")
        phi_dir = os.path.dirname(phi_file)
        if phi_dir:
            os.makedirs(phi_dir, exist_ok=True)

        try:
            np.savetxt(
                phi_file,
                phi_angles,
                fmt="%.6f",
                header="Phi angles (degrees)",
                comments="# ",
            )

            # Save wavevector q list
            q_file = os.path.join(data_folder, "wavevector_q_list.txt")
            np.savetxt(
                q_file,
                q_values,
                fmt="%.8e",
                header="Wavevector q (1/Angstrom)",
                comments="# ",
            )

            logger.debug(f"Text files saved: {phi_file}, {q_file}")
        except OSError as e:
            logger.warning(f"Could not save text files (non-fatal): {e}")

    def _validate_loaded_data(
        self,
        data: dict[str, Any],
        validation_settings: dict[str, bool],
    ) -> None:
        """Perform validation on loaded data.

        Args:
            data: Loaded data dictionary
            validation_settings: Validation configuration
        """
        if validation_settings.get("physics_checks", False):
            self._perform_physics_validation(data)

        if validation_settings.get("data_quality", False):
            self._perform_data_quality_checks(
                data,
                validation_settings.get("comprehensive", False),
            )

    def _perform_physics_validation(self, data: dict[str, Any]) -> None:
        """Perform physics-based validation using v2 PhysicsConstants."""
        if not HAS_PHYSICS_VALIDATION:
            logger.warning(
                "Physics validation requested but v2 physics module not available",
            )
            return

        # Validate q-range
        q_values = (
            np.array(data["wavevector_q_list"])
            if HAS_JAX
            else data["wavevector_q_list"]
        )
        if np.any(q_values < PhysicsConstants.Q_MIN_TYPICAL):
            logger.warning(
                f"Some q-values below typical range: {PhysicsConstants.Q_MIN_TYPICAL}",
            )
        if np.any(q_values > PhysicsConstants.Q_MAX_TYPICAL):
            logger.warning(
                f"Some q-values above typical range: {PhysicsConstants.Q_MAX_TYPICAL}",
            )

        # Validate time parameters
        dt = self.analyzer_config.get("dt", 1.0)
        if dt < PhysicsConstants.TIME_MIN_XPCS:
            logger.warning(
                f"Time step dt={dt}s below typical XPCS minimum: {PhysicsConstants.TIME_MIN_XPCS}s",
            )

        logger.info("Physics validation completed")

    def _perform_data_quality_checks(
        self,
        data: dict[str, Any],
        comprehensive: bool = False,
    ) -> None:
        """Perform data quality validation."""
        c2_exp = np.array(data["c2_exp"]) if HAS_JAX else data["c2_exp"]

        # Basic checks
        if np.any(~np.isfinite(c2_exp)):
            logger.error("Correlation data contains non-finite values (NaN or Inf)")

        if np.any(c2_exp < 0):
            logger.warning("Correlation data contains negative values")

        # Check for reasonable correlation values (should be around 1.0 at t=0)
        diagonal_values = np.array([c2_exp[i].diagonal() for i in range(len(c2_exp))])
        mean_diagonal = np.nanmean(diagonal_values[:, 0])  # t=0 correlation
        if not (0.5 < mean_diagonal < 2.0):
            logger.warning(
                f"Unusual t=0 correlation value: {mean_diagonal:.3f} (expected ~1.0)",
            )

        if comprehensive:
            # Additional comprehensive checks
            logger.info("Performing comprehensive data quality analysis...")

            # Check correlation decay
            decay_rates = []
            for i in range(len(c2_exp)):
                diag = c2_exp[i].diagonal()
                if len(diag) > 10:
                    decay_rate = (diag[0] - diag[10]) / diag[0]
                    decay_rates.append(decay_rate)

            if decay_rates:
                mean_decay = np.nanmean(decay_rates)
                logger.info(
                    f"Mean correlation decay over 10 time steps: {mean_decay:.3f}",
                )

        logger.info("Data quality validation completed")

    def _initialize_quality_control(self) -> Any | None:
        """Initialize quality control system if enabled."""
        try:
            quality_config = self.config.get("quality_control", {})
            if not quality_config.get("enabled", False):
                logger.debug("Quality control disabled in configuration")
                return None

            # Import quality control system
            from xpcsjax.data.quality_controller import (
                DataQualityController,
                QualityControlStage,
            )

            logger.info("Initializing data quality control system")
            controller = DataQualityController(self.config)

            # Store reference to stage enum for convenience
            controller.QualityControlStage = QualityControlStage  # type: ignore

            return controller

        except ImportError as e:
            logger.warning(f"Quality control system not available: {e}")
            return None
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Narrowed from broad Exception: only catch configuration/setup errors.
            # MemoryError, SystemExit, KeyboardInterrupt must propagate.
            logger.error(f"Failed to initialize quality control: {e}")
            return None

    def _get_quality_report_path(self) -> str:
        """Generate path for quality control report."""
        data_folder = self.exp_config.get("data_folder_path", "./")
        data_file = self.exp_config.get("data_file_name", "unknown")
        data_file_base = os.path.splitext(data_file)[0]

        # Create quality reports subdirectory
        quality_dir = os.path.join(data_folder, "quality_reports")
        os.makedirs(quality_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = int(time.time())
        quality_filename = f"{data_file_base}_quality_report_{timestamp}.json"

        return os.path.join(quality_dir, quality_filename)

    @log_performance(threshold=0.5)
    def _apply_preprocessing_pipeline(
        self,
        data: dict[str, Any],
        quality_controller: Any | None = None,
        quality_results: list | None = None,
    ) -> dict[str, Any]:
        """Apply preprocessing pipeline to loaded data if enabled.

        Args:
            data: Raw data loaded from HDF5 files

        Returns:
            Processed data after applying preprocessing pipeline
        """
        try:
            # Check if preprocessing is enabled
            preprocessing_config = self.config.get("preprocessing", {})
            if not preprocessing_config.get("enabled", False):
                logger.debug("Preprocessing pipeline disabled in configuration")
                return data

            logger.info("Applying preprocessing pipeline to loaded data")

            # Import preprocessing pipeline
            from xpcsjax.data.preprocessing import PreprocessingPipeline

            # Create and execute preprocessing pipeline
            pipeline = PreprocessingPipeline(self.config)
            result = pipeline.process(data)

            if result.success:
                logger.info("Preprocessing pipeline completed successfully")
                logger.info(f"Pipeline stages executed: {len(result.stage_results)}")

                # Log stage results
                successful_stages = sum(result.stage_results.values())
                total_stages = len(result.stage_results)
                logger.info(f"Successful stages: {successful_stages}/{total_stages}")

                # Quality control validation after preprocessing
                if quality_controller and quality_results:
                    preprocessing_validation_result = (
                        quality_controller.validate_data_stage(
                            result.data,
                            quality_controller.QualityControlStage.PREPROCESSED_DATA,
                            previous_result=(
                                quality_results[-1] if quality_results else None
                            ),
                        )
                    )
                    quality_results.append(preprocessing_validation_result)

                    if not preprocessing_validation_result.passed:
                        logger.warning(
                            f"Preprocessing quality validation failed: score={preprocessing_validation_result.metrics.overall_score:.1f}",
                        )

                # Save provenance if requested
                if preprocessing_config.get("save_provenance", False):
                    provenance_path = self._get_provenance_path()
                    pipeline.save_provenance(result.provenance, provenance_path)

                # Log warnings if any
                if result.provenance.warnings:
                    for warning in result.provenance.warnings:
                        logger.warning(f"Preprocessing warning: {warning}")

                return result.data
            else:
                logger.error("Preprocessing pipeline failed")

                # Log errors
                for error in result.provenance.errors:
                    logger.error(f"Preprocessing error: {error}")

                # Return original data if fallback is enabled
                if preprocessing_config.get("fallback_on_failure", True):
                    logger.warning(
                        "Falling back to original data after preprocessing failure",
                    )
                    return data
                else:
                    raise XPCSDataFormatError(
                        "Preprocessing pipeline failed and fallback disabled",
                    )

        except ImportError as e:
            logger.warning(
                f"Preprocessing pipeline not available: {e}. Using original data.",
            )
            return data
        except (ValueError, KeyError, IndexError, RuntimeError) as e:
            # Narrowed from broad Exception: only catch expected processing errors.
            # Programming bugs (AttributeError, TypeError) and system errors
            # (MemoryError, KeyboardInterrupt) must propagate without swallowing.
            logger.error(f"Unexpected error in preprocessing pipeline: {e}")

            # Check fallback setting
            preprocessing_config = self.config.get("preprocessing", {})
            if preprocessing_config.get("fallback_on_failure", True):
                logger.warning(
                    "Falling back to original data after preprocessing error",
                )
                return data
            else:
                raise XPCSDataFormatError(f"Preprocessing pipeline failed: {e}") from e

    def _get_provenance_path(self) -> str:
        """Generate path for saving preprocessing provenance."""
        # Use data folder as base
        data_folder = self.exp_config.get("data_folder_path", "./")

        # Create provenance subdirectory
        provenance_dir = os.path.join(data_folder, "preprocessing_provenance")
        os.makedirs(provenance_dir, exist_ok=True)

        # Generate filename based on data file and timestamp
        data_file = self.exp_config.get("data_file_name", "unknown")
        data_file_base = os.path.splitext(data_file)[0]
        timestamp = int(time.time())

        provenance_filename = (
            f"{data_file_base}_preprocessing_provenance_{timestamp}.json"
        )
        return os.path.join(provenance_dir, provenance_filename)


# Convenience function for simple usage
@log_performance(threshold=1.0)
def load_xpcs_data(
    config_path: str | dict | None = None,
    config_dict: dict | None = None,
) -> XpcsDataset:
    """Convenience function to load XPCS data from configuration file or dict.

    Supports both YAML and JSON configuration files with auto-detection,
    or direct configuration dictionary for programmatic use (backward compatible).

    Args:
        config_path: Path to YAML/JSON config file, OR dict for backward compatibility
        config_dict: Configuration dictionary (alternative to config_path)

    Returns:
        Dictionary containing loaded experimental data with JAX arrays when available

    Example:
        >>> # From config file
        >>> data = load_xpcs_data(config_path="xpcs_config.yaml")
        >>> print(data.keys())
        dict_keys(['wavevector_q_list', 'phi_angles_list', 't1', 't2', 'c2_exp'])

        >>> # From dict (backward compatible - positional)
        >>> config = {"data_file": "experiment.h5", "analysis_mode": "static_isotropic"}
        >>> data = load_xpcs_data(config)

        >>> # From dict (keyword argument)
        >>> data = load_xpcs_data(config_dict=config)
    """
    # Backward compatibility: if config_path is a dict, treat it as config_dict
    if isinstance(config_path, dict):
        if config_dict is not None:
            raise ValueError(
                "Cannot provide both config_path as dict and config_dict parameter"
            )
        config_dict = config_path
        config_path = None

    loader = XPCSDataLoader(config_path=config_path, config_dict=config_dict)
    # Wrap in the typed XpcsDataset (a dict subclass): key-indexed access is
    # unchanged, but callers gain the typed .c2/.phi accessors and schema.
    from xpcsjax.data.dataset import XpcsDataset

    return XpcsDataset(loader.load_experimental_data())


# Export main classes and functions
__all__ = [
    "XPCSDataLoader",
    "load_xpcs_data",
    "XPCSDataFormatError",
    "XPCSDependencyError",
    "XPCSConfigurationError",
    "load_xpcs_config",
]
