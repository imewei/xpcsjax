"""Load XPCS correlation data from APS / APS-U HDF5 files into JAX-ready arrays.

The :class:`XPCSDataLoader` and the :func:`load_xpcs_data` convenience wrapper read
homodyne / heterodyne XPCS correlation matrices from disk, reconstruct the full
correlation matrix from its stored half, apply mandatory diagonal correction, and
return float64 JAX arrays (with a NumPy fallback) ready to hand to
:func:`xpcsjax.fit_nlsq`.

Notes
-----
The loader supports two on-disk formats, auto-detected from the HDF5 layout and
selected by the ``experimental_data.data_type`` config key:

* ``"aps_old"`` -- legacy APS format.
* ``"aps_u"`` -- unified (APS-U) format.

No other ``data_type`` strings are accepted.

Capabilities:

* YAML-first configuration with JSON support (see :func:`load_xpcs_config`).
* Smart NPZ caching to avoid reloading large HDF5 files.
* Auto-detection of APS vs APS-U format.
* Half-matrix reconstruction for correlation matrices.
* Mandatory diagonal correction applied post-load.
* JAX array output (float64) with NumPy fallback.
* Integration with structured logging and physics validation.

Runtime validation runs unconditionally at the I/O boundary: loaded arrays are
checked for finite values (no NaN/inf, except ``wavevector_q_list`` which
tolerates NaN — see below), square 2-D correlation matrices, bounded
allocation size, and monotonic time axes. Any violation raises
:class:`XPCSDataFormatError`.

See Also
--------
load_xpcs_data : Convenience wrapper returning a typed dataset.
load_xpcs_config : Load a YAML/JSON configuration file.
xpcsjax.fit_nlsq : Fit the loaded correlation data.
"""

from __future__ import annotations

import json
import logging
import os
import string
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xpcsjax.data.dataset import XpcsDataset
else:
    NDArray = Any

# numpy, h5py, jax, jaxlib, and pyyaml are all pyproject.toml hard dependencies
# (pyproject.toml:1-13); xpcsjax.utils.logging and xpcsjax.core.* are in-tree
# modules. None of these imports can fail in any supported install, so the
# ``except ImportError`` fallbacks/HAS_* flags previously here were dead
# branches (2026-09-15 review, finding B3) — same stance already documented
# below for xpcsjax.data.memory_manager.
import jax.numpy as jnp
import numpy as np
import yaml

from xpcsjax.core.diagonal_correction import (
    _is_jax_array,
    apply_diagonal_correction_batch,
)
from xpcsjax.core.jax_backend import jax_available
from xpcsjax.core.physics import PhysicsConstants
from xpcsjax.utils.logging import (
    get_logger,
    log_calls,
    log_exception,
    log_performance,
    log_phase,
)

_YAML_ERROR: type[Exception] = yaml.YAMLError

# xpcsjax.data.memory_manager is an internal sibling module (not an optional
# external package) and its own hard dependency, psutil, is a required
# (non-extra) install — this import cannot fail in any supported install, so
# no soft-fail guard is needed.
# D6 (2026-09-15 review): the NPZ cache load/save/validate block and the two
# HDF5 format readers were extracted into their own modules as thin moves.
# Names re-exported below for backward compatibility (existing tests import
# several of them directly off this module).
from xpcsjax.data import hdf5_readers as _hdf5_readers  # noqa: E402
from xpcsjax.data import npz_cache as _npz_cache  # noqa: E402
from xpcsjax.data.hdf5_readers import (  # noqa: E402, F401
    guard_aps_u_intermediate_allocation as _guard_aps_u_intermediate_allocation,
)
from xpcsjax.data.memory_manager import AdvancedMemoryManager  # noqa: E402
from xpcsjax.data.npz_cache import (  # noqa: E402, F401
    hash_filter_config as _hash_filter_config,
)
from xpcsjax.data.npz_cache import (  # noqa: E402
    migrate_cache_template as _migrate_cache_template,
)
from xpcsjax.data.npz_cache import (  # noqa: E402, F401
    peek_npz_array_header as _peek_npz_array_header,
)

logger = get_logger(__name__)


def _maybe_apply_mandatory_diagonal_correction(
    data: dict[str, Any],
    fallback_correct_diagonal_batch: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Apply the mandatory post-load diagonal correction, unless already done.

    Preprocessing's CORRECT_DIAGONAL stage sets ``data["_diagonal_corrected"]
    = True`` on success (xpcsjax/data/preprocessing.py's ``_execute_stage``).
    Re-applying the mandatory 'basic' correction on top of that would
    silently discard whatever method the user configured
    (statistical/interpolation) — see Finding #2 of the 2026-07-23
    debug-audit-fixes spec.
    """
    if data.get("_diagonal_corrected", False):
        logger.debug(
            "Skipping mandatory diagonal correction: preprocessing already "
            "corrected the diagonal (_diagonal_corrected=True)"
        )
        return data

    logger.debug("Applying mandatory diagonal correction to correlation matrices")
    data["c2_exp"] = apply_diagonal_correction_batch(data["c2_exp"])
    return data


class XPCSDataFormatError(Exception):
    """Raised when XPCS data format is not recognized or invalid."""


class CacheStaleError(XPCSDataFormatError):
    """Raised when a cache's source-file identity (name/size) no longer matches.

    Unlike other cache-validation failures (q-vector/filter/dt mismatches,
    which are genuine format incompatibilities the caller must resolve by
    hand), this is a cache MISS: the loader catches it, logs a warning, and
    falls through to reloading from the HDF5 source and overwriting the
    stale cache (review finding A2 follow-up, 2026-09-15). A source-file
    mtime-only change (a content-preserving touch -- ``cp`` without ``-p``,
    ``rsync``, a backup restore) does not raise this; only a name or size
    change does.
    """


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

# Upper bound on the *total* in-memory correlation buffer. ``_check_frame_count``
# bounds only the time axis; the real allocation is ``(n_matrices, n_t, n_t)``
# and a crafted file with a legal ``n_t`` but a huge matrix count can still
# demand hundreds of GB (SEC-2 / DoS). This caps the product in bytes. Datasets
# that legitimately exceed this go through the streaming/stratified path, not the
# in-memory loader, so a generous fixed ceiling is safe.
MAX_CORRELATION_ALLOC_BYTES = 64 * 1024**3  # 64 GiB


def _check_allocation_budget(n_matrices: int, n_t: int, itemsize: int, *, source: str) -> None:
    """Reject an ``(n_matrices, n_t, n_t)`` buffer that exceeds the byte ceiling.

    The per-axis :func:`_check_frame_count` cap is necessary but not sufficient:
    the quantity that actually drives the allocation is the product
    ``n_matrices * n_t * n_t * itemsize``. A file declaring a legal ``n_t`` but
    tens of thousands of correlation matrices still triggers a multi-hundred-GB
    allocation. Raises ``XPCSDataFormatError`` when the requested buffer exceeds
    :data:`MAX_CORRELATION_ALLOC_BYTES`.
    """
    if n_matrices < 0:
        raise XPCSDataFormatError(
            f"Invalid correlation matrix count {n_matrices} from {source!r} (must be non-negative)."
        )
    total_bytes = n_matrices * n_t * n_t * itemsize
    if total_bytes > MAX_CORRELATION_ALLOC_BYTES:
        raise XPCSDataFormatError(
            f"Refusing to allocate {total_bytes / 1024**3:.1f} GiB for "
            f"{n_matrices}x{n_t}x{n_t} correlation matrices from {source!r} "
            f"(exceeds the {MAX_CORRELATION_ALLOC_BYTES / 1024**3:.0f} GiB cap). "
            "Use the streaming/stratified path for legitimately huge datasets, "
            "or raise MAX_CORRELATION_ALLOC_BYTES."
        )


def _check_square_matrix(shape: tuple[int, ...], *, source: str) -> None:
    """Validate that a stored half-matrix is 2-D and square before allocation.

    The allocation derives ``(n_t, n_t)`` from ``shape[0]`` only; a non-square
    (or non-2-D) stored dataset would otherwise drive a buffer sized purely on
    the first axis and fail later inside the ``c2_half + c2_half.T``
    reconstruction. Reject it cheaply at the I/O boundary (SEC-2).
    """
    if len(shape) != 2 or shape[0] != shape[1]:
        raise XPCSDataFormatError(
            f"Correlation dataset from {source!r} has shape {shape}; expected a "
            "square 2-D half-matrix (n_t, n_t)."
        )


def _validate_loaded_arrays(data: dict[str, Any], *, source: str) -> None:
    """Hard-fail (DATA-2) on corrupt loaded correlation arrays at the I/O boundary.

    Enforces the project's I/O contract *unconditionally*, rather than behind an
    opt-in flag:

    * **Finite values** — no NaN/inf in any loaded array, EXCEPT
      ``wavevector_q_list`` which tolerates NaN (legitimate bad-pixel masking,
      one entry per (q, phi) pair) but still hard-rejects inf. Corrupt data
      must stop the run, not silently drive a numerically wrong fit.
    * **Bounded buffer** — a 3-D ``c2_exp`` is re-checked for square trailing
      axes, frame count, and total allocation budget. This also guards the
      ``.npz`` cache path (threat-03), which otherwise bypasses the HDF5
      allocation guard.
    * **Monotonic time axes** — ``t1``/``t2`` are ``[0, dt, 2*dt, ...]`` by
      construction and must be non-decreasing. (Monotonicity is intentionally
      NOT asserted on ``wavevector_q_list``: in XPCS the q-list holds one entry
      per (q, phi) pair and is legitimately non-monotonic.)

    Raises ``XPCSDataFormatError`` on any violation.
    """
    for key in ("c2_exp", "t1", "t2", "phi_angles_list"):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size and not np.all(np.isfinite(arr)):
            raise XPCSDataFormatError(
                f"{key} from {source!r} contains NaN/inf values; refusing to "
                "proceed with corrupt correlation data."
            )

    # wavevector_q_list gets its own, NaN-tolerant check: NaN there is
    # legitimate (one entry per (q, phi) pair; a bad/masked detector pixel
    # legitimately produces NaN at that pair's q-value), but inf/-inf still
    # indicates corrupt data and must keep hard-failing.
    if "wavevector_q_list" in data:
        q_arr = np.asarray(data["wavevector_q_list"])
        if q_arr.size and np.isinf(q_arr).any():
            raise XPCSDataFormatError(
                f"wavevector_q_list from {source!r} contains inf values; "
                "refusing to proceed with corrupt correlation data."
            )

    c2 = np.asarray(data.get("c2_exp"))
    if c2.ndim == 3:
        _check_square_matrix((int(c2.shape[-2]), int(c2.shape[-1])), source=source)
        _check_frame_count(int(c2.shape[-1]), source=source)
        _check_allocation_budget(
            int(c2.shape[0]), int(c2.shape[-1]), c2.dtype.itemsize, source=source
        )
    else:
        # A non-3-D shape (2-D, 1-D, 4-D, ...) is a malformed I/O-boundary input
        # and must hard-fail here rather than silently skip the square/frame/
        # allocation guards and surface a confusing downstream shape error later.
        raise XPCSDataFormatError(
            f"c2_exp from {source!r} has shape {c2.shape}; expected a 3-D "
            "(n_phi, n_t, n_t) correlation-matrix stack."
        )

    for key in ("t1", "t2"):
        if key not in data:
            continue
        t = np.asarray(data[key])
        if t.ndim == 1 and t.size > 1 and not np.all(np.diff(t) >= 0):
            raise XPCSDataFormatError(
                f"{key} from {source!r} is not non-decreasing; the correlation "
                "time axis must be monotonic."
            )


def _check_frame_count(n_frames: int, *, source: str) -> None:
    """Validate a correlation-matrix time dimension before allocating on it.

    Raises ``XPCSDataFormatError`` if ``n_frames`` is non-positive or exceeds
    :data:`MAX_CORRELATION_FRAMES`. This guards the I/O boundary against an
    unbounded allocation driven by an untrusted file's declared shape.
    """
    if n_frames <= 0:
        raise XPCSDataFormatError(
            f"Invalid correlation frame count {n_frames} from {source!r} (must be positive)."
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
    null byte. Platform-agnostic by construction.
    """
    if any(tok in name for tok in _UNSAFE_FILENAME_TOKENS):
        raise ValueError(f"Unsafe cache filename from template: {name!r}")


def load_xpcs_config(config_path: str | Path) -> dict[str, Any]:
    """Load an XPCS configuration from a YAML or JSON file.

    YAML is the primary format; JSON files are loaded and returned with the same
    nested structure (no schema conversion is performed). The file extension
    selects the parser: ``.yaml`` / ``.yml`` use PyYAML's safe loader, ``.json``
    uses :mod:`json`.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to a YAML (``.yaml`` / ``.yml``) or JSON (``.json``) configuration
        file.

    Returns
    -------
    dict
        The parsed configuration dictionary.

    Raises
    ------
    XPCSConfigurationError
        If the file does not exist, has an unsupported extension, or fails to
        parse.
    XPCSDependencyError
        If a YAML file is given but PyYAML is not installed.

    See Also
    --------
    XPCSDataLoader : Consumes the returned configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")

    try:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            # Native YAML loading
            with open(config_path, encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            # A09-1: basename only at INFO; full path at DEBUG (no dir-layout disclosure).
            logger.info(f"Loaded YAML configuration: {Path(config_path).name}")
            logger.debug(f"Config full path: {config_path}")
            return config

        if config_path.suffix.lower() == ".json":
            # JSON loading with structure conversion
            with open(config_path, encoding="utf-8") as f:
                json_config: dict[str, Any] = json.load(f)

            logger.info(f"Loaded JSON configuration (converted to YAML): {Path(config_path).name}")
            logger.debug(f"Config full path: {config_path}")
            logger.info("Consider migrating to YAML format for better readability")

            # Convert JSON structure to YAML-style (for now, keep identical structure)
            # In future, can add more sophisticated conversion via existing converter
            return json_config

        raise XPCSConfigurationError(
            f"Unsupported configuration format: {config_path.suffix}. "
            f"Supported formats: .yaml, .yml, .json",
        )

    except (_YAML_ERROR, json.JSONDecodeError) as e:
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
        """Initialize the loader from a config file path or an in-memory dict.

        Exactly one of ``config_path`` or ``config_dict`` must be provided.

        Parameters
        ----------
        config_path : str, optional
            Path to a YAML or JSON configuration file. Mutually exclusive with
            ``config_dict``.
        config_dict : dict, optional
            Configuration dictionary, used in place of ``config_path`` for
            programmatic use. Mutually exclusive with ``config_path``.
        configure_logging : bool, default True
            Whether to apply the logging configuration found in the config.
        generate_quality_reports : bool, default False
            Whether to write data-quality reports during loading. Reports are
            only generated when explicitly requested (e.g. when plotting
            experimental data), never during normal optimization runs.

        Raises
        ------
        ValueError
            If both or neither of ``config_path`` and ``config_dict`` are given.
        XPCSDependencyError
            If a required dependency (e.g. ``h5py``) is unavailable.
        XPCSConfigurationError
            If the configuration is invalid.
        """
        # Check for required dependencies
        self._check_dependencies()

        # Store whether to generate quality reports (only for --plot-experimental-data)
        self.generate_quality_reports = generate_quality_reports

        # DATA-1: detectable record of degraded fallbacks (filtering/preprocessing
        # crashes that silently substitute the optimizer's input). Downstream code
        # can inspect this to tell a crash-fallback from an intended no-op.
        self.load_degradations: list[str] = []

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
        # Resolve ${ENV_VAR} and a leading ~ in the data folder path so configs
        # can stay machine-agnostic (e.g. fixtures point at ${XPCSJAX_DATA_ROOT}).
        # expandvars/expanduser are no-ops on plain absolute paths, so this cannot
        # perturb any existing config or the rtol=1e-10 parity baselines. The
        # traversal/security checks downstream run on the *expanded* value.
        _folder = self.exp_config.get("data_folder_path")
        if isinstance(_folder, str) and _folder:
            self.exp_config["data_folder_path"] = os.path.expanduser(os.path.expandvars(_folder))
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
        """Check for required dependencies.

        numpy and h5py are pyproject.toml hard dependencies, so a missing
        install fails at the top-of-module ``import numpy``/``import h5py``
        (well before this constructor runs), not here. Kept as a no-op call
        site for API stability; :class:`XPCSDependencyError` stays public.
        """

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
            "memory_pressure_monitoring": True,
        }

        if "performance" not in self.config:
            self.config["performance"] = {}

        for key, default_value in performance_defaults.items():
            if key not in self.config["performance"]:
                self.config["performance"][key] = default_value

    def _init_performance_components(self) -> None:
        """Initialize performance optimization components.

        ``memory_manager`` is constructed: its background pressure-monitor
        thread has a real, documented side effect (WARNING logs when memory
        pressure crosses the 75%/90% thresholds — see
        ``docs/source/theory/heterodyne_memory_strategy.rst``), regardless of
        whether anyone calls a method on the returned object.

        A ``performance_engine`` attribute (the multi-level cache,
        memory-mapped chunked loading, and prefetching engine formerly in
        ``xpcsjax.data.performance_engine``) used to also live here, always
        ``None`` — an audit found XPCSDataLoader never called anything on it
        besides ``shutdown()`` in :meth:`close`, and every actual feature it
        offered was reachable only through the also-dead
        ``AdvancedDatasetOptimizer``, never invoked in production. Both the
        module and the attribute were removed in the 2026-09-15 review
        (finding B1).
        """
        self.memory_manager = None

        # ``performance.performance_engine_enabled`` used to gate the (deleted)
        # PerformanceEngine; the only knob left here is memory-pressure monitoring.
        performance_config = self.config.get("performance", {})
        try:
            # Initialize memory manager
            if performance_config.get("memory_pressure_monitoring", True):
                self.memory_manager = AdvancedMemoryManager(self.config)
                logger.info("Advanced memory manager initialized")

        except Exception as e:
            log_exception(
                logger,
                e,
                context={"operation": "init_performance_components"},
                level=logging.DEBUG,
            )
            logger.info("Falling back to basic optimization")
            self.memory_manager = None

    def close(self) -> None:
        """Shut down the memory manager, if constructed.

        Safe to call multiple times.
        """
        if self.memory_manager is not None:
            try:
                self.memory_manager.shutdown()
            except Exception as e:  # pragma: no cover - defensive only
                logger.warning(f"Error shutting down memory manager: {e}")
            finally:
                self.memory_manager = None

    def __enter__(self) -> XPCSDataLoader:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, calling :meth:`close`."""
        self.close()

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
            raise ValueError(f"Path traversal detected in data file path: {data_file_path}")

        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found: {Path(data_file_path).name}")
            logger.debug(f"Full path checked: {data_file_path}")
            logger.info("File will be checked again during data loading")

    def _get_output_format(self) -> str:
        """Get output array format from configuration."""
        format_val: Any = self.v2_config.get("output_format", "auto")
        return str(format_val)

    def _should_perform_validation(self) -> dict[str, bool]:
        """Get validation settings from configuration."""
        validation_level = self.v2_config.get("validation_level", "basic")
        return {
            "physics_checks": self.v2_config.get("physics_validation", False),
            "data_quality": validation_level != "none",
            "comprehensive": validation_level == "full",
        }

    def _convert_arrays_to_target_format(
        self,
        data: dict[str, NDArray],
    ) -> dict[str, Any]:
        """Convert arrays to the target format based on configuration.

        Parameters
        ----------
        data
            Dictionary with numpy arrays.

        Returns
        -------
        dict
            Dictionary with arrays in the target format (JAX or numpy).
        """
        output_format = self._get_output_format()

        if output_format == "jax" and jax_available:
            logger.debug("Converting arrays to JAX format")
            return {
                k: jnp.asarray(np.ascontiguousarray(v), dtype=jnp.float64)
                if isinstance(v, np.ndarray)
                else v
                for k, v in data.items()
            }

        if output_format == "auto" and jax_available:
            logger.debug("Auto-selecting JAX format (available)")
            return {
                k: jnp.asarray(np.ascontiguousarray(v), dtype=jnp.float64)
                if isinstance(v, np.ndarray)
                else v
                for k, v in data.items()
            }

        if output_format == "auto":
            logger.debug("Auto-selecting numpy format (JAX not available)")

        return data  # Keep numpy format

    @log_performance(threshold=0.5)
    def load_experimental_data(self) -> dict[str, Any]:
        """Load experimental correlation data, resolving cache then raw HDF5.

        Resolution order: a direct ``.npz`` override, then the NPZ cache (when
        caching is enabled), then the raw HDF5 file. The loaded data passes
        through optional quality control, the preprocessing pipeline, conversion
        to the target array backend (JAX or NumPy), and mandatory diagonal
        correction before being returned.

        Returns
        -------
        dict
            Mapping with at least the following keys:

            ``wavevector_q_list``
                Array of scattering wavevector ``q`` values (one entry per
                ``(q, phi)`` pair; intentionally not monotonic).
            ``phi_angles_list``
                Array of azimuthal ``phi`` angles.
            ``t1``, ``t2``
                1-D monotonic correlation time axes.
            ``c2_exp``
                Experimental correlation data, diagonal-corrected.

            When a degraded fallback occurred during loading, ``_degraded`` is
            ``True`` and ``load_degradations`` lists the per-event reasons.

        Raises
        ------
        FileNotFoundError
            If neither the cache file nor the raw HDF5 file exists.
        XPCSDataFormatError
            If the loaded arrays fail I/O-boundary validation (non-finite
            values, non-square matrices, oversized allocation, or a
            non-monotonic time axis).
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

        validate_save_path(cache_path, allowed_extensions=(".npz",), require_parent_exists=False)

        # Track the genuine pre-filter correlation-matrix count for the
        # quality-control retention metric. Reset per load; only the HDF paths
        # populate it (a cache hit has no pre-filter baseline to compare against).
        self._n_prefilter_matrices: int | None = None

        # If user provided a direct NPZ path, prefer it
        direct_path = os.path.join(data_folder, data_file) if data_file else ""
        data: dict[str, Any] | None = None
        if direct_path.endswith(".npz") and os.path.exists(direct_path):
            logger.info(f"Loading data from NPZ override: {Path(direct_path).name}")
            logger.debug(f"NPZ full path: {direct_path}")
            data = self._load_from_cache(direct_path)

        # Otherwise, try cache then raw HDF
        elif (
            os.path.exists(cache_path)
            and self.v2_config.get("cache_strategy", "intelligent") != "none"
        ):
            # A09-1: log only the basename at INFO; full path at DEBUG so log
            # artifacts don't disclose the user's home/dataset directory layout.
            logger.info(f"Loading cached data from: {Path(cache_path).name}")
            logger.debug(f"Cache full path: {cache_path}")
            try:
                data = self._load_from_cache(cache_path)
            except CacheStaleError as exc:
                # A2 follow-up: a source-file name/size mismatch is a cache
                # MISS, not a hard failure -- fall through below to reload
                # from the HDF5 and overwrite the stale cache.
                logger.warning(
                    f"{exc} Cache source changed; reloading from HDF5 and overwriting the cache."
                )

        if data is None:
            # Load from raw HDF file (fresh load, or stale-cache fallback).
            hdf_path = os.path.join(data_folder, data_file)
            if not os.path.exists(hdf_path):
                raise FileNotFoundError(
                    f"Neither cache file {cache_path} nor HDF file {hdf_path} exists",
                )

            logger.info(f"Loading raw data from: {Path(hdf_path).name}")
            logger.debug(f"HDF full path: {hdf_path}")
            data = self._load_from_hdf(hdf_path)

            # Save to cache if caching enabled
            if self.v2_config.get("cache_strategy", "intelligent") != "none":
                logger.info(f"Saving processed data to cache: {Path(cache_path).name}")
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
            # Hand the genuine pre-filter matrix count to the controller so the
            # retention metric compares post-filter vs the TRUE pre-filter count
            # (the loader already filtered `data` before either QC stage runs, so
            # the RAW-stage shape is NOT a valid baseline). None on cache hits.
            quality_controller._prefilter_matrix_count = self._n_prefilter_matrices
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

        # Apply mandatory diagonal correction (post-load for consistent behavior),
        # unless preprocessing's CORRECT_DIAGONAL stage already corrected it.
        data = _maybe_apply_mandatory_diagonal_correction(data, self._correct_diagonal_batch)

        # Final quality control validation. Initialized to None (not implicitly
        # assumed-set) so the `_degraded` check below never risks an
        # UnboundLocalError if this variable's use ever drifts out of sync
        # with the `if quality_controller:` guard that populates it
        # (CodeQL py/uninitialized-local-variable).
        final_validation_result: Any | None = None
        if quality_controller:
            final_validation_result = quality_controller.validate_data_stage(
                data,
                quality_controller.QualityControlStage.FINAL_DATA,
                previous_result=quality_results[-1] if quality_results else None,
            )
            quality_results.append(final_validation_result)

            # DATA-1: validate_data_stage's `.passed` for the FINAL_DATA stage is not
            # checked anywhere else in this method — quality control is otherwise
            # advisory-only, so a dataset that fails every quality gate would return
            # successfully with no trace beyond scattered per-issue log lines a caller
            # could easily miss. Surface it loudly and fold it into the `_degraded`
            # signal below so it is visible both in logs and to any programmatic
            # caller, without changing the (still non-raising) contract.
            if not final_validation_result.passed:
                error_issues = [
                    issue.message
                    for issue in final_validation_result.issues
                    if issue.severity == "error"
                ]
                logger.error(
                    "Final data quality check FAILED (score=%.1f, %d error issue(s)): %s "
                    "-- data is being returned anyway; inspect before trusting fit results.",
                    final_validation_result.metrics.overall_score,
                    len(error_issues),
                    "; ".join(error_issues) or "see quality report for details",
                )

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

        # DATA-1: surface the accumulated degradation signal on the returned dict so
        # a programmatic caller can gate on it rather than only watching ERROR logs.
        # `load_degradations` carries the per-event reasons (angle-filter fallbacks,
        # preprocessing crashes); `_degraded` is the single boolean to branch on.
        if self.load_degradations:
            data["load_degradations"] = list(self.load_degradations)
        data["_degraded"] = (
            bool(self.load_degradations)
            or bool(data.get("_preprocessing_degraded", False))
            or (final_validation_result is not None and not final_validation_result.passed)
        )

        return data

    @log_performance(threshold=0.2)
    def _load_from_cache(self, cache_path: str) -> dict[str, Any]:
        """Load data from NPZ cache file with q-vector validation.

        See :func:`xpcsjax.data.npz_cache.load_from_cache` (D6 extraction).
        """
        return _npz_cache.load_from_cache(self, cache_path)

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
        # DATA-2: unconditional finite/monotonic/shape validation at the I/O
        # boundary — corrupt data hard-fails here rather than silently driving
        # a numerically wrong fit.
        _validate_loaded_arrays(data, source=hdf_path)
        return data

    @log_performance(threshold=0.1)
    def _detect_format(self, hdf_path: str) -> str:
        """Detect whether an HDF5 file is APS old or APS-U new format.

        See :func:`xpcsjax.data.hdf5_readers.detect_format` (D6 extraction).
        """
        return _hdf5_readers.detect_format(hdf_path)

    @log_performance(threshold=0.8)
    def _load_aps_old_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS old format HDF5 file.

        See :func:`xpcsjax.data.hdf5_readers.load_aps_old_format` (D6 extraction).
        """
        return _hdf5_readers.load_aps_old_format(self, hdf_path)

    @log_performance(threshold=0.8)
    def _load_aps_u_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS-U new format HDF5 file using processed_bins mapping.

        See :func:`xpcsjax.data.hdf5_readers.load_aps_u_format` (D6 extraction).
        """
        return _hdf5_readers.load_aps_u_format(self, hdf_path)

    def _reconstruct_full_matrix(self, c2_half: NDArray) -> NDArray:
        """Reconstruct full correlation matrix from half matrix (APS storage format).

        Based on pyXPCSViewer's approach:
        c2 = c2_half + c2_half.T
        c2[diag] /= 2

        Note: Diagonal correction is now applied post-load for consistent behavior.
        """
        c2_full = c2_half + c2_half.T
        # Correct diagonal (was doubled in addition)
        diag_indices = np.diag_indices(c2_half.shape[0])
        c2_full[diag_indices] /= 2

        return c2_full  # type: ignore[no-any-return]

    # Performance Optimization (Spec 006 - FR-006, FR-006a): Batch diagonal correction
    def _correct_diagonal_batch(self, c2_matrices: NDArray) -> NDArray:
        """Apply diagonal correction to all matrices in a batch.

        .. deprecated::
            Use :func:`xpcsjax.core.diagonal_correction.apply_diagonal_correction_batch`
            instead. This method is kept for backward compatibility only.

        Pre-allocates the output array and uses direct assignment instead of a
        list-append pattern (Spec 006 - FR-006); expected memory reduction 30%.

        Parameters
        ----------
        c2_matrices
            Correlation matrices, shape ``(n_phi, n_t1, n_t2)``.

        Returns
        -------
        numpy.ndarray
            Corrected matrices with the same shape as the input.
        """
        n_phi = c2_matrices.shape[0]
        size = c2_matrices.shape[1]

        # FR-006: Pre-allocate output array (avoid list append)
        if _is_jax_array(c2_matrices):
            # JAX path: use vmap for vectorized correction (FR-006a)
            return self._correct_diagonal_batch_jax(c2_matrices)  # type: ignore
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
        """Apply vectorized diagonal correction using JAX ``vmap``.

        Uses :func:`jax.vmap` for parallel diagonal correction across all
        angles (Spec 006 - FR-006a); expected speedup 2-4x.

        Parameters
        ----------
        c2_matrices
            JAX array of shape ``(n_phi, n_t1, n_t2)``.

        Returns
        -------
        Any
            Corrected matrices with the same shape as the input.
        """
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
        *,
        record_degradation: bool = True,
    ) -> NDArray | None:
        """Get indices for comprehensive data filtering based on configuration.

        Implements multi-criteria filtering including:

        - Q-range filtering based on wavevector values.
        - Phi angle filtering (integrates with ``phi_filtering.py``).
        - Quality-based filtering using correlation matrix properties.
        - Frame-based filtering with configurable criteria.
        - Combined filtering with AND/OR logic.

        Parameters
        ----------
        dqlist
            Array of q-values (wavevector magnitudes).
        dphilist
            Array of phi angles in degrees.
        correlation_matrices
            Optional list of correlation matrices used for quality filtering.
        record_degradation
            Whether a fallback-to-all-data-points outcome should append to
            ``self.load_degradations`` (DATA-1 signal). The APS-U quality-
            filtering path (:mod:`xpcsjax.data.hdf5_readers`) calls this
            twice per load -- a phi/q metadata pre-filter pass, then the
            final quality-filter pass on the narrowed candidates -- against
            the SAME ``data_filtering`` config; passing ``False`` on the
            metadata pre-filter pass avoids double-recording one fallback
            (review finding, 2026-09-15).

        Returns
        -------
        numpy.ndarray or None
            Array of selected indices, or ``None`` if no filtering is applied.
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
                selection_fraction = selected_count / total_count if total_count > 0 else 0.0

                logger.info(
                    f"Data filtering completed: {selected_count}/{total_count} "
                    f"data points selected ({selection_fraction:.2%})",
                )

                if filtering_result.fallback_used and record_degradation:
                    # DATA-1 parity with the `except` branch below: a fallback
                    # substitutes ALL data points for the subset the config asked
                    # for (empty filter result, or a caught filtering error inside
                    # apply_filtering). That is the same silent substitution of the
                    # optimizer's input, so it must leave the same programmatic
                    # signal -- a bare WARNING is not distinguishable downstream.
                    self._record_degradation(
                        "data filtering fell back to all data points "
                        f"({selected_count}/{total_count}); "
                        f"reasons: {filtering_result.warnings or filtering_result.errors}"
                    )

                # Additional integration with phi filtering for compatibility
                return self._integrate_with_phi_filtering(
                    filtering_result.selected_indices,
                    dphilist,
                    filtering_result,
                )

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
                if record_degradation:
                    # DATA-1: record the degraded substitution (all angles used)
                    # instead of a bare WARNING the caller cannot distinguish.
                    self._record_degradation(
                        f"angle filtering crashed ({e}); fell back to all angles"
                    )
                return None
            raise XPCSDataFormatError(f"Data filtering failed: {e}") from e

    def _record_degradation(self, reason: str) -> None:
        """Record a degraded-fallback event so it is detectable downstream (DATA-1).

        When filtering or preprocessing crashes and the loader silently
        substitutes the optimizer's input, this leaves a programmatic signal on
        ``self.load_degradations`` and logs at ERROR (the same severity as the
        originating failure) so an automated pipeline can gate on it rather than
        proceeding blind with degraded data.
        """
        self.load_degradations.append(reason)
        logger.error("Load degradation: %s", reason)

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
        """Select the q-vector index closest to the config value (no tolerance).

        Parameters
        ----------
        dqlist
            Array of available q-vector values.

        Returns
        -------
        int
            Index of the selected q-vector within ``dqlist``.
        """
        # Get target q-vector from configuration
        scattering_config = self.analyzer_config.get("scattering", {})
        config_q = scattering_config.get("wavevector_q", 0.0054)

        logger.debug(f"Target q-vector: {config_q:.6f} A^-1")

        # Find closest q-vector to target, skipping any NaN-masked entries
        # (wavevector_q_list may legitimately contain NaN for bad-pixel masking).
        try:
            closest_idx = int(np.nanargmin(np.abs(dqlist - config_q)))
        except ValueError as e:
            raise ValueError("wavevector_q_list contains no finite q-values") from e
        selected_q = dqlist[closest_idx]
        deviation = abs(selected_q - config_q)

        logger.info(
            f"Selected closest q-vector: {selected_q:.6f} AA^-1 (target: {config_q:.6f} AA^-1, index: {closest_idx}, deviation: {deviation:.6f} AA^-1)",
        )

        return closest_idx

    def _apply_frame_slicing_to_selected_q(self, c2_matrices: NDArray) -> NDArray:
        """Apply frame slicing to already q-filtered correlation matrices.

        Parameters
        ----------
        c2_matrices
            Correlation matrices for the selected q-vector, shape
            ``(n_phi, full_frames, full_frames)``.

        Returns
        -------
        numpy.ndarray
            Frame-sliced correlation matrices, shape
            ``(n_phi, sliced_frames, sliced_frames)``.
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
            logger.warning(f"end_frame adjusted to {max_frames} (was {original_end_frame})")

        # Fail loudly on an empty frame window rather than silently returning a
        # degenerate (n_phi, 0, 0) stack. A start_frame at/after the last
        # available frame slips past both bounds checks above (the default
        # end_frame=-1 resolves to max_frames), so guard the window explicitly.
        if start_frame >= end_frame:
            raise XPCSDataFormatError(
                f"start_frame ({start_frame + 1}) is at or after the last available "
                f"frame ({max_frames}); the resulting correlation window is empty. "
                f"Choose start_frame < {max_frames}.",
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

        # Record the actually-applied (clamped) frame window in 1-based inclusive
        # coordinates so cache metadata reports the true extent instead of
        # re-deriving it from the raw (possibly out-of-range) config values.
        self._applied_start_frame = start_frame + 1  # 0-based start -> 1-based
        self._applied_end_frame = end_frame  # 0-based exclusive end == 1-based last

        return c2_exp

    def _calculate_time_arrays(self, matrix_size: int) -> NDArray:
        r"""Calculate the 1D time array for correlation analysis.

        Returns a 1D array that is converted to 2D meshgrids by the NLSQ
        wrapper as needed.

        Time starts from 0 (frame 0 corresponds to ``t=0``). The ``t=0``
        exclusion for D(t) singularity prevention is handled during analysis,
        not caching.

        Parameters
        ----------
        matrix_size
            Number of time points (frames after slicing).

        Returns
        -------
        numpy.ndarray
            1D time array ``[0, dt, 2*dt, ..., (N-1)*dt]``.
        """
        dt = self.analyzer_config.get("dt", 1.0)

        # Create 1D time array starting from 0
        # Last point at index (N-1), not N
        time_max = dt * (matrix_size - 1)
        return np.linspace(0, time_max, matrix_size)

    def _source_hdf_stat(self) -> tuple[str, int, int] | None:
        """Return the configured source HDF5 file's identity for cache keying.

        See :func:`xpcsjax.data.npz_cache.source_hdf_stat` (D6 extraction).
        """
        return _npz_cache.source_hdf_stat(self)

    @log_performance(threshold=0.3)
    def _save_to_cache(self, data: dict[str, Any], cache_path: str) -> None:
        """Save processed data to NPZ cache file with q-vector metadata.

        See :func:`xpcsjax.data.npz_cache.save_to_cache` (D6 extraction).
        """
        _npz_cache.save_to_cache(self, data, cache_path)

    @staticmethod
    def _require_nonempty_selection(final_indices: np.ndarray, *, selected_q: float) -> np.ndarray:
        """Return *final_indices*, or raise if the (q, phi) selection is empty.

        An empty selection means the configured q-vector / phi filter matched no
        data. Falling back to index 0 here would silently return correlation data
        for an unrelated q-vector (audit C9), so abort loudly instead — parity
        with the APS-old branch.
        """
        if len(final_indices) == 0:
            raise XPCSDataFormatError(
                f"No (q, phi) pairs matched the selected q-vector "
                f"{selected_q:.6f} AA^-1 after phi filtering. Check the configured "
                f"wavevector_q and phi-angle range against the dataset.",
            )
        return final_indices

    def _validate_cache_q_vector(self, cache_metadata: dict[str, Any]) -> None:
        """Validate that cached q-vector is compatible with current configuration."""
        _npz_cache.validate_cache_q_vector(self, cache_metadata)

    @log_performance(threshold=0.1)
    def _save_text_files(self, data: dict[str, Any]) -> None:
        """Save phi_angles and wavevector_q lists to text files."""
        # Get output directory
        phi_folder = self.exp_config.get("phi_angles_path", "./")
        data_folder = self.exp_config.get("data_folder_path", "./")

        # Convert JAX arrays to numpy for text file saving
        phi_angles = np.array(data["phi_angles_list"])
        q_values = np.array(data["wavevector_q_list"])

        # Route the config-controlled output directories through get_safe_output_dir
        # so a phi_angles_path / data_folder_path containing '..' cannot write these
        # side-output files outside the intended tree. Filenames are fixed. Text-file
        # writing remains non-fatal: a traversal rejection or filesystem error is
        # logged and skipped rather than aborting the data load.
        from xpcsjax.utils.path_validation import PathValidationError, get_safe_output_dir

        try:
            phi_file = get_safe_output_dir(phi_folder) / "phi_angles_list.txt"
            np.savetxt(
                phi_file,
                phi_angles,
                fmt="%.6f",
                header="Phi angles (degrees)",
                comments="# ",
            )

            # Save wavevector q list
            q_file = get_safe_output_dir(data_folder) / "wavevector_q_list.txt"
            np.savetxt(
                q_file,
                q_values,
                fmt="%.8e",
                header="Wavevector q (1/Angstrom)",
                comments="# ",
            )

            logger.debug(f"Text files saved: {phi_file.name}, {q_file.name}")
        except (OSError, PathValidationError) as e:
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
        # Validate q-range
        q_values = np.array(data["wavevector_q_list"])
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
        c2_exp = np.array(data["c2_exp"])

        # Basic checks
        if np.any(~np.isfinite(c2_exp)):
            raise XPCSDataFormatError(
                "Correlation data contains non-finite values (NaN or Inf)",
            )

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
        """Apply the preprocessing pipeline to loaded data if enabled.

        Parameters
        ----------
        data
            Raw data loaded from HDF5 files.
        quality_controller
            Optional quality controller used during preprocessing.
        quality_results
            Optional list collecting quality-check results.

        Returns
        -------
        dict
            Processed data after applying the preprocessing pipeline (returned
            unchanged when preprocessing is disabled).
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
                    preprocessing_validation_result = quality_controller.validate_data_stage(
                        result.data,
                        quality_controller.QualityControlStage.PREPROCESSED_DATA,
                        previous_result=(quality_results[-1] if quality_results else None),
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
            logger.error("Preprocessing pipeline failed")

            # Log errors
            for error in result.provenance.errors:
                logger.error(f"Preprocessing error: {error}")

            # Return original data if fallback is enabled
            if preprocessing_config.get("fallback_on_failure", True):
                # DATA-1: degraded path — the fit runs on un-preprocessed
                # data. Record it and tag the result so it is detectable.
                self._record_degradation(
                    "preprocessing pipeline failed; fell back to original data"
                )
                if isinstance(data, dict):
                    data["_preprocessing_degraded"] = True
                return data
            raise XPCSDataFormatError(
                "Preprocessing pipeline failed and fallback disabled",
            )

        except ImportError as e:
            logger.warning(f"Preprocessing pipeline not available: {e}.")
            # Check fallback setting (same gate as the other two error paths above).
            preprocessing_config = self.config.get("preprocessing", {})
            if preprocessing_config.get("fallback_on_failure", True):
                self._record_degradation(
                    f"preprocessing pipeline unavailable ({e}); fell back to original data"
                )
                if isinstance(data, dict):
                    data["_preprocessing_degraded"] = True
                return data
            raise XPCSDataFormatError(
                f"Preprocessing pipeline unavailable and fallback disabled: {e}"
            ) from e
        except (ValueError, KeyError, IndexError, RuntimeError) as e:
            # Narrowed from broad Exception: only catch expected processing errors.
            # Programming bugs (AttributeError, TypeError) and system errors
            # (MemoryError, KeyboardInterrupt) must propagate without swallowing.
            logger.error(f"Unexpected error in preprocessing pipeline: {e}")

            # Check fallback setting
            preprocessing_config = self.config.get("preprocessing", {})
            if preprocessing_config.get("fallback_on_failure", True):
                # DATA-1: degraded path — record and tag so it is detectable.
                self._record_degradation(
                    f"preprocessing pipeline crashed ({e}); fell back to original data"
                )
                if isinstance(data, dict):
                    data["_preprocessing_degraded"] = True
                return data
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

        provenance_filename = f"{data_file_base}_preprocessing_provenance_{timestamp}.json"
        return os.path.join(provenance_dir, provenance_filename)


# Convenience function for simple usage
@log_performance(threshold=1.0)
def load_xpcs_data(
    config_path: str | dict | None = None,
    config_dict: dict | None = None,
) -> XpcsDataset:
    """Load XPCS data from a configuration file or dictionary.

    Thin convenience wrapper over :class:`XPCSDataLoader`: it builds the loader,
    runs :meth:`XPCSDataLoader.load_experimental_data`, and wraps the result in a
    typed :class:`~xpcsjax.data.dataset.XpcsDataset`. The result is ready to pass
    to :func:`xpcsjax.fit_nlsq`.

    Exactly one configuration source must be supplied. For backward
    compatibility, a ``dict`` passed positionally as ``config_path`` is treated
    as ``config_dict``.

    Parameters
    ----------
    config_path : str or dict, optional
        Path to a YAML/JSON configuration file. A ``dict`` may be passed here for
        backward compatibility, in which case it is treated as ``config_dict``.
        Mutually exclusive with ``config_dict``.
    config_dict : dict, optional
        Configuration dictionary, for programmatic use. Mutually exclusive with a
        non-``None`` ``config_path``.

    Returns
    -------
    xpcsjax.data.dataset.XpcsDataset
        A ``dict`` subclass holding the loaded experimental data (float64 JAX
        arrays when JAX is available, otherwise NumPy). Key-indexed access is
        unchanged; the typed ``.c2`` / ``.phi`` / ``.t1`` / ``.t2`` accessors are
        also available. Keys include ``wavevector_q_list``, ``phi_angles_list``,
        ``t1``, ``t2``, and ``c2_exp``.

    Raises
    ------
    ValueError
        If both ``config_path`` (as a dict) and ``config_dict`` are provided, or
        if neither configuration source is given.
    FileNotFoundError
        If the configured data file cannot be found.
    XPCSDataFormatError
        If the loaded data fails I/O-boundary validation.

    See Also
    --------
    XPCSDataLoader : The underlying loader class.
    load_xpcs_config : Load a YAML/JSON configuration file.
    xpcsjax.fit_nlsq : Fit the loaded correlation data with NLSQ.

    Examples
    --------
    Load from a config file:

    >>> data = load_xpcs_data(config_path="xpcs_config.yaml")
    >>> sorted(data.keys())
    ['c2_exp', 'phi_angles_list', 't1', 't2', 'wavevector_q_list']
    >>> data.c2.shape  # typed accessor for c2_exp
    (1, 100, 100)

    Load from a dictionary (positional, backward compatible):

    >>> config = {"data_file": "experiment.h5", "analysis_mode": "static_isotropic"}
    >>> data = load_xpcs_data(config)

    Load from a dictionary (keyword argument):

    >>> data = load_xpcs_data(config_dict=config)
    """
    # Backward compatibility: if config_path is a dict, treat it as config_dict
    if isinstance(config_path, dict):
        if config_dict is not None:
            raise ValueError("Cannot provide both config_path as dict and config_dict parameter")
        config_dict = config_path
        config_path = None

    # The performance engine / memory manager are load-scoped: nothing
    # downstream needs them alive after this call returns. Without the
    # context manager, each call leaked a monitoring thread (and everything
    # it transitively keeps alive) for the life of the process.
    with XPCSDataLoader(config_path=config_path, config_dict=config_dict) as loader:
        # Wrap in the typed XpcsDataset (a dict subclass): key-indexed access
        # is unchanged, but callers gain the typed .c2/.phi accessors and
        # schema.
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
