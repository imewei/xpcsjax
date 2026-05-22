"""Unified Diagonal Correction Module.

This module provides a single source of truth for diagonal correction of
two-time correlation matrices (C₂). It consolidates implementations from:
- xpcsjax/core/physics_utils.py (JAX, basic)
- xpcsjax/data/xpcs_loader.py (NumPy/JAX, basic, batch)
- xpcsjax/data/preprocessing.py (NumPy, basic/statistical/interpolation)

Diagonal correction removes the bright autocorrelation peak at t₁=t₂ by
interpolating diagonal values from adjacent off-diagonal elements. This is
a critical preprocessing step for XPCS analysis.

Usage:
    # Single matrix (auto-detect backend)
    from xpcsjax.core.diagonal_correction import apply_diagonal_correction
    c2_corrected = apply_diagonal_correction(c2_matrix)

    # Batch processing (auto-detect backend)
    from xpcsjax.core.diagonal_correction import apply_diagonal_correction_batch
    c2_batch_corrected = apply_diagonal_correction_batch(c2_matrices)

    # Force specific backend or method
    c2_corrected = apply_diagonal_correction(c2_matrix, method="statistical", backend="numpy")

References:
    - pyXPCSViewer: https://github.com/AdvancedPhotonSource/pyXPCSViewer
    - XPCS Analysis: He et al. PNAS 2024, doi:10.1073/pnas.2401162121

Version: 2.14.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Optional imports
try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    from jax import jit, vmap

    HAS_JAX = True
except ImportError:
    from collections.abc import Callable
    from typing import TypeVar

    HAS_JAX = False
    jnp = np  # type: ignore[misc]

    _F = TypeVar("_F", bound=Callable[..., object])

    def jit(f: _F) -> _F:  # type: ignore[no-redef]  # noqa: E731, UP047
        """No-op decorator when JAX is unavailable."""
        return f

    vmap = None  # type: ignore[assignment]

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = get_logger(__name__)

# Type aliases
Backend = Literal["numpy", "jax", "auto"]
Method = Literal["basic", "statistical", "interpolation"]
Estimator = Literal["mean", "median", "trimmed_mean"]


# =============================================================================
# PUBLIC API
# =============================================================================


def apply_diagonal_correction(
    c2_mat: ArrayLike,
    method: Method = "basic",
    backend: Backend = "auto",
    **config: Any,
) -> np.ndarray | jnp.ndarray:
    """Apply diagonal correction to a single correlation matrix.

    This function replaces the diagonal elements (t₁=t₂) with interpolated
    values from adjacent off-diagonal elements, removing the autocorrelation
    peak and isolating cross-correlation dynamics.

    Args:
        c2_mat: Two-time correlation matrix with shape (N, N).
                Must be square matrix with N >= 2.
        method: Correction method.
            - "basic": Adjacent off-diagonal interpolation (default, fastest)
            - "statistical": Robust estimators with configurable window
            - "interpolation": Linear interpolation
        backend: Array backend.
            - "auto": Detect from input type (default)
            - "numpy": Force NumPy operations
            - "jax": Force JAX operations (JIT-compiled)
        **config: Method-specific configuration options.

            For "statistical":

            - window_size (int): Window size for neighbor collection. Default: 3
            - estimator (str): "mean", "median", or "trimmed_mean". Default: "median"
            - trim_fraction (float): Trim fraction for trimmed_mean. Default: 0.2

            For "interpolation":

            - interpolation_method (str): "linear" (only supported option). Default: "linear"

    Returns:
        Corrected correlation matrix with same shape and backend as input.

    Example:
        >>> import numpy as np
        >>> c2 = np.array([[5.0, 1.2, 1.1],
        ...                [1.2, 5.0, 1.3],
        ...                [1.1, 1.3, 5.0]])
        >>> c2_corrected = apply_diagonal_correction(c2)
        >>> # Diagonal now contains interpolated values, not 5.0

    Note:
        The "basic" method is fastest and recommended for optimization loops.
        Use "statistical" or "interpolation" for data preprocessing where
        robustness to outliers is more important than speed.
    """
    # Determine backend
    actual_backend = _resolve_backend(c2_mat, backend)

    # Dispatch to appropriate implementation
    if actual_backend == "jax":
        if method != "basic":
            logger.warning(
                f"JAX backend only supports 'basic' method, got '{method}'. "
                "Using 'basic' method."
            )
        return _diagonal_correction_jax(c2_mat)
    else:
        return _diagonal_correction_numpy(c2_mat, method, config)


def apply_diagonal_correction_batch(
    c2_matrices: ArrayLike,
    method: Method = "basic",
    backend: Backend = "auto",
    **config: Any,
) -> np.ndarray | jnp.ndarray:
    """Apply diagonal correction to a batch of correlation matrices.

    Efficiently processes multiple correlation matrices, using vectorized
    operations (vmap) for JAX backend or pre-allocated arrays for NumPy.

    Args:
        c2_matrices: Batch of correlation matrices with shape (n_phi, N, N).
        method: Correction method (same as apply_diagonal_correction).
        backend: Array backend (same as apply_diagonal_correction).
        **config: Method-specific configuration (same as apply_diagonal_correction).

    Returns:
        Corrected matrices with same shape (n_phi, N, N) and backend as input.

    Example:
        >>> import numpy as np
        >>> # 3 angles, each with 100x100 correlation matrix
        >>> c2_batch = np.random.randn(3, 100, 100)
        >>> c2_corrected = apply_diagonal_correction_batch(c2_batch)
        >>> c2_corrected.shape
        (3, 100, 100)

    Performance:
        - JAX backend: Uses jax.vmap for parallel processing (2-4x speedup)
        - NumPy backend: Pre-allocates output array, reuses index arrays
    """
    # Determine backend
    actual_backend = _resolve_backend(c2_matrices, backend)

    # Dispatch to appropriate implementation
    if actual_backend == "jax":
        if method != "basic":
            logger.warning(
                f"JAX backend only supports 'basic' method, got '{method}'. "
                "Using 'basic' method."
            )
        return _diagonal_correction_batch_jax(c2_matrices)
    else:
        return _diagonal_correction_batch_numpy(c2_matrices, method, config)


# =============================================================================
# BACKEND RESOLUTION
# =============================================================================


def _resolve_backend(arr: ArrayLike, backend: Backend) -> Literal["numpy", "jax"]:
    """Resolve the actual backend to use based on input and preference."""
    if backend == "jax":
        if not HAS_JAX:
            logger.warning("JAX not available, falling back to NumPy")
            return "numpy"
        return "jax"
    elif backend == "numpy":
        return "numpy"
    else:  # auto
        # Auto-detect based on input type
        # Note: NumPy 2.x arrays have .device attribute for array API compliance,
        # so we need to check the actual type, not just presence of .device
        if _is_jax_array(arr):
            return "jax"
        return "numpy"


def _is_jax_array(arr: ArrayLike) -> bool:
    """Check if array is a JAX array.

    Note: NumPy 2.x arrays have .device attribute for array API compliance,
    so we check the module name to distinguish JAX arrays from NumPy arrays.
    """
    if not HAS_JAX:
        return False
    # Check module name to handle both jax.Array and jaxlib types
    type_module = type(arr).__module__
    return type_module.startswith(("jax", "jaxlib"))


# =============================================================================
# JAX IMPLEMENTATIONS (JIT-compiled)
# =============================================================================

if HAS_JAX:

    @jit
    def _diagonal_correction_jax_core(c2_mat: jnp.ndarray) -> jnp.ndarray:
        """Core JAX implementation of basic diagonal correction (JIT-compiled).

        Algorithm:
        1. Extract side band: elements at (i, i+1) for i=0..N-2 (symmetrized)
        2. Compute diagonal values as average of adjacent off-diagonals:
           - diag[0] = side_band[0] (edge: one neighbor)
           - diag[i] = (side_band[i-1] + side_band[i]) / 2 for i=1..N-2
           - diag[N-1] = side_band[N-2] (edge: one neighbor)
        3. Replace diagonal via a single scatter .at[diag_indices].set()
        """
        size = c2_mat.shape[0]
        if size <= 1:
            return c2_mat  # Nothing to correct for 1x1 or empty matrix

        # Extract side band: off-diagonal elements adjacent to main diagonal
        indices_i = jnp.arange(size - 1)
        indices_j = jnp.arange(1, size)
        side_band = 0.5 * (c2_mat[indices_i, indices_j] + c2_mat[indices_j, indices_i])

        # Compute diagonal values directly from side_band slices (no scatter ops):
        # edges get one neighbor, interior points get average of two neighbors.
        # This replaces 2 scatter-add ops + 1 division on a zeros array.
        diag_val = jnp.concatenate(
            [
                side_band[:1],
                (side_band[:-1] + side_band[1:]) * 0.5,
                side_band[-1:],
            ]
        )

        # Replace diagonal with computed values (single scatter write)
        diag_indices = jnp.diag_indices(size)
        return c2_mat.at[diag_indices].set(diag_val)

    def _diagonal_correction_jax(c2_mat: ArrayLike) -> jnp.ndarray:
        """JAX implementation wrapper (handles type conversion)."""
        c2_jax = jnp.asarray(c2_mat)
        result: jnp.ndarray = _diagonal_correction_jax_core(c2_jax)
        return result

    # Hoist vmap to module level to prevent re-tracing on every call
    _vmapped_diagonal_correction = vmap(_diagonal_correction_jax_core, in_axes=0)

    def _diagonal_correction_batch_jax(c2_matrices: ArrayLike) -> jnp.ndarray:
        """Batch JAX implementation using vmap."""
        c2_jax = jnp.asarray(c2_matrices)
        result: jnp.ndarray = _vmapped_diagonal_correction(c2_jax)
        return result

else:
    # Fallback stubs when JAX is not available
    def _diagonal_correction_jax(c2_mat: ArrayLike) -> np.ndarray:  # type: ignore[misc]
        """Fallback to NumPy when JAX not available."""
        return _diagonal_correction_numpy(c2_mat, "basic", {})

    def _diagonal_correction_batch_jax(c2_matrices: ArrayLike) -> np.ndarray:  # type: ignore[misc]
        """Fallback to NumPy when JAX not available."""
        return _diagonal_correction_batch_numpy(c2_matrices, "basic", {})


# =============================================================================
# NUMPY IMPLEMENTATIONS
# =============================================================================


def _diagonal_correction_numpy(
    c2_mat: ArrayLike,
    method: Method = "basic",
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """NumPy implementation of diagonal correction with multiple methods."""
    config = config or {}
    c2_np = np.asarray(c2_mat)

    if method == "basic":
        return _basic_correction_numpy(c2_np)
    elif method == "statistical":
        return _statistical_correction_numpy(c2_np, config)
    elif method == "interpolation":
        return _interpolation_correction_numpy(c2_np, config)
    else:  # Defensive fallback for unknown method
        logger.warning(f"Unknown method '{method}', using 'basic'")  # type: ignore[unreachable]
        return _basic_correction_numpy(c2_np)


def _diagonal_correction_batch_numpy(
    c2_matrices: ArrayLike,
    method: Method = "basic",
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Batch NumPy implementation with pre-allocated arrays."""
    config = config or {}
    c2_np = np.asarray(c2_matrices)
    n_phi = c2_np.shape[0]
    size = c2_np.shape[1]

    # Pre-allocate output array
    c2_corrected = np.empty_like(c2_np)

    if method == "basic":
        if size <= 1:
            return c2_np.copy()

        # Optimized batch processing for basic method
        # Pre-compute normalization and index arrays (reused for all matrices)
        norm = np.ones(size)
        norm[1:-1] = 2
        idx_upper = np.arange(size - 1)
        idx_lower = np.arange(1, size)
        diag_indices = np.diag_indices(size)

        for i in range(n_phi):
            c2_mat = c2_np[i]
            # Extract side band values (average both diagonals for symmetry)
            side_band = 0.5 * (
                c2_mat[idx_upper, idx_lower] + c2_mat[idx_lower, idx_upper]
            )

            # Compute diagonal values
            diag_val = np.zeros(size)
            diag_val[:-1] += side_band
            diag_val[1:] += side_band

            # Copy and apply correction
            c2_corrected[i] = c2_mat.copy()
            c2_corrected[i][diag_indices] = diag_val / norm
    else:
        # Generic batch processing for other methods
        for i in range(n_phi):
            c2_corrected[i] = _diagonal_correction_numpy(c2_np[i], method, config)

    return c2_corrected


# =============================================================================
# METHOD IMPLEMENTATIONS (NumPy)
# =============================================================================


def _basic_correction_numpy(c2_mat: np.ndarray) -> np.ndarray:
    """Basic diagonal correction using adjacent off-diagonal interpolation.

    This is the fastest method and matches pyXPCSViewer's implementation.
    """
    size = c2_mat.shape[0]
    if size <= 1:
        return c2_mat.copy()

    # Extract side band: off-diagonal elements adjacent to main diagonal
    idx_upper = np.arange(size - 1)
    idx_lower = np.arange(1, size)
    side_band = 0.5 * (c2_mat[idx_upper, idx_lower] + c2_mat[idx_lower, idx_upper])

    # Compute diagonal values as average of adjacent off-diagonal elements
    diag_val = np.zeros(size)
    diag_val[:-1] += side_band  # Add left neighbors
    diag_val[1:] += side_band  # Add right neighbors

    # Normalize by number of neighbors (1 for edges, 2 for middle)
    norm = np.ones(size)
    norm[1:-1] = 2

    # Apply correction (always copy to avoid modifying input)
    c2_corrected = c2_mat.copy()
    np.fill_diagonal(c2_corrected, diag_val / norm)

    return c2_corrected


def _statistical_correction_numpy(
    c2_mat: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Statistical diagonal correction using robust estimators.

    Collects neighboring off-diagonal values within a window and applies
    a statistical estimator (mean, median, or trimmed mean).
    """
    c2_corrected = c2_mat.copy()
    size = c2_mat.shape[0]

    # Configuration
    window_size = config.get("window_size", 3)
    estimator: Estimator = config.get("estimator", "median")
    trim_fraction = config.get("trim_fraction", 0.2)

    for i in range(size):
        # Collect neighboring off-diagonal values
        neighbors = []
        for offset in range(1, min(window_size + 1, size)):
            if i - offset >= 0:
                neighbors.append(c2_mat[i - offset, i])
                neighbors.append(c2_mat[i, i - offset])
            if i + offset < size:
                neighbors.append(c2_mat[i + offset, i])
                neighbors.append(c2_mat[i, i + offset])

        if neighbors:
            neighbors_arr = np.array(neighbors)

            # Apply statistical estimator (NaN-safe: neighbors come from raw HDF5 data)
            if estimator == "median":
                c2_corrected[i, i] = np.nanmedian(neighbors_arr)
            elif estimator == "mean":
                c2_corrected[i, i] = np.nanmean(neighbors_arr)
            elif estimator == "trimmed_mean":
                if HAS_SCIPY:
                    # Remove NaN before trimmed mean — scipy trim_mean propagates NaN
                    finite_neighbors = neighbors_arr[np.isfinite(neighbors_arr)]
                    if finite_neighbors.size > 0:
                        c2_corrected[i, i] = stats.trim_mean(
                            finite_neighbors, trim_fraction
                        )
                    else:
                        c2_corrected[i, i] = np.nan
                else:
                    # Fallback to median if scipy not available
                    c2_corrected[i, i] = np.nanmedian(neighbors_arr)
            else:  # Defensive fallback for unknown estimator
                logger.warning(f"Unknown estimator '{estimator}', using median")  # type: ignore[unreachable]
                c2_corrected[i, i] = np.nanmedian(neighbors_arr)

    return c2_corrected


def _interpolation_correction_numpy(
    c2_mat: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Interpolation-based diagonal correction.

    Uses neighboring off-diagonal values for linear interpolation.
    """
    c2_corrected = c2_mat.copy()
    size = c2_mat.shape[0]

    if size <= 1:
        return c2_corrected

    interp_method = config.get("interpolation_method", "linear")

    for i in range(size):
        if 0 < i < size - 1:
            # Use symmetrized adjacent off-diagonal values on both sides of
            # the diagonal. This matches the basic correction's side-band
            # interpretation while keeping the interpolation method explicit.
            left = 0.5 * (c2_mat[i - 1, i] + c2_mat[i, i - 1])
            right = 0.5 * (c2_mat[i + 1, i] + c2_mat[i, i + 1])
            y_points = [left, right]

            if interp_method == "linear":
                c2_corrected[i, i] = np.nanmean(y_points)
            elif interp_method == "cubic":
                raise NotImplementedError(
                    "Cubic diagonal correction is not yet implemented. "
                    "Use interpolation_method='linear' (the only supported option)."
                )
            else:
                c2_corrected[i, i] = np.nanmean(y_points)
        elif i == 0:
            # Edge case: use next off-diagonal value
            c2_corrected[i, i] = 0.5 * (c2_mat[0, 1] + c2_mat[1, 0])
        elif i == size - 1:
            # Edge case: use previous off-diagonal value
            c2_corrected[i, i] = 0.5 * (
                c2_mat[size - 2, size - 1] + c2_mat[size - 1, size - 2]
            )

    return c2_corrected


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_diagonal_correction_methods() -> list[str]:
    """Return list of available correction methods."""
    return ["basic", "statistical", "interpolation"]


def get_available_backends() -> list[str]:
    """Return list of available backends."""
    backends = ["numpy"]
    if HAS_JAX:
        backends.append("jax")
    return backends


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Public API
    "apply_diagonal_correction",
    "apply_diagonal_correction_batch",
    # Utility functions
    "get_diagonal_correction_methods",
    "get_available_backends",
]
