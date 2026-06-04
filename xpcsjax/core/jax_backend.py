"""JAX Computational Backend for Homodyne
==========================================

High-performance JAX-based implementation of the core mathematical operations
for xpcsjax scattering analysis. Provides JIT-compiled functions with automatic
differentiation capabilities for optimization.

This module provides JAX-based computational kernels
that offer superior performance and automatic differentiation
for gradient-based optimization methods.

Key Features:
- JIT compilation for optimal performance
- Automatic differentiation (grad, hessian) for optimization
- Vectorized operations with vmap for parallelization
- CPU-only: optimized for multi-core NUMA architectures (no GPU/TPU)
- Memory-efficient operations for large correlation matrices
- Numerical stability enhancements

Physical Model Implementation:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²

Where g₁ = g₁_diffusion × g₁_shear captures:
- Anomalous diffusion: g₁_diff = exp[-q²/2 ∫ D(t')dt']
- Time-dependent shear: g₁_shear = [sinc(Φ)]²
"""

from collections.abc import Callable

# Handle JAX import with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    # Fallback to numpy when JAX is not available
    import numpy as jnp  # type: ignore[no-redef]

    JAX_AVAILABLE = False

    # Import NumPy-based gradients for graceful fallback. The optional
    # numpy_gradients module is not shipped (this fallback is only reachable when
    # JAX is unavailable); the try/except degrades gracefully via the
    # NUMPY_GRADIENTS_AVAILABLE flag.
    try:
        from xpcsjax.core.numpy_gradients import (  # pyright: ignore[reportMissingImports]
            numpy_gradient,
            numpy_hessian,
        )

        NUMPY_GRADIENTS_AVAILABLE = True
    except ImportError:
        NUMPY_GRADIENTS_AVAILABLE = False

    # Create fallback decorators
    def jit(func: Callable) -> Callable:  # type: ignore[no-redef,misc,unused-ignore]
        """No-op JIT decorator for NumPy fallback."""
        return func

    def vmap(func: Callable, *args: object, **kwargs: object) -> Callable:  # type: ignore[no-redef,misc,unused-ignore]
        """Simple vectorization fallback using Python loops."""

        def vectorized_func(inputs: object, *vargs: object, **vkwargs: object) -> object:
            if hasattr(inputs, "__iter__") and not isinstance(inputs, str):
                return [func(inp, *vargs, **vkwargs) for inp in inputs]
            return func(inputs, *vargs, **vkwargs)

        return vectorized_func

    def grad(func: Callable, argnums: int = 0) -> Callable:  # type: ignore[no-redef,misc,unused-ignore]
        """Intelligent fallback gradient function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_gradient_fallback(func, argnums)
        else:
            return _create_no_gradient_fallback(
                func.__name__ if hasattr(func, "__name__") else "function",
            )

    def hessian(func: Callable, argnums: int = 0) -> Callable:  # type: ignore[no-redef,misc,unused-ignore]
        """Intelligent fallback Hessian function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_hessian_fallback(func, argnums)
        else:
            return _create_no_hessian_fallback(
                func.__name__ if hasattr(func, "__name__") else "function",
            )


import threading
from collections import OrderedDict
from functools import partial, wraps
from typing import Any, cast

from xpcsjax.core.physics_utils import (
    PI,
    safe_sinc,
)
from xpcsjax.core.physics_utils import (
    calculate_diffusion_coefficient as _calculate_diffusion_coefficient_impl_jax,
)
from xpcsjax.core.physics_utils import (
    calculate_shear_rate as _calculate_shear_rate_impl_jax,
)
from xpcsjax.core.physics_utils import (
    create_time_integral_matrix as _create_time_integral_matrix_impl_jax,
)
from xpcsjax.core.physics_utils import (
    trapezoid_cumsum as _trapezoid_cumsum,
)
from xpcsjax.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# Performance tracking for fallback warnings
_performance_warned: set[str] = set()
_fallback_stats = {
    "gradient_calls": 0,
    "hessian_calls": 0,
    "jit_bypassed": 0,
    "vmap_loops": 0,
}

# Meshgrid cache for repeated computations with same time arrays
# Key: (t1_hash, t2_hash) where hash includes shape, dtype, and content digest
# Value: (t1_grid, t2_grid) JAX arrays
# Performance Optimization (Spec 001 - FR-002): LRU eviction for better cache utilization
_meshgrid_cache: OrderedDict[tuple, tuple] = OrderedDict()
_MESHGRID_CACHE_MAX_SIZE = 64  # Increased for 23-angle datasets (v2.11.0+)
# Guards every mutation of _meshgrid_cache (move_to_end, popitem, insert, clear)
# and _cache_stats — the cache is a process-global shared by all threads.
_cache_lock = threading.Lock()

# Performance Optimization (Spec 006 - FR-010, T040-T042): Cache statistics
_cache_stats: dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "skipped_large": 0,
    "skipped_traced": 0,
}


# Define exception types for array hash key computation
# JAX raises ConcretizationTypeError when accessing traced values inside JIT
if JAX_AVAILABLE:
    _jax_tracer_exceptions: list[type[Exception]] = [
        TypeError,
        jax.errors.ConcretizationTypeError,  # type: ignore[attr-defined,unused-ignore]
        jax.errors.TracerArrayConversionError,  # type: ignore[attr-defined,unused-ignore]
    ]
    # UnexpectedTracerError is raised by np.asarray() on traced values in
    # newer JAX versions — add it when available.
    _unexpected = getattr(jax.errors, "UnexpectedTracerError", None)
    if _unexpected is not None:
        _jax_tracer_exceptions.append(_unexpected)
    _ARRAY_HASH_EXCEPTIONS: tuple[type[Exception], ...] = tuple(  # type: ignore[no-redef,unused-ignore]
        _jax_tracer_exceptions
    )
else:
    _ARRAY_HASH_EXCEPTIONS: tuple[type[Exception], ...] = (TypeError,)  # type: ignore[no-redef,unused-ignore]


def _get_array_hash_key(arr: "jnp.ndarray") -> tuple | None:
    """Create a hashable key from array properties.

    Uses (length, quartile samples, dtype_str) to detect both endpoint
    differences and interior spacing differences (e.g. non-uniform grids).
    Sampling quartile points keeps the cost to ≤3 element accesses while
    making same-endpoint / different-interior collisions astronomically rare.

    Returns None if the array is a traced abstract value (inside JIT context).
    ``float(arr[i])`` raises ConcretizationTypeError on traced arrays, which
    is caught by ``_ARRAY_HASH_EXCEPTIONS``.  ``np.asarray`` is deliberately
    avoided here because it raises ``UnexpectedTracerError`` which is a
    side-effect rather than a clean value-access error.
    """
    try:
        n = int(arr.shape[0])
        if n == 0:
            # Empty array: no endpoints to sample — skip caching (sentinel None)
            return None
        if n <= 4:
            # Short arrays: include every element to guarantee uniqueness
            interior: tuple = tuple(float(arr[i]) for i in range(1, n - 1))
        else:
            # Sample quartile points to distinguish non-uniform spacing
            interior = (float(arr[n // 4]), float(arr[n // 2]), float(arr[3 * n // 4]))
        return (n, float(arr[0]), interior, float(arr[n - 1]), str(arr.dtype))
    except _ARRAY_HASH_EXCEPTIONS:
        # Inside JIT tracing — concrete values unavailable
        return None


def get_cached_meshgrid(t1: "jnp.ndarray", t2: "jnp.ndarray") -> tuple:
    """Get or create cached meshgrid for time arrays.

    For repeated calls with the same time arrays (common in optimization loops),
    this avoids recreating the same meshgrid ~23 times per iteration (once per phi).

    When called inside a JIT context (traced arrays), caching is skipped and
    meshgrid is created directly (the JIT will handle caching via tracing).

    Performance Optimization (Spec 006 - FR-010, T041):
    Increments hit/miss counters for cache monitoring.

    Args:
        t1: First time array (1D)
        t2: Second time array (1D)

    Returns:
        Tuple of (t1_grid, t2_grid) with indexing="ij"
    """
    global _meshgrid_cache, _cache_stats

    # Only cache 1D arrays that need meshgrid expansion
    if t1.ndim != 1 or t2.ndim != 1:
        return t1, t2

    # Don't cache large pooled data (element-wise matched, shouldn't mesh)
    # Use safe len check for JAX tracing compatibility
    try:
        n1 = len(t1)
        if n1 > 2000:
            with _cache_lock:
                _cache_stats["skipped_large"] += 1  # T041: Track skipped large arrays
            return t1, t2
    except TypeError:
        # Inside JIT tracing - skip stats AND caching
        if t1.shape[0] > 2000:
            return t1, t2

    # Try to create cache key - may fail inside JIT context
    t1_key = _get_array_hash_key(t1)
    t2_key = _get_array_hash_key(t2)

    # If inside JIT context, skip caching and create meshgrid directly
    if t1_key is None or t2_key is None:
        t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
        return t1_grid, t2_grid

    key = (t1_key, t2_key)

    with _cache_lock:
        if key in _meshgrid_cache:
            # Performance Optimization (Spec 001 - FR-002, T019): LRU - mark recent
            _meshgrid_cache.move_to_end(key)
            _cache_stats["hits"] += 1  # T041: Increment hit counter
            return _meshgrid_cache[key]
        _cache_stats["misses"] += 1  # T041: Increment miss counter

    # Create meshgrid outside the lock (pure XLA work — don't serialize it).
    t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")

    with _cache_lock:
        # Performance Optimization (Spec 001 - FR-002, T020): LRU eviction
        if len(_meshgrid_cache) >= _MESHGRID_CACHE_MAX_SIZE and key not in _meshgrid_cache:
            # Remove least recently used entry (first in OrderedDict)
            _meshgrid_cache.popitem(last=False)
            _cache_stats["evictions"] += 1  # T041: Track evictions
        _meshgrid_cache[key] = (t1_grid, t2_grid)
    return t1_grid, t2_grid


def clear_meshgrid_cache() -> None:
    """Clear the meshgrid cache.

    Call this when switching between datasets or when memory is constrained.
    """
    global _meshgrid_cache
    with _cache_lock:
        _meshgrid_cache.clear()


# Performance Optimization (Spec 006 - FR-010, T042): Cache stats utility
def get_cache_stats() -> dict[str, int | float]:
    """Get meshgrid cache statistics.

    Performance Optimization (Spec 006 - FR-010, T042):
    Returns cache hit/miss statistics for monitoring and optimization.

    Returns:
        Dictionary with cache statistics:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - evictions: Number of cache evictions
        - skipped_large: Arrays too large for caching
        - skipped_traced: Skipped due to JIT tracing
        - hit_rate: Cache hit rate (hits / total lookups)
        - cache_size: Current number of cached entries
    """
    total_lookups = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = _cache_stats["hits"] / total_lookups if total_lookups > 0 else 0.0

    return {
        **_cache_stats,
        "hit_rate": hit_rate,
        "cache_size": len(_meshgrid_cache),
        "max_cache_size": _MESHGRID_CACHE_MAX_SIZE,
    }


def reset_cache_stats() -> None:
    """Reset cache statistics counters.

    Performance Optimization (Spec 006 - FR-010):
    Call before benchmarking to get clean statistics.
    """
    global _cache_stats
    with _cache_lock:
        _cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "skipped_large": 0,
            "skipped_traced": 0,
        }


# Global flags for availability checking
jax_available = JAX_AVAILABLE
numpy_gradients_available = NUMPY_GRADIENTS_AVAILABLE if not JAX_AVAILABLE else False


if not JAX_AVAILABLE:
    if NUMPY_GRADIENTS_AVAILABLE:
        logger.warning(
            "JAX not available - using NumPy gradients fallback.\n"
            "Performance will be 10-50x slower than JAX.\n"
            "Install JAX for optimal performance: pip install jax",
        )
    else:
        logger.error(
            "Neither JAX nor NumPy gradients available.\n"
            "Install NumPy gradients: pip install scipy\n"
            "Or install JAX for optimal performance: pip install jax",
        )


def _create_gradient_fallback(func: Callable, argnums: int = 0) -> Callable:
    """Create intelligent gradient fallback with performance monitoring."""
    func_name = getattr(func, "__name__", "unknown")

    @wraps(func)
    def fallback_gradient(*args: object, **kwargs: object) -> object:
        _fallback_stats["gradient_calls"] += 1

        # Issue performance warning (once per function)
        if func_name not in _performance_warned:
            logger.warning(
                f"Using NumPy gradient fallback for {func_name}. "
                f"Expected 10-50x performance degradation. "
                f"Install JAX for optimal performance.",
            )
            _performance_warned.add(func_name)

        # Use numpy_gradient with appropriate configuration
        grad_func = numpy_gradient(func, argnums)
        return grad_func(*args, **kwargs)

    return fallback_gradient


def _create_hessian_fallback(func: Callable, argnums: int = 0) -> Callable:
    """Create intelligent Hessian fallback with performance monitoring."""
    func_name = getattr(func, "__name__", "unknown")

    @wraps(func)
    def fallback_hessian(*args: object, **kwargs: object) -> object:
        _fallback_stats["hessian_calls"] += 1

        # Issue performance warning (once per function)
        if func_name not in _performance_warned:
            logger.warning(
                f"Using NumPy Hessian fallback for {func_name}. "
                f"Expected 50-200x performance degradation. "
                f"Install JAX for optimal performance.",
            )
            _performance_warned.add(func_name)

        # Use numpy_hessian with appropriate configuration
        hess_func = numpy_hessian(func, argnums)
        return hess_func(*args, **kwargs)

    return fallback_hessian


def _create_no_gradient_fallback(func_name: str) -> Callable:
    """Create informative gradient fallback when no numerical differentiation is available."""

    def no_gradient_available(*args: object, **kwargs: object) -> object:
        error_msg = (
            f"Gradient computation not available for {func_name}.\n"
            f"Install NumPy gradients support or JAX:\n"
            f"  pip install scipy (for numerical differentiation)\n"
            f"  pip install jax (recommended for optimal performance)\n"
            f"\nCurrently available backends: None"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    return no_gradient_available


def _create_no_hessian_fallback(func_name: str) -> Callable:
    """Create informative Hessian fallback when no numerical differentiation is available."""

    def no_hessian_available(*args: object, **kwargs: object) -> object:
        error_msg = (
            f"Hessian computation not available for {func_name}.\n"
            f"Install NumPy gradients support or JAX:\n"
            f"  pip install scipy (for numerical differentiation)\n"
            f"  pip install jax (recommended for optimal performance)\n"
            f"\nCurrently available backends: None"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    return no_hessian_available


# Core physics computations with discrete numerical integration
# Note: _calculate_diffusion_coefficient_impl_jax, _calculate_shear_rate_impl_jax,
# and _create_time_integral_matrix_impl_jax are now imported from physics_utils.py
@jit
def _compute_g1_diffusion_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    dt: float,
    time_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute diffusion contribution to g1 using reference implementation approach.

    Algorithm (following reference v1 exactly):
    1. Extract time array (t1 = t2 = t, same time points)
    2. Calculate D(t) = D₀ t^α + D_offset at each time point
    3. Create integral matrix using cumulative sums: matrix[i,j] = |∫D(t)dt from i to j|
    4. Compute g1[i,j] = exp(-wavevector_q_squared_half_dt * matrix[i,j])

    Physical model: g₁_diff[i,j] = exp[-q²/2 * dt * ∫|tᵢ-tⱼ| D(t')dt']
    Where: D(t) = D₀ t^α + D_offset
    And: wavevector_q_squared_half_dt = 0.5 * q² * dt (from configuration)

    FORMULA VERIFICATION (matches reference exactly):
    Reference: self.wavevector_q_squared_half_dt = 0.5 * self.wavevector_q_squared * self.dt
    Which is: wavevector_q_squared_half_dt = 0.5 * (q²) * dt

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        dt: Time step from experimental configuration (time per frame)
        time_grid: Caller-provided time grid for cumulative trapezoid integration.
            When provided, used directly instead of building an internal grid.
            Required for element-wise mode to cover the full data time range.

    Returns:
        Diffusion contribution to g1 correlation function
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # P0-2: Dispatch element-wise mode based on dimensionality only (not size threshold).
    # 1D t1 = element-wise paired data; 2D t1 = meshgrid (NLSQ matrix mode).
    # The old `safe_len(t1) > 2000` heuristic caused small shards (<= 2000 pts)
    # to fall into matrix mode, producing wrong shapes.
    is_elementwise = t1.ndim == 1

    if is_elementwise:
        # ELEMENT-WISE MODE: Use cumulative trapezoid for accurate integration
        t1_arr = jnp.atleast_1d(t1)
        t2_arr = jnp.atleast_1d(t2)

        # P0-1: Use caller-provided time_grid instead of fixed 10001-point grid.
        # The old hardcoded MAX_GRID_SIZE=10001 truncated integrals for datasets
        # with t_max > 10000*dt, silently biasing g1 values.
        # Element-wise callers always provide time_grid via model_kwargs.
        # NLSQ callers use matrix mode (t1.ndim==2), so this branch is not reached.
        if time_grid is not None:
            time_grid_used = time_grid
        else:
            # Legacy fallback for direct calls: use static max size for JIT compat
            # T3-1: Use jnp.result_type(dt) to infer dtype from context instead of
            # hardcoding float64, which silently downcasts to float32 without X64.
            _FALLBACK_GRID_SIZE = 10001
            grid_indices = jnp.arange(_FALLBACK_GRID_SIZE, dtype=jnp.result_type(dt))
            time_grid_used = grid_indices * dt

        grid_size = time_grid_used.shape[0]

        # Compute D(t) on grid and build cumulative trapezoid
        D_grid = _calculate_diffusion_coefficient_impl_jax(time_grid_used, D0, alpha, D_offset)
        D_cumsum = _trapezoid_cumsum(D_grid)

        # Map times to grid indices using searchsorted (FR-007: clamp to valid range)
        max_index = grid_size - 1
        idx1 = jnp.clip(jnp.searchsorted(time_grid_used, t1_arr, side="left"), 0, max_index)
        idx2 = jnp.clip(jnp.searchsorted(time_grid_used, t2_arr, side="left"), 0, max_index)

        # Lookup integrals with smooth abs for gradient stability (FR-008).
        # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision).
        epsilon_abs = 1e-12
        D_integral = jnp.sqrt((D_cumsum[idx2] - D_cumsum[idx1]) ** 2 + epsilon_abs)

    else:
        # MATRIX MODE: Standard approach for small datasets or meshgrids
        # Step 1: Extract time array (t1 and t2 should be identical)
        # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
        if t1.ndim == 2:
            # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
            # So extract first COLUMN to get unique t1 values
            time_array = t1[:, 0]  # Extract first column for unique t1 values
        elif t1.ndim == 0:
            # Handle 0-dimensional (scalar) input
            time_array = jnp.atleast_1d(t1)
        else:
            # Handle 1D and other cases
            time_array = jnp.atleast_1d(t1)

        # Step 2: Calculate D(t) at each time point
        D_t = _calculate_diffusion_coefficient_impl_jax(time_array, D0, alpha, D_offset)

        # Step 3: Create diffusion integral matrix using cumulative sums
        # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
        D_integral = _create_time_integral_matrix_impl_jax(D_t)

    # Step 4: Compute g1 correlation using log-space for numerical stability
    # This matches reference: g1 = exp(-wavevector_q_squared_half_dt * D_integral)
    #
    # LOG-SPACE CALCULATION FIX (Oct 2025):
    # Computing in log-space preserves precision across full dynamic range.
    # Old approach: clip(g1, 1e-10, 1.0) caused artificial plateaus (~16% of data)
    # New approach: clip in log-space, then exp() - no artificial plateaus
    log_g1 = -wavevector_q_squared_half_dt * D_integral

    # Clip in log-space to prevent numerical overflow/underflow
    # -700 → exp(-700) ≈ 1e-304 (near machine precision)
    # 0 → exp(0) = 1.0 (maximum physical value)
    log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)

    # Compute exponential — log_g1_bounded is already clipped to [-700, 0],
    # so jnp.exp is safe (no overflow risk).
    g1_result: jnp.ndarray = jnp.exp(log_g1_bounded)

    # P1-2: Removed jnp.minimum(g1_result, 1.0) — the log-space clip above
    # (jnp.clip(log_g1, -700, 0)) already guarantees g1 = exp(log_g1) ≤ 1.0.
    # The hard min killed gradients at g1=1.0 (diagonal elements), harming NUTS.
    return g1_result


@jit
def _compute_g1_shear_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    sinc_prefactor: float,
    dt: float,
    time_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute shear contribution to g1 using reference implementation approach.

    Algorithm (following reference v1 exactly):
    1. Extract time array (t1 = t2 = t, same time points)
    2. Calculate γ̇(t) = γ̇₀ t^β + γ̇_offset at each time point
    3. Create integral matrix using cumulative sums: matrix[i,j] = |∫γ̇(t)dt from i to j|
    4. Compute sinc²[i,j] for each phi angle

    Physical model: g₁_shear = [sinc(Φ)]²
    Where: Φ = sinc_prefactor * cos(φ₀-φ) * ∫|tᵢ-tⱼ| γ̇(t') dt'
    And: γ̇(t) = γ̇₀ t^β + γ̇_offset
    And: sinc_prefactor = 0.5/π * q * L * dt (from configuration)

    FORMULA VERIFICATION (matches reference exactly):
    Reference: self.sinc_prefactor = 0.5 / np.pi * self.wavevector_q * self.stator_rotor_gap * self.dt
    Which is: sinc_prefactor = 0.5/π * q * L * dt
    Where L = stator_rotor_gap (sample-detector distance)

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        dt: Time step from experimental configuration (time per frame)

    Returns:
        Shear contribution to g1 correlation function (sinc² values)
    """
    # Check params length - if < 7, we're in static mode (no shear)
    if params.shape[0] < 7:
        # Return ones matching input dimensionality (g1_shear = 1)
        if t1.ndim == 1:
            # Element-wise mode (flat arrays for heatmap generation):
            # return 1D ones to match g1_diff shape in _compute_g1_total_core
            return jnp.ones_like(t1)
        else:
            # Matrix mode: return (n_phi, n_times, n_times) to broadcast with g1_diff
            phi_array = jnp.atleast_1d(phi)
            n_phi = phi_array.shape[0]
            n_times = t1.shape[0]
            return jnp.ones((n_phi, n_times, n_times))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # P0-2: Dispatch element-wise mode based on dimensionality only (not size threshold).
    # Same rationale as _compute_g1_diffusion_core.
    is_elementwise = t1.ndim == 1

    if is_elementwise:
        # ELEMENT-WISE MODE: Use cumulative trapezoid for accurate integration
        t1_arr = jnp.atleast_1d(t1)
        t2_arr = jnp.atleast_1d(t2)

        # P0-1: Use caller-provided time_grid instead of fixed 10001-point grid.
        if time_grid is not None:
            time_grid_used = time_grid
        else:
            # T3-1: Use jnp.result_type(dt) instead of hardcoding float64.
            _FALLBACK_GRID_SIZE = 10001
            grid_indices = jnp.arange(_FALLBACK_GRID_SIZE, dtype=jnp.result_type(dt))
            time_grid_used = grid_indices * dt

        grid_size = time_grid_used.shape[0]

        # Compute γ̇(t) on grid and build cumulative trapezoid
        gamma_grid = _calculate_shear_rate_impl_jax(
            time_grid_used, gamma_dot_0, beta, gamma_dot_offset
        )
        gamma_cumsum = _trapezoid_cumsum(gamma_grid)

        # Map times to grid indices using searchsorted (FR-007: clamp to valid range)
        max_index = grid_size - 1
        idx1 = jnp.clip(jnp.searchsorted(time_grid_used, t1_arr, side="left"), 0, max_index)
        idx2 = jnp.clip(jnp.searchsorted(time_grid_used, t2_arr, side="left"), 0, max_index)

        # Lookup integrals with smooth abs for gradient stability (FR-008).
        # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision).
        epsilon_abs = 1e-12
        gamma_integral = jnp.sqrt((gamma_cumsum[idx2] - gamma_cumsum[idx1]) ** 2 + epsilon_abs)
        n_times = t1_arr.shape[0]

    else:
        # MATRIX MODE: Standard approach for small datasets or meshgrids
        # Step 1: Extract time array (t1 and t2 should be identical)
        # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
        if t1.ndim == 2:
            # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
            # So extract first COLUMN to get unique t1 values
            time_array = t1[:, 0]  # Extract first column for unique t1 values
        elif t1.ndim == 0:
            # Handle 0-dimensional (scalar) input
            time_array = jnp.atleast_1d(t1)
        else:
            # Handle 1D and other cases
            time_array = jnp.atleast_1d(t1)

        # Step 2: Calculate γ̇(t) at each time point
        gamma_t = _calculate_shear_rate_impl_jax(
            time_array,
            gamma_dot_0,
            beta,
            gamma_dot_offset,
        )

        # Step 3: Create shear integral matrix using cumulative sums
        # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫γ̇(t)dt from i to j|
        # Create shear integral matrix using cumulative sums
        gamma_integral = _create_time_integral_matrix_impl_jax(gamma_t)
        n_times = time_array.shape[0]

    # Ensure phi is a 1D array regardless of input shape.
    # Handles (1, 1, 1, 23), (23,), scalar, etc. uniformly.
    # reshape(-1) avoids a Python while loop that would cause JIT retracing.
    phi_array = jnp.asarray(phi, dtype=jnp.result_type(phi)).reshape(-1)
    n_phi = phi_array.shape[0]

    if is_elementwise:
        # ELEMENT-WISE MODE: phi, gamma_integral are all 1D arrays (n,)
        # Each element i has its own phi[i] value (per-angle scaling)
        # Compute phase: Φ[i] = sinc_prefactor × cos(φ₀-phi[i]) × gamma_integral[i]

        # Element-wise computation (no broadcasting needed!)
        angle_diff = jnp.deg2rad(phi0 - phi_array)  # shape: (n,)
        cos_term = jnp.cos(angle_diff)  # shape: (n,)
        prefactor = sinc_prefactor * cos_term  # shape: (n,)
        phase = prefactor * gamma_integral  # shape: (n,)

        # Compute sinc² values: [sinc(Φ)]² for all elements
        sinc_val = safe_sinc(phase)
        sinc2_result: jnp.ndarray = sinc_val**2  # shape: (n,)

    else:
        # MATRIX MODE: Use vmap over phi to avoid O(n_phi × N²) peak memory.
        # Broadcasting would create a (n_phi, n_times, n_times) 3D tensor in one
        # shot — for n_phi=23, N=1001 that is 23 × 10^6 = 23M elements (~185 MB).
        # vmap applies the single-phi kernel sequentially-but-fused by XLA, keeping
        # peak working memory at O(N²) while returning the stacked (n_phi, N, N) result.

        # Ensure gamma_integral has the expected 2D shape
        if gamma_integral.ndim != 2:
            raise ValueError(
                f"gamma_integral should be 2D, got shape {gamma_integral.shape}",
            )

        def _sinc2_for_one_phi(phi_scalar: jnp.ndarray) -> jnp.ndarray:
            """Compute sinc²(Φ) for a single phi angle. Shape: (n_times, n_times)."""
            angle_diff = jnp.deg2rad(phi0 - phi_scalar)
            prefactor = sinc_prefactor * jnp.cos(angle_diff)
            phase = prefactor * gamma_integral  # (n_times, n_times)
            result: jnp.ndarray = safe_sinc(phase) ** 2
            return result

        # vmap over the phi array axis — each call gets a scalar phi element
        sinc2_result = vmap(_sinc2_for_one_phi)(phi_array)  # (n_phi, n_times, n_times)

    return sinc2_result


@jit
def _compute_g1_total_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    dt: float,
    time_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute total g1 correlation function as product of diffusion and shear.

    Following reference implementation:
    g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]

    Physical constraint: 0 < g₁(t) ≤ 1

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        dt: Time step from experimental configuration (time per frame)
        time_grid: Caller-provided time grid for element-wise cumulative trapezoid.
            Threaded through to diffusion/shear core functions.

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)
        or (n_total,) in element-wise mode.
    """
    # Compute diffusion contribution
    g1_diff = _compute_g1_diffusion_core(
        params, t1, t2, wavevector_q_squared_half_dt, dt, time_grid=time_grid
    )

    # Compute shear contribution
    g1_shear = _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt, time_grid=time_grid)

    # CRITICAL FIX (Nov 2025): Handle element-wise vs matrix mode
    # Element-wise mode: both g1_diff and g1_shear are 1D (shape (n,))
    # Matrix mode: g1_diff is 2D (n_times, n_times), g1_shear is 3D (n_phi, n_times, n_times)
    # Note: element-wise branch only valid for single-angle (P=1).
    is_elementwise = g1_diff.ndim == 1 and g1_shear.ndim == 1

    if is_elementwise:
        # ELEMENT-WISE MODE: Simple element-wise multiplication
        # g1_diff: (n,), g1_shear: (n,) → g1_total: (n,)
        g1_total = g1_diff * g1_shear
    else:
        # MATRIX MODE: Broadcast diffusion term to match shear dimensions
        # g1_diff: (n_times, n_times) → (n_phi, n_times, n_times) via broadcast
        n_phi = g1_shear.shape[0]
        g1_diff_broadcasted = jnp.broadcast_to(
            g1_diff[None, :, :],
            (n_phi, g1_diff.shape[0], g1_diff.shape[1]),
        )
        # g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]
        g1_total = g1_diff_broadcasted * g1_shear

    # P1-2: Keep only gradient-safe lower floor (prevents log(0)).
    # Upper clip removed — g1_diff is already bounded ≤ 1.0 from log-space clip,
    # and g1_shear (sinc²) is naturally bounded ≤ 1.0. Hard upper clips kill
    # gradients at the boundary, harming NUTS exploration.
    # Use jnp.where instead of jnp.maximum for gradient safety at the floor.
    epsilon = 1e-10
    g1_bounded: jnp.ndarray = jnp.where(g1_total > epsilon, g1_total, epsilon)

    return g1_bounded


@jit
def _compute_g2_scaled_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Core homodyne equation: g₂ = offset + contrast × [g₁]²

    The homodyne scattering equation is g₂ = 1 + β×g₁², where the baseline "1"
    is the constant background. In our implementation, this baseline is included
    in the offset parameter (offset ≈ 1.0 for physical measurements).

    For theoretical fits: Use offset=1.0, contrast=1.0 to get g₂ = 1 + g₁²
    For experimental fits: offset and contrast are free parameters centered around 1.0 and 0.5

    Physical constraint: 0.5 < g2 ≤ 2.5

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        contrast: Contrast parameter (β in literature) - typically [0, 1]
        offset: Baseline level (includes the "1" from physics) - typically ~1.0
        dt: Time step from experimental configuration (time per frame) [seconds]

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    g1 = _compute_g1_total_core(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        dt,
    )

    # Homodyne physics: g₂ = offset + contrast × [g₁]²
    # The baseline "1" is included in the offset parameter (offset ≈ 1.0 for physical data)
    g2 = offset + contrast * g1**2

    # P0-3: Removed hard jnp.clip(g2, 0.5, 2.5) — it kills gradients at boundaries.
    # For NLSQ (TRF optimizer), the bounds are enforced via parameter bounds, not g2 clipping.
    # For NUTS/MCMC, hard clips create zero-gradient plateaus that stall the sampler.
    # Physical range (0.5-2.5) is enforced through parameter priors instead.
    return g2  # type: ignore[no-any-return]


# =============================================================================
# COMPATIBILITY WRAPPER FUNCTIONS
# =============================================================================
# Re-export diagonal correction from unified module for backward compatibility.
# See xpcsjax/core/diagonal_correction.py for the canonical implementation.
from xpcsjax.core.diagonal_correction import (  # noqa: F401, E402
    apply_diagonal_correction,
    apply_diagonal_correction_batch,
)


def compute_g1_diffusion(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    q: float,
    dt: float | None = None,
) -> jnp.ndarray:
    """Wrapper function that computes g1 diffusion using configuration dt.

    IMPORTANT: The dt parameter should come from configuration, not be computed.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        q: Scattering wave vector magnitude
        dt: Time step from configuration (REQUIRED for correct physics)

    Returns:
        Diffusion contribution to g1 correlation function
    """
    # Handle 1D time arrays by creating meshgrids (cached for performance)
    # The cache avoids recreating the same meshgrid ~23 times per iteration (once per phi)
    t1, t2 = get_cached_meshgrid(t1, t2)

    # Use dt from configuration (REQUIRED for correct physics)
    # If dt not provided, estimate from time array as fallback.
    # P2-R7-02: After get_cached_meshgrid, t1 is always 2D, so the 1D branch
    # was dead code. Only the 2D case is needed.
    if dt is None:
        # FALLBACK: Estimate from 2D meshgrid (first column = unique t1 values)
        time_array = t1[:, 0]
        dt_value = float(time_array[1] - time_array[0]) if time_array.shape[0] > 1 else 1.0
    else:
        dt_value = dt

    # Compute the pre-computed factor using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value

    return jnp.asarray(
        _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt, dt_value)
    )


def compute_g1_shear(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
) -> jnp.ndarray:
    """Wrapper function that computes g1 shear using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        Shear contribution to g1 correlation function (sinc² values)

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids (cached for performance)
    t1, t2 = get_cached_meshgrid(t1, t2)

    # Compute the physics factor using configuration dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return jnp.asarray(_compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt))


def compute_g1_total(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float | None,
) -> jnp.ndarray:
    """Wrapper function that computes total g1 using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids (cached for performance)
    t1, t2 = get_cached_meshgrid(t1, t2)

    # Compute physics factors using configuration dt.
    # P2-R6-03: dt is REQUIRED — physics factors (sinc prefactor, q^2*dt) are
    # dt-dependent and there is no safe default frame rate across beamlines.
    # Raising here (on a Python-level None, before tracing) rather than silently
    # substituting a value prevents physically wrong fits from passing unnoticed.
    if dt is None:
        raise ValueError(
            "compute_g1_total: dt must be provided explicitly (seconds). "
            "Physics factors are dt-dependent; there is no safe default frame rate."
        )
    dt_value = dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
    sinc_prefactor = 0.5 / PI * q * L * dt_value

    return jnp.asarray(
        _compute_g1_total_core(
            params,
            t1,
            t2,
            phi,
            wavevector_q_squared_half_dt,
            sinc_prefactor,
            dt_value,
        )
    )


def compute_g2_scaled(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float | None,
) -> jnp.ndarray:
    """Wrapper function that computes g2 using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids (cached for performance).
    # get_cached_meshgrid may skip meshgrid for large 1D arrays (>2000 elements,
    # assumed to be element-wise matched pooled data). For the public API we
    # always need 2D matrix output, so fall back to explicit meshgrid.
    t1, t2 = get_cached_meshgrid(t1, t2)
    if t1.ndim == 1 and t2.ndim == 1:
        t1, t2 = jnp.meshgrid(t1, t2, indexing="ij")

    # Compute physics factors using configuration dt.
    # P2-R6-03: dt is REQUIRED — physics factors are dt-dependent and there is no
    # safe default frame rate. Raise (on a Python-level None, before tracing) rather
    # than silently substituting a value that yields physically wrong fits.
    if dt is None:
        raise ValueError(
            "compute_g2_scaled: dt must be provided explicitly (seconds). "
            "Physics factors are dt-dependent; there is no safe default frame rate."
        )
    dt_value = dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
    sinc_prefactor = 0.5 / PI * q * L * dt_value

    return jnp.asarray(
        _compute_g2_scaled_core(
            params,
            t1,
            t2,
            phi,
            wavevector_q_squared_half_dt,
            sinc_prefactor,
            contrast,
            offset,
            dt_value,
        )
    )


@partial(jit, static_argnums=(4, 5, 8))
def compute_g2_scaled_with_factors(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-optimized g2 computation using pre-computed physics factors.

    This is the hybrid architecture functional core - accepts pre-computed
    factors directly, avoiding runtime computation. Suitable for use with
    HomodyneModel where factors are computed once at initialization.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids for correlation calculation
        phi: Scattering angles [degrees]
        wavevector_q_squared_half_dt: Pre-computed factor (0.5 * q² * dt)
        sinc_prefactor: Pre-computed factor (q * L * dt / 2π)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from experimental configuration (time per frame) [seconds]

    Returns:
        g2 correlation function with scaled fitting

    Note:
        This function is JIT-compiled for maximum performance.
        Use with HomodyneModel for best results.
    """
    # Handle 1D time arrays by creating meshgrids.
    # This function is only called from the NLSQ path (HomodyneModel), which
    # always passes 2D grids. The 1D branch is a safety net for external
    # callers passing 1D time vectors.
    if t1.ndim == 1 and t2.ndim == 1:
        t1, t2 = jnp.meshgrid(t1, t2, indexing="ij")

    # Call core computation with pre-computed factors
    return jnp.asarray(
        _compute_g2_scaled_core(
            params,
            t1,
            t2,
            phi,
            wavevector_q_squared_half_dt,
            sinc_prefactor,
            contrast,
            offset,
            dt,
        )
    )


@partial(jit, static_argnums=(6, 7, 10))
def compute_chi_squared(
    params: jnp.ndarray,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Compute chi-squared goodness of fit.

    χ² = Σᵢ [(data_i - theory_i) / σᵢ]²

    Args:
        params: Physical parameters
        data: Experimental correlation data
        sigma: Measurement uncertainties
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        dt: Time step from configuration

    Returns:
        Chi-squared value
    """
    theory = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)
    # Guard against zero sigma: replace with inf so residual becomes 0 for masked pixels
    # (finite / inf = 0, so zero-sigma points contribute nothing to chi-squared).
    safe_sigma = jnp.where(sigma > 0, sigma, jnp.inf)
    residuals = (data - theory) / safe_sigma
    return jnp.sum(residuals**2)


# Automatic differentiation functions with intelligent fallback
# These will work with either JAX or NumPy fallbacks
# Pre-JIT compiled for 50-100x faster first call (avoids compilation overhead)
gradient_g2 = jit(grad(compute_g2_scaled, argnums=0))  # Gradient w.r.t. params
hessian_g2 = jit(hessian(compute_g2_scaled, argnums=0))  # Hessian w.r.t. params

gradient_chi2 = jit(grad(compute_chi_squared, argnums=0))  # Gradient of chi-squared
hessian_chi2 = jit(hessian(compute_chi_squared, argnums=0))  # Hessian of chi-squared

# Module-level vmapped functions — created once to avoid per-call re-tracing.
# params_batch axis 0 is batched; all other args are broadcast unchanged.
_vmap_g2_scaled = vmap(
    compute_g2_scaled,
    in_axes=(0, None, None, None, None, None, None, None, None),
)
_vmap_chi_squared = vmap(
    compute_chi_squared,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None),
)


# Vectorized versions for batch computation
@log_performance(threshold=0.1)
def vectorized_g2_computation(
    params_batch: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float | None = None,
) -> jnp.ndarray:
    """Vectorized g2 computation for multiple parameter sets.

    Uses JAX vmap for efficient parallel computation.

    Args:
        params_batch: Batch of parameter arrays, shape (n_batch, n_params)
        t1, t2: Time arrays for correlation calculation
        phi: Scattering angles
        q: Wavevector magnitude [Å⁻¹]
        L: Beam width [Å]
        contrast: Contrast parameter
        offset: Baseline offset
        dt: Time step from configuration [seconds]. MUST be provided for correct physics.
    """
    # dt is REQUIRED for correct physics factors; there is no safe default frame
    # rate. Raise on a Python-level None before vmap tracing rather than silently
    # substituting a value that yields physically wrong batch results.
    if dt is None:
        raise ValueError(
            "dt must be provided explicitly (seconds) for batch g2/chi-squared "
            "computation; physics factors are dt-dependent."
        )
    dt_value = dt

    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt_value)
            results.append(result)
        return jnp.stack(results)

    # JAX vectorized version — use module-level vmapped function to avoid re-tracing
    return _vmap_g2_scaled(params_batch, t1, t2, phi, q, L, contrast, offset, dt_value)


@log_performance(threshold=0.05)
def batch_chi_squared(
    params_batch: jnp.ndarray,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float | None = None,
) -> jnp.ndarray:
    """Compute chi-squared for multiple parameter sets efficiently.

    Args:
        params_batch: Batch of parameter arrays, shape (n_batch, n_params)
        data: Experimental g2 data
        sigma: Uncertainty in data
        t1, t2: Time arrays for correlation calculation
        phi: Scattering angles
        q: Wavevector magnitude [Å⁻¹]
        L: Beam width [Å]
        contrast: Contrast parameter
        offset: Baseline offset
        dt: Time step from configuration [seconds]. MUST be provided for correct physics.
    """
    # dt is REQUIRED for correct physics factors; there is no safe default frame
    # rate. Raise on a Python-level None before vmap tracing rather than silently
    # substituting a value that yields physically wrong batch results.
    if dt is None:
        raise ValueError(
            "dt must be provided explicitly (seconds) for batch g2/chi-squared "
            "computation; physics factors are dt-dependent."
        )
    dt_value = dt

    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_chi_squared(
                params,
                data,
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                contrast,
                offset,
                dt_value,
            )
            results.append(result)
        return jnp.array(results)

    # JAX vectorized version — use module-level vmapped function to avoid re-tracing
    return jnp.asarray(
        _vmap_chi_squared(
            params_batch,
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            contrast,
            offset,
            dt_value,
        )
    )


# Utility functions for optimization
def validate_backend() -> dict[str, Any]:
    """Validate computational backends with comprehensive diagnostics."""
    results: dict[str, Any] = {
        "jax_available": JAX_AVAILABLE,
        "numpy_gradients_available": numpy_gradients_available,
        "gradient_support": False,
        "hessian_support": False,
        "backend_type": "unknown",
        "performance_estimate": "unknown",
        "recommendations": cast(list[str], []),
        "fallback_stats": _fallback_stats.copy(),
        "test_results": cast(dict[str, str], {}),
    }

    # Determine backend type and performance characteristics
    if JAX_AVAILABLE:
        results["backend_type"] = "jax_native"
        results["performance_estimate"] = "optimal (1x)"
    elif numpy_gradients_available:
        results["backend_type"] = "numpy_fallback"
        results["performance_estimate"] = "degraded (10-50x slower)"
        cast(list[str], results["recommendations"]).append(
            "Install JAX for optimal performance: pip install jax",
        )
    else:
        results["backend_type"] = "none"
        results["performance_estimate"] = "unavailable"
        cast(list[str], results["recommendations"]).extend(
            [
                "Install JAX for optimal performance: pip install jax",
                "Or install scipy for basic functionality: pip install scipy",
            ],
        )

    # Test basic computation
    try:
        test_params = jnp.array([100.0, 0.0, 10.0])
        test_t1 = jnp.array([0.0, 0.001, 0.002])
        test_t2 = jnp.array([0.0, 0.001, 0.002])
        test_q = 0.01

        # Test forward computation
        compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        cast(dict[str, str], results["test_results"])["forward_computation"] = "success"

        # Test gradient computation
        try:
            grad_func = grad(compute_g1_diffusion, argnums=0)
            grad_func(test_params, test_t1, test_t2, test_q)
            results["gradient_support"] = True
            cast(dict[str, str], results["test_results"])["gradient_computation"] = "success"

            if not JAX_AVAILABLE:
                cast(dict[str, str], results["test_results"])["gradient_method"] = "numpy_fallback"

        except ImportError as e:
            cast(dict[str, str], results["test_results"])["gradient_computation"] = (
                f"failed: {str(e)}"
            )
            logger.warning(f"Gradient computation not available: {e}")
        except Exception as e:
            cast(dict[str, str], results["test_results"])["gradient_computation"] = (
                f"error: {str(e)}"
            )
            logger.error(f"Gradient computation failed: {e}")

        # Test hessian computation
        try:
            hess_func = hessian(compute_g1_diffusion, argnums=0)
            hess_func(test_params, test_t1, test_t2, test_q)
            results["hessian_support"] = True
            cast(dict[str, str], results["test_results"])["hessian_computation"] = "success"

            if not JAX_AVAILABLE:
                cast(dict[str, str], results["test_results"])["hessian_method"] = "numpy_fallback"

        except ImportError as e:
            cast(dict[str, str], results["test_results"])["hessian_computation"] = (
                f"failed: {str(e)}"
            )
            logger.warning(f"Hessian computation not available: {e}")
        except Exception as e:
            cast(dict[str, str], results["test_results"])["hessian_computation"] = (
                f"error: {str(e)}"
            )
            logger.error(f"Hessian computation failed: {e}")

        logger.info(f"Backend validation completed: {results['backend_type']} mode")

    except Exception as e:
        logger.error(f"Basic computation test failed: {e}")
        cast(dict[str, str], results["test_results"])["forward_computation"] = f"failed: {str(e)}"

    return results


def get_device_info() -> dict[str, Any]:
    """Get comprehensive device and backend information."""
    if not JAX_AVAILABLE:
        fallback_info: dict[str, Any] = {
            "available": False,
            "devices": cast(list[str], []),
            "backend": "numpy_fallback" if numpy_gradients_available else "none",
            "fallback_active": True,
            "performance_impact": ("10-50x slower" if numpy_gradients_available else "unavailable"),
            "recommendations": cast(list[str], []),
        }

        if numpy_gradients_available:
            cast(list[str], fallback_info["recommendations"]).append(
                "Install JAX for optimal performance: pip install jax",
            )
            fallback_info["fallback_stats"] = _fallback_stats.copy()
        else:
            cast(list[str], fallback_info["recommendations"]).extend(
                [
                    "Install JAX for optimal performance: pip install jax",
                    "Or install scipy for basic functionality: pip install scipy",
                ],
            )

        return fallback_info

    try:
        devices = jax.devices()
        return {
            "available": True,
            "devices": [str(d) for d in devices],
            "backend": jax.default_backend(),
            "device_count": len(devices),
            "fallback_active": False,
            "performance_impact": "optimal (native JAX)",
            "recommendations": ["JAX is available and configured correctly"],
        }
    except Exception as e:
        logger.warning(f"Could not get JAX device info: {e}")
        return {
            "available": True,
            "devices": ["unknown"],
            "backend": "unknown",
            "error": str(e),
            "fallback_active": False,
        }


def get_performance_summary() -> dict[str, Any]:
    """Get performance summary and recommendations."""
    return {
        "backend_type": (
            "jax_native"
            if JAX_AVAILABLE
            else ("numpy_fallback" if numpy_gradients_available else "none")
        ),
        "jax_available": JAX_AVAILABLE,
        "numpy_gradients_available": numpy_gradients_available,
        "fallback_stats": _fallback_stats.copy(),
        "performance_multiplier": (
            "1x" if JAX_AVAILABLE else ("10-50x" if numpy_gradients_available else "N/A")
        ),
        "recommendations": _get_performance_recommendations(),
    }


def _get_performance_recommendations() -> list[str]:
    """Get performance optimization recommendations."""
    recommendations = []

    if not JAX_AVAILABLE:
        recommendations.append(
            "[PERF] Install JAX for 10-50x performance improvement: pip install jax",
        )

        if not numpy_gradients_available:
            recommendations.append(
                "[PERF] Install scipy for basic numerical differentiation: pip install scipy",
            )
        else:
            recommendations.append("[OK] NumPy gradients available as fallback")

    if JAX_AVAILABLE:
        # P2-R7-04: This is a CPU-only package — no GPU/TPU detection.
        # Report CPU device count for parallel processing hints only.
        try:
            import jax

            devices = jax.devices("cpu")
            if len(devices) > 1:
                recommendations.append(
                    f"[INFO] {len(devices)} CPU devices available for parallel processing",
                )
        except Exception:
            logger.debug("Device inspection failed; proceeding without device hints")

    return recommendations


# Export main functions
__all__ = [
    "jax_available",
    "numpy_gradients_available",
    "compute_g1_diffusion",
    "compute_g1_shear",
    "compute_g1_total",
    "compute_g2_scaled",
    "compute_chi_squared",
    "gradient_g2",
    "hessian_g2",
    "gradient_chi2",
    "hessian_chi2",
    "vectorized_g2_computation",
    "batch_chi_squared",
    "validate_backend",
    "get_device_info",
    "get_performance_summary",  # New performance monitoring
]
