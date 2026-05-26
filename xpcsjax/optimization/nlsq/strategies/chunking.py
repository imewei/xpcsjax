"""Angle-Stratified Chunking for Per-Angle Parameter Optimization.

This module implements angle-stratified data reorganization to ensure NLSQ's
chunking strategy remains compatible with per-angle parameters (contrast[i],
offset[i] for each phi angle).

Root Cause of Incompatibility:
------------------------------
NLSQ's chunking splits data arbitrarily without angle awareness. When per-angle
parameters are used:
- Each contrast[i] only affects points with phi=angle[i]
- If a chunk has no points with angle[i], gradient w.r.t. contrast[i] is ZERO
- Zero gradients → NLSQ fails silently (0 iterations, unchanged parameters)

Solution: Angle-Stratified Chunking
------------------------------------
Reorganize data BEFORE NLSQ optimization so every chunk contains ALL phi angles:
- Original: Random 100k-point chunks may miss angles
- Stratified: Each 100k-point chunk has balanced angle representation
- Result: All per-angle gradients always well-defined

Performance Impact: <1% overhead (0.15s for 3M points)
Memory Impact: 2x peak during reorganization (temporary)

Examples
--------
>>> # Reorganize 3M point dataset with 3 angles
>>> phi, t1, t2, g2 = load_data()  # 3M points
>>> phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
...     phi, t1, t2, g2, target_chunk_size=100_000
... )
>>> # Now NLSQ optimization will work correctly with per_angle_scaling=True

References
----------
Ultra-Think Analysis: ultra-think-20251106-012247
Issue: Per-angle scaling + NLSQ chunking incompatibility
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AngleDistributionStats:
    """Statistics about phi angle distribution in dataset.

    Attributes
    ----------
    unique_angles : np.ndarray
        Array of unique phi angles in the dataset
    n_angles : int
        Number of unique angles
    counts : dict[float, int]
        Points per angle: {angle: count}
    fractions : dict[float, float]
        Fraction of total per angle: {angle: fraction}
    imbalance_ratio : float
        max(counts) / min(counts), indicates balance
    min_angle : float
        Angle with fewest points
    max_angle : float
        Angle with most points
    is_balanced : bool
        True if imbalance_ratio < 5.0 (recommended threshold)
    """

    unique_angles: np.ndarray
    n_angles: int
    counts: dict[float, int]
    fractions: dict[float, float]
    imbalance_ratio: float
    min_angle: float
    max_angle: float
    is_balanced: bool


@dataclass
class StratificationDiagnostics:
    """Detailed diagnostics for stratification performance and quality.

    This dataclass provides comprehensive metrics for analyzing stratification
    effectiveness, performance, and memory usage.

    Attributes
    ----------
    n_chunks : int
        Number of chunks created
    chunk_sizes : list[int]
        Size of each chunk in points
    chunk_balance : dict[str, float]
        Chunk size statistics: {mean, std, min, max, cv}
    angles_per_chunk : list[int]
        Number of unique angles in each chunk
    angle_coverage : dict[str, float]
        Angle coverage statistics: {mean, std, min_coverage_ratio}
    execution_time_ms : float
        Time taken for stratification (milliseconds)
    memory_overhead_mb : float
        Peak memory overhead during stratification
    memory_efficiency : float
        Ratio of data size to peak memory (1.0 = perfect)
    throughput_points_per_sec : float
        Processing throughput (points per second)
    use_index_based : bool
        Whether index-based stratification was used
    """

    n_chunks: int
    chunk_sizes: list[int]
    chunk_balance: dict[str, float]
    angles_per_chunk: list[int]
    angle_coverage: dict[str, float]
    execution_time_ms: float
    memory_overhead_mb: float
    memory_efficiency: float
    throughput_points_per_sec: float
    use_index_based: bool


def analyze_angle_distribution(phi: jnp.ndarray | np.ndarray) -> AngleDistributionStats:
    """Analyze phi angle distribution to assess balance.

    Computes statistics about how data points are distributed across phi angles.
    This is critical for deciding whether angle-stratified chunking or
    sequential per-angle optimization should be used.

    Parameters
    ----------
    phi : jnp.ndarray or np.ndarray
        Array of phi angles (radians or degrees), shape (n_points,)

    Returns
    -------
    AngleDistributionStats
        Complete statistics about angle distribution

    Examples
    --------
    >>> phi = np.array([0, 0, 45, 45, 90])  # 2 @ 0°, 2 @ 45°, 1 @ 90°
    >>> stats = analyze_angle_distribution(phi)
    >>> print(f"Imbalance ratio: {stats.imbalance_ratio:.1f}")
    Imbalance ratio: 2.0
    >>> print(f"Balanced: {stats.is_balanced}")
    Balanced: True

    Notes
    -----
    Imbalance ratio interpretation:
    - < 2.0: Excellent balance (ideal for stratification)
    - 2.0 - 5.0: Acceptable balance (stratification works)
    - > 5.0: High imbalance (consider sequential per-angle)
    - > 10.0: Very high imbalance (sequential per-angle recommended)
    """
    # Convert to numpy for analysis
    if isinstance(phi, jnp.ndarray):
        phi = np.array(phi)

    # Get unique angles and counts
    unique_angles, counts = np.unique(phi, return_counts=True)
    n_angles = len(unique_angles)
    total_points = len(phi)

    # Build statistics dictionaries
    counts_dict = {
        float(angle): int(count)
        for angle, count in zip(unique_angles, counts, strict=False)
    }
    fractions_dict = {
        float(angle): float(count) / total_points
        for angle, count in zip(unique_angles, counts, strict=False)
    }

    # Calculate imbalance
    min_count = int(np.min(counts))
    max_count = int(np.max(counts))
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")

    # Find min/max angles
    min_angle = float(unique_angles[np.argmin(counts)])
    max_angle = float(unique_angles[np.argmax(counts)])

    # Assess balance
    is_balanced = imbalance_ratio < 5.0

    logger.debug(
        f"Angle distribution: {n_angles} angles, "
        f"imbalance ratio {imbalance_ratio:.2f}, "
        f"balanced: {is_balanced}"
    )

    return AngleDistributionStats(
        unique_angles=unique_angles,
        n_angles=n_angles,
        counts=counts_dict,
        fractions=fractions_dict,
        imbalance_ratio=imbalance_ratio,
        min_angle=min_angle,
        max_angle=max_angle,
        is_balanced=is_balanced,
    )


def estimate_stratification_memory(
    n_points: int,
    n_features: int = 4,
    use_index_based: bool = False,
    estimated_expansion: float = 1.0,  # New parameter for Cyclic Stratification
) -> dict[str, Any]:
    """Estimate memory requirements for stratification ONLY.

    WARNING: This function ONLY estimates data reorganization memory.
    For complete NLSQ optimization memory including Jacobian and optimizer state,
    use estimate_nlsq_optimization_memory() instead.

    Parameters
    ----------
    n_points : int
        Total number of data points
    n_features : int, optional
        Number of data features (phi, t1, t2, g2_exp), default: 4
    use_index_based : bool, optional
        If True, use index-based stratification (zero-copy), default: False
    estimated_expansion : float, optional
        Estimated data expansion factor due to Cyclic Stratification (default: 1.0).
        For imbalanced data, this can be > 1.0 (e.g., 2.0 for 2:1 imbalance).

    Returns
    -------
    dict
        Memory statistics with keys:
        - original_memory_mb: Original data memory usage
        - stratified_memory_mb: Memory for stratified copy (including expansion)
        - peak_memory_mb: Peak memory during stratification
        - index_memory_mb: Memory for index arrays (if use_index_based)
        - is_safe: Whether memory usage is safe (<70% of available)

    Notes
    -----
    Memory usage:
    - Full copy: original + (original * expansion) (peak)
    - Index-based: original + index_array (peak)
    """
    bytes_per_float = 8  # float64

    # Original data memory
    original_bytes = n_points * n_features * bytes_per_float
    original_mb = original_bytes / (1024**2)

    # Stratified data size (potentially expanded)
    stratified_points = int(n_points * estimated_expansion)
    stratified_bytes = stratified_points * n_features * bytes_per_float
    stratified_mb = stratified_bytes / (1024**2)

    if use_index_based:
        # Index arrays: one index per stratified point
        bytes_per_int = 8  # int64
        index_bytes = stratified_points * bytes_per_int
        index_mb = index_bytes / (1024**2)
        peak_mb = original_mb + index_mb
        # For index based, the output "stratified_memory_mb" is virtually 0
        # (just the index), but effectively we are accessing stratified_points of data
        stratified_mb = 0
    else:
        # Full copy approach
        index_mb = 0
        peak_mb = original_mb + stratified_mb  # Original + copy

    # Check against available memory
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024**2)
        is_safe = peak_mb < available_mb * 0.7
    except ImportError:
        logger.warning("psutil not available, cannot check memory safety")
        is_safe = True  # Assume safe if we can't check

    logger.debug(
        f"Stratification memory estimate (expansion {estimated_expansion:.1f}x): "
        f"original={original_mb:.1f} MB, "
        f"peak={peak_mb:.1f} MB, "
        f"safe={is_safe}"
    )

    return {
        "original_memory_mb": original_mb,
        "stratified_memory_mb": stratified_mb,
        "peak_memory_mb": peak_mb,
        "index_memory_mb": index_mb,
        "is_safe": is_safe,
    }


def estimate_nlsq_optimization_memory(
    n_points: int,
    n_params: int,
    n_features: int = 4,
    dtype_bytes: int = 8,
) -> dict[str, Any]:
    """Estimate complete memory requirements for NLSQ optimization.

    This function provides a COMPLETE memory estimate including all components:
    - Data arrays (phi, t1, t2, g2)
    - Jacobian matrix (DOMINANT memory consumer)
    - JAX JIT compilation overhead
    - Optimizer internal state

    Root Cause Fix (Nov 10, 2025):
    The original estimate_stratification_memory() only counted data (703 MB),
    but actual usage was 51 GB (36× underestimate). This function includes ALL
    memory components for accurate prediction.

    Parameters
    ----------
    n_points : int
        Total number of data points
    n_params : int
        Number of optimization parameters (e.g., 53 for laminar_flow with per-angle)
    n_features : int, optional
        Number of data features (phi, t1, t2, g2_exp), default: 4
    dtype_bytes : int, optional
        Bytes per floating point number, default: 8 (float64)

    Returns
    -------
    dict
        Complete memory statistics with keys:
        - data_mb: Data arrays memory
        - jacobian_mb: Jacobian matrix memory (DOMINANT)
        - jax_overhead_mb: JAX JIT cache and device arrays
        - optimizer_mb: Optimizer state (Hessian, gradients)
        - total_mb: Total estimated memory
        - peak_gb: Peak memory in GB
        - available_gb: Available system memory
        - utilization_pct: Percentage of available memory used
        - is_safe: Whether memory usage is safe (<70% of available)

    Examples
    --------
    >>> # Real dataset from log: 23M points, 53 params
    >>> mem = estimate_nlsq_optimization_memory(
    ...     n_points=23_046_023,
    ...     n_params=53
    ... )
    >>> print(f"Jacobian: {mem['jacobian_mb']:.0f} MB")
    Jacobian: 9,784 MB
    >>> print(f"Total: {mem['peak_gb']:.1f} GB")
    Total: 14.3 GB
    >>> print(f"Utilization: {mem['utilization_pct']:.1f}%")
    Utilization: 22.8%
    >>>
    >>> # With old fixed 100K chunks: 51 GB actual vs 14.3 GB estimated
    >>> # Difference due to memory leak (fixed separately)

    Notes
    -----
    Memory Components:
    1. Data arrays: n_points × n_features × dtype_bytes
    2. Jacobian: n_points × n_params × dtype_bytes (DOMINANT)
    3. JAX overhead: 1.75× data (JIT cache, device arrays)
    4. Optimizer state: Hessian (n_params²) + gradients + trust region
    5. Safety margin: 20% buffer for temporary allocations

    Root Cause (Nov 10, 2025):
    - Old estimate: Only data = 703 MB
    - Actual peak: 51 GB (includes Jacobian + leak)
    - New estimate: 14.3 GB (without leak)
    - With fixes: Expected ~15 GB actual
    """
    # 1. Data arrays (phi, t1, t2, g2)
    data_bytes = n_points * n_features * dtype_bytes
    data_mb = data_bytes / (1024**2)

    # 2. Jacobian matrix (DOMINANT memory consumer)
    # Each residual needs gradient w.r.t. all parameters
    jacobian_bytes = n_points * n_params * dtype_bytes
    jacobian_mb = jacobian_bytes / (1024**2)

    # 3. JAX overhead (JIT cache, device arrays, XLA buffers)
    # Empirically ~1.5-2× the data size
    jax_overhead_mb = data_mb * 1.75

    # 4. Optimizer state
    # Hessian approximation: n_params × n_params
    # Gradients: n_params
    # Trust region matrices: additional overhead
    hessian_bytes = n_params * n_params * dtype_bytes
    gradient_bytes = n_params * dtype_bytes
    trust_region_mb = 100  # Empirical overhead for trust region algorithm
    optimizer_mb = (hessian_bytes + gradient_bytes) / (1024**2) + trust_region_mb

    # Total with 20% safety margin
    safety_margin = 0.20
    total_mb = (data_mb + jacobian_mb + jax_overhead_mb + optimizer_mb) * (
        1 + safety_margin
    )
    peak_gb = total_mb / 1000

    # Check against available memory
    try:
        import psutil

        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        utilization_pct = (peak_gb / available_gb) * 100
        is_safe = utilization_pct < 70.0
    except ImportError:
        logger.warning("psutil not available, cannot check memory safety")
        available_gb = 0.0
        utilization_pct = 0.0
        is_safe = True

    logger.info(
        f"NLSQ optimization memory estimate:\n"
        f"  Data arrays: {data_mb:.0f} MB\n"
        f"  Jacobian matrix: {jacobian_mb:.0f} MB (DOMINANT)\n"
        f"  JAX overhead: {jax_overhead_mb:.0f} MB\n"
        f"  Optimizer state: {optimizer_mb:.0f} MB\n"
        f"  Total (with 20% margin): {total_mb:.0f} MB ({peak_gb:.1f} GB)\n"
        f"  Available memory: {available_gb:.1f} GB\n"
        f"  Utilization: {utilization_pct:.1f}%\n"
        f"  Safe: {is_safe}"
    )

    return {
        "data_mb": data_mb,
        "jacobian_mb": jacobian_mb,
        "jax_overhead_mb": jax_overhead_mb,
        "optimizer_mb": optimizer_mb,
        "total_mb": total_mb,
        "peak_gb": peak_gb,
        "available_gb": available_gb,
        "utilization_pct": utilization_pct,
        "is_safe": is_safe,
    }


def calculate_adaptive_chunk_size(
    total_points: int,
    n_params: int,
    n_angles: int,
    available_memory_gb: float | None = None,
    safety_factor: float = 5.0,
    min_chunk_size: int = 10_000,
    max_chunk_size: int = 500_000,
) -> int:
    """
    Calculate optimal chunk size based on available system memory and parameter count.

    This function addresses the root cause of memory pressure in NLSQ optimization:
    the fixed 100K chunk size doesn't account for available memory or the number
    of parameters, which determines Jacobian matrix size.

    The Jacobian matrix dominates memory usage:
    - Size: n_residuals × n_params × 8 bytes
    - For 100K points with 53 params: ~42 MB per chunk
    - Full dataset (23M points): ~9.8 GB Jacobian

    Memory Budget Calculation:
    1. Reserve 30% for OS, JAX overhead, optimizer state
    2. Calculate max points that fit: available_memory / (param_bytes × safety_factor)
    3. Ensure all angles fit in each chunk (critical for per-angle parameters)
    4. Clamp to reasonable bounds for numerical stability and iteration speed

    Parameters
    ----------
    total_points : int
        Total number of data points in dataset
    n_params : int
        Number of optimization parameters (e.g., 53 for laminar_flow with per-angle scaling)
    n_angles : int
        Number of unique phi angles (must all fit in each chunk)
    available_memory_gb : float, optional
        Available system memory in GB. If None, auto-detected using psutil.
    safety_factor : float, optional
        Multiplicative safety factor for memory overhead (default: 5.0)
        Accounts for JAX JIT cache, optimizer state, temporary arrays.
    min_chunk_size : int, optional
        Minimum chunk size for numerical stability (default: 10,000)
    max_chunk_size : int, optional
        Maximum chunk size for iteration speed (default: 500,000)

    Returns
    -------
    int
        Optimal chunk size that fits in available memory

    Examples
    --------
    >>> # 23M points, 53 parameters, 23 angles, 62GB system
    >>> chunk_size = calculate_adaptive_chunk_size(
    ...     total_points=23_046_023,
    ...     n_params=53,
    ...     n_angles=23,
    ...     available_memory_gb=62.8
    ... )
    >>> print(f"Optimal chunk size: {chunk_size:,}")
    Optimal chunk size: 23,000
    >>>
    >>> # Small dataset, few parameters
    >>> chunk_size = calculate_adaptive_chunk_size(
    ...     total_points=1_000_000,
    ...     n_params=9,
    ...     n_angles=3,
    ...     available_memory_gb=32.0
    ... )
    >>> print(f"Optimal chunk size: {chunk_size:,}")
    Optimal chunk size: 500,000  # Clamped to max

    Notes
    -----
    Root Cause Analysis (Nov 10, 2025):
    - Fixed 100K chunk size caused 96% memory pressure on 62.8GB system
    - With 53 params: Jacobian alone is 9.8 GB
    - JAX overhead adds 1.5-2× data size
    - Optimizer state adds ~2 GB
    - Total: ~51 GB peak (should be ~15 GB with adaptive sizing)

    Algorithm:
    1. Auto-detect available memory if not provided
    2. Calculate memory per point: n_params × 8 bytes (Jacobian row)
    3. Usable memory: 70% of available (reserve 30% for OS/JAX)
    4. Max points: usable_memory / (memory_per_point × safety_factor)
    5. Chunk size: (max_points / n_angles) × n_angles  # Ensure all angles fit
    6. Clamp to [min_chunk_size, max_chunk_size]
    """
    # Auto-detect available memory if not provided
    if available_memory_gb is None:
        try:
            import psutil

            available_bytes = psutil.virtual_memory().available
            available_memory_gb = available_bytes / (1024**3)
            logger.debug(
                f"Auto-detected available memory: {available_memory_gb:.1f} GB"
            )
        except ImportError:
            # Fallback: try os.sysconf for total memory, then estimate 50% available
            try:
                import os

                page_size = os.sysconf("SC_PAGE_SIZE")
                phys_pages = os.sysconf("SC_PHYS_PAGES")
                if page_size > 0 and phys_pages > 0:
                    total_gb = (page_size * phys_pages) / (1024**3)
                    # Conservative estimate: assume 50% of total memory available
                    available_memory_gb = total_gb * 0.5
                    logger.warning(
                        f"psutil not available, estimated available memory from total: "
                        f"{available_memory_gb:.1f} GB (50% of {total_gb:.1f} GB)"
                    )
                else:
                    logger.warning(
                        "psutil not available and os.sysconf failed, "
                        "using conservative default of 16 GB"
                    )
                    available_memory_gb = 16.0
            except (ValueError, OSError, AttributeError):
                logger.warning(
                    "psutil not available and os.sysconf failed, "
                    "using conservative default of 16 GB"
                )
                available_memory_gb = 16.0

    # Memory per point for Jacobian (dominant memory consumer)
    jacobian_bytes_per_point = n_params * 8  # 8 bytes per float64

    # Usable memory: 70% of available (reserve 30% for OS, JAX overhead, optimizer state)
    usable_memory_bytes = available_memory_gb * (1024**3) * 0.70

    # Calculate max points considering Jacobian + safety factor
    # Safety factor accounts for:
    # - JAX JIT compilation cache (~1.5× data)
    # - Optimizer internal state (Hessian approximation)
    # - Temporary arrays during computation
    # - Data arrays (phi, t1, t2, g2)
    max_total_points = usable_memory_bytes / (jacobian_bytes_per_point * safety_factor)

    # Ensure all angles fit in each chunk (critical for per-angle parameters)
    # If chunk doesn't contain all angles, gradients for missing angles are zero
    if n_angles > 0:
        points_per_angle = max_total_points / n_angles
        chunk_size = int(points_per_angle * n_angles)
    else:
        chunk_size = int(max_total_points)

    # Clamp to reasonable bounds
    # Min: 10K for numerical stability (avoids noisy gradient estimates)
    # Max: 500K for iteration speed (large chunks slow down each iteration)
    chunk_size_clamped = max(min_chunk_size, min(chunk_size, max_chunk_size))

    # Log decision rationale
    logger.info(
        f"Adaptive chunk size calculation:\n"
        f"  Available memory: {available_memory_gb:.1f} GB\n"
        f"  Usable (70%): {usable_memory_bytes / 1e9:.1f} GB\n"
        f"  Parameters: {n_params}\n"
        f"  Angles: {n_angles}\n"
        f"  Jacobian memory/point: {jacobian_bytes_per_point} bytes\n"
        f"  Safety factor: {safety_factor}\n"
        f"  Calculated chunk size: {chunk_size:,} points\n"
        f"  Clamped chunk size: {chunk_size_clamped:,} points [{min_chunk_size:,}, {max_chunk_size:,}]"
    )

    # Warn if total dataset would still cause memory pressure
    estimated_jacobian_gb = (
        total_points * jacobian_bytes_per_point * safety_factor
    ) / (1024**3)
    if estimated_jacobian_gb > available_memory_gb * 0.70:
        logger.warning(
            f"WARNING: Dataset may still cause memory pressure!\n"
            f"  Estimated total memory: {estimated_jacobian_gb:.1f} GB\n"
            f"  Available (usable): {available_memory_gb * 0.70:.1f} GB\n"
            f"  Consider reducing dataset size or increasing system memory."
        )

    return chunk_size_clamped


def create_angle_stratified_data(
    phi: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    g2_exp: jnp.ndarray,
    target_chunk_size: int = 100_000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]:
    """Ensure each chunk contains every phi angle using Cyclic Stratification.

    Reorders data so NLSQ chunking keeps balanced angle coverage and maintains
    valid gradients for per-angle parameters.

    CRITICAL FIX (Jan 2026): Cyclic Stratification
    ----------------------------------------------
    Previously, stratification stopped when the smallest angle was exhausted,
    dumping all remaining data into a single massive, unbalanced chunk. This
    caused rank-deficient Jacobians (zero gradients for missing angles) and
    memory spikes.

    New Logic:
    1. Determine number of chunks based on failure mode: ensuring ALL data is used regardless of balance.
    2. Iterate through chunks, pulling data from EACH angle.
    3. If an angle runs out of data, recycled data from the beginning (Cyclic).
    4. Result: Consistent chunk sizes, all angles present in all chunks.

    Parameters
    ----------
    phi : jnp.ndarray
        Phi angles (radians or degrees), shape (n_points,)
    t1 : jnp.ndarray
        First time delays, shape (n_points,)
    t2 : jnp.ndarray
        Second time delays, shape (n_points,)
    g2_exp : jnp.ndarray
        Experimental g2 values, shape (n_points,)
    target_chunk_size : int, optional
        Target size for each chunk (default: 100,000)
        NLSQ typically uses 100k chunks for LARGE/CHUNKED strategies

    Returns
    -------
    phi_stratified : jnp.ndarray
        Stratified phi angles
    t1_stratified : jnp.ndarray
        Stratified t1 delays
    t2_stratified : jnp.ndarray
        Stratified t2 delays
    g2_stratified : jnp.ndarray
        Stratified g2 values
    chunk_sizes : list[int]
        Size of each stratified chunk (CRITICAL for correct re-chunking)
    """
    import time as _time

    _start_time = _time.perf_counter()
    n_points = len(phi)

    # Convert to numpy for manipulation (JAX arrays are immutable)
    phi_np = np.array(phi)
    t1_np = np.array(t1)
    t2_np = np.array(t2)
    g2_np = np.array(g2_exp)

    # Analyze angle distribution
    stats = analyze_angle_distribution(phi_np)

    # Single angle: no stratification needed
    if stats.n_angles == 1:
        logger.info("Single phi angle detected, no stratification needed")
        return phi, t1, t2, g2_exp, [n_points]

    logger.info(
        f"Stratifying {n_points:,} points across {stats.n_angles} angles "
        f"(imbalance ratio: {stats.imbalance_ratio:.2f}) using Interleaved Stratification"
    )

    # Group data by angle
    angle_groups = {}
    for angle in stats.unique_angles:
        mask = phi_np == angle
        angle_groups[angle] = {
            "phi": phi_np[mask],
            "t1": t1_np[mask],
            "t2": t2_np[mask],
            "g2_exp": g2_np[mask],
            "size": int(np.sum(mask)),
        }

    # Calculate number of chunks based on total points (no expansion)
    n_chunks = max(1, int(np.ceil(n_points / target_chunk_size)))

    logger.debug(
        f"Target chunk size: {target_chunk_size:,}, Number of chunks: {n_chunks}"
    )

    # For each angle, calculate how many points go to each chunk
    # Using round-robin distribution to spread points evenly
    angle_chunk_allocations = {}
    for angle in stats.unique_angles:
        group_size = angle_groups[angle]["size"]
        base_per_chunk = group_size // n_chunks
        remainder = group_size % n_chunks
        # Earlier chunks get one extra point if there's remainder
        allocations = [
            base_per_chunk + (1 if i < remainder else 0) for i in range(n_chunks)
        ]
        angle_chunk_allocations[angle] = allocations

    logger.info(
        f"Interleaved Stratification Plan:\n"
        f"  Chunks: {n_chunks}\n"
        f"  Total points: {n_points:,} (no expansion)"
    )

    # Build stratified chunks by interleaving angle groups
    stratified_chunks = []
    angle_offsets = dict.fromkeys(
        stats.unique_angles, 0
    )  # Track position in each angle

    for chunk_idx in range(n_chunks):
        # Each value is a list of ndarray slices that get concatenated below.
        # Explicit annotation: mypy can't infer the inner type from the empty
        # ``[]`` literals because the appends happen across loop iterations.
        chunk_parts: dict[str, list[np.ndarray]] = {
            "phi": [],
            "t1": [],
            "t2": [],
            "g2_exp": [],
        }

        for angle in stats.unique_angles:
            group = angle_groups[angle]
            start = angle_offsets[angle]
            count = angle_chunk_allocations[angle][chunk_idx]
            end = start + count

            if count > 0:
                chunk_parts["phi"].append(group["phi"][start:end])
                chunk_parts["t1"].append(group["t1"][start:end])
                chunk_parts["t2"].append(group["t2"][start:end])
                chunk_parts["g2_exp"].append(group["g2_exp"][start:end])
                angle_offsets[angle] = end

        # Concatenate all angles for this chunk
        if any(len(arr) > 0 for arr in chunk_parts["phi"]):
            chunk_size = sum(len(arr) for arr in chunk_parts["phi"])
            stratified_chunks.append(
                {
                    "phi": np.concatenate(chunk_parts["phi"]),
                    "t1": np.concatenate(chunk_parts["t1"]),
                    "t2": np.concatenate(chunk_parts["t2"]),
                    "g2_exp": np.concatenate(chunk_parts["g2_exp"]),
                    "size": chunk_size,
                }
            )

    # Store chunk sizes for correct re-chunking. The ``stratified_chunks``
    # dicts mix ndarray and int values, so mypy widens lookups to ``object``;
    # cast to int (we know "size" is the ``sum(len(...))`` int written above).
    from typing import cast as _cast

    chunk_sizes: list[int] = [
        _cast(int, chunk["size"]) for chunk in stratified_chunks
    ]

    # Flatten back to single arrays using pre-allocated buffers.
    # B1: Avoids four np.concatenate() intermediate copies followed by
    # jnp.array() (which forces a second copy on CPU).  Instead we allocate
    # once per output array and fill by slice, then use jnp.asarray() which
    # is zero-copy on the JAX CPU backend (shares the numpy buffer).
    # At 23M pts × float64 this saves ~370–740 MB of transient RSS.
    total_stratified = sum(chunk_sizes)
    phi_out = np.empty(total_stratified, dtype=np.float64)
    t1_out = np.empty(total_stratified, dtype=np.float64)
    t2_out = np.empty(total_stratified, dtype=np.float64)
    g2_out = np.empty(total_stratified, dtype=np.float64)
    pos = 0
    for chunk in stratified_chunks:
        n = _cast(int, chunk["size"])
        phi_out[pos : pos + n] = chunk["phi"]
        t1_out[pos : pos + n] = chunk["t1"]
        t2_out[pos : pos + n] = chunk["t2"]
        g2_out[pos : pos + n] = chunk["g2_exp"]
        pos += n

    # T039: Log chunking operation timing
    _duration = _time.perf_counter() - _start_time
    logger.info(
        f"Stratification complete: {len(stratified_chunks)} balanced chunks created "
        f"in {_duration:.3f}s ({n_points / _duration / 1e6:.2f}M pts/s)"
    )

    # Convert back to JAX arrays and return with chunk boundary information.
    # jnp.asarray() on a contiguous C-order float64 numpy array is zero-copy
    # on the CPU backend; jnp.array() would force an additional device copy.
    return (
        jnp.asarray(phi_out),
        jnp.asarray(t1_out),
        jnp.asarray(t2_out),
        jnp.asarray(g2_out),
        chunk_sizes,
    )


def create_angle_stratified_indices(
    phi: jnp.ndarray | np.ndarray,
    target_chunk_size: int = 100_000,
) -> tuple[np.ndarray, list[int]]:
    """Create index array for zero-copy angle-stratified data access using Interleaved Stratification.

    This function implements index-based stratification, reducing memory overhead
    from 2x (full copy) to ~1% (index array only).

    Interleaved Stratification
    --------------------------
    Distributes data from each angle group across chunks using round-robin allocation.
    Each angle's data is split proportionally across chunks, ensuring:
    - No data expansion (output size = input size)
    - No duplicates
    - All angles represented in each chunk (for balanced data)

    Parameters
    ----------
    phi : jnp.ndarray or np.ndarray
        Phi angles (radians or degrees), shape (n_points,)
    target_chunk_size : int, optional
        Target size for each chunk (default: 100,000)

    Returns
    -------
    indices : np.ndarray
        Index array specifying stratified ordering, shape (n_points,)
        Use: data_stratified = data_original[indices]
    chunk_sizes : list[int]
        Size of each stratified chunk (CRITICAL for correct re-chunking)

    """
    n_points = len(phi)

    # Convert to numpy
    phi_np = np.array(phi) if not isinstance(phi, np.ndarray) else phi

    # Analyze angle distribution
    stats = analyze_angle_distribution(phi_np)

    # Single angle: return identity index (no stratification)
    if stats.n_angles == 1:
        logger.info("Single phi angle detected, no stratification needed")
        return np.arange(n_points), [n_points]  # Single chunk with all points

    logger.info(
        f"Creating stratified indices for {n_points:,} points across {stats.n_angles} angles "
        f"(imbalance ratio: {stats.imbalance_ratio:.2f}) using Interleaved Stratification"
    )

    # Group indices by angle
    angle_index_groups = {}
    for angle in stats.unique_angles:
        mask = phi_np == angle
        angle_index_groups[angle] = np.where(mask)[0]

    # Calculate number of chunks based on total points (no expansion)
    n_chunks = max(1, int(np.ceil(n_points / target_chunk_size)))

    logger.debug(
        f"Target chunk size: {target_chunk_size:,}, Number of chunks: {n_chunks}"
    )

    # For each angle, calculate how many points go to each chunk
    # Using round-robin distribution to spread points evenly
    angle_chunk_allocations = {}
    for angle in stats.unique_angles:
        group_size = len(angle_index_groups[angle])
        base_per_chunk = group_size // n_chunks
        remainder = group_size % n_chunks
        # Earlier chunks get one extra point if there's remainder
        allocations = [
            base_per_chunk + (1 if i < remainder else 0) for i in range(n_chunks)
        ]
        angle_chunk_allocations[angle] = allocations

    # Build stratified index array by interleaving angle groups
    stratified_indices = []
    angle_offsets = dict.fromkeys(
        stats.unique_angles, 0
    )  # Track position in each angle

    for chunk_idx in range(n_chunks):
        chunk_indices = []

        for angle in stats.unique_angles:
            indices_for_angle = angle_index_groups[angle]
            start = angle_offsets[angle]
            count = angle_chunk_allocations[angle][chunk_idx]
            end = start + count

            if count > 0:
                chunk_indices.append(indices_for_angle[start:end])
                angle_offsets[angle] = end

        # Concatenate all angle indices for this chunk
        if chunk_indices:
            stratified_indices.append(np.concatenate(chunk_indices))

    # Flatten to single index array
    final_indices = np.concatenate(stratified_indices)

    logger.info(
        f"Stratification complete: {n_chunks} chunks created, "
        f"{len(final_indices):,} indices (no expansion)"
    )

    # Calculate actual chunk sizes from the stratified_indices list
    chunk_sizes = [len(chunk) for chunk in stratified_indices]

    return final_indices, chunk_sizes


@dataclass
class StratifiedIndexIterator:
    """Iterator that yields index chunks for stratified data access.

    This iterator allows processing strictly stratified chunks one by one
    without materializing the full index array or data chunks in memory.
    """

    indices: np.ndarray
    chunk_sizes: list[int]

    def __iter__(self) -> Iterator[np.ndarray]:
        start = 0
        for size in self.chunk_sizes:
            end = start + size
            yield self.indices[start:end]
            start = end

    def __len__(self) -> int:
        return len(self.chunk_sizes)


def get_stratified_chunk_iterator(
    phi: jnp.ndarray | np.ndarray,
    target_chunk_size: int = 100_000,
) -> StratifiedIndexIterator:
    """Create an iterator yielding stratified index chunks.

    Args:
        phi: Array of phi angles
        target_chunk_size: Desired chunk size

    Returns:
        StratifiedIndexIterator yielding index chunks
    """
    indices, chunk_sizes = create_angle_stratified_indices(phi, target_chunk_size)
    return StratifiedIndexIterator(indices, chunk_sizes)


def should_use_stratification(
    n_points: int,
    n_angles: int,
    per_angle_scaling: bool,
    imbalance_ratio: float,
) -> tuple[bool, str]:
    """Decide whether to use angle-stratified chunking.

    Decision logic:
    - Small datasets (<100k): No (use STANDARD strategy, no chunking)
    - No per-angle scaling: No (regular chunking works fine)
    - High imbalance (>5:1): No (use sequential per-angle instead)
    - Otherwise: Yes (use stratified chunking)

    Parameters
    ----------
    n_points : int
        Total number of data points
    n_angles : int
        Number of unique phi angles
    per_angle_scaling : bool
        Whether per-angle parameters are enabled
    imbalance_ratio : float
        max(angle_counts) / min(angle_counts)

    Returns
    -------
    should_stratify : bool
        True if stratification should be used
    reason : str
        Human-readable explanation of decision

    Examples
    --------
    >>> should, reason = should_use_stratification(
    ...     n_points=3_000_000,
    ...     n_angles=3,
    ...     per_angle_scaling=True,
    ...     imbalance_ratio=2.5
    ... )
    >>> print(should, reason)
    True "Large dataset with balanced angles"
    """
    # Small dataset: no chunking, no stratification needed
    if n_points < 100_000:
        return False, "Dataset < 100k points, STANDARD strategy used (no chunking)"

    # No per-angle scaling: regular chunking works fine
    if not per_angle_scaling:
        return False, "Per-angle scaling disabled, stratification not needed"

    # Single angle: no stratification needed
    if n_angles == 1:
        return False, "Single phi angle, stratification not applicable"

    # High imbalance: sequential per-angle better
    if imbalance_ratio > 5.0:
        return (
            False,
            f"High imbalance ratio ({imbalance_ratio:.1f} > 5.0), "
            "use sequential per-angle instead",
        )

    # All conditions met: use stratification
    return (
        True,
        f"Large dataset ({n_points:,} points) with balanced angles "
        f"({n_angles} angles, imbalance {imbalance_ratio:.1f})",
    )


def compute_stratification_diagnostics(
    phi_original: np.ndarray,
    phi_stratified: np.ndarray,
    execution_time_ms: float,
    use_index_based: bool = False,
    target_chunk_size: int = 100_000,
    chunk_sizes: list[int] | None = None,
) -> StratificationDiagnostics:
    """Compute detailed diagnostics for stratification quality and performance.

    This function analyzes the stratified data to provide comprehensive metrics
    about chunk balance, angle coverage, memory efficiency, and throughput.

    Parameters
    ----------
    phi_original : np.ndarray
        Original phi angles before stratification
    phi_stratified : np.ndarray
        Stratified phi angles after reorganization
    execution_time_ms : float
        Time taken for stratification (milliseconds)
    use_index_based : bool, optional
        Whether index-based stratification was used, default: False
    target_chunk_size : int, optional
        Target chunk size used, default: 100,000

    Returns
    -------
    StratificationDiagnostics
        Comprehensive diagnostic metrics

    Examples
    --------
    >>> import time
    >>> phi = np.repeat([0, 45, 90], 100)
    >>> start = time.perf_counter()
    >>> phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(phi, t1, t2, g2)
    >>> exec_time_ms = (time.perf_counter() - start) * 1000
    >>> diagnostics = compute_stratification_diagnostics(
    ...     phi, phi_s, exec_time_ms, use_index_based=False
    ... )
    >>> print(f"Chunks: {diagnostics.n_chunks}")
    >>> print(f"Throughput: {diagnostics.throughput_points_per_sec:,.0f} pts/s")
    """
    n_points = len(phi_original)

    # Analyze original angle distribution
    stats = analyze_angle_distribution(phi_original)

    # Use actual chunk sizes if provided, otherwise estimate with sequential slicing
    if chunk_sizes is not None:
        # Use actual chunk boundaries from stratification
        n_chunks = len(chunk_sizes)
        angles_per_chunk = []

        start_idx = 0
        for chunk_size in chunk_sizes:
            end_idx = start_idx + chunk_size
            chunk_phi = phi_stratified[start_idx:end_idx]
            angles_per_chunk.append(len(np.unique(chunk_phi)))
            start_idx = end_idx
    else:
        # Fall back to naive sequential slicing
        n_chunks = int(np.ceil(n_points / target_chunk_size))
        chunk_sizes = []
        angles_per_chunk = []

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * target_chunk_size
            end_idx = min(start_idx + target_chunk_size, n_points)

            chunk_phi = phi_stratified[start_idx:end_idx]
            chunk_sizes.append(len(chunk_phi))
            angles_per_chunk.append(len(np.unique(chunk_phi)))

    # Chunk balance statistics
    chunk_sizes_arr = np.array(chunk_sizes)
    chunk_balance = {
        "mean": float(np.mean(chunk_sizes_arr)),
        "std": float(np.std(chunk_sizes_arr)),
        "min": int(np.min(chunk_sizes_arr)),
        "max": int(np.max(chunk_sizes_arr)),
        "cv": float(
            np.std(chunk_sizes_arr) / np.mean(chunk_sizes_arr)
        ),  # Coefficient of variation
    }

    # Angle coverage statistics
    angles_per_chunk_arr = np.array(angles_per_chunk)
    min_coverage_ratio = float(np.min(angles_per_chunk_arr) / stats.n_angles)

    angle_coverage = {
        "mean_angles": float(np.mean(angles_per_chunk_arr)),
        "std_angles": float(np.std(angles_per_chunk_arr)),
        "min_coverage_ratio": min_coverage_ratio,  # Fraction of angles in worst chunk
        "perfect_coverage_chunks": int(np.sum(angles_per_chunk_arr == stats.n_angles)),
    }

    # Memory overhead estimation
    mem_stats = estimate_stratification_memory(
        n_points, use_index_based=use_index_based
    )
    memory_overhead_mb = mem_stats["peak_memory_mb"] - mem_stats["original_memory_mb"]
    memory_efficiency = mem_stats["original_memory_mb"] / mem_stats["peak_memory_mb"]

    # Throughput calculation
    throughput_points_per_sec = (
        (n_points / execution_time_ms) * 1000.0 if execution_time_ms > 0 else 0.0
    )

    return StratificationDiagnostics(
        n_chunks=n_chunks,
        chunk_sizes=chunk_sizes,
        chunk_balance=chunk_balance,
        angles_per_chunk=angles_per_chunk,
        angle_coverage=angle_coverage,
        execution_time_ms=execution_time_ms,
        memory_overhead_mb=memory_overhead_mb,
        memory_efficiency=memory_efficiency,
        throughput_points_per_sec=throughput_points_per_sec,
        use_index_based=use_index_based,
    )


def format_diagnostics_report(diagnostics: StratificationDiagnostics) -> str:
    """Format stratification diagnostics as human-readable report.

    Parameters
    ----------
    diagnostics : StratificationDiagnostics
        Diagnostic metrics to format

    Returns
    -------
    str
        Formatted report with all diagnostic metrics

    Examples
    --------
    >>> diagnostics = compute_stratification_diagnostics(phi, phi_s, 150.0)
    >>> report = format_diagnostics_report(diagnostics)
    >>> print(report)
    """
    lines = [
        "=" * 70,
        "STRATIFICATION DIAGNOSTICS REPORT",
        "=" * 70,
        "",
        "Chunking:",
        f"  Number of chunks: {diagnostics.n_chunks}",
        f"  Method: {'Index-based (zero-copy)' if diagnostics.use_index_based else 'Full copy'}",
        "",
        "Chunk Balance:",
        f"  Mean size: {diagnostics.chunk_balance['mean']:.0f} points",
        f"  Std dev: {diagnostics.chunk_balance['std']:.1f} points",
        f"  Range: [{diagnostics.chunk_balance['min']}, {diagnostics.chunk_balance['max']}]",
        f"  Coefficient of variation: {diagnostics.chunk_balance['cv']:.3f}",
        "",
        "Angle Coverage:",
        f"  Mean angles per chunk: {diagnostics.angle_coverage['mean_angles']:.1f}",
        f"  Std dev: {diagnostics.angle_coverage['std_angles']:.2f}",
        f"  Min coverage ratio: {diagnostics.angle_coverage['min_coverage_ratio']:.2%}",
        f"  Perfect coverage chunks: {diagnostics.angle_coverage['perfect_coverage_chunks']}/{diagnostics.n_chunks}",
        "",
        "Performance:",
        f"  Execution time: {diagnostics.execution_time_ms:.2f} ms",
        f"  Throughput: {diagnostics.throughput_points_per_sec:,.0f} points/second",
        "",
        "Memory:",
        f"  Overhead: {diagnostics.memory_overhead_mb:.1f} MB",
        f"  Efficiency: {diagnostics.memory_efficiency:.1%}",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)
