"""Memory Management and Unified Strategy Selection for NLSQ Optimization.

Provides adaptive memory threshold detection and unified memory-based strategy
selection for NLSQ optimization (v2.13.0+).

Key Features:
- Cross-platform system memory detection (psutil + os.sysconf fallback)
- Adaptive threshold calculation based on available memory
- Unified memory-based strategy selection (NLSQStrategy, select_nlsq_strategy)
- Environment variable override support (NLSQ_MEMORY_FRACTION)
- Safe fraction clamping to prevent OOM or underutilization

Strategy Selection (v2.13.0+):
    >>> from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy
    >>> decision = select_nlsq_strategy(n_points=100_000_000, n_params=53)
    >>> print(decision.strategy.value)  # 'standard', 'out_of_core', or 'hybrid_streaming'

Memory Threshold:
    >>> from xpcsjax.optimization.nlsq.memory import get_adaptive_memory_threshold
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> print(f"Threshold: {threshold_gb:.1f} GB")
"""

import os
import warnings
from typing import Any

from xpcsjax.utils.logging import get_logger, log_phase

# Check if NLSQ MemoryManager is available (v0.4+)
try:
    _NLSQ_MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    _nlsq_get_memory_manager = None
    _NLSQ_MEMORY_MANAGER_AVAILABLE = False

# Module-level logger
logger = get_logger(__name__)

# Default memory fraction and environment variable name
DEFAULT_MEMORY_FRACTION = 0.75
MEMORY_FRACTION_ENV_VAR = "NLSQ_MEMORY_FRACTION"
FALLBACK_THRESHOLD_GB = 16.0
MIN_MEMORY_FRACTION = 0.1
MAX_MEMORY_FRACTION = 0.9


def detect_total_system_memory() -> float | None:
    """Detect total system memory in bytes using multiple methods.

    Returns
    -------
    float | None
        Total system memory in bytes, or None if detection fails.

    Notes
    -----
    Detection priority:
    1. psutil.virtual_memory().total (preferred, cross-platform)
    2. os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') (Linux fallback)
    """
    # Method 1: psutil (preferred, cross-platform)
    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
        if total_bytes > 0:
            return float(total_bytes)
    except ImportError:
        logger.debug("psutil not available, trying os.sysconf fallback")
    except (OSError, ValueError, AttributeError) as e:
        logger.debug(f"psutil memory detection failed: {e}")

    # Method 2: os.sysconf (Linux/Unix fallback)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            total_bytes = page_size * phys_pages
            return float(total_bytes)
    except (ValueError, OSError, AttributeError) as e:
        logger.debug(f"os.sysconf memory detection failed: {e}")

    return None


def get_adaptive_memory_threshold(
    memory_fraction: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute adaptive memory threshold based on system memory.

    The memory threshold determines when NLSQ switches to streaming mode
    for memory-bounded optimization. Instead of a fixed 16 GB threshold,
    this function computes an adaptive threshold as a fraction of total
    system memory.

    Parameters
    ----------
    memory_fraction : float | None, optional
        Fraction of total system memory to use as threshold (0.1 to 0.9).
        If None, uses:
        1. Environment variable NLSQ_MEMORY_FRACTION (if set)
        2. Default value of 0.75 (75% of total memory)

    Returns
    -------
    threshold_gb : float
        Memory threshold in gigabytes.
    info : dict
        Diagnostic information with keys:
        - 'total_memory_gb': Detected total system memory (GB)
        - 'memory_fraction': Fraction used
        - 'source': How the fraction was determined ('argument', 'env', 'default')
        - 'detection_method': How memory was detected ('psutil', 'sysconf', 'fallback')

    Notes
    -----
    - If total memory cannot be detected, falls back to 16.0 GB with a warning.
    - Memory fraction is clamped to [0.1, 0.9] for safety.
    - Environment variable NLSQ_MEMORY_FRACTION can override the default.

    Examples
    --------
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> pct = info['memory_fraction'] * 100
    >>> tot = info['total_memory_gb']
    >>> print(f"Threshold: {threshold_gb:.1f} GB ({pct:.0f}% of {tot:.1f} GB)")
    Threshold: 24.0 GB (75% of 32.0 GB)

    >>> # Override with specific fraction
    >>> threshold_gb, _ = get_adaptive_memory_threshold(memory_fraction=0.5)

    >>> # Override via environment variable
    >>> import os
    >>> os.environ["NLSQ_MEMORY_FRACTION"] = "0.6"
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> assert info['source'] == 'env'
    """
    info: dict[str, Any] = {}

    # Step 1: Determine memory fraction
    fraction_source = "default"
    effective_fraction = DEFAULT_MEMORY_FRACTION

    if memory_fraction is not None:
        # Use explicit argument
        effective_fraction = memory_fraction
        fraction_source = "argument"
    else:
        # Check environment variable
        env_value = os.environ.get(MEMORY_FRACTION_ENV_VAR)
        if env_value is not None:
            try:
                effective_fraction = float(env_value)
                fraction_source = "env"
            except ValueError:
                warnings.warn(
                    f"Invalid {MEMORY_FRACTION_ENV_VAR}='{env_value}', "
                    f"using default {DEFAULT_MEMORY_FRACTION}",
                    UserWarning,
                    stacklevel=2,
                )

    # Step 2: Clamp fraction to safe range
    original_fraction = effective_fraction
    effective_fraction = max(
        MIN_MEMORY_FRACTION, min(effective_fraction, MAX_MEMORY_FRACTION)
    )

    if effective_fraction != original_fraction:
        warnings.warn(
            f"Memory fraction {original_fraction} clamped to "
            f"[{MIN_MEMORY_FRACTION}, {MAX_MEMORY_FRACTION}]: "
            f"using {effective_fraction}",
            UserWarning,
            stacklevel=2,
        )

    info["memory_fraction"] = effective_fraction
    info["source"] = fraction_source

    # Step 3: Detect total system memory
    total_bytes = detect_total_system_memory()

    if total_bytes is not None:
        total_gb = total_bytes / (1024**3)
        threshold_gb = total_gb * effective_fraction

        # Determine detection method for logging
        try:
            import psutil  # noqa: F401

            info["detection_method"] = "psutil"
        except ImportError:
            info["detection_method"] = "sysconf"

        info["total_memory_gb"] = total_gb

        logger.info(
            f"Adaptive memory threshold: {threshold_gb:.1f} GB "
            f"({effective_fraction * 100:.0f}% of {total_gb:.1f} GB total, "
            f"source={fraction_source}, method={info['detection_method']})"
        )

        return threshold_gb, info

    # Step 4: Fallback if memory detection fails
    warnings.warn(
        f"Could not detect system memory. "
        f"Using fallback threshold of {FALLBACK_THRESHOLD_GB} GB. "
        "Install psutil for accurate memory detection: pip install psutil",
        UserWarning,
        stacklevel=2,
    )

    info["total_memory_gb"] = 0.0
    info["detection_method"] = "fallback"

    logger.warning(
        f"Memory detection failed. Using fallback threshold: {FALLBACK_THRESHOLD_GB} GB"
    )

    return FALLBACK_THRESHOLD_GB, info


def estimate_peak_memory_gb(
    n_points: int,
    n_params: int,
    bytes_per_element: int = 8,
    jacobian_overhead: float = 6.5,
) -> float:
    """Estimate peak memory usage for full Jacobian optimization.

    Parameters
    ----------
    n_points : int
        Number of data points
    n_params : int
        Number of parameters
    bytes_per_element : int, optional
        Bytes per float element (default: 8 for float64)
    jacobian_overhead : float, optional
        Multiplicative factor accounting for:
        - Base Jacobian matrix (n_points × n_params)
        - Autodiff intermediate tensors (~2×)
        - Stratified array padding and copies (~1.5×)
        - JIT compilation intermediates (~1.5×)
        - Optimizer working memory (residuals, QR decomp, etc.)
        Default: 6.5 (empirically validated for 23M+ point datasets)

    Returns
    -------
    float
        Estimated peak memory in gigabytes
    """
    # Jacobian matrix: n_points × n_params × bytes
    jacobian_bytes = n_points * n_params * bytes_per_element

    # Total with autodiff + stratification + JIT overhead
    peak_bytes = jacobian_bytes * jacobian_overhead

    return peak_bytes / (1024**3)


from dataclasses import dataclass  # noqa: E402
from enum import Enum  # noqa: E402


class NLSQStrategy(Enum):
    """NLSQ optimization strategy based on memory constraints."""

    STANDARD = "standard"  # In-memory full Jacobian
    OUT_OF_CORE = "out_of_core"  # Chunk-wise J^T J accumulation
    HYBRID_STREAMING = "hybrid_streaming"  # L-BFGS warmup + streaming GN


@dataclass(frozen=True)
class StrategyDecision:
    """Result of unified memory-based strategy selection.

    Attributes
    ----------
    strategy : NLSQStrategy
        Selected optimization strategy
    threshold_gb : float
        Memory threshold used for decision (GB)
    index_memory_gb : float
        Memory required for int64 index array (GB)
    peak_memory_gb : float
        Estimated peak memory for full Jacobian (GB)
    reason : str
        Human-readable explanation of decision
    """

    strategy: NLSQStrategy
    threshold_gb: float
    index_memory_gb: float
    peak_memory_gb: float
    reason: str


def select_nlsq_strategy(
    n_points: int,
    n_params: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> StrategyDecision:
    """Unified memory-based NLSQ strategy selection.

    Implements a pure memory-based decision tree:

    1. If index array > threshold → HYBRID_STREAMING (extreme scale)
    2. Elif peak memory > threshold → OUT_OF_CORE (large scale)
    3. Else → STANDARD (in-memory)

    Parameters
    ----------
    n_points : int
        Number of data points
    n_params : int
        Number of optimization parameters
    memory_fraction : float, optional
        Fraction of system RAM to use as threshold (default: 0.75)

    Returns
    -------
    StrategyDecision
        Decision with strategy, metrics, and rationale

    Examples
    --------
    >>> decision = select_nlsq_strategy(100_000_000, 53)
    >>> print(decision.strategy.value)
    'out_of_core'
    >>> print(decision.reason)
    'Peak memory (12.8 GB) exceeds threshold (24.0 GB)'
    """
    # T038: Add timing for memory strategy selection.
    # The with-block covers the full decision tree so that log_phase
    # captures the complete selection time, not just the metric computation.
    with log_phase("memory_strategy_selection", logger=logger):
        # Get unified memory threshold
        threshold_gb, _ = get_adaptive_memory_threshold(memory_fraction)

        # Compute memory metrics
        index_memory_bytes = n_points * 8  # int64 indices
        index_memory_gb = index_memory_bytes / (1024**3)

        # Handle edge case: n_params=0 means we can't estimate properly
        if n_params <= 0:
            peak_memory_gb = 0.0
        else:
            peak_memory_gb = estimate_peak_memory_gb(n_points, n_params)

        logger.debug(
            f"Memory strategy analysis: "
            f"n_points={n_points:,}, n_params={n_params}, "
            f"index={index_memory_gb:.2f} GB, peak={peak_memory_gb:.2f} GB, "
            f"threshold={threshold_gb:.2f} GB"
        )

        # Decision tree (check index FIRST - extreme case)
        if index_memory_gb > threshold_gb:
            # Performance Optimization (Spec 001 - T051): Log auto-streaming mode activation
            logger.info(
                f"Auto-switching to HYBRID_STREAMING: index array ({index_memory_gb:.2f} GB) "
                f"exceeds threshold ({threshold_gb:.2f} GB)"
            )
            return StrategyDecision(
                strategy=NLSQStrategy.HYBRID_STREAMING,
                threshold_gb=threshold_gb,
                index_memory_gb=index_memory_gb,
                peak_memory_gb=peak_memory_gb,
                reason=(
                    f"Index array ({index_memory_gb:.2f} GB) exceeds "
                    f"threshold ({threshold_gb:.2f} GB)"
                ),
            )

        if peak_memory_gb > threshold_gb:
            # Performance Optimization (Spec 001 - T051): Log auto-streaming mode activation
            logger.info(
                f"Auto-switching to OUT_OF_CORE: peak memory ({peak_memory_gb:.2f} GB) "
                f"exceeds threshold ({threshold_gb:.2f} GB)"
            )
            return StrategyDecision(
                strategy=NLSQStrategy.OUT_OF_CORE,
                threshold_gb=threshold_gb,
                index_memory_gb=index_memory_gb,
                peak_memory_gb=peak_memory_gb,
                reason=(
                    f"Peak memory ({peak_memory_gb:.2f} GB) exceeds "
                    f"threshold ({threshold_gb:.2f} GB)"
                ),
            )

        return StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=threshold_gb,
            index_memory_gb=index_memory_gb,
            peak_memory_gb=peak_memory_gb,
            reason=(
                f"Memory fits: {peak_memory_gb:.2f} GB < {threshold_gb:.2f} GB threshold"
            ),
        )


__all__ = [
    "DEFAULT_MEMORY_FRACTION",
    "MEMORY_FRACTION_ENV_VAR",
    "FALLBACK_THRESHOLD_GB",
    "detect_total_system_memory",
    "get_adaptive_memory_threshold",
    "estimate_peak_memory_gb",
    # Unified strategy selection (v2.13.0)
    "NLSQStrategy",
    "StrategyDecision",
    "select_nlsq_strategy",
]
