"""Memory-aware strategy selection for NLSQ optimization (**heterodyne** flavor).

This module mirrors :mod:`xpcsjax.optimization.nlsq.memory` (the homodyne flavor)
but uses heterodyne-shaped naming — ``STANDARD``, ``LARGE``, ``STREAMING`` —
because the two-component residual layout has a different memory footprint
than homodyne's. Do not collapse the two modules together: the strategy names
are load-bearing in callers (``wrapper.py`` imports homodyne names;
``heterodyne_core.py`` imports the names from this module).

Estimates peak memory usage from Jacobian size and selects between
standard (in-memory), large (chunked J^T J), and streaming (L-BFGS
warmup + streaming Gauss-Newton) strategies.

Strategy decision tree:
    1. Index array alone > threshold  ->  STREAMING  (extreme scale)
    2. Peak Jacobian memory > threshold  ->  LARGE  (out-of-core chunks)
    3. Otherwise  ->  STANDARD  (full in-memory Jacobian)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_FRACTION: float = 0.75
"""Fraction of system RAM used as the memory threshold."""

FALLBACK_THRESHOLD_GB: float = 16.0
"""Threshold (GB) when system memory cannot be detected."""

MEMORY_FRACTION_ENV_VAR: str = "HETERODYNE_MEMORY_FRACTION"
"""Environment variable that overrides *memory_fraction*."""

_MIN_FRACTION: float = 0.1
_MAX_FRACTION: float = 0.9
_JACOBIAN_OVERHEAD: float = 6.5
"""Overhead factor: base Jacobian + autodiff intermediates + JIT + workspace."""


# ---------------------------------------------------------------------------
# Strategy enum & decision dataclass
# ---------------------------------------------------------------------------


class NLSQStrategy(Enum):
    """NLSQ optimization strategy based on memory constraints."""

    STANDARD = "standard"
    LARGE = "large"
    STREAMING = "streaming"


@dataclass(frozen=True)
class StrategyDecision:
    """Result of memory-based strategy selection.

    Attributes
    ----------
    strategy : NLSQStrategy
        Selected optimization strategy.
    threshold_gb : float
        Memory threshold used for the decision (GB).
    peak_memory_gb : float
        Estimated peak memory for the full Jacobian (GB).
    reason : str
        Human-readable explanation of the decision.
    """

    strategy: NLSQStrategy
    threshold_gb: float
    peak_memory_gb: float
    reason: str


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------


def detect_total_system_memory() -> float | None:
    """Detect total system memory in GB.

    Tries ``psutil`` first, then ``os.sysconf`` (Linux/macOS).

    Returns
    -------
    float | None
        Total memory in GB, or ``None`` if detection fails.
    """
    # Method 1: psutil (preferred, cross-platform)
    try:
        import psutil

        total = psutil.virtual_memory().total
        if total > 0:
            return float(total) / (1024**3)
    except ImportError:
        logger.debug("psutil not available, trying os.sysconf fallback")
    except (OSError, ValueError, AttributeError) as exc:
        logger.debug("psutil memory detection failed: %s", exc)

    # Method 2: os.sysconf (Linux/Unix)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return float(page_size * phys_pages) / (1024**3)
    except (ValueError, OSError, AttributeError) as exc:
        logger.debug("os.sysconf memory detection failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_peak_memory_gb(
    n_points: int,
    n_params: int,
    *,
    bytes_per_element: int = 8,
    jacobian_overhead: float = _JACOBIAN_OVERHEAD,
) -> float:
    """Estimate peak memory for full-Jacobian NLSQ optimization.

    The dominant cost is the Jacobian matrix ``(n_points x n_params)``
    multiplied by an overhead factor that accounts for autodiff
    intermediates, JIT compilation buffers, and optimizer workspace.

    Parameters
    ----------
    n_points : int
        Residual vector length.
    n_params : int
        Number of varying parameters.
    bytes_per_element : int
        Bytes per array element (default 8 for float64).
    jacobian_overhead : float
        Multiplicative overhead factor (default 6.5).

    Returns
    -------
    float
        Estimated peak memory in GB.
    """
    jacobian_bytes = n_points * n_params * bytes_per_element
    return (jacobian_bytes * jacobian_overhead) / (1024**3)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def _get_memory_threshold(memory_fraction: float) -> float:
    """Compute memory threshold in GB.

    Checks ``HETERODYNE_MEMORY_FRACTION`` env-var, clamps the fraction
    to ``[0.1, 0.9]``, and falls back to :data:`FALLBACK_THRESHOLD_GB`
    when detection fails.
    """
    # Environment override
    env_val = os.environ.get(MEMORY_FRACTION_ENV_VAR)
    if env_val is not None:
        try:
            memory_fraction = float(env_val)
        except ValueError:
            logger.warning(
                "Invalid %s=%r, using default=%.2f",
                MEMORY_FRACTION_ENV_VAR,
                env_val,
                memory_fraction,
            )

    # Clamp
    memory_fraction = max(_MIN_FRACTION, min(_MAX_FRACTION, memory_fraction))

    total_gb = detect_total_system_memory()
    if total_gb is None:
        logger.warning(
            "Could not detect system memory; using fallback threshold %.1f GB",
            FALLBACK_THRESHOLD_GB,
        )
        return FALLBACK_THRESHOLD_GB

    threshold = total_gb * memory_fraction
    logger.debug(
        "System memory: %.1f GB, threshold: %.1f GB (%.0f%%)",
        total_gb,
        threshold,
        memory_fraction * 100,
    )
    return threshold


def select_nlsq_strategy(
    n_points: int,
    n_params: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> StrategyDecision:
    """Select NLSQ strategy based on estimated memory usage.

    Decision tree (evaluated top-down):

    1. **STREAMING** — index array alone exceeds threshold (extreme scale).
    2. **LARGE** — peak Jacobian memory exceeds threshold.
    3. **STANDARD** — everything fits in memory.

    Parameters
    ----------
    n_points : int
        Number of data points.
    n_params : int
        Number of varying parameters.
    memory_fraction : float
        Fraction of system memory to use as threshold (default 0.75).

    Returns
    -------
    StrategyDecision
        Decision with selected strategy and rationale.
    """
    threshold_gb = _get_memory_threshold(memory_fraction)

    # Index array cost (int64 per point)
    index_gb = (n_points * 8) / (1024**3)

    peak_gb = estimate_peak_memory_gb(n_points, n_params) if n_params > 0 else 0.0

    logger.debug(
        "Strategy analysis: n_points=%s, n_params=%d, "
        "index=%.2f GB, peak=%.2f GB, threshold=%.2f GB",
        f"{n_points:,}",
        n_params,
        index_gb,
        peak_gb,
        threshold_gb,
    )

    # 1. Extreme scale — even the index array blows memory
    if index_gb > threshold_gb:
        reason = f"Index array ({index_gb:.2f} GB) exceeds threshold ({threshold_gb:.2f} GB)"
        logger.info("Auto-selecting STREAMING: %s", reason)
        return StrategyDecision(
            strategy=NLSQStrategy.STREAMING,
            threshold_gb=threshold_gb,
            peak_memory_gb=peak_gb,
            reason=reason,
        )

    # 2. Large scale — Jacobian doesn't fit
    if peak_gb > threshold_gb:
        reason = f"Peak memory ({peak_gb:.2f} GB) exceeds threshold ({threshold_gb:.2f} GB)"
        logger.info("Auto-selecting LARGE: %s", reason)
        return StrategyDecision(
            strategy=NLSQStrategy.LARGE,
            threshold_gb=threshold_gb,
            peak_memory_gb=peak_gb,
            reason=reason,
        )

    # 3. Standard — fits in memory
    reason = f"Peak memory ({peak_gb:.2f} GB) within threshold ({threshold_gb:.2f} GB)"
    logger.debug("Selecting STANDARD: %s", reason)
    return StrategyDecision(
        strategy=NLSQStrategy.STANDARD,
        threshold_gb=threshold_gb,
        peak_memory_gb=peak_gb,
        reason=reason,
    )


__all__ = [
    "DEFAULT_MEMORY_FRACTION",
    "FALLBACK_THRESHOLD_GB",
    "MEMORY_FRACTION_ENV_VAR",
    "NLSQStrategy",
    "StrategyDecision",
    "detect_total_system_memory",
    "estimate_peak_memory_gb",
    "select_nlsq_strategy",
]
