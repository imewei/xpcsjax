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

# Shared concurrency detector (single source of truth for the divisor precedence:
# explicit arg > XPCSJAX_FIT_CONCURRENCY > PYTEST_XDIST_WORKER_COUNT > 1). Only the
# strategy *names* differ between the homodyne and heterodyne flavors; the
# overcommit-prevention divisor is identical, so it is imported, not duplicated.
from xpcsjax.optimization.nlsq.memory import (
    _detect_fit_concurrency,
)
from xpcsjax.optimization.nlsq.memory import (
    detect_available_system_memory as _detect_available_bytes,
)
from xpcsjax.optimization.nlsq.memory import (
    detect_total_system_memory as _detect_total_bytes,
)
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


_BYTES_PER_GB: float = 1024**3


def detect_total_system_memory() -> float | None:
    """Detect total system memory in GB.

    Thin GB-returning wrapper over the homodyne flavor
    (:func:`xpcsjax.optimization.nlsq.memory.detect_total_system_memory`), which
    returns bytes. The detection logic is shared, not duplicated.

    Returns
    -------
    float | None
        Total memory in GB, or ``None`` if detection fails.
    """
    total_bytes = _detect_total_bytes()
    return None if total_bytes is None else total_bytes / _BYTES_PER_GB


def detect_available_system_memory() -> float | None:
    """Detect currently *available* system memory in GB.

    Thin GB-returning wrapper over the homodyne flavor
    (:func:`xpcsjax.optimization.nlsq.memory.detect_available_system_memory`),
    which returns bytes. Available memory is the preferred budget basis.

    Returns
    -------
    float | None
        Available memory in GB, or ``None`` if detection fails.
    """
    available_bytes = _detect_available_bytes()
    return None if available_bytes is None else available_bytes / _BYTES_PER_GB


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


def _get_memory_threshold(memory_fraction: float, *, concurrency: int | None = None) -> float:
    """Compute memory threshold in GB (concurrency-aware).

    Checks ``HETERODYNE_MEMORY_FRACTION`` env-var, clamps the fraction to
    ``[0.1, 0.9]``, prefers *available* over total memory as the basis, divides
    by the detected fit concurrency (overcommit prevention — see the homodyne
    flavor), and falls back to :data:`FALLBACK_THRESHOLD_GB` when detection fails.
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

    divisor = _detect_fit_concurrency(concurrency)

    # Available is the preferred basis; total is the secondary basis.
    basis_gb = detect_available_system_memory()
    memory_basis = "available"
    if basis_gb is None:
        basis_gb = detect_total_system_memory()
        memory_basis = "total"
    if basis_gb is None:
        logger.warning(
            "Could not detect system memory; using fallback threshold %.1f GB",
            FALLBACK_THRESHOLD_GB,
        )
        return FALLBACK_THRESHOLD_GB

    threshold = basis_gb * memory_fraction / divisor
    logger.debug(
        "System memory: %.1f GB %s / %d concurrent, threshold: %.1f GB (%.0f%%)",
        basis_gb,
        memory_basis,
        divisor,
        threshold,
        memory_fraction * 100,
    )
    return threshold


def select_nlsq_strategy(
    n_points: int,
    n_params: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    *,
    concurrency: int | None = None,
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
    concurrency : int | None, optional
        Number of fits expected to run concurrently; the budget is divided by
        this so N parallel fits don't each claim the whole box. ``None``
        auto-detects from the fit-pool / pytest-xdist env-vars (default 1).

    Returns
    -------
    StrategyDecision
        Decision with selected strategy and rationale.
    """
    threshold_gb = _get_memory_threshold(memory_fraction, concurrency=concurrency)

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
