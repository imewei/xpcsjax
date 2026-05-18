"""Shared Data Types for Homodyne Data Layer
=============================================

Common dataclasses and type definitions shared across the data module.
Extracted to prevent circular imports between optimization.py and performance_engine.py.

This module provides foundation types that both modules need without
creating import cycles.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetInfo:
    """Information about dataset characteristics for optimization."""

    size: int
    category: str  # "small", "medium", "large"
    memory_usage_mb: float
    recommended_chunk_size: int
    recommended_batch_size: int
    use_progressive_loading: bool
    compression_ratio: float | None = None


@dataclass
class ProcessingStrategy:
    """Processing strategy for different dataset sizes."""

    chunk_size: int
    batch_size: int
    memory_limit_mb: float
    use_caching: bool
    use_compression: bool
    parallel_workers: int
    jax_config: dict[str, Any]


__all__ = [
    "DatasetInfo",
    "ProcessingStrategy",
]
