"""Advanced Performance Engine for Massive XPCS Datasets - Homodyne
===================================================================

High-performance data processing engine for handling massive XPCS datasets (>1GB)
with memory-mapped I/O, intelligent chunking, parallel processing, and multi-level caching.

This module provides advanced performance optimizations beyond the basic optimization.py:

- Memory-mapped HDF5 access for files too large to fit in memory
- Adaptive chunking based on memory pressure and data characteristics
- Multi-threaded parallel processing with proper synchronization
- Smart prefetching and background loading with predictive access patterns
- Multi-level caching (memory, SSD, HDD) with intelligent eviction
- Performance monitoring and bottleneck identification
- Graceful degradation when optimizations fail

Key Features:
- Progressive loading of correlation matrices without loading entire datasets
- Intelligent buffer management for large correlation matrix collections
- Cross-chunk correlation analysis for maintaining data integrity
- Background processing of quality control and preprocessing
- Compressed caching with fast decompression optimized for repeated access
- Real-time performance metrics and automatic tuning
"""

from __future__ import annotations

import hashlib
import os
import pickle  # nosec B403: internal cache serialization only
import threading
import time
import types
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import psutil

# Core dependencies with graceful fallback
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# Optional compression library with fallback. zstd ships as a compiled
# extension without bundled source, so Pyright cannot resolve a source module.
try:
    import zstd  # pyright: ignore[reportMissingModuleSource]

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    zstd = None

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

# JAX integration for array acceleration
F = TypeVar("F", bound=Callable[..., Any])
try:
    import jax.numpy as jnp

    from xpcsjax.core.jax_backend import jax_available

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax_available = False
    jnp: types.ModuleType = np  # type: ignore[no-redef]

    def device_put(x: Any) -> Any:  # type: ignore[misc]
        return x

    def device_get(x: Any) -> Any:
        return x


# V2 system integration - import types from types.py to avoid circular imports
try:
    from xpcsjax.data.types import DatasetInfo
    from xpcsjax.utils.logging import get_logger, log_calls, log_performance

    HAS_V2_LOGGING = True
except ImportError:
    import logging

    HAS_V2_LOGGING = False

    def get_logger(name: str) -> Any:  # type: ignore[misc]
        return logging.getLogger(name)

    def log_performance(*args: Any, **kwargs: Any) -> Callable[[F], F]:  # type: ignore[misc]
        return lambda f: f

    def log_calls(*args: Any, **kwargs: Any) -> Callable[[F], F]:  # type: ignore[misc]
        return lambda f: f

    DatasetInfo = None  # type: ignore[assignment,misc]

# Import DatasetOptimizer lazily to avoid circular imports
DatasetOptimizer = None  # Will be imported when needed

logger = get_logger(__name__)


class PerformanceEngineError(Exception):
    """Base exception for performance engine errors."""


class MemoryPressureError(PerformanceEngineError):
    """Raised when memory pressure becomes critical."""


class CacheError(PerformanceEngineError):
    """Raised when cache operations fail."""


@dataclass
class PerformanceMetrics:
    """Real-time performance monitoring metrics."""

    loading_speed_mbps: float = 0.0
    memory_usage_mb: float = 0.0
    memory_pressure: float = 0.0  # 0.0-1.0 scale
    cache_hit_rate: float = 0.0
    cpu_utilization: float = 0.0
    io_wait_time: float = 0.0
    chunk_processing_rate: float = 0.0  # chunks/second
    parallel_efficiency: float = 0.0  # 0.0-1.0 scale
    bottleneck_type: str | None = None  # "memory", "io", "cpu", "cache"

    # Performance history for trending
    history_size: int = field(default=100, init=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=100), init=False)

    def update(self, **kwargs: Any) -> None:
        """Update metrics and maintain history."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Add snapshot to history
        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "loading_speed_mbps": self.loading_speed_mbps,
            "memory_usage_mb": self.memory_usage_mb,
            "cache_hit_rate": self.cache_hit_rate,
            "cpu_utilization": self.cpu_utilization,
        }
        self._history.append(snapshot)

    def get_trend(self, metric: str, window: int = 10) -> float:
        """Get trend for a specific metric (-1.0 to 1.0, negative=declining)."""
        if len(self._history) < window:
            return 0.0

        recent_values = [h.get(metric, 0.0) for h in list(self._history)[-window:]]
        if not recent_values or max(recent_values) == min(recent_values):
            return 0.0

        # Simple linear trend calculation
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        slope = float(np.polyfit(x, y, 1)[0])

        # Normalize slope to -1.0 to 1.0 range
        value_range = max(recent_values) - min(recent_values)
        if value_range > 0:
            normalized_slope = slope / value_range
            return float(max(-1.0, min(1.0, normalized_slope)))
        return 0.0


@dataclass
class ChunkInfo:
    """Information about a data chunk for intelligent processing."""

    index: int
    size: int
    memory_size_mb: float
    complexity_score: float  # Based on correlation matrix properties
    priority: int  # 1=highest, 10=lowest
    access_pattern: str  # "sequential", "random", "predictive"
    estimated_processing_time: float
    dependencies: list[int] = field(default_factory=list)
    cache_key: str | None = None


class MemoryMapManager:
    """Manager for memory-mapped access to large HDF5 files.

    Provides efficient access to correlation matrices without loading entire datasets.
    Implements intelligent buffer management and progressive access patterns.
    """

    def __init__(self, max_open_files: int = 64, buffer_size_mb: float = 512.0):
        """Initialize memory map manager.

        Args:
            max_open_files: Maximum number of simultaneously open memory maps
            buffer_size_mb: Size of internal buffers in MB
        """
        self.max_open_files = max_open_files
        self.buffer_size_mb = buffer_size_mb
        self._open_maps: dict[str, Any] = {}
        self._access_counts: dict[str, int] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.RLock()

        logger.info(
            f"Memory map manager initialized: max_files={max_open_files}, "
            f"buffer_size={buffer_size_mb}MB",
        )

    @contextmanager
    def open_memory_mapped_hdf5(self, file_path: str, mode: str = "r") -> Iterator[Any]:
        """Context manager for memory-mapped HDF5 file access.

        Args:
            file_path: Path to HDF5 file
            mode: File access mode

        Yields:
            Memory-mapped h5py.File object
        """
        if not HAS_H5PY:
            raise PerformanceEngineError("h5py required for memory-mapped access")

        file_path = str(file_path)

        # Acquire lock to check/register the file, then release before yielding
        # so callers don't hold the lock during potentially long I/O operations.
        with self._lock:
            if file_path in self._open_maps:
                self._access_counts[file_path] += 1
                self._last_access[file_path] = time.time()
                hdf_file = self._open_maps[file_path]
            else:
                # Clean up old mappings if needed
                self._cleanup_old_mappings()

                try:
                    hdf_file = h5py.File(
                        file_path,
                        mode,
                        rdcc_nbytes=int(self.buffer_size_mb * 1024 * 1024),
                    )
                    self._open_maps[file_path] = hdf_file
                    self._access_counts[file_path] = 1
                    self._last_access[file_path] = time.time()
                    logger.debug(f"Opened memory-mapped HDF5: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to open memory-mapped HDF5 {file_path}: {e}")
                    raise

        # Yield outside the lock — callers can use the file handle without
        # blocking other threads from registering/opening files.
        yield hdf_file

    def _cleanup_old_mappings(self) -> None:
        """Clean up old memory mappings to stay under limits."""
        if len(self._open_maps) < self.max_open_files:
            return

        # Sort by last access time, oldest first
        sorted_files = sorted(self._last_access.items(), key=lambda x: x[1])

        # Close oldest files until under limit
        files_to_close = len(self._open_maps) - self.max_open_files + 5  # Extra buffer
        for file_path, _ in sorted_files[:files_to_close]:
            try:
                self._open_maps[file_path].close()
                del self._open_maps[file_path]
                del self._access_counts[file_path]
                del self._last_access[file_path]
                logger.debug(f"Closed old memory mapping: {file_path}")
            except OSError as e:
                logger.warning(f"Error closing memory mapping {file_path}: {e}")

    def close_all(self) -> None:
        """Close all memory mappings."""
        with self._lock:
            for file_path, hdf_file in list(self._open_maps.items()):
                try:
                    hdf_file.close()
                except OSError as e:
                    logger.warning(f"Error closing {file_path}: {e}")

            self._open_maps.clear()
            self._access_counts.clear()
            self._last_access.clear()
            logger.info("All memory mappings closed")


class AdaptiveChunker:
    """Intelligent chunking system that adapts based on memory pressure and data characteristics.

    Provides smart chunk size determination, cross-chunk correlation analysis,
    and adaptive chunking that adjusts based on processing performance feedback.
    """

    def __init__(
        self,
        base_chunk_size: int = 100000,
        memory_threshold: float = 0.8,
        performance_feedback_window: int = 10,
    ):
        """Initialize adaptive chunker.

        Args:
            base_chunk_size: Base chunk size for normal conditions
            memory_threshold: Memory usage threshold for adaptation (0.0-1.0)
            performance_feedback_window: Number of chunks to consider for performance feedback
        """
        self.base_chunk_size = base_chunk_size
        self.memory_threshold = memory_threshold
        self.performance_feedback_window = performance_feedback_window

        # Performance tracking for adaptation
        self._chunk_performance: deque = deque(maxlen=performance_feedback_window)
        self._optimal_chunk_size = base_chunk_size
        self._last_adaptation_time = 0.0
        self._adaptation_cooldown = 30.0  # Seconds between adaptations

        logger.info(
            f"Adaptive chunker initialized: base_size={base_chunk_size}, "
            f"memory_threshold={memory_threshold}",
        )

    def calculate_optimal_chunk_size(
        self,
        total_size: int,
        data_complexity: float = 1.0,
        available_memory_mb: float | None = None,
    ) -> int:
        """Calculate optimal chunk size based on current conditions.

        Args:
            total_size: Total size of data to be chunked
            data_complexity: Complexity score (1.0=normal, >1.0=complex)
            available_memory_mb: Available memory in MB

        Returns:
            Optimal chunk size for current conditions
        """
        # Get current memory status
        if available_memory_mb is None:
            memory_info = psutil.virtual_memory()
            available_memory_mb = memory_info.available / (1024 * 1024)
            memory_pressure = memory_info.percent / 100.0
        else:
            memory_pressure = max(
                0.0,
                1.0 - available_memory_mb / 8192,
            )  # Assume 8GB baseline

        # Base chunk size adjustment
        chunk_size = self._optimal_chunk_size

        # Adjust for memory pressure
        if memory_pressure > self.memory_threshold:
            memory_factor = max(
                0.1,
                1.0 - (memory_pressure - self.memory_threshold) * 2.0,
            )
            chunk_size = int(chunk_size * memory_factor)

        # Adjust for data complexity
        complexity_factor = 1.0 / max(1.0, data_complexity)
        chunk_size = int(chunk_size * complexity_factor)

        # Ensure minimum and maximum bounds
        min_chunk_size = max(
            1000,
            total_size // 1000,
        )  # At least 1000 points or 0.1% of total
        max_chunk_size = min(
            self.base_chunk_size * 4,
            total_size // 2,
        )  # At most 4x base or 50% of total

        chunk_size = max(min_chunk_size, min(max_chunk_size, chunk_size))

        logger.debug(
            f"Calculated optimal chunk size: {chunk_size} "
            f"(memory_pressure={memory_pressure:.2f}, complexity={data_complexity:.2f})",
        )

        return chunk_size

    def create_chunk_plan(self, total_size: int, chunk_size: int) -> list[ChunkInfo]:
        """Create intelligent chunk processing plan.

        Args:
            total_size: Total size of data
            chunk_size: Size of each chunk

        Returns:
            List of ChunkInfo objects with processing plan
        """
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)
            actual_size = end_idx - start_idx

            # Estimate memory size (assuming 8 bytes per data point plus overhead)
            memory_size_mb = (actual_size * 8 * 4) / (
                1024 * 1024
            )  # 4x overhead for processing

            # Calculate complexity score (simplified - could be enhanced with data analysis)
            complexity_score = 1.0
            if i == 0 or i == num_chunks - 1:
                complexity_score = 1.2  # Edge chunks might need special handling

            # Set priority (middle chunks can often be processed in parallel)
            if num_chunks <= 3:
                priority = 1  # All chunks high priority for small datasets
            elif i == 0 or i == num_chunks - 1:
                priority = 1  # Edge chunks high priority
            else:
                priority = 2  # Middle chunks can wait

            # Estimate processing time based on size and complexity
            base_processing_rate = 50000  # points per second
            estimated_processing_time = (
                actual_size * complexity_score
            ) / base_processing_rate

            chunk_info = ChunkInfo(
                index=i,
                size=actual_size,
                memory_size_mb=memory_size_mb,
                complexity_score=complexity_score,
                priority=priority,
                access_pattern="sequential",
                estimated_processing_time=estimated_processing_time,
                cache_key=f"chunk_{i}_{actual_size}",
            )

            chunks.append(chunk_info)

        logger.info(
            f"Created chunk plan: {num_chunks} chunks, "
            f"avg_size={np.mean([c.size for c in chunks]):.0f}, "
            f"total_memory={sum(c.memory_size_mb for c in chunks):.1f}MB",
        )

        return chunks

    def update_performance_feedback(
        self,
        chunk_info: ChunkInfo,
        actual_processing_time: float,
        success: bool = True,
    ) -> None:
        """Update performance feedback for adaptive optimization.

        Args:
            chunk_info: Information about processed chunk
            actual_processing_time: Actual time taken to process chunk
            success: Whether processing was successful
        """
        performance_ratio = chunk_info.estimated_processing_time / max(
            actual_processing_time,
            0.001,
        )

        feedback = {
            "chunk_size": chunk_info.size,
            "estimated_time": chunk_info.estimated_processing_time,
            "actual_time": actual_processing_time,
            "performance_ratio": performance_ratio,
            "success": success,
            "timestamp": time.time(),
        }

        self._chunk_performance.append(feedback)

        # Adapt chunk size if we have enough data and cooldown period has passed
        current_time = time.time()
        if (
            len(self._chunk_performance) >= self.performance_feedback_window
            and current_time - self._last_adaptation_time > self._adaptation_cooldown
        ):
            self._adapt_chunk_size()
            self._last_adaptation_time = current_time

    def _adapt_chunk_size(self) -> None:
        """Adapt optimal chunk size based on performance feedback."""
        if not self._chunk_performance:
            return

        # Calculate average performance metrics
        successful_chunks = [p for p in self._chunk_performance if p["success"]]
        if not successful_chunks:
            return

        avg_performance_ratio = np.mean(
            [p["performance_ratio"] for p in successful_chunks],
        )
        # Note: avg_chunk_size calculation removed - was unused

        # Adapt based on performance
        if avg_performance_ratio < 0.8:  # Processing slower than expected
            # Reduce chunk size
            new_optimal_size = int(self._optimal_chunk_size * 0.8)
            logger.info(
                f"Reducing optimal chunk size: {self._optimal_chunk_size} -> {new_optimal_size} "
                f"(avg_performance_ratio={avg_performance_ratio:.2f})",
            )
        elif avg_performance_ratio > 1.2:  # Processing faster than expected
            # Increase chunk size
            new_optimal_size = int(self._optimal_chunk_size * 1.1)
            logger.info(
                f"Increasing optimal chunk size: {self._optimal_chunk_size} -> {new_optimal_size} "
                f"(avg_performance_ratio={avg_performance_ratio:.2f})",
            )
        else:
            return  # No change needed

        # Apply bounds
        new_optimal_size = max(1000, min(self.base_chunk_size * 8, new_optimal_size))
        self._optimal_chunk_size = new_optimal_size


class MultiLevelCache:
    """Advanced multi-level caching system with intelligent eviction.

    Implements memory cache, SSD cache, and HDD cache with intelligent
    cache coherence management and compressed caching for memory efficiency.
    """

    def __init__(
        self,
        memory_cache_mb: float = 1024.0,
        ssd_cache_mb: float = 8192.0,
        hdd_cache_mb: float = 32768.0,
        compression_level: int = 3,
    ):
        """Initialize multi-level cache system.

        Args:
            memory_cache_mb: Memory cache size in MB
            ssd_cache_mb: SSD cache size in MB
            hdd_cache_mb: HDD cache size in MB
            compression_level: Compression level for caching (1-22, higher=better compression)
        """
        self.memory_cache_mb = memory_cache_mb
        self.ssd_cache_mb = ssd_cache_mb
        self.hdd_cache_mb = hdd_cache_mb
        self.compression_level = compression_level

        # Memory cache (LRU with size limit)
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_usage_mb = 0.0

        # Access statistics for intelligent eviction
        self._access_counts: dict[str, int] = {}
        self._access_times: dict[str, float] = {}
        self._access_frequencies: dict[str, deque] = {}  # For frequency analysis

        # Cache hierarchy paths
        _xdg_cache = os.environ.get("XDG_CACHE_HOME", "")
        if _xdg_cache:
            self._cache_base_path = Path(_xdg_cache) / "xpcsjax"
        else:
            self._cache_base_path = Path.home() / ".cache" / "xpcsjax"
        self._ssd_cache_path = self._cache_base_path / "ssd"
        self._hdd_cache_path = self._cache_base_path / "hdd"

        # Create cache directories
        self._ssd_cache_path.mkdir(parents=True, exist_ok=True)
        self._hdd_cache_path.mkdir(parents=True, exist_ok=True)

        # Cache usage tracking
        self._ssd_usage_mb = 0.0
        self._hdd_usage_mb = 0.0

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"Multi-level cache initialized: memory={memory_cache_mb}MB, "
            f"ssd={ssd_cache_mb}MB, hdd={hdd_cache_mb}MB, compression={compression_level}",
        )

    def get(self, key: str) -> Any | None:
        """Get item from cache hierarchy (memory -> SSD -> HDD).

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        with self._lock:
            current_time = time.time()

            # Check memory cache first
            if key in self._memory_cache:
                item = self._memory_cache.pop(key)
                self._memory_cache[key] = item  # Move to end (most recent)
                self._update_access_stats(key, current_time)
                logger.debug(f"Cache hit (memory): {key}")
                return item

            # Check SSD cache
            ssd_path = self._ssd_cache_path / f"{key}.zstd"
            if ssd_path.exists():
                try:
                    item = self._load_from_disk(ssd_path)
                    # Promote to memory cache
                    self._put_memory(key, item, current_time)
                    self._update_access_stats(key, current_time)
                    logger.debug(f"Cache hit (SSD): {key}")
                    return item
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to load from SSD cache {key}: {e}")

            # Check HDD cache
            hdd_path = self._hdd_cache_path / f"{key}.zstd"
            if hdd_path.exists():
                try:
                    item = self._load_from_disk(hdd_path)
                    # Promote to memory and SSD cache
                    self._put_memory(key, item, current_time)
                    self._put_ssd(key, item)
                    self._update_access_stats(key, current_time)
                    logger.debug(f"Cache hit (HDD): {key}")
                    return item
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to load from HDD cache {key}: {e}")

            logger.debug(f"Cache miss: {key}")
            return None

    def put(self, key: str, item: Any, priority: int = 5) -> None:
        """Put item in cache hierarchy with intelligent placement.

        Args:
            key: Cache key
            item: Item to cache
            priority: Priority level (1=highest, 10=lowest)
        """
        # Determine promotion decisions while holding the lock
        with self._lock:
            current_time = time.time()

            # Always try to put in memory cache first
            self._put_memory(key, item, current_time, priority)

            # Decide SSD/HDD promotion without performing disk I/O
            should_cache_ssd = (
                priority <= 3 or self._get_access_frequency(key) > 0.1
            )  # >0.1 accesses per minute
            should_cache_hdd = (
                priority <= 2 or self._get_access_frequency(key) > 1.0
            )  # >1 access per minute

        # Perform disk writes outside the lock
        if should_cache_ssd:
            try:
                self._put_ssd(key, item)
            except OSError as e:
                logger.warning(f"Failed to cache to SSD {key}: {e}")

        if should_cache_hdd:
            try:
                self._put_hdd(key, item)
            except OSError as e:
                logger.warning(f"Failed to cache to HDD {key}: {e}")

    def _put_memory(
        self,
        key: str,
        item: Any,
        current_time: float,
        priority: int = 5,
    ) -> None:
        """Put item in memory cache with size management."""
        # Calculate item size
        item_size_mb = self._estimate_size_mb(item)

        # Remove existing item if present
        if key in self._memory_cache:
            old_item = self._memory_cache.pop(key)
            old_size_mb = self._estimate_size_mb(old_item)
            self._memory_usage_mb -= old_size_mb

        # Make space if needed
        while (
            self._memory_usage_mb + item_size_mb > self.memory_cache_mb
            and len(self._memory_cache) > 0
        ):
            self._evict_from_memory()

        # Add new item
        if self._memory_usage_mb + item_size_mb <= self.memory_cache_mb:
            self._memory_cache[key] = item
            self._memory_usage_mb += item_size_mb
            self._update_access_stats(key, current_time)

    def _put_ssd(self, key: str, item: Any) -> None:
        """Put item in SSD cache with size management."""
        try:
            ssd_path = self._ssd_cache_path / f"{key}.zstd"
            item_size_mb = self._save_to_disk(ssd_path, item)

            # Update usage tracking
            self._ssd_usage_mb += item_size_mb

            # Clean up if over limit
            while self._ssd_usage_mb > self.ssd_cache_mb:
                self._evict_from_ssd()

        except (OSError, ValueError) as e:
            logger.warning(f"Failed to cache to SSD {key}: {e}")

    def _put_hdd(self, key: str, item: Any) -> None:
        """Put item in HDD cache with size management."""
        try:
            hdd_path = self._hdd_cache_path / f"{key}.zstd"
            item_size_mb = self._save_to_disk(hdd_path, item)

            # Update usage tracking
            self._hdd_usage_mb += item_size_mb

            # Clean up if over limit
            while self._hdd_usage_mb > self.hdd_cache_mb:
                self._evict_from_hdd()

        except (OSError, ValueError) as e:
            logger.warning(f"Failed to cache to HDD {key}: {e}")

    def _save_to_disk(self, file_path: Path, item: Any) -> float:
        """Save compressed item to disk and return size in MB."""
        try:
            # Serialize item
            serialized = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)

            # Compress with zstd if available, otherwise save uncompressed
            if HAS_ZSTD:
                compressed = zstd.compress(serialized, self.compression_level)
            else:
                compressed = serialized  # Fallback to uncompressed

            # Write to disk
            with open(file_path, "wb") as f:
                f.write(compressed)
            os.chmod(file_path, 0o600)

            size_mb = len(compressed) / (1024 * 1024)
            return size_mb

        except (OSError, ValueError) as e:
            logger.error(f"Failed to save {file_path}: {e}")
            raise

    def _load_from_disk(self, file_path: Path) -> Any:
        """Load and decompress item from disk.

        Security model: deserialization is unsafe on attacker-controlled
        input. We mitigate with three defense-in-depth invariants enforced
        BEFORE the cache item is decoded, so a compromised cache directory
        cannot escalate to arbitrary code execution:

        1. **Path containment** — the resolved path must live under
           ``self._cache_base_path``. Stops symlink escapes and
           ``../../../etc/passwd``-style traversal if the key derivation
           ever started letting user input flow through ``key``.
        2. **Ownership** — on POSIX, the file must be owned by the
           current uid. A malicious co-tenant, or an artifact dropped
           by another user with the same path, fails the gate.
           (Windows: skipped — POSIX ownership semantics don't apply;
           rely on path containment + mode check.)
        3. **Mode** — the file must not be group- or world-writable
           (``0o077`` bits clear). Matches the 0o600 mode
           ``_save_to_disk`` writes; rejecting weaker modes catches
           tampering after creation.

        External input still enters only as HDF5 / NumPy arrays at I/O
        boundaries (those are guarded by ``allow_pickle=False`` in
        ``xpcs_loader.py``). This loader is for the internal compute
        cache only.
        """
        # ------------------------------------------------------------------
        # Defense-in-depth gates (run before deserialization)
        # ------------------------------------------------------------------
        resolved = file_path.resolve()
        cache_root = self._cache_base_path.resolve()
        try:
            resolved.relative_to(cache_root)
        except ValueError as exc:
            raise OSError(
                f"refusing to load {file_path}: resolves to {resolved}, "
                f"outside cache root {cache_root}. Possible symlink escape."
            ) from exc

        try:
            st = resolved.stat()
        except OSError as exc:
            raise OSError(f"cache file {resolved} missing or unreadable") from exc

        # POSIX ownership check. On Windows ``getuid``/``st_uid`` ownership
        # semantics are unavailable, so this defense-in-depth layer is absent
        # there. Surface that explicitly (M-6) rather than silently running with
        # a weaker posture — the path-containment and mode checks still apply.
        if hasattr(os, "getuid"):
            current_uid = os.getuid()
            if st.st_uid != current_uid:
                raise OSError(
                    f"refusing to load {resolved}: owned by uid {st.st_uid}, "
                    f"current uid is {current_uid}. Possible cache poisoning."
                )
        else:
            logger.warning(
                "Cache ownership verification is not available on this platform "
                "(no os.getuid); loading %s with reduced tamper-resistance. "
                "Ensure the cache directory is not writable by other users.",
                resolved,
            )

        # Mode check — group/world must have no permissions. ``_save_to_disk``
        # writes 0o600; if the mode drifted, refuse to load.
        permissive_bits = st.st_mode & 0o077
        if permissive_bits:
            raise OSError(
                f"refusing to load {resolved}: mode={st.st_mode:#o} grants "
                f"group/world access ({permissive_bits:#o}). Possible tamper."
            )

        try:
            with open(resolved, "rb") as f:
                data = f.read()

            # Decompress if zstd available, otherwise data is already uncompressed
            if HAS_ZSTD:
                serialized = zstd.decompress(data)
            else:
                serialized = data  # Fallback to uncompressed

            # Deserialize (trusted internal cache only)
            item = pickle.loads(serialized)  # nosec B301
            return item

        except (OSError, ValueError) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def _estimate_size_mb(self, item: Any) -> float:
        """Estimate memory size of item in MB."""
        if hasattr(item, "nbytes"):  # numpy array
            return float(item.nbytes / (1024 * 1024))
        elif isinstance(item, (list, tuple)):
            total_size = 0.0
            for sub_item in item:
                total_size += self._estimate_size_mb(sub_item)
            return total_size
        elif isinstance(item, dict):
            total_size = 0.0
            for key, value in item.items():
                total_size += self._estimate_size_mb(key) + self._estimate_size_mb(
                    value,
                )
            return total_size
        else:
            # Rough estimate based on pickle size
            try:
                return float(
                    len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
                    / (1024 * 1024)
                )
            except (pickle.PicklingError, TypeError, AttributeError):
                return 0.1  # Conservative estimate

    def _update_access_stats(self, key: str, current_time: float) -> None:
        """Update access statistics for intelligent caching decisions."""
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._access_times[key] = current_time

        # Track access frequency
        if key not in self._access_frequencies:
            self._access_frequencies[key] = deque(maxlen=10)
        self._access_frequencies[key].append(current_time)

    def _get_access_frequency(self, key: str) -> float:
        """Get access frequency (accesses per minute) for a key."""
        if key not in self._access_frequencies:
            return 0.0

        accesses = list(self._access_frequencies[key])
        if len(accesses) < 2:
            return 0.0

        time_span = accesses[-1] - accesses[0]
        if time_span <= 0:
            return 0.0

        return float((len(accesses) - 1) / (time_span / 60.0))  # accesses per minute

    def _evict_from_memory(self) -> None:
        """Evict least valuable item from memory cache."""
        if not self._memory_cache:
            return

        # Use LRU with access frequency weighting
        # Higher value_score = less recent + less frequent = should be evicted
        best_eviction_key = None
        highest_eviction_score = float("-inf")

        current_time = time.time()

        for key in list(self._memory_cache.keys()):
            # Calculate value score (higher = less valuable = better eviction candidate)
            recency_score = current_time - self._access_times.get(
                key,
                0,
            )  # Higher = less recent
            frequency_score = 1.0 / (
                self._get_access_frequency(key) + 0.1
            )  # Higher = less frequent
            value_score = recency_score * frequency_score

            if value_score > highest_eviction_score:
                highest_eviction_score = value_score
                best_eviction_key = key

        if best_eviction_key:
            evicted_item = self._memory_cache.pop(best_eviction_key)
            evicted_size = self._estimate_size_mb(evicted_item)
            self._memory_usage_mb -= evicted_size
            logger.debug(
                f"Evicted from memory: {best_eviction_key} ({evicted_size:.1f}MB)",
            )

    def _evict_from_ssd(self) -> None:
        """Evict oldest file from SSD cache."""
        try:
            ssd_files = list(self._ssd_cache_path.glob("*.zstd"))
            if not ssd_files:
                return

            # Sort by modification time, oldest first
            oldest_file = min(ssd_files, key=lambda x: x.stat().st_mtime)
            file_size_mb = oldest_file.stat().st_size / (1024 * 1024)

            oldest_file.unlink()
            self._ssd_usage_mb -= file_size_mb
            logger.debug(f"Evicted from SSD: {oldest_file.name} ({file_size_mb:.1f}MB)")

        except OSError as e:
            logger.warning(f"Error evicting from SSD cache: {e}")

    def _evict_from_hdd(self) -> None:
        """Evict oldest file from HDD cache."""
        try:
            hdd_files = list(self._hdd_cache_path.glob("*.zstd"))
            if not hdd_files:
                return

            # Sort by modification time, oldest first
            oldest_file = min(hdd_files, key=lambda x: x.stat().st_mtime)
            file_size_mb = oldest_file.stat().st_size / (1024 * 1024)

            oldest_file.unlink()
            self._hdd_usage_mb -= file_size_mb
            logger.debug(f"Evicted from HDD: {oldest_file.name} ({file_size_mb:.1f}MB)")

        except OSError as e:
            logger.warning(f"Error evicting from HDD cache: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Perform disk I/O outside the lock to avoid blocking other threads.
        # glob() can be slow (network filesystems, large directories).
        ssd_items = len(list(self._ssd_cache_path.glob("*.zstd")))
        hdd_items = len(list(self._hdd_cache_path.glob("*.zstd")))
        with self._lock:
            memory_items = len(self._memory_cache)

            return {
                "memory_cache": {
                    "items": memory_items,
                    "usage_mb": self._memory_usage_mb,
                    "limit_mb": self.memory_cache_mb,
                    "utilization": self._memory_usage_mb / self.memory_cache_mb
                    if self.memory_cache_mb > 0
                    else 0.0,
                },
                "ssd_cache": {
                    "items": ssd_items,
                    "usage_mb": self._ssd_usage_mb,
                    "limit_mb": self.ssd_cache_mb,
                    "utilization": self._ssd_usage_mb / self.ssd_cache_mb
                    if self.ssd_cache_mb > 0
                    else 0.0,
                },
                "hdd_cache": {
                    "items": hdd_items,
                    "usage_mb": self._hdd_usage_mb,
                    "limit_mb": self.hdd_cache_mb,
                    "utilization": self._hdd_usage_mb / self.hdd_cache_mb
                    if self.hdd_cache_mb > 0
                    else 0.0,
                },
                "total_items": memory_items + ssd_items + hdd_items,
            }


class PerformanceEngine:
    """Main performance engine coordinating all optimization components.

    Orchestrates memory-mapped I/O, intelligent chunking, parallel processing,
    smart prefetching, and multi-level caching for optimal performance.
    """

    @log_calls(include_args=False)
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize performance engine with configuration.

        Args:
            config: Performance configuration dictionary
        """
        self.config = config or {}
        self.performance_config = self.config.get("performance", {})

        # Initialize components
        self._init_memory_manager()
        self._init_chunker()
        self._init_cache()
        self._init_parallel_executor()

        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self._monitoring_enabled = self.performance_config.get("monitoring", {}).get(
            "enabled",
            True,
        )
        self._monitoring_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Prefetching and background loading
        self._prefetch_enabled = self.performance_config.get("prefetching", {}).get(
            "enabled",
            True,
        )
        self._prefetch_queue: deque = deque(maxlen=100)
        self._background_executor: ThreadPoolExecutor | None = None

        if self._monitoring_enabled:
            self._start_performance_monitoring()

        if self._prefetch_enabled:
            self._start_background_processing()

        logger.info("Performance engine initialized with advanced optimizations")

    def _init_memory_manager(self) -> None:
        """Initialize memory map manager."""
        memory_config = self.performance_config.get("memory_mapping", {})
        max_files = memory_config.get("max_open_files", 64)
        buffer_size = memory_config.get("buffer_size_mb", 512.0)

        self.memory_manager = MemoryMapManager(max_files, buffer_size)

    def _init_chunker(self) -> None:
        """Initialize adaptive chunker."""
        chunking_config = self.performance_config.get("chunking", {})
        base_chunk_size = chunking_config.get("base_chunk_size", 100000)
        memory_threshold = chunking_config.get("memory_threshold", 0.8)
        feedback_window = chunking_config.get("performance_feedback_window", 10)

        self.chunker = AdaptiveChunker(
            base_chunk_size,
            memory_threshold,
            feedback_window,
        )

    def _init_cache(self) -> None:
        """Initialize multi-level cache system."""
        cache_config = self.performance_config.get("caching", {})
        memory_cache_mb = cache_config.get("memory_cache_mb", 1024.0)
        ssd_cache_mb = cache_config.get("ssd_cache_mb", 8192.0)
        hdd_cache_mb = cache_config.get("hdd_cache_mb", 32768.0)
        compression_level = cache_config.get("compression_level", 3)

        self.cache = MultiLevelCache(
            memory_cache_mb,
            ssd_cache_mb,
            hdd_cache_mb,
            compression_level,
        )

    def _init_parallel_executor(self) -> None:
        """Initialize parallel processing executor."""
        parallel_config = self.performance_config.get("parallel", {})
        max_workers = parallel_config.get("max_workers", min(os.cpu_count() or 1, 8))

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PerformanceEngine",
        )

    def _start_performance_monitoring(self) -> None:
        """Start background performance monitoring."""
        self._monitoring_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            name="PerformanceMonitoring",
            daemon=True,
        )
        self._monitoring_thread.start()
        logger.debug("Performance monitoring started")

    def _start_background_processing(self) -> None:
        """Start background processing for prefetching."""
        self._background_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="BackgroundProcessing",
        )
        logger.debug("Background processing started")

    def _performance_monitoring_loop(self) -> None:
        """Main performance monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                self._update_performance_metrics()
                self._detect_bottlenecks()
                time.sleep(1.0)  # Update every second
            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")
                time.sleep(5.0)  # Longer sleep on error

    def _update_performance_metrics(self) -> None:
        """Update real-time performance metrics."""
        # Memory metrics
        memory_info = psutil.virtual_memory()
        self.metrics.update(
            memory_usage_mb=memory_info.used / (1024 * 1024),
            memory_pressure=memory_info.percent / 100.0,
        )

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics.update(cpu_utilization=cpu_percent / 100.0)

        # Cache metrics
        cache_stats = self.cache.get_cache_stats()
        if cache_stats["memory_cache"]["items"] > 0:
            with self.cache._lock:
                total_requests = sum(self.cache._access_counts.values())
                cache_hits = len(
                    [
                        k
                        for k in self.cache._memory_cache.keys()
                        if k in self.cache._access_counts
                    ],
                )
            cache_hit_rate = cache_hits / max(total_requests, 1)
            self.metrics.update(cache_hit_rate=cache_hit_rate)

    def _detect_bottlenecks(self) -> None:
        """Detect and classify performance bottlenecks."""
        # Memory bottleneck
        if self.metrics.memory_pressure > 0.9:
            self.metrics.bottleneck_type = "memory"
        # CPU bottleneck
        elif self.metrics.cpu_utilization > 0.95:
            self.metrics.bottleneck_type = "cpu"
        # Cache bottleneck
        elif self.metrics.cache_hit_rate < 0.5:
            self.metrics.bottleneck_type = "cache"
        # I/O bottleneck (heuristic)
        elif self.metrics.cpu_utilization < 0.3 and self.metrics.memory_pressure < 0.7:
            self.metrics.bottleneck_type = "io"
        else:
            self.metrics.bottleneck_type = None

    @log_performance(threshold=2.0)
    def load_correlation_matrices_optimized(
        self,
        hdf_path: str,
        data_keys: list[str],
        chunk_info: list[ChunkInfo] | None = None,
    ) -> Any:  # Returns np.ndarray or jax.Array
        """Load correlation matrices with full performance optimization.

        Args:
            hdf_path: Path to HDF5 file
            data_keys: List of correlation matrix keys to load
            chunk_info: Optional chunk information for processing

        Returns:
            Array of correlation matrices
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(hdf_path, data_keys)

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Loaded {len(data_keys)} correlation matrices from cache")
            return cached_result

        try:
            # Use memory-mapped access for large files
            with self.memory_manager.open_memory_mapped_hdf5(hdf_path) as hdf_file:
                # Determine if we need chunked processing
                estimated_memory_mb = (
                    len(data_keys) * 64 * 64 * 8 / (1024 * 1024)
                )  # Rough estimate

                if (
                    chunk_info or estimated_memory_mb > 1024
                ):  # Use chunking for large datasets
                    logger.info(
                        f"Using chunked processing for {len(data_keys)} correlation matrices",
                    )
                    correlation_matrices = self._load_matrices_chunked(
                        hdf_file,
                        data_keys,
                        chunk_info,
                    )
                else:
                    logger.info(
                        f"Using direct loading for {len(data_keys)} correlation matrices",
                    )
                    correlation_matrices = self._load_matrices_direct(
                        hdf_file,
                        data_keys,
                    )

            # Convert to JAX arrays if available and requested
            output_format = self.config.get("v2_features", {}).get(
                "output_format",
                "auto",
            )
            if output_format in ["jax", "auto"] and HAS_JAX and jax_available:
                correlation_matrices = jnp.array(correlation_matrices)

            # Cache the result
            cache_priority = (
                3 if len(data_keys) > 100 else 5
            )  # High priority for large datasets
            self.cache.put(cache_key, correlation_matrices, priority=cache_priority)

            # Update performance metrics
            loading_time = time.time() - start_time
            data_size_mb = correlation_matrices.nbytes / (1024 * 1024)
            loading_speed = data_size_mb / max(loading_time, 0.001)

            self.metrics.update(
                loading_speed_mbps=loading_speed,
                chunk_processing_rate=len(data_keys) / max(loading_time, 0.001),
            )

            logger.info(
                f"Loaded {len(data_keys)} correlation matrices in {loading_time:.2f}s "
                f"({loading_speed:.1f} MB/s)",
            )

            return correlation_matrices

        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Failed to load correlation matrices: {e}")
            raise PerformanceEngineError(
                f"Correlation matrix loading failed: {e}",
            ) from e

    def _load_matrices_chunked(
        self,
        hdf_file: Any,
        data_keys: list[str],
        chunk_info: list[ChunkInfo] | None = None,
    ) -> Any:  # Returns np.ndarray
        """Load correlation matrices using chunked parallel processing."""
        if chunk_info is None:
            # Create chunk plan
            chunk_size = self.chunker.calculate_optimal_chunk_size(len(data_keys))
            chunk_info = self.chunker.create_chunk_plan(len(data_keys), chunk_size)

        # Process chunks in parallel
        future_to_chunk = {}

        for chunk in chunk_info:
            start_idx = chunk.index * chunk.size
            end_idx = min(start_idx + chunk.size, len(data_keys))
            chunk_keys = data_keys[start_idx:end_idx]

            future = self.executor.submit(
                self._load_matrix_chunk,
                hdf_file,
                chunk_keys,
                chunk,
            )
            future_to_chunk[future] = chunk

        # Collect results
        all_matrices = []
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk_start_time = time.time()
                chunk_matrices = future.result()
                chunk_processing_time = time.time() - chunk_start_time

                all_matrices.extend(chunk_matrices)

                # Update chunker performance feedback
                self.chunker.update_performance_feedback(
                    chunk,
                    chunk_processing_time,
                    success=True,
                )

            except Exception as e:
                logger.error(f"Chunk {chunk.index} failed: {e}")
                # Update chunker with failure
                self.chunker.update_performance_feedback(chunk, 0.0, success=False)
                raise

        return np.array(all_matrices)

    def _load_matrices_direct(
        self, hdf_file: Any, data_keys: list[str]
    ) -> Any:  # Returns np.ndarray
        """Load correlation matrices directly without chunking."""
        matrices = []

        # Determine the HDF5 group structure
        if "exchange/C2T_all" in hdf_file:
            # APS old format
            c2t_group = hdf_file["exchange/C2T_all"]
        elif "xpcs/twotime/correlation_map" in hdf_file:
            # APS-U format
            c2t_group = hdf_file["xpcs/twotime/correlation_map"]
        else:
            raise PerformanceEngineError(
                "Cannot determine HDF5 correlation matrix location",
            )

        for key in data_keys:
            if key in c2t_group:
                c2_half = c2t_group[key][()]
                # Reconstruct full matrix
                c2_full = self._reconstruct_full_matrix(c2_half)
                matrices.append(c2_full)

        return np.array(matrices)

    def _load_matrix_chunk(
        self,
        hdf_file: Any,
        chunk_keys: list[str],
        chunk_info: ChunkInfo,
    ) -> list[Any]:  # Returns list[np.ndarray]
        """Load a chunk of correlation matrices."""
        matrices = []

        # Determine the HDF5 group structure
        if "exchange/C2T_all" in hdf_file:
            c2t_group = hdf_file["exchange/C2T_all"]
        elif "xpcs/twotime/correlation_map" in hdf_file:
            c2t_group = hdf_file["xpcs/twotime/correlation_map"]
        else:
            raise PerformanceEngineError(
                "Cannot determine HDF5 correlation matrix location",
            )

        for key in chunk_keys:
            if key in c2t_group:
                c2_half = c2t_group[key][()]
                c2_full = self._reconstruct_full_matrix(c2_half)
                matrices.append(c2_full)

        return matrices

    def _reconstruct_full_matrix(
        self, c2_half: Any
    ) -> Any:  # Takes and returns np.ndarray
        """Reconstruct full correlation matrix from half matrix."""
        c2_full = c2_half + c2_half.T
        diag_indices = np.diag_indices(c2_half.shape[0])
        c2_full[diag_indices] /= 2
        return c2_full

    def _generate_cache_key(self, hdf_path: str, data_keys: list[str]) -> str:
        """Generate cache key for correlation matrices."""
        # Use file path, modification time, and hash of data keys
        file_stat = os.stat(hdf_path)
        file_info = f"{hdf_path}:{file_stat.st_mtime}:{file_stat.st_size}"

        # Hash the data keys for shorter cache key
        keys_hash = hashlib.sha256(",".join(sorted(data_keys)).encode()).hexdigest()[:8]

        return (
            f"corr_matrices_{keys_hash}_{file_info.replace('/', '_').replace(':', '_')}"
        )

    def prefetch_data(
        self,
        hdf_path: str,
        data_keys: list[str],
        priority: int = 5,
    ) -> Future:
        """Schedule data for background prefetching.

        Args:
            hdf_path: Path to HDF5 file
            data_keys: List of data keys to prefetch
            priority: Priority level for prefetching

        Returns:
            Future object for tracking prefetch operation
        """
        if not self._prefetch_enabled or not self._background_executor:
            # Return a dummy future that immediately returns None
            dummy_future: Future[None] = Future()
            dummy_future.set_result(None)
            return dummy_future

        cache_key = self._generate_cache_key(hdf_path, data_keys)

        # Check if already cached
        if self.cache.get(cache_key) is not None:
            dummy_future2: Future[None] = Future()
            dummy_future2.set_result(None)
            return dummy_future2

        # Schedule for background loading
        future = self._background_executor.submit(
            self._background_load_data,
            hdf_path,
            data_keys,
            cache_key,
            priority,
        )

        logger.debug(f"Scheduled background prefetch for {len(data_keys)} data keys")
        return future

    def _background_load_data(
        self,
        hdf_path: str,
        data_keys: list[str],
        cache_key: str,
        priority: int,
    ) -> None:
        """Background data loading for prefetching."""
        try:
            # Load data using optimized loading
            correlation_matrices = self.load_correlation_matrices_optimized(
                hdf_path,
                data_keys,
            )

            # Cache with specified priority
            self.cache.put(cache_key, correlation_matrices, priority=priority)

            logger.debug(
                f"Background prefetch completed for {len(data_keys)} data keys",
            )

        except OSError as e:
            logger.warning(f"Background prefetch failed: {e}")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.cache.get_cache_stats()

        report = {
            "performance_metrics": {
                "loading_speed_mbps": self.metrics.loading_speed_mbps,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "memory_pressure": self.metrics.memory_pressure,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "cpu_utilization": self.metrics.cpu_utilization,
                "chunk_processing_rate": self.metrics.chunk_processing_rate,
                "bottleneck_type": self.metrics.bottleneck_type,
            },
            "cache_statistics": cache_stats,
            "chunker_status": {
                "optimal_chunk_size": self.chunker._optimal_chunk_size,
                "performance_history_length": len(self.chunker._chunk_performance),
            },
            "system_info": {
                "cpu_count": os.cpu_count(),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
        }

        # Add performance trends
        for metric in ["loading_speed_mbps", "memory_usage_mb", "cache_hit_rate"]:
            trend = self.metrics.get_trend(metric)
            report["performance_metrics"][f"{metric}_trend"] = trend

        return report

    def shutdown(self) -> None:
        """Shutdown performance engine and cleanup resources."""
        logger.info("Shutting down performance engine")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop monitoring thread
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        # Shutdown executors
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        if self._background_executor:
            self._background_executor.shutdown(wait=True)

        # Close memory mappings
        if hasattr(self, "memory_manager"):
            self.memory_manager.close_all()

        logger.info("Performance engine shutdown complete")

    def __enter__(self) -> PerformanceEngine:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


# Export main classes and functions
__all__ = [
    "PerformanceEngine",
    "PerformanceMetrics",
    "MemoryMapManager",
    "AdaptiveChunker",
    "MultiLevelCache",
    "ChunkInfo",
    "PerformanceEngineError",
    "MemoryPressureError",
    "CacheError",
]
