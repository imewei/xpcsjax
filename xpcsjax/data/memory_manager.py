"""Advanced Memory Manager for Homodyne Performance Engine
==========================================================

Intelligent memory management system for massive XPCS datasets with dynamic allocation,
memory pools, pressure monitoring, and optimization strategies.

This module provides:

- Dynamic memory allocation based on available system resources
- Memory pool management for efficient buffer reuse
- Memory pressure monitoring and adaptive responses
- Garbage collection optimization to prevent fragmentation
- Virtual memory optimization for large datasets
- Memory-efficient data structures and algorithms

Key Features:
- Real-time memory pressure detection and response
- Intelligent memory allocation strategies based on workload patterns
- Memory pool recycling to minimize allocation overhead
- Background memory optimization and cleanup
- Integration with system virtual memory for handling datasets larger than RAM
- Proactive memory management to prevent out-of-memory conditions
"""

import atexit
import gc
import logging
import mmap
import os
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import psutil

# Core dependencies
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# JAX for array and memory management
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore[misc]

    def device_put(x):  # type: ignore[no-untyped-def, misc]
        return x

    def device_get(x):  # type: ignore[no-untyped-def]
        return x


# V2 system integration
try:
    from xpcsjax.utils.logging import (
        get_logger,
        log_calls,
        log_exception,
        log_once,
        log_performance,
        logged_errors,
    )

    HAS_V2_LOGGING = True
except ImportError:
    from contextlib import contextmanager as _contextmanager

    HAS_V2_LOGGING = False

    def get_logger(name):  # type: ignore[no-untyped-def,misc]
        return logging.getLogger(name)

    def log_performance(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        return lambda f: f

    def log_calls(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        return lambda f: f

    def log_exception(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        return None

    def log_once(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        return None

    @_contextmanager  # type: ignore[misc]
    def logged_errors(*args, **kwargs):  # type: ignore[no-untyped-def,misc]
        # Fallback when the Phase-1 helper is unavailable: behave as a
        # pure-swallow guard so control flow is unchanged.
        try:
            yield
        except Exception:
            pass


logger = get_logger(__name__)

T = TypeVar("T")

# Global registry of active MemoryPressureMonitor instances to prevent threads running on exit
_active_monitors: weakref.WeakSet[Any] = weakref.WeakSet()


def _cleanup_active_monitors() -> None:
    """Clean up all active memory pressure monitors on interpreter exit."""
    for monitor in list(_active_monitors):
        with logged_errors(
            logger,
            "atexit_stop_monitoring",
            policy="suppress",
            level=logging.DEBUG,
            once_key=f"{id(monitor)}:memmgr:atexit_stop_monitoring",
        ):
            monitor.stop_monitoring()


atexit.register(_cleanup_active_monitors)


class MemoryManagerError(Exception):
    """Base exception for memory manager errors."""


class MemoryPressureError(MemoryManagerError):
    """Raised when memory pressure becomes critical."""


class AllocationError(MemoryManagerError):
    """Raised when memory allocation fails."""


@dataclass
class MemoryStats:
    """Comprehensive memory statistics and monitoring."""

    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    memory_pressure: float = 0.0  # 0.0-1.0 scale

    # Pool statistics
    allocated_pools: int = 0
    active_pools: int = 0
    pool_memory_gb: float = 0.0
    pool_efficiency: float = 0.0

    # Allocation patterns
    allocation_rate: float = 0.0  # allocations per second
    deallocation_rate: float = 0.0  # deallocations per second
    fragmentation_ratio: float = 0.0  # 0.0-1.0, higher = more fragmented

    # System pressure indicators
    swap_usage_gb: float = 0.0
    page_faults_per_sec: float = 0.0
    gc_collections_per_min: float = 0.0

    # Performance impact
    allocation_latency_ms: float = 0.0
    memory_throughput_mbps: float = 0.0

    def update_system_stats(self) -> None:
        """Update system memory statistics."""
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()

        self.total_memory_gb = memory_info.total / (1024**3)
        self.available_memory_gb = memory_info.available / (1024**3)
        self.used_memory_gb = memory_info.used / (1024**3)
        self.memory_pressure = memory_info.percent / 100.0
        self.swap_usage_gb = swap_info.used / (1024**3)

    def get_pressure_level(self) -> str:
        """Get human-readable pressure level."""
        if self.memory_pressure < 0.6:
            return "low"
        elif self.memory_pressure < 0.8:
            return "moderate"
        elif self.memory_pressure < 0.9:
            return "high"
        else:
            return "critical"


@dataclass
class MemoryPool:
    """Memory pool for efficient buffer reuse."""

    pool_id: str
    buffer_size: int
    max_buffers: int
    buffers: deque = field(default_factory=deque)
    allocated_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate for this pool."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / max(total_requests, 1)

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage of this pool in MB."""
        return (len(self.buffers) * self.buffer_size * 8) / (1024 * 1024)

    def get_buffer(self) -> np.ndarray | None:
        """Get a buffer from the pool."""
        self.last_access_time = time.time()

        if self.buffers:
            self.hit_count += 1
            return self.buffers.popleft()  # type: ignore[no-any-return]
        else:
            self.miss_count += 1
            if self.allocated_count < self.max_buffers:
                buffer = np.empty(self.buffer_size, dtype=np.float64)
                self.allocated_count += 1
                return buffer
            return None

    def return_buffer(self, buffer: np.ndarray) -> None:
        """Return a buffer to the pool."""
        if len(self.buffers) < self.max_buffers:
            # Clear buffer contents for security
            buffer.fill(0.0)
            self.buffers.append(buffer)


class MemoryPressureMonitor:
    """Real-time memory pressure monitoring with adaptive responses.

    Monitors system memory usage and triggers appropriate responses
    to prevent out-of-memory conditions.
    """

    def __init__(
        self,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.9,
        monitoring_interval: float = 1.0,
    ):
        """Initialize memory pressure monitor.

        Args:
            warning_threshold: Memory pressure threshold for warnings (0.0-1.0)
            critical_threshold: Memory pressure threshold for critical actions (0.0-1.0)
            monitoring_interval: Monitoring interval in seconds
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_interval = monitoring_interval

        self.stats = MemoryStats()
        self._monitoring_active = False
        self._monitoring_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Pressure response callbacks
        self._warning_callbacks: list[Callable] = []
        self._critical_callbacks: list[Callable] = []
        self._recovery_callbacks: list[Callable] = []

        # Pressure state tracking to prevent duplicate logs
        self._last_pressure_state: str = "normal"  # "normal", "warning", "critical"
        self._recovery_logged: bool = False  # Track if recovery was already logged
        self._warning_logged: bool = False  # Track if warning was already logged
        self._critical_logged: bool = False  # Track if critical was already logged

        # GC rate limiting to avoid wasteful calls
        self._last_gc_freed: int = -1  # Objects freed in last GC (-1 = never run)
        self._consecutive_zero_gc: int = 0  # Count of consecutive GC calls that freed 0

        # Pressure history for trend analysis
        self._pressure_history: deque = deque(maxlen=300)  # 5 minutes at 1s intervals

        # Register this monitor
        _active_monitors.add(self)

        logger.info(
            f"Memory pressure monitor initialized: warning={warning_threshold}, "
            f"critical={critical_threshold}",
        )

    def start_monitoring(self) -> None:
        """Start background memory pressure monitoring."""
        if self._monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        self._monitoring_active = True
        self._shutdown_event.clear()

        # Register this monitor in active list
        _active_monitors.add(self)

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MemoryPressureMonitor",
            daemon=True,
        )
        self._monitoring_thread.start()

        logger.info("Memory pressure monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory pressure monitoring."""
        self._monitoring_active = False
        self._shutdown_event.set()
        _active_monitors.discard(self)

        with logged_errors(
            logger, "join_monitoring_thread", policy="suppress", level=logging.DEBUG
        ):
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=2.0)

        logger.info("Memory pressure monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                self._update_stats()
                self._check_pressure_levels()
                self._shutdown_event.wait(self.monitoring_interval)
            except Exception as exc:
                log_once(
                    logger,
                    logging.DEBUG,
                    f"{id(self)}:memmgr:monitoring_loop",
                    "Memory monitoring error: %s",
                    exc,
                )
                with logged_errors(
                    logger,
                    "monitoring_loop_backoff_wait",
                    policy="suppress",
                    level=logging.DEBUG,
                ):
                    self._shutdown_event.wait(5.0)  # Longer wait on error

    def _update_stats(self) -> None:
        """Update memory statistics."""
        with logged_errors(
            logger,
            "update_stats",
            policy="suppress",
            level=logging.DEBUG,
            once_key=f"{id(self)}:memmgr:update_stats",
        ):
            self.stats.update_system_stats()

            # Add to pressure history
            pressure_snapshot = {
                "timestamp": time.time(),
                "pressure": self.stats.memory_pressure,
                "available_gb": self.stats.available_memory_gb,
                "swap_usage_gb": self.stats.swap_usage_gb,
            }
            self._pressure_history.append(pressure_snapshot)

    def _check_pressure_levels(self) -> None:
        """Check memory pressure levels and trigger responses.

        Only logs state transitions to avoid log spam. Callbacks are triggered
        on state entry only, not every monitoring cycle.
        """
        current_pressure = self.stats.memory_pressure
        new_state = "normal"

        if current_pressure >= self.critical_threshold:
            new_state = "critical"
            # Only trigger on state transition to critical
            if not self._critical_logged:
                self._trigger_critical_response()
                self._critical_logged = True
                self._warning_logged = False  # Reset warning flag
            self._recovery_logged = False  # Reset recovery flag
        elif current_pressure >= self.warning_threshold:
            new_state = "warning"
            # Only trigger on state transition to warning
            if not self._warning_logged:
                self._trigger_warning_response()
                self._warning_logged = True
                self._critical_logged = False  # Reset critical flag
            self._recovery_logged = False  # Reset recovery flag
        elif current_pressure < self.warning_threshold * 0.8:  # Recovery threshold
            new_state = "normal"
            # Only trigger recovery callback on state transition (not every cycle)
            if (
                self._last_pressure_state in ("warning", "critical")
                and not self._recovery_logged
            ):
                self._trigger_recovery_response()
                self._recovery_logged = True
            # Reset warning/critical flags when recovered
            self._warning_logged = False
            self._critical_logged = False

        # Update state
        self._last_pressure_state = new_state

    def _trigger_warning_response(self) -> None:
        """Trigger warning-level memory pressure response."""
        logger.warning(
            f"Memory pressure warning: {self.stats.memory_pressure:.1%} "
            f"(available: {self.stats.available_memory_gb:.1f}GB)",
        )

        for callback in self._warning_callbacks:
            try:
                callback(self.stats)
            except Exception as exc:
                log_once(
                    logger,
                    logging.DEBUG,
                    f"{id(self)}:memmgr:warning_callback:{id(callback)}",
                    "Warning callback failed: %s",
                    exc,
                )

    def _trigger_critical_response(self) -> None:
        """Trigger critical-level memory pressure response."""
        logger.critical(
            f"Critical memory pressure: {self.stats.memory_pressure:.1%} "
            f"(available: {self.stats.available_memory_gb:.1f}GB)",
        )

        for callback in self._critical_callbacks:
            try:
                callback(self.stats)
            except Exception as exc:
                log_once(
                    logger,
                    logging.DEBUG,
                    f"{id(self)}:memmgr:critical_callback:{id(callback)}",
                    "Critical callback failed: %s",
                    exc,
                )

    def _trigger_recovery_response(self) -> None:
        """Trigger recovery-level response when pressure decreases."""
        for callback in self._recovery_callbacks:
            try:
                callback(self.stats)
            except Exception as exc:
                log_once(
                    logger,
                    logging.DEBUG,
                    f"{id(self)}:memmgr:recovery_callback:{id(callback)}",
                    "Recovery callback failed: %s",
                    exc,
                )

    def register_warning_callback(
        self,
        callback: Callable[[MemoryStats], None],
    ) -> None:
        """Register callback for warning-level memory pressure."""
        self._warning_callbacks.append(callback)

    def register_critical_callback(
        self,
        callback: Callable[[MemoryStats], None],
    ) -> None:
        """Register callback for critical-level memory pressure."""
        self._critical_callbacks.append(callback)

    def register_recovery_callback(
        self,
        callback: Callable[[MemoryStats], None],
    ) -> None:
        """Register callback for memory pressure recovery."""
        self._recovery_callbacks.append(callback)

    def get_pressure_trend(self, window_minutes: int = 5) -> str:
        """Get memory pressure trend over specified window.

        Args:
            window_minutes: Analysis window in minutes

        Returns:
            Trend description: "increasing", "decreasing", "stable"
        """
        if len(self._pressure_history) < 10:
            return "insufficient_data"

        cutoff_time = time.time() - (window_minutes * 60)
        recent_pressures = [
            h["pressure"]
            for h in self._pressure_history
            if h["timestamp"] > cutoff_time
        ]

        if len(recent_pressures) < 5:
            return "insufficient_data"

        # Simple trend analysis
        first_half = recent_pressures[: len(recent_pressures) // 2]
        second_half = recent_pressures[len(recent_pressures) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        change = (second_avg - first_avg) / max(first_avg, 1e-10)

        if change > 0.05:
            return "increasing"
        elif change < -0.05:
            return "decreasing"
        else:
            return "stable"

    def __del__(self) -> None:
        """Destructor to ensure cleanup when garbage collected."""
        with logged_errors(
            logger, "monitor_del_stop", policy="suppress", level=logging.DEBUG
        ):
            self.stop_monitoring()


class AdvancedMemoryManager:
    """Advanced memory manager with intelligent allocation strategies.

    Provides dynamic memory allocation, pool management, pressure monitoring,
    and optimization strategies for massive XPCS datasets.
    """

    @log_calls(include_args=False)
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize advanced memory manager.

        Args:
            config: Memory management configuration
        """
        self.config = config or {}
        self.memory_config = self.config.get("memory", {})

        # Memory pools for different buffer sizes
        self._pools: dict[int, MemoryPool] = {}
        self._pools_lock = threading.RLock()

        # Memory pressure monitoring
        warning_threshold = self.memory_config.get("warning_threshold", 0.75)
        critical_threshold = self.memory_config.get("critical_threshold", 0.9)
        monitoring_interval = self.memory_config.get("monitoring_interval", 1.0)

        self.pressure_monitor = MemoryPressureMonitor(
            warning_threshold,
            critical_threshold,
            monitoring_interval,
        )

        # Register pressure response callbacks
        self.pressure_monitor.register_warning_callback(self._handle_memory_warning)
        self.pressure_monitor.register_critical_callback(self._handle_memory_critical)
        self.pressure_monitor.register_recovery_callback(self._handle_memory_recovery)

        # Memory allocation tracking
        self._allocation_history: deque = deque(maxlen=1000)
        self._total_allocated_mb = 0.0
        self._allocation_lock = threading.Lock()

        # Garbage collection optimization
        self._gc_optimization_enabled = self.memory_config.get("gc_optimization", True)
        self._last_gc_time = 0.0
        self._gc_threshold_multiplier = 2.0  # Reduce GC frequency under pressure

        # GC rate limiting (owned by this class, not by pressure_monitor)
        self._consecutive_zero_gc: int = 0
        self._last_gc_freed: int = 0

        # Virtual memory support
        self._virtual_memory_enabled = self.memory_config.get("virtual_memory", True)
        default_vm_path = str(
            Path(os.path.expanduser("~/.cache/xpcsjax/vm")) / "xpcsjax_vm"
        )
        self._virtual_memory_path = self.memory_config.get(
            "virtual_memory_path",
            default_vm_path,
        )

        # Start monitoring
        if self.memory_config.get("enable_monitoring", True):
            self.pressure_monitor.start_monitoring()

        logger.info("Advanced memory manager initialized")

    @contextmanager
    def managed_allocation(  # type: ignore[no-untyped-def]
        self,
        size: int,
        dtype: np.dtype = np.float64,  # type: ignore[assignment]
        pool_enabled: bool = True,
    ):
        """Context manager for managed memory allocation.

        Args:
            size: Number of elements to allocate
            dtype: Data type for allocation
            pool_enabled: Whether to use memory pooling

        Yields:
            Allocated array
        """
        buffer = None
        pool_id = None

        try:
            # Attempt to get from pool first
            if pool_enabled and dtype == np.float64:
                buffer, pool_id = self._get_from_pool(size)

            # Allocate new buffer if pool failed
            if buffer is None:
                buffer = self._allocate_buffer(size, dtype)

            yield buffer

        finally:
            # Return to pool or cleanup
            if buffer is not None:
                if pool_id and pool_enabled:
                    self._return_to_pool(buffer, pool_id)
                else:
                    del buffer
                    if self._gc_optimization_enabled:
                        self._optimize_garbage_collection()

    def _get_from_pool(self, size: int) -> tuple[np.ndarray | None, str | None]:
        """Get buffer from appropriate memory pool."""
        with self._pools_lock:
            # Find appropriate pool (use next power of 2 for size)
            pool_size = 1
            while pool_size < size:
                pool_size *= 2

            pool_id_int = pool_size  # Use int as key

            # Get or create pool
            if pool_id_int not in self._pools:
                max_buffers = max(
                    4,
                    min(32, int(1024 * 1024 * 1024 / (pool_size * 8))),
                )  # ~1GB max per pool
                self._pools[pool_id_int] = MemoryPool(
                    pool_id=f"pool_{pool_size}",
                    buffer_size=pool_size,
                    max_buffers=max_buffers,
                )

            pool = self._pools[pool_id_int]
            buffer = pool.get_buffer()

            if buffer is not None:
                view = buffer[:size]  # Return view of correct size
                return view, f"pool_{pool_size}"

            return None, None

    def _return_to_pool(self, buffer: np.ndarray, pool_id: str) -> None:
        """Return buffer to memory pool."""
        with self._pools_lock:
            # Extract size from pool_id like "pool_1024"
            try:
                pool_size = int(pool_id.split("_")[1])
                if pool_size in self._pools:
                    pool = self._pools[pool_size]
                    # Walk the base chain to find the original pool buffer.
                    # Guard: only return if the base has the expected pool size;
                    # mismatched size means we have a view-of-view chain where
                    # an intermediate view is not the original allocation.
                    base_buffer = buffer.base if buffer.base is not None else buffer
                    if base_buffer.size != pool_size:
                        # Try one more level up (view of view)
                        if (
                            base_buffer.base is not None
                            and base_buffer.base.size == pool_size
                        ):
                            base_buffer = base_buffer.base
                        else:
                            # Cannot recover original buffer; do not corrupt pool
                            logger.debug(
                                f"Skipping pool return: base size {base_buffer.size} "
                                f"!= pool size {pool_size}"
                            )
                            return
                    pool.return_buffer(base_buffer)
            except (ValueError, IndexError):
                pass

    def _allocate_buffer(self, size: int, dtype: np.dtype) -> np.ndarray:
        """Allocate new memory buffer with tracking."""
        start_time = time.time()

        try:
            # Check memory pressure before allocation
            if (
                self.pressure_monitor.stats.memory_pressure
                > self.pressure_monitor.critical_threshold
            ):
                self._emergency_memory_cleanup()

            # Attempt allocation
            buffer = np.empty(size, dtype=dtype)

            # Track allocation
            allocation_time = time.time() - start_time
            buffer_size_mb = buffer.nbytes / (1024 * 1024)

            with self._allocation_lock:
                self._total_allocated_mb += buffer_size_mb
                self._allocation_history.append(
                    {
                        "timestamp": time.time(),
                        "size_mb": buffer_size_mb,
                        "allocation_time_ms": allocation_time * 1000,
                        "success": True,
                    },
                )

            logger.debug(
                f"Allocated {buffer_size_mb:.1f}MB buffer in {allocation_time * 1000:.1f}ms",
            )
            return buffer

        except MemoryError as e:
            # Handle allocation failure
            allocation_time = time.time() - start_time

            with self._allocation_lock:
                self._allocation_history.append(
                    {
                        "timestamp": time.time(),
                        "size_mb": size * np.dtype(dtype).itemsize / (1024 * 1024),
                        "allocation_time_ms": allocation_time * 1000,
                        "success": False,
                    },
                )

            logger.error(f"Memory allocation failed: {size} elements of {dtype}")

            # Try emergency cleanup and retry once
            self._emergency_memory_cleanup()

            try:
                buffer = np.empty(size, dtype=dtype)
                logger.warning("Memory allocation succeeded after emergency cleanup")
                return buffer
            except MemoryError:
                # If still fails, try virtual memory if enabled
                if self._virtual_memory_enabled:
                    return self._allocate_virtual_memory(size, dtype)
                else:
                    raise AllocationError(
                        f"Failed to allocate {size} elements of {dtype}",
                    ) from e

    def _allocate_virtual_memory(self, size: int, dtype: np.dtype) -> np.ndarray:
        """Allocate virtual memory-backed array for very large datasets."""
        try:
            # Create memory-mapped file
            element_size = np.dtype(dtype).itemsize
            total_bytes = size * element_size

            if total_bytes == 0:
                logger.warning("Zero-size virtual memory allocation requested")
                return np.empty(0, dtype=dtype)

            # Ensure virtual memory directory exists
            os.makedirs(os.path.dirname(self._virtual_memory_path), exist_ok=True)

            # Create unique filename
            vm_file = (
                f"{self._virtual_memory_path}_{int(time.time())}_{os.getpid()}.dat"
            )

            # Create sparse file: seek to the last byte and write one zero.
            # This avoids allocating total_bytes in RAM just to populate zeros —
            # the OS fills unwritten extents with zero pages on demand.
            with open(vm_file, "wb") as f:
                f.seek(total_bytes - 1)
                f.write(b"\x00")
            os.chmod(vm_file, 0o600)

            # Register cleanup on process exit (best-effort)
            def _cleanup_vm(path: str = vm_file) -> None:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass

            atexit.register(_cleanup_vm)

            # Memory map the file — keep fh open for mmap lifetime
            fh = open(vm_file, "r+b")
            mm = mmap.mmap(fh.fileno(), 0)

            # Create numpy array from memory map
            buffer = np.ndarray(
                shape=(total_bytes // np.dtype(dtype).itemsize,), dtype=dtype, buffer=mm
            )

            # Store references to keep mmap and file handle alive
            buffer._xpcsjax_mmap = mm  # type: ignore[attr-defined]
            buffer._xpcsjax_fh = fh  # type: ignore[attr-defined]
            buffer._xpcsjax_vm_file = vm_file  # type: ignore[attr-defined]

            logger.info(
                f"Allocated {total_bytes / (1024 * 1024):.1f}MB virtual memory buffer",
            )
            return buffer

        except Exception as e:
            logger.error(f"Virtual memory allocation failed: {e}")
            raise AllocationError("Virtual memory allocation failed") from e

    def _handle_memory_warning(self, stats: MemoryStats) -> None:
        """Handle memory pressure warning.

        Includes GC rate-limiting to avoid wasteful calls when GC cannot
        free memory (e.g., during JAX/NumPy-heavy workloads where memory
        is actively referenced).
        """
        logger.warning("Memory pressure warning - triggering optimization")

        # Trigger garbage collection with rate-limiting
        # Skip GC if previous calls consistently freed 0 objects (JAX/NumPy workload)
        if self._gc_optimization_enabled:
            # Skip GC if we've had 3+ consecutive zero-result collections
            # This indicates memory is in use by JAX/NumPy, not collectable
            if self._consecutive_zero_gc >= 3:
                logger.debug(
                    "Skipping GC - previous calls freed 0 objects "
                    "(memory likely in JAX/NumPy arrays)"
                )
            else:
                collected = gc.collect()
                logger.debug(f"Garbage collection freed {collected} objects")

                # Track consecutive zero-result collections
                if collected == 0:
                    self._consecutive_zero_gc += 1
                else:
                    self._consecutive_zero_gc = 0
                self._last_gc_freed = collected

        # Clean up old pools
        with logged_errors(
            logger, "cleanup_old_pools", policy="suppress", level=logging.DEBUG
        ):
            self._cleanup_old_pools()

        # Adjust GC thresholds to be more aggressive
        if self._gc_optimization_enabled:
            with logged_errors(
                logger, "warning_gc_threshold", policy="suppress", level=logging.DEBUG
            ):
                current_thresholds = gc.get_threshold()
                new_thresholds = tuple(
                    int(t / self._gc_threshold_multiplier) for t in current_thresholds
                )
                gc.set_threshold(*new_thresholds)

    def _handle_memory_critical(self, stats: MemoryStats) -> None:
        """Handle critical memory pressure."""
        logger.critical("Critical memory pressure - performing emergency cleanup")

        with logged_errors(
            logger, "emergency_cleanup", policy="suppress", level=logging.DEBUG
        ):
            self._emergency_memory_cleanup()

        # More aggressive GC threshold adjustment
        if self._gc_optimization_enabled:
            with logged_errors(
                logger, "critical_gc_threshold", policy="suppress", level=logging.DEBUG
            ):
                current_thresholds = gc.get_threshold()
                new_thresholds = tuple(
                    int(t / (self._gc_threshold_multiplier * 2)) for t in current_thresholds
                )
                gc.set_threshold(*new_thresholds)

    def _handle_memory_recovery(self, stats: MemoryStats) -> None:
        """Handle memory pressure recovery."""
        logger.info("Memory pressure recovered - restoring normal operation")

        # Restore normal GC thresholds
        if self._gc_optimization_enabled:
            with logged_errors(
                logger, "recovery_gc_threshold", policy="suppress", level=logging.DEBUG
            ):
                # Reset to default thresholds
                gc.set_threshold(700, 10, 10)

    def _emergency_memory_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")

        # Clear all memory pools
        with logged_errors(
            logger, "emergency_clear_pools", policy="suppress", level=logging.DEBUG
        ):
            with self._pools_lock:
                for pool in self._pools.values():
                    pool.buffers.clear()
                self._pools.clear()

        # Force garbage collection multiple times
        for _ in range(3):
            try:
                collected = gc.collect()
                logger.debug(f"Emergency GC collected {collected} objects")
            except Exception as exc:
                log_once(
                    logger,
                    logging.DEBUG,
                    f"{id(self)}:memmgr:emergency_gc",
                    "Emergency GC failed: %s",
                    exc,
                )

        # JAX memory cleanup if available
        if HAS_JAX:
            try:
                # CRITICAL FIX (Nov 10, 2025): jax.clear_backends() removed in JAX 0.4.0+
                # Use jax.clear_caches() for JAX 0.8.0 compatibility
                # This clears JIT compilation cache and helps release device memory
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
                    logger.debug("Cleared JAX compilation cache")
                else:
                    logger.debug("JAX clear_caches() not available (older JAX version)")
            except Exception as exc:
                log_exception(
                    logger,
                    exc,
                    context={"operation": "jax_memory_cleanup"},
                    level=logging.DEBUG,
                )

    def _cleanup_old_pools(self) -> None:
        """Clean up old or unused memory pools."""
        current_time = time.time()
        cleanup_threshold = 300  # 5 minutes

        with self._pools_lock:
            pools_to_remove = []

            for pool_id, pool in self._pools.items():
                if current_time - pool.last_access_time > cleanup_threshold:
                    if pool.hit_rate < 0.1:  # Low hit rate
                        pools_to_remove.append(pool_id)

            for pool_id in pools_to_remove:
                pool = self._pools[pool_id]
                pool.buffers.clear()
                del self._pools[pool_id]
                logger.debug(f"Cleaned up unused pool: {pool_id}")

    def _optimize_garbage_collection(self) -> None:
        """Optimize garbage collection based on current conditions."""
        if not self._gc_optimization_enabled:
            return

        current_time = time.time()

        # Don't run GC too frequently
        if current_time - self._last_gc_time < 1.0:
            return

        # Run GC if under memory pressure
        if self.pressure_monitor.stats.memory_pressure > 0.8:
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Proactive GC collected {collected} objects")

        self._last_gc_time = current_time

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self._pools_lock:
            pool_stats = {}
            total_pool_memory = 0.0

            for pool_id, pool in self._pools.items():
                pool_memory = pool.memory_usage_mb
                total_pool_memory += pool_memory

                pool_stats[pool_id] = {
                    "buffer_size": pool.buffer_size,
                    "buffer_count": len(pool.buffers),
                    "allocated_count": pool.allocated_count,
                    "max_buffers": pool.max_buffers,
                    "hit_rate": pool.hit_rate,
                    "memory_usage_mb": pool_memory,
                }

        # Calculate allocation statistics
        with self._allocation_lock:
            recent_allocations = [
                a
                for a in self._allocation_history
                if time.time() - a["timestamp"] < 60  # Last minute
            ]

            successful_allocations = [a for a in recent_allocations if a["success"]]

            avg_allocation_time = 0.0
            if successful_allocations:
                avg_allocation_time = sum(
                    a["allocation_time_ms"] for a in successful_allocations
                ) / len(successful_allocations)

            allocation_success_rate = len(successful_allocations) / max(
                len(recent_allocations),
                1,
            )

        return {
            "system_memory": {
                "total_gb": self.pressure_monitor.stats.total_memory_gb,
                "available_gb": self.pressure_monitor.stats.available_memory_gb,
                "used_gb": self.pressure_monitor.stats.used_memory_gb,
                "pressure": self.pressure_monitor.stats.memory_pressure,
                "pressure_level": self.pressure_monitor.stats.get_pressure_level(),
                "pressure_trend": self.pressure_monitor.get_pressure_trend(),
            },
            "pool_management": {
                "active_pools": len(self._pools),
                "total_pool_memory_mb": total_pool_memory,
                "pool_stats": pool_stats,
            },
            "allocation_performance": {
                "total_allocated_mb": self._total_allocated_mb,
                "avg_allocation_time_ms": avg_allocation_time,
                "allocation_success_rate": allocation_success_rate,
                "recent_allocations": len(recent_allocations),
            },
            "optimization_status": {
                "gc_optimization_enabled": self._gc_optimization_enabled,
                "virtual_memory_enabled": self._virtual_memory_enabled,
                "monitoring_active": self.pressure_monitor._monitoring_active,
            },
        }

    def optimize_for_workload(self, workload_type: str, dataset_size_gb: float) -> None:
        """Optimize memory management for specific workload characteristics.

        Args:
            workload_type: Type of workload ("streaming", "batch", "interactive")
            dataset_size_gb: Expected dataset size in GB
        """
        logger.info(
            f"Optimizing memory management for {workload_type} workload, "
            f"dataset size: {dataset_size_gb:.1f}GB",
        )

        if workload_type == "streaming":
            # Optimize for streaming workload
            self._gc_threshold_multiplier = 1.5  # More frequent GC
            self.pressure_monitor.warning_threshold = 0.7  # Earlier warning

        elif workload_type == "batch":
            # Optimize for batch processing
            self._gc_threshold_multiplier = 3.0  # Less frequent GC
            self.pressure_monitor.warning_threshold = 0.8  # Later warning

        elif workload_type == "interactive":
            # Optimize for interactive use
            self._gc_threshold_multiplier = 2.0  # Balanced GC
            self.pressure_monitor.warning_threshold = 0.75  # Standard warning

        # Adjust pool sizes based on dataset size
        if dataset_size_gb > 10.0:
            # Large dataset - bigger pools
            with self._pools_lock:
                for pool in self._pools.values():
                    pool.max_buffers = min(pool.max_buffers * 2, 64)

        logger.info(f"Memory optimization applied for {workload_type} workload")

    def cleanup_virtual_memory(self) -> None:
        """Clean up any virtual memory files."""
        try:
            vm_dir = os.path.dirname(self._virtual_memory_path)
            if os.path.exists(vm_dir):
                for file in os.listdir(vm_dir):
                    if file.startswith(os.path.basename(self._virtual_memory_path)):
                        try:
                            os.remove(os.path.join(vm_dir, file))
                            logger.debug(f"Cleaned up virtual memory file: {file}")
                        except Exception as exc:
                            log_once(
                                logger,
                                logging.DEBUG,
                                f"{id(self)}:memmgr:cleanup_vm_file",
                                "Failed to cleanup virtual memory file %s: %s",
                                file,
                                exc,
                            )
        except Exception as exc:
            log_exception(
                logger,
                exc,
                context={"operation": "cleanup_virtual_memory"},
                level=logging.DEBUG,
            )

    def shutdown(self) -> None:
        """Shutdown memory manager and cleanup resources."""
        logger.info("Shutting down advanced memory manager")

        # Stop monitoring
        with logged_errors(
            logger, "shutdown_stop_monitoring", policy="suppress", level=logging.DEBUG
        ):
            self.pressure_monitor.stop_monitoring()

        # Clear all pools
        with logged_errors(
            logger, "shutdown_clear_pools", policy="suppress", level=logging.DEBUG
        ):
            with self._pools_lock:
                for pool in self._pools.values():
                    pool.buffers.clear()
                self._pools.clear()

        # Cleanup virtual memory files
        with logged_errors(
            logger, "shutdown_cleanup_vm", policy="suppress", level=logging.DEBUG
        ):
            self.cleanup_virtual_memory()

        # Final garbage collection
        with logged_errors(
            logger, "shutdown_gc", policy="suppress", level=logging.DEBUG
        ):
            gc.collect()

        logger.info("Advanced memory manager shutdown complete")

    def __enter__(self) -> "AdvancedMemoryManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Context manager exit."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor to ensure cleanup when garbage collected."""
        with logged_errors(
            logger, "manager_del_shutdown", policy="suppress", level=logging.DEBUG
        ):
            self.shutdown()


# Export main classes and functions
__all__ = [
    "AdvancedMemoryManager",
    "MemoryPressureMonitor",
    "MemoryStats",
    "MemoryPool",
    "MemoryManagerError",
    "MemoryPressureError",
    "AllocationError",
]
