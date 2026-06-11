"""HPC CPU device optimization with intelligent configuration.

Provides CPU-only device detection, configuration, and optimization
for high-performance computing environments.

GPU support removed - CPU-only optimization focus.

Key Features:
- Automatic CPU device detection and optimal configuration
- HPC CPU optimization for 36/128-core nodes
- Performance benchmarking and optimization
- NUMA-aware configuration
- Multi-core thread allocation strategies

Usage:
    from xpcsjax.device import configure_optimal_device
    config = configure_optimal_device()
"""

from __future__ import annotations

import logging
from typing import Any

# Suppress JAX backend warnings and messages (CPU-only)
# - TPU backend warnings (not available on standard systems)
# - GPU fallback warnings (expected behavior for CPU-only installation)
# - Backend initialization INFO messages
# IMPORTANT: Don't set JAX_PLATFORMS - let JAX auto-detect available backend

# Suppress JAX backend logs (set to ERROR to hide GPU fallback warnings)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

from xpcsjax.utils.logging import get_logger  # noqa: E402 - After logging config

logger = get_logger(__name__)

# Import CPU-specific module (re-exported for public API)
try:
    from xpcsjax.device.cpu import (  # noqa: F401
        benchmark_cpu_performance,
        configure_cpu_hpc,
        detect_cpu_info,
        get_optimal_batch_size,
    )

    HAS_CPU_MODULE = True
except ImportError as e:
    logger.warning(f"CPU optimization module not available: {e}")
    HAS_CPU_MODULE = False


def configure_optimal_device(
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """Automatically configure the optimal CPU device for xpcsjax analysis.

    Configures optimized CPU settings for HPC environments.

    Parameters
    ----------
    cpu_threads : int, optional
        Number of CPU threads to use. If None, auto-detects optimal count.

    Returns
    -------
    dict
        Device configuration summary with performance hints
    """
    logger.info("Configuring optimal CPU device for xpcsjax analysis")

    config_result = {
        "device_type": "cpu",
        "configuration_successful": False,
        "performance_ready": False,
        "recommendations": [],
        "warnings": [],
        "device_info": {},
    }

    # Configure CPU optimization
    return _configure_cpu_optimal(config_result, cpu_threads)


def _configure_cpu_optimal(
    config_result: dict[str, Any], cpu_threads: int | None
) -> dict[str, Any]:
    """Configure optimal CPU settings."""
    logger.info("Configuring CPU optimization...")

    try:
        if HAS_CPU_MODULE:
            cpu_config = configure_cpu_hpc(
                num_threads=cpu_threads,
                enable_hyperthreading=False,  # Usually better for HPC
                numa_policy="auto",
                memory_optimization="standard",
            )

            config_result.update(
                {
                    "device_type": "cpu",
                    "configuration_successful": True,
                    "performance_ready": True,
                    "device_info": cpu_config,
                    "recommendations": [
                        f"CPU optimization configured for {cpu_config['threads_configured']} threads",
                        "Multi-core CPU optimizations enabled",
                    ],
                },
            )

            logger.info(
                f"[OK] CPU configuration successful with {cpu_config['threads_configured']} threads",
            )

        else:
            # Minimal CPU configuration if module not available
            import multiprocessing
            import os

            num_cores = multiprocessing.cpu_count()
            os.environ["OMP_NUM_THREADS"] = str(num_cores)
            os.environ["JAX_PLATFORMS"] = "cpu"

            config_result.update(
                {
                    "device_type": "cpu",
                    "configuration_successful": True,
                    "performance_ready": False,
                    "recommendations": [
                        f"Basic CPU configuration with {num_cores} cores",
                        "Install psutil for advanced CPU optimization",
                    ],
                },
            )

            logger.info("[OK] Basic CPU configuration completed")

    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as e:
        logger.error(f"CPU configuration failed: {e}")
        config_result.update(
            {
                "device_type": "cpu",
                "configuration_successful": False,
                "performance_ready": False,
                "warnings": [f"CPU configuration failed: {e}"],
            },
        )

    return config_result


def get_device_status() -> dict[str, Any]:
    """Get current device status and capabilities.

    Returns
    -------
    dict
        Comprehensive CPU device status information
    """
    status: dict[str, Any] = {
        "timestamp": None,
        "cpu_info": {},
        "recommendations": [],
        "performance_estimate": "unknown",
    }

    try:
        import datetime

        status["timestamp"] = datetime.datetime.now().isoformat()

        # Get CPU information
        if HAS_CPU_MODULE:
            status["cpu_info"] = detect_cpu_info()
        else:
            import multiprocessing

            status["cpu_info"] = {
                "logical_cores": multiprocessing.cpu_count(),
                "optimization_available": False,
            }

        # Generate performance estimate based on CPU cores
        if status["cpu_info"].get("physical_cores", 0) >= 32:
            status["performance_estimate"] = "high"
            status["recommendations"].append(
                "High-core-count CPU detected - excellent performance expected",
            )
        elif status["cpu_info"].get("physical_cores", 0) >= 16:
            status["performance_estimate"] = "medium-high"
            status["recommendations"].append(
                "Multi-core CPU detected - good performance expected",
            )
        else:
            status["performance_estimate"] = "medium"
            status["recommendations"].append(
                "Standard CPU configuration - adequate performance expected",
            )

    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as e:
        logger.error(f"Device status check failed: {e}")
        status["error"] = str(e)

    return status


def benchmark_device_performance(
    test_size: int = 5000,
) -> dict[str, Any]:
    """Benchmark CPU device performance for optimization planning.

    Parameters
    ----------
    test_size : int, default 5000
        Size of benchmark computation

    Returns
    -------
    dict
        Benchmark results with performance metrics
    """
    logger.info(f"Benchmarking CPU performance (test_size={test_size})")

    benchmark_results: dict[str, Any] = {
        "device_type": "cpu",
        "test_size": test_size,
        "results": {},
        "recommendations": [],
    }

    try:
        # Benchmark CPU
        if HAS_CPU_MODULE:
            logger.info("Running CPU benchmark...")
            cpu_results = benchmark_cpu_performance(test_size=test_size)
            benchmark_results["results"]["cpu"] = cpu_results
            benchmark_results["recommendations"].append(
                "CPU benchmark completed - see results for performance metrics",
            )
        else:
            benchmark_results["recommendations"].append(
                "Install psutil for detailed CPU benchmarking",
            )

    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as e:
        logger.error(f"Device benchmarking failed: {e}")
        benchmark_results["error"] = str(e)

    return benchmark_results


# Main exports. Literal __all__ + conditional ``+=`` so Pyright can analyze it
# (reportUnsupportedDunderAll); assigning a dynamically-built variable was not
# statically supported. The CPU symbols are statically importable (the
# try-import above), so they pass the dunder-all presence check.
__all__ = [
    # Primary device configuration
    "configure_optimal_device",
    # Device information
    "get_device_status",
    "benchmark_device_performance",
    # Status flags
    "HAS_CPU_MODULE",
]

# Add CPU-specific exports if available
if HAS_CPU_MODULE:
    __all__ += [
        "benchmark_cpu_performance",
        "configure_cpu_hpc",
        "detect_cpu_info",
        "get_optimal_batch_size",
    ]
