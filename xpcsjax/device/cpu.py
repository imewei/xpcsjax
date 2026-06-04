"""HPC CPU Optimization for Homodyne
====================================

CPU-primary optimization strategies for high-performance computing environments.
Optimized for 36/128-core HPC nodes with intelligent thread management and
JAX CPU configuration.

Key Features:
- CPU core detection and optimal thread allocation
- JAX CPU-specific optimizations for HPC environments
- Memory-efficient processing strategies
- NUMA-aware configuration
- Intel/AMD architecture detection and optimization

HPC Environment Support:
- 36-core HPC nodes (typical cluster setup)
- 128-core HPC nodes (high-end clusters)
- Multi-socket NUMA systems
- Intel Xeon and AMD EPYC processors
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess  # nosec B404
from typing import Any

import psutil

from xpcsjax.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


def _jax_backend_initialized() -> bool:
    """Return True if a JAX backend is already live.

    XLA reads ``XLA_FLAGS`` exactly once, when the backend is first initialized.
    After that, mutating the env var has no effect. Checking the live-backend
    registry (best-effort across JAX versions, with a ``sys.modules`` fallback)
    lets ``configure_cpu_jax`` warn instead of silently writing ignored flags.
    """
    try:
        from jax._src import xla_bridge

        return bool(getattr(xla_bridge, "_backends", None))
    except Exception as exc:
        import sys

        log_exception(
            logger,
            exc,
            context={
                "operation": "jax_backend_live_probe",
                "fallback": "sys.modules",
            },
            level=logging.DEBUG,
        )
        return "jax" in sys.modules


# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore[assignment]


def detect_cpu_info() -> dict[str, Any]:
    """Detect CPU architecture and capabilities for optimization.

    Returns
    -------
    dict
        CPU information including cores, architecture, and optimization hints
    """
    info: dict[str, Any] = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "numa_nodes": 1,  # Default
        "cpu_brand": "Unknown",
        "supports_avx": False,
        "supports_avx512": False,
        "optimization_flags": [],
    }

    try:
        # Try to get CPU brand information
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpuinfo = f.read()
                for line in cpuinfo.split("\n"):
                    if "model name" in line and info["cpu_brand"] == "Unknown":
                        info["cpu_brand"] = line.split(":")[1].strip()
                    elif "flags" in line or "Features" in line:
                        flags = line.split(":")[1].strip().split()
                        if not info["supports_avx"]:
                            info["supports_avx"] = "avx" in flags
                        if not info["supports_avx512"]:
                            info["supports_avx512"] = any("avx512" in flag for flag in flags)

        # Detect NUMA topology
        try:
            lscpu_path = shutil.which("lscpu")
            if lscpu_path:
                result = subprocess.run(  # nosec B603
                    [lscpu_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "NUMA node(s):" in line:
                            info["numa_nodes"] = int(line.split(":")[1].strip())
                            break
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.debug("NUMA detection via lscpu failed: %s", exc)

        # Set optimization recommendations
        cpu_brand = info["cpu_brand"]
        optimization_flags = info["optimization_flags"]
        if not isinstance(cpu_brand, str):
            raise TypeError(f"Expected str for cpu_brand, got {type(cpu_brand).__name__}")
        if not isinstance(optimization_flags, list):
            raise TypeError(
                f"Expected list for optimization_flags, got {type(optimization_flags).__name__}"
            )
        if "Intel" in cpu_brand:
            optimization_flags.append("intel_mkl")
        elif "AMD" in cpu_brand:
            optimization_flags.append("amd_blis")

        if info["supports_avx512"]:
            optimization_flags.append("avx512")
        elif info["supports_avx"]:
            optimization_flags.append("avx2")

    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.warning(f"Could not detect full CPU information: {e}")

    return info


def configure_cpu_hpc(
    num_threads: int | None = None,
    enable_hyperthreading: bool = False,
    numa_policy: str = "auto",
    memory_optimization: str = "standard",
    enable_onednn: bool = False,
) -> dict[str, Any]:
    """Configure JAX and system for HPC CPU optimization.

    Optimizes thread allocation, memory usage, and computational efficiency
    for HPC environments with 36/128-core nodes.

    Parameters
    ----------
    num_threads : int, optional
        Number of threads to use. If None, auto-detects optimal count.
    enable_hyperthreading : bool, default False
        Whether to use hyperthreading. Usually disabled for HPC.
    numa_policy : str, default "auto"
        NUMA memory policy ("auto", "local", "interleave")
    memory_optimization : str, default "standard"
        Memory optimization level ("minimal", "standard", "aggressive")
    enable_onednn : bool, default False
        Enable Intel oneDNN optimizations for matrix operations.
        Only recommended for Intel CPUs with matrix-heavy workloads.
        XPCS analysis is element-wise dominated, so benefit is minimal.
        Set to True to benchmark potential improvements.

    Returns
    -------
    dict
        Configuration summary and performance hints
    """
    logger.info("Configuring CPU optimization for HPC environment")

    cpu_info = detect_cpu_info()

    # Determine optimal thread count
    if num_threads is None:
        if enable_hyperthreading:
            num_threads = cpu_info["logical_cores"]
        else:
            num_threads = cpu_info["physical_cores"]

        # For HPC environments, often better to leave some cores for system
        if num_threads >= 32:
            num_threads = max(num_threads - 4, 32)  # Reserve 4 cores for system
        elif num_threads >= 16:
            num_threads = max(num_threads - 2, 16)  # Reserve 2 cores for system

    logger.info(
        f"Using {num_threads} threads on {cpu_info['physical_cores']} physical cores",
    )

    # Configure environment variables for optimal performance
    env_vars = _set_cpu_environment_variables(
        num_threads,
        cpu_info,
        numa_policy,
        memory_optimization,
    )

    # Build configuration summary with proper typing
    config_summary: dict[str, Any] = dict(env_vars)

    # Configure JAX for CPU optimization
    if JAX_AVAILABLE:
        jax_config = _configure_jax_cpu(num_threads, cpu_info, enable_onednn)
        config_summary.update(jax_config)

    config_summary.update(
        {
            "cpu_info": cpu_info,
            "threads_configured": num_threads,
            "hyperthreading_enabled": enable_hyperthreading,
            "numa_policy": numa_policy,
            "memory_optimization": memory_optimization,
            "onednn_enabled": enable_onednn,
        },
    )

    logger.info(
        f"HPC CPU configuration completed: {num_threads} threads, "
        f"{cpu_info['numa_nodes']} NUMA nodes",
    )

    return config_summary


def _set_cpu_environment_variables(
    num_threads: int,
    cpu_info: dict[str, Any],
    numa_policy: str,
    memory_optimization: str,
) -> dict[str, str]:
    """Set environment variables for optimal CPU performance."""

    env_vars = {}

    # OpenMP configuration
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if "OMP_PROC_BIND" not in os.environ:
        os.environ["OMP_PROC_BIND"] = "true"
    if "OMP_PLACES" not in os.environ:
        os.environ["OMP_PLACES"] = "cores"
    env_vars["OMP_NUM_THREADS"] = str(num_threads)

    # Intel MKL configuration (if Intel CPU)
    if "intel" in cpu_info.get("optimization_flags", []):
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_DOMAIN_NUM_THREADS"] = f"MKL_BLAS={num_threads}"
        env_vars["MKL_NUM_THREADS"] = str(num_threads)

    # BLAS configuration
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    env_vars["OPENBLAS_NUM_THREADS"] = str(num_threads)

    # Memory optimization
    if memory_optimization == "aggressive":
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
        os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"
    elif memory_optimization == "standard":
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"

    # NUMA policy
    if numa_policy == "local" and cpu_info["numa_nodes"] > 1:
        os.environ["NUMA_POLICY"] = "local"
    elif numa_policy == "interleave" and cpu_info["numa_nodes"] > 1:
        os.environ["NUMA_POLICY"] = "interleave"

    return env_vars


def _configure_jax_cpu(
    num_threads: int,
    cpu_info: dict[str, Any],
    enable_onednn: bool = False,
) -> dict[str, Any]:
    """Configure JAX for optimal CPU performance.

    Parameters
    ----------
    num_threads : int
        Number of threads to use
    cpu_info : dict
        CPU information from detect_cpu_info()
    enable_onednn : bool, default False
        Enable Intel oneDNN optimizations (experimental for XPCS workloads)

    Returns
    -------
    dict
        JAX configuration summary
    """
    jax_config: dict[str, Any] = {}

    try:
        # Force CPU platform.
        #
        # ADR: GPU acceleration deliberately not supported (2026-03).
        # Homodyne's float64 physics (epsilon_abs=1e-12, params spanning 7
        # orders of magnitude) rules out consumer GPUs: RTX 4090 delivers
        # ~1.3 TFLOPS float64 (1:64 ratio) vs ~0.5-1.0 TFLOPS on a 20-core
        # CPU -- a 1.3-2.6x advantage erased by PCIe round-trips forced by
        # the NLSQ C extension (~70ms/iteration) and CPU-only XLA flags.
        # Viable only with datacenter GPUs (A100/H100, 1:2 float64 ratio)
        # and a jaxopt-based optimizer rewrite.
        os.environ["JAX_PLATFORMS"] = "cpu"
        jax_config["platform"] = "cpu"

        # Note: x64 precision automatically enabled by nlsq import (when imported before JAX)
        # No manual jax.config.update("jax_enable_x64", True) needed
        # Reference: https://nlsq.readthedocs.io/en/latest/guides/advanced_features.html
        jax_config["x64_enabled"] = True  # Verified by nlsq import

        # Disable traceback filtering for better error debugging (NLSQ recommendation)
        jax.config.update("jax_traceback_filtering", "off")
        jax_config["traceback_filtering"] = "off"

        # Build XLA flags based on CPU capabilities and user preferences
        # NOTE: The constant_folding pass is disabled in xpcsjax/__init__.py
        # BEFORE JAX import to prevent slow_operation_alarm for large datasets.
        # Additional CPU-specific flags are set here.
        xla_flags = [
            "--xla_cpu_multi_thread_eigen=true",
        ]

        # Add AVX-512 optimizations if supported
        if cpu_info.get("supports_avx512"):
            xla_flags.extend(
                [
                    "--xla_cpu_enable_fast_math=true",
                    "--xla_cpu_fast_math_honor_nans=true",
                    "--xla_cpu_fast_math_honor_infs=true",
                    # --xla_cpu_enable_xla_runtime removed: dropped in JAX 0.8+,
                    # now causes FATAL parse error from parse_flags_from_env.cc.
                ]
            )
            jax_config["optimizations"] = "avx512_enabled"
        else:
            jax_config["optimizations"] = "standard"

        # Add oneDNN optimization if requested (experimental for XPCS)
        if enable_onednn:
            # Only enable on Intel CPUs where it's likely to help
            if "Intel" in cpu_info.get("cpu_brand", ""):
                xla_flags.append("--xla_cpu_use_onednn=true")
                jax_config["onednn"] = "enabled"
                logger.info(
                    "Intel oneDNN enabled (experimental for XPCS workloads). "
                    "Benchmark to verify performance improvements."
                )
            else:
                logger.warning("oneDNN requested but CPU is not Intel. Skipping oneDNN.")
                jax_config["onednn"] = "skipped_non_intel"
        else:
            jax_config["onednn"] = "disabled"

        # Merge CPU-specific XLA flags into existing flags (preserving
        # device_count and other flags set by xpcsjax/__init__.py).
        # CRITICAL: Do NOT overwrite — __init__.py sets
        # --xla_force_host_platform_device_count and
        # --xla_disable_hlo_passes which must be preserved.
        # H-1: XLA parses XLA_FLAGS once at backend init. If the backend is
        # already live (the common case — xpcsjax/__init__.py imports JAX at
        # import time, before this runs), writing the env var here is silently
        # ignored. Warn and skip rather than write stale flags and imply success.
        if _jax_backend_initialized():
            logger.warning(
                "JAX backend already initialized; CPU XLA flags %s cannot take "
                "effect (XLA_FLAGS is read once, at init). Set them before "
                "importing xpcsjax/JAX. Skipping env mutation.",
                xla_flags,
            )
            jax_config["xla_flags_applied"] = False
        else:
            existing_flags = os.environ.get("XLA_FLAGS", "")
            for flag in xla_flags:
                flag_name = flag.split("=")[0]
                if flag_name not in existing_flags:
                    existing_flags = existing_flags + " " + flag
            os.environ["XLA_FLAGS"] = existing_flags.strip()
            jax_config["xla_flags_applied"] = True

        # Memory optimization
        jax.config.update("jax_default_device", jax.devices("cpu")[0])

        logger.info("JAX CPU configuration completed successfully")

    except (OSError, ValueError, RuntimeError, AttributeError) as e:
        logger.warning(f"JAX CPU configuration failed: {e}")
        jax_config["error"] = str(e)

    return jax_config


def configure_cpu_threading(num_threads: int | None = None) -> dict[str, Any]:
    """Configure CPU threading for NLSQ optimization.

    Performance Optimization (Spec 001 - FR-005, T024): Simplified threading
    configuration for NLSQ initialization. Calls configure_cpu_hpc() with
    sensible defaults for optimization workloads.

    Parameters
    ----------
    num_threads : int, optional
        Number of threads to use. If None, auto-detects optimal count
        based on physical cores.

    Returns
    -------
    dict
        Configuration summary including thread count and XLA settings.
    """
    return configure_cpu_hpc(
        num_threads=num_threads,
        enable_hyperthreading=False,  # Physical cores only for HPC
        numa_policy="auto",
        memory_optimization="standard",
        enable_onednn=False,  # Disabled by default for XPCS workloads
    )


def get_optimal_batch_size(
    data_size: int,
    available_memory_gb: float | None = None,
    target_memory_usage: float = 0.7,
) -> int:
    """Calculate optimal batch size for CPU processing.

    Parameters
    ----------
    data_size : int
        Total size of data to process
    available_memory_gb : float, optional
        Available memory in GB. If None, auto-detects.
    target_memory_usage : float, default 0.7
        Target fraction of memory to use

    Returns
    -------
    int
        Optimal batch size for processing
    """
    if available_memory_gb is None:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Estimate memory usage per data point (rough heuristic)
    memory_per_point_mb = 0.1  # 100 KB per data point (conservative estimate)
    total_memory_mb = available_memory_gb * 1024

    # Calculate batch size that uses target fraction of memory
    optimal_batch_size = int(
        (total_memory_mb * target_memory_usage) / memory_per_point_mb,
    )

    # Ensure batch size is reasonable
    optimal_batch_size = max(min(optimal_batch_size, data_size), 1000)

    logger.info(
        f"Optimal batch size: {optimal_batch_size} (memory: {available_memory_gb:.1f}GB)",
    )

    return optimal_batch_size


def benchmark_cpu_performance(
    test_size: int = 10000,
    num_iterations: int = 5,
) -> dict[str, float]:
    """Benchmark CPU performance for optimization planning.

    Parameters
    ----------
    test_size : int, default 10000
        Size of test computation
    num_iterations : int, default 5
        Number of benchmark iterations

    Returns
    -------
    dict
        Benchmark results with timing information
    """
    logger.info(f"Running CPU benchmark with {test_size} data points")

    import time

    import numpy as np

    results: dict[str, Any] = {"numpy_performance": [], "cpu_info": detect_cpu_info()}

    # NumPy benchmark
    for _i in range(num_iterations):
        start_time = time.perf_counter()

        # Simulate typical XPCS computation
        x = np.random.randn(test_size, test_size)
        y = np.fft.fft2(x)
        z = np.abs(y) ** 2
        _ = np.sum(z)

        end_time = time.perf_counter()
        results["numpy_performance"].append(end_time - start_time)

    # JAX benchmark (if available)
    if JAX_AVAILABLE:
        results["jax_performance"] = []

        @jax.jit
        def jax_computation(x: Any) -> Any:
            y = jnp.fft.fft2(x)
            z = jnp.abs(y) ** 2
            return jnp.sum(z)

        # Warm up JIT
        test_array = jnp.array(np.random.randn(100, 100))
        _ = jax_computation(test_array)

        for _i in range(num_iterations):
            start_time = time.perf_counter()

            x_jax = jnp.array(np.random.randn(test_size, test_size))
            result = jax_computation(x_jax)
            result.block_until_ready()  # Ensure computation completes

            end_time = time.perf_counter()
            results["jax_performance"].append(end_time - start_time)

    # Calculate statistics
    results["numpy_mean_time"] = np.mean(results["numpy_performance"])
    results["numpy_std_time"] = np.std(results["numpy_performance"])

    if JAX_AVAILABLE:
        results["jax_mean_time"] = np.mean(results["jax_performance"])
        results["jax_std_time"] = np.std(results["jax_performance"])
        results["jax_speedup"] = results["numpy_mean_time"] / results["jax_mean_time"]

    logger.info(f"Benchmark completed. NumPy: {results['numpy_mean_time']:.3f}s avg")
    if JAX_AVAILABLE:
        logger.info(
            f"JAX: {results['jax_mean_time']:.3f}s avg "
            f"(speedup: {results.get('jax_speedup', 0):.2f}x)",
        )

    return results
