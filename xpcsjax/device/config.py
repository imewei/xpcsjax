"""Hardware detection and configuration helpers for xpcsjax NLSQ optimization.

Detects hardware capabilities (CPU core count, available memory, JAX availability)
and exposes them via :class:`HardwareConfig`. Used for NLSQ thread and memory
budget decisions.
"""

import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import Literal

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from xpcsjax.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware configuration for NLSQ optimization.

    This dataclass encapsulates all detected hardware information needed
    for NLSQ thread and memory budget decisions.

    Attributes
    ----------
    platform : {'cpu'}
        Primary compute platform (CPU-only)
    num_devices : int
        Number of available CPU devices
    memory_per_device_gb : float
        Available system memory in GB
    num_nodes : int
        Number of cluster nodes (1 for standalone)
    cores_per_node : int
        Number of physical CPU cores per node
    total_memory_gb : float
        Total system memory in GB
    cluster_type : {'pbs', 'slurm', 'standalone', None}
        Detected cluster scheduler type
    recommended_backend : str
        Recommended execution backend based on hardware
        Options: 'pjit', 'multiprocessing', 'pbs', 'slurm'
    max_parallel_shards : int
        Maximum number of shards that can run in parallel
        - Multi-node cluster: num_nodes * cores_per_node
        - CPU: cores_per_node

    Examples
    --------
    >>> hw = detect_hardware()
    >>> print(hw.platform)
    'cpu'
    >>> print(hw.max_parallel_shards)
    4
    >>> print(hw.recommended_backend)
    'multiprocessing'
    """

    platform: Literal["cpu"]
    num_devices: int
    memory_per_device_gb: float
    num_nodes: int
    cores_per_node: int
    total_memory_gb: float
    cluster_type: Literal["pbs", "slurm", "standalone"] | None
    recommended_backend: str
    max_parallel_shards: int


def detect_hardware() -> HardwareConfig:
    """Auto-detect hardware configuration for NLSQ optimization.

    This function performs comprehensive hardware detection to inform
    NLSQ thread and memory budget decisions.

    Detection Logic
    ---------------
    1. **JAX Devices**: Query JAX for CPU devices (CPU-only)
    2. **System Memory**: Query total system memory via psutil
       - Fallback: Assume 32GB if psutil unavailable
    3. **Cluster Environment**: Check environment variables
       - PBS: PBS_JOBID, PBS_NODEFILE
       - Slurm: SLURM_JOB_NUM_NODES, SLURM_CPUS_ON_NODE
       - Standalone: Neither PBS nor Slurm detected
    4. **CPU Resources**: Count physical cores using psutil
    5. **Backend Recommendation**: Select optimal backend based on:
       - Multi-node cluster → PBS/Slurm backend
       - CPU standalone → multiprocessing backend

    Returns
    -------
    HardwareConfig
        Comprehensive hardware configuration for NLSQ

    Examples
    --------
    >>> hw = detect_hardware()
    >>> print(hw.platform)
    'cpu'
    >>> print(hw.num_devices)
    4
    >>> print(hw.memory_per_device_gb)
    64.0
    >>> print(hw.cluster_type)
    'pbs'
    >>> print(hw.recommended_backend)
    'pbs'

    Notes
    -----
    - Detection is robust with multiple fallback mechanisms
    - Cluster detection requires environment variables set by scheduler
    - CPU core count excludes hyperthreading for accurate parallelism
    - CPU-only; JAX will always report platform='cpu'
    """
    logger.info("Detecting hardware configuration...")

    # Step 1: Detect JAX devices
    # Use the actual active backend, not just first device in list
    # When JAX_PLATFORMS="cpu,gpu", devices[0] may be CPU even if GPU is active
    try:
        # Try new API first (newer JAX), fall back to legacy API
        try:
            from jax.extend import backend as jax_backend

            backend = jax_backend.get_backend()
        except (ImportError, AttributeError):
            # Legacy API for older JAX
            import importlib

            xla_bridge = importlib.import_module("jax.lib.xla_bridge")
            backend = xla_bridge.get_backend()

        platform = backend.platform
        devices = backend.devices()
        num_devices = len(devices)
        logger.info(f"JAX devices detected: {num_devices} {platform} device(s)")
    except Exception as e:
        log_exception(
            logger,
            e,
            context={
                "operation": "jax_device_detection",
                "fallback": "cpu",
            },
            level=logging.WARNING,
        )
        platform = "cpu"
        num_devices = 1

    # Step 2: Query system memory (CPU-only)
    if HAS_PSUTIL:
        memory_gb = psutil.virtual_memory().total / 1e9
        logger.info(f"System memory detected: {memory_gb:.2f} GB")
    else:
        logger.warning("psutil not available. Assuming 32 GB system memory")
        memory_gb = 32.0

    # Step 3: Detect cluster environment
    cluster_type: Literal["pbs", "slurm", "standalone"] | None = None
    num_nodes = 1

    if "PBS_JOBID" in os.environ:
        cluster_type = "pbs"
        # Parse PBS_NODEFILE for node count
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile, encoding="utf-8") as f:
                    # Strip whitespace and skip blank lines before deduplication;
                    # PBS nodefiles often contain a trailing newline or blank lines.
                    num_nodes = len(
                        {line.strip() for line in f.read().splitlines() if line.strip()}
                    )
                logger.info(f"PBS cluster detected: {num_nodes} nodes")
            except Exception as e:
                log_exception(
                    logger,
                    e,
                    context={
                        "operation": "parse_pbs_nodefile",
                        "nodefile": nodefile,
                        "fallback_num_nodes": 1,
                    },
                    level=logging.WARNING,
                )
                num_nodes = 1
        else:
            logger.debug("PBS_JOBID present but PBS_NODEFILE not found")
            num_nodes = 1

    elif "SLURM_JOB_NUM_NODES" in os.environ:
        cluster_type = "slurm"
        try:
            num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
            logger.info(f"Slurm cluster detected: {num_nodes} nodes")
        except ValueError:
            logger.warning("Failed to parse SLURM_JOB_NUM_NODES")
            num_nodes = 1

    else:
        cluster_type = "standalone"
        num_nodes = 1
        logger.info("Standalone system detected (no cluster scheduler)")

    # Step 4: Detect CPU cores
    if HAS_PSUTIL:
        # Use physical cores (exclude hyperthreading)
        cores_per_node = psutil.cpu_count(logical=False) or 1
        total_memory_gb = psutil.virtual_memory().total / 1e9
        logger.info(f"CPU cores detected: {cores_per_node} physical cores")
    else:
        logger.warning("psutil not available. Using multiprocessing for CPU count")
        cores_per_node = multiprocessing.cpu_count()
        total_memory_gb = memory_gb  # Use previously detected value

    # Step 5: Recommend backend and calculate max parallel shards (CPU-only)
    recommended_backend: str
    if cluster_type in ["pbs", "slurm"] and num_nodes > 1:
        # Multi-node cluster: Use PBS/Slurm backend
        recommended_backend = cluster_type
        max_parallel_shards = num_nodes * cores_per_node
        logger.info(
            f"Recommended backend: {recommended_backend} "
            f"(max {max_parallel_shards} parallel shards)"
        )
    else:
        # CPU standalone: Use multiprocessing backend
        recommended_backend = "multiprocessing"
        max_parallel_shards = cores_per_node
        logger.info(
            f"Recommended backend: multiprocessing (max {max_parallel_shards} parallel shards)"
        )

    # Construct and return HardwareConfig
    hw_config = HardwareConfig(
        platform=platform,
        num_devices=num_devices,
        memory_per_device_gb=memory_gb,
        num_nodes=num_nodes,
        cores_per_node=cores_per_node,
        total_memory_gb=total_memory_gb,
        cluster_type=cluster_type,
        recommended_backend=recommended_backend,
        max_parallel_shards=max_parallel_shards,
    )

    logger.info(f"Hardware detection complete: {hw_config.platform} platform")
    return hw_config


# Export public API
__all__ = [
    "HardwareConfig",
    "detect_hardware",
]
