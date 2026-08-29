"""HPC CPU device optimization with intelligent configuration.

Provides CPU-only device detection, configuration, and optimization
for high-performance computing environments.

GPU support removed - CPU-only optimization focus.

Key Features:
- HPC CPU optimization for 36/128-core nodes
- NUMA-aware configuration
- Multi-core thread allocation strategies

Usage:
    from xpcsjax.device.cpu import configure_cpu_threading
    config = configure_cpu_threading()
"""

from __future__ import annotations

import logging

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

# Re-exported for public API. `xpcsjax.device.cpu` is an internal sibling
# module (not an optional external package) and its own hard dependency,
# psutil, is a required (non-extra) install — this import cannot fail in any
# supported install, so no soft-fail guard is needed.
from xpcsjax.device.cpu import configure_cpu_hpc, detect_cpu_info  # noqa: E402,F401

__all__ = [
    "configure_cpu_hpc",
    "detect_cpu_info",
]
