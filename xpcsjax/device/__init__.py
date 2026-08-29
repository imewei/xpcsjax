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

# Import CPU-specific module (re-exported for public API)
try:
    from xpcsjax.device.cpu import (  # noqa: F401
        configure_cpu_hpc,
        detect_cpu_info,
    )

    HAS_CPU_MODULE = True
except ImportError as e:
    logger.warning(f"CPU optimization module not available: {e}")
    HAS_CPU_MODULE = False


# Main exports. Literal __all__ + conditional ``+=`` so Pyright can analyze it
# (reportUnsupportedDunderAll); assigning a dynamically-built variable was not
# statically supported. The CPU symbols are statically importable (the
# try-import above), so they pass the dunder-all presence check.
__all__ = [
    # Status flags
    "HAS_CPU_MODULE",
]

# Add CPU-specific exports if available
if HAS_CPU_MODULE:
    __all__ += [
        "configure_cpu_hpc",
        "detect_cpu_info",
    ]
