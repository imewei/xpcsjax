"""xpcsjax — unified JAX-native XPCS NLSQ fitting.

Public API (lazy-loaded — heavy deps like JAX import on first use):

    from xpcsjax import load_xpcs_data, fit_nlsq, ConfigManager

    data = load_xpcs_data("config.yaml")
    result = fit_nlsq(data, "config.yaml")
    print(result.parameters)
    result.save("output/")

Env setup at import time is mirrored verbatim from homodyne/__init__.py.
"""
from __future__ import annotations

# ============================================================================
# Standard library imports
# ============================================================================
import importlib
import logging
import os
from typing import TYPE_CHECKING

# ============================================================================
# JAX CPU Device Configuration (MUST be set before JAX import)
# ============================================================================
# Mirrored verbatim from homodyne/__init__.py:
#   - xla_force_host_platform_device_count=4: enables parallel evaluation paths
#   - xla_disable_hlo_passes=constant_folding: prevents > 1 s slow-compilation
#     warnings on HYBRID_STREAMING strategy (23M+ points) where data arrays
#     are captured in JIT closures. Performance impact: minimal (< 5 ms/call).
_DEFAULT_XLA_FLAGS = [
    "--xla_force_host_platform_device_count=4",
    "--xla_disable_hlo_passes=constant_folding",
]

# JAX must be in float64 for parameters spanning 6+ orders of magnitude.
# This env var MUST be set BEFORE the first JAX import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = " ".join(_DEFAULT_XLA_FLAGS)
else:
    existing = os.environ["XLA_FLAGS"]
    flags_to_add = []
    for flag in _DEFAULT_XLA_FLAGS:
        flag_name = flag.split("=")[0]
        if flag_name not in existing:
            flags_to_add.append(flag)
    if flags_to_add:
        os.environ["XLA_FLAGS"] += " " + " ".join(flags_to_add)

# Suppress NLSQ GPU warnings (v0.1 CPU-only; GPU support is v0.2+)
os.environ.setdefault("NLSQ_SKIP_GPU_CHECK", "1")

# Suppress JAX backend logs (GPU fallback warnings on CPU-only systems)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

# ============================================================================
# Version
# ============================================================================
__version__ = "0.1.0.dev0"

# ============================================================================
# Lazy public API
# ============================================================================
_LAZY_EXPORTS = {
    "load_xpcs_data": "xpcsjax.data",
    "fit_nlsq": "xpcsjax.optimization.nlsq",
    "ConfigManager": "xpcsjax.config",
    "HomodyneModel": "xpcsjax.core",
    "HeterodyneModel": "xpcsjax.core",
    "OptimizationResult": "xpcsjax.io",
}

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager  # noqa: F401
    from xpcsjax.core import HeterodyneModel, HomodyneModel  # noqa: F401
    from xpcsjax.data import load_xpcs_data  # noqa: F401
    from xpcsjax.io import OptimizationResult  # noqa: F401
    from xpcsjax.optimization.nlsq import fit_nlsq  # noqa: F401


def __getattr__(name: str):  # noqa: D401
    """Lazy attribute loader for the documented public API."""
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'xpcsjax' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
