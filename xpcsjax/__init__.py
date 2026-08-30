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

# ============================================================================
# JAX CPU Device Configuration (MUST be set before JAX import)
# ============================================================================
# Mirrored from homodyne/__init__.py, with one xpcsjax-specific adaptation
# (concurrency gating, see _xla_host_device_count):
#   - xla_force_host_platform_device_count: enables parallel evaluation paths
#   - xla_disable_hlo_passes=constant_folding: prevents > 1 s slow-compilation
#     warnings on HYBRID_STREAMING strategy (23M+ points) where data arrays
#     are captured in JIT closures. Performance impact: minimal (< 5 ms/call).


def _detect_worker_count() -> int:
    """Concurrent fit-process count from the pool / pytest-xdist env-vars (>=1).

    Inlined here (not imported from ``optimization.nlsq.memory``) on purpose:
    this runs *before* the first JAX import, and importing that module could pull
    JAX in early and defeat the env-before-import ordering this block exists for.
    """
    for _env_var in ("XPCSJAX_FIT_CONCURRENCY", "PYTEST_XDIST_WORKER_COUNT"):
        _raw = os.environ.get(_env_var)
        if _raw:
            try:
                return max(1, int(_raw))
            except ValueError:
                pass
    return 1


def _xla_host_device_count(worker_count: int) -> int:
    """Return the number of host CPU devices to force for XLA.

    A lone fit benefits from 4 host devices (parallel evaluation paths). But
    under parallelism (pytest-xdist or the production multistart / accumulator
    pools) each of N worker processes forcing 4 devices means 4*N logical devices
    contending for the physical cores — wasted per-worker compile/buffer overhead
    and RAM. So drop to a single device per process whenever more than one fit
    runs concurrently.
    """
    return 4 if worker_count <= 1 else 1


_WORKER_COUNT = _detect_worker_count()
_DEFAULT_XLA_FLAGS = [
    f"--xla_force_host_platform_device_count={_xla_host_device_count(_WORKER_COUNT)}",
    "--xla_disable_hlo_passes=constant_folding",
    # Always-applicable (no CPU-model detection needed), duplicated here so
    # it actually takes effect: device.cpu.configure_cpu_hpc() also builds
    # this same flag (plus CPU-model-dependent extras -- AVX-512 fast-math,
    # oneDNN -- that DO need a runtime probe and so can't move here), but its
    # call site runs at fit time, after xpcsjax/JAX is already imported --
    # always too late for XLA_FLAGS to take effect. Pre-setting it here means
    # device.cpu's copy is a redundant no-op by the time it runs (it finds
    # this flag already present and skips both the write and the warning).
    "--xla_cpu_multi_thread_eigen=true",
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

# Pin JAX to CPU (CPU-only; no GPU support). Setting this at
# package import time (before any jax import) is the *only* place this works
# reliably — spawn-pool worker init runs *after* the worker's `import jax`,
# which is why xpcsjax.viz.nlsq_plots.{_worker_init_cpu_only,_render_one_angle_worker}
# can't set this themselves. Child processes inherit os.environ from the parent.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# Cap each worker's JAX arena under parallelism so one fit process can't claim
# the whole device. No-op on the CPU backend (the v0.1 target), but a correct
# per-worker bound for any future GPU backend; setdefault leaves an explicit
# user/operator override untouched.
if _WORKER_COUNT > 1:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{max(0.05, 0.9 / _WORKER_COUNT):.4f}")

# Suppress NLSQ GPU warnings (CPU-only; no GPU support)
os.environ.setdefault("NLSQ_SKIP_GPU_CHECK", "1")

# Suppress JAX backend logs (GPU fallback warnings on CPU-only systems)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

# ============================================================================
# Version
# ============================================================================
__version__ = "0.1.6"

# ============================================================================
# Lazy public API
# ============================================================================
_LAZY_EXPORTS = {
    "load_xpcs_data": "xpcsjax.data",
    "fit_nlsq": "xpcsjax.optimization.nlsq",
    "ConfigManager": "xpcsjax.config",
    "generate_nlsq_plots": "xpcsjax.viz",
    "HomodyneModel": "xpcsjax.core",
    "HeterodyneModel": "xpcsjax.core",
    "OptimizationResult": "xpcsjax.optimization.nlsq.results",
}

# TYPE_CHECKING block for IDE / Pyright static visibility. All submodules
# below now export their public symbol, so the original deferral comment
# (Tasks 6/11/15/19/20/28) is resolved.
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from xpcsjax.config import ConfigManager
    from xpcsjax.core import HeterodyneModel, HomodyneModel
    from xpcsjax.data import load_xpcs_data
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz import generate_nlsq_plots


def __getattr__(name: str):  # noqa: D401
    """Lazy attribute loader for the documented public API."""
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'xpcsjax' has no attribute {name!r}")


# Literal __all__ for Pyright's reportUnsupportedDunderAll; kept in sync
# with _LAZY_EXPORTS by the runtime assertion below.
__all__ = [
    "load_xpcs_data",
    "fit_nlsq",
    "ConfigManager",
    "generate_nlsq_plots",
    "HomodyneModel",
    "HeterodyneModel",
    "OptimizationResult",
]

assert set(__all__) == set(_LAZY_EXPORTS), (
    "xpcsjax public API mismatch between __all__ and _LAZY_EXPORTS"
)
