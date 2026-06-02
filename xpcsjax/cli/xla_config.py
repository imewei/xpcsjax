"""XLA configuration helper for JAX on CPU.

Two distinct use cases:

1.  **Library import (effective)** — external scripts can call
    ``from xpcsjax.cli.xla_config import configure_xla`` *before* their
    first ``import xpcsjax`` (or any other JAX consumer) to override the
    defaults that ``xpcsjax/__init__.py`` would otherwise set. JAX reads
    ``XLA_FLAGS`` exactly once at backend init, so ordering matters.

2.  **CLI entry (informational)** — ``xpcsjax-config-xla`` only reports
    settings. It does NOT mutate the running process's JAX backend
    because by the time the entry-point script runs,
    ``xpcsjax/__init__.py`` has already been imported and JAX backend
    init has fired. For persistent CLI-level overrides use
    ``xpcsjax-post-install --xla-mode {auto|nlsq|<N>}`` instead — that
    writes ``$VIRTUAL_ENV/etc/xpcsjax/xla_mode`` which the activation
    script reads on every shell entry.
"""

from __future__ import annotations

import argparse
import os


def configure_xla(
    num_threads: int | None = None,
    disable_jit: bool = False,
    enable_x64: bool = True,
) -> dict[str, str]:
    """Set XLA / JAX env vars for CPU execution.

    Args:
        num_threads: CPU thread count for XLA (``None`` leaves unset).
        disable_jit: If True, set ``JAX_DISABLE_JIT=1`` (debugging).
        enable_x64: If True, set ``JAX_ENABLE_X64=1`` (required for
            xpcsjax — parameters span 6+ orders of magnitude).

    Returns:
        Mapping of env vars that were set.
    """
    env_vars: dict[str, str] = {}

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    env_vars["JAX_PLATFORM_NAME"] = "cpu"

    if num_threads is not None:
        existing = os.environ.get("XLA_FLAGS", "")
        new_flags = (
            "--xla_cpu_multi_thread_eigen=true"
            f" --intra_op_parallelism_threads={num_threads}"
        )
        if new_flags not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {new_flags}".strip()
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        env_vars["XLA_FLAGS"] = os.environ["XLA_FLAGS"]
        env_vars["OMP_NUM_THREADS"] = str(num_threads)
        env_vars["MKL_NUM_THREADS"] = str(num_threads)

    if disable_jit:
        os.environ["JAX_DISABLE_JIT"] = "1"
        env_vars["JAX_DISABLE_JIT"] = "1"

    if enable_x64:
        os.environ["JAX_ENABLE_X64"] = "1"
        env_vars["JAX_ENABLE_X64"] = "1"

    return env_vars


def get_cpu_info() -> dict[str, int | float | str]:
    """Return a compact dict of CPU / RAM metrics (uses psutil)."""
    import psutil

    info: dict[str, int | float | str] = {
        "cpu_count": psutil.cpu_count() or 1,
        "physical_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
    }
    mem = psutil.virtual_memory()
    info["available_memory_gb"] = round(mem.available / (1024**3), 1)
    info["total_memory_gb"] = round(mem.total / (1024**3), 1)
    return info


def main() -> None:
    """``xpcsjax-config-xla`` / ``xj-config-xla`` entry point.

    This command is **informational only**. By the time it runs, the
    parent shim has already executed ``import xpcsjax`` (which triggered
    JAX backend init), so any env-var mutation we perform here cannot
    take effect on the current process. The command prints what WOULD be
    set; to persist a configuration for future shells, use
    ``xpcsjax-post-install --xla-mode``.
    """
    parser = argparse.ArgumentParser(
        prog="xpcsjax-config-xla",
        description=(
            "Show the XLA / JAX configuration xpcsjax would apply. "
            "Informational only — use xpcsjax-post-install --xla-mode to "
            "persist a setting that future shells will pick up."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU thread count to preview (default: auto-detect physical cores).",
    )
    parser.add_argument(
        "--no-x64",
        action="store_true",
        help="Preview without 64-bit precision (NOT recommended — xpcsjax requires float64).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Preview with JIT disabled (debugging only).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print CPU/RAM info and exit.",
    )

    args = parser.parse_args()

    if args.info:
        info = get_cpu_info()
        print("CPU Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return

    # Build the env-var preview WITHOUT mutating os.environ — calling
    # configure_xla() here would write env vars JAX has already read.
    threads = args.threads
    if threads is None:
        physical = get_cpu_info().get("physical_cores", 4)
        threads = int(physical) if isinstance(physical, int) else 4

    preview: dict[str, str] = {"JAX_PLATFORM_NAME": "cpu"}
    preview["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=true"
        f" --intra_op_parallelism_threads={threads}"
    )
    preview["OMP_NUM_THREADS"] = str(threads)
    preview["MKL_NUM_THREADS"] = str(threads)
    if args.debug:
        preview["JAX_DISABLE_JIT"] = "1"
    if not args.no_x64:
        preview["JAX_ENABLE_X64"] = "1"

    print("XLA Configuration (preview — not applied to this process):")
    for key, value in preview.items():
        print(f"  {key}={value}")
    print("")
    print("To make a setting persistent for future shells, run:")
    print("  xpcsjax-post-install --xla-mode auto       # or 'nlsq' / <N>")


if __name__ == "__main__":
    main()
