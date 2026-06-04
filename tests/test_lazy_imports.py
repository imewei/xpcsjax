"""Verify top-level imports are lazy and that homodyne's env setup is mirrored."""

import subprocess
import sys
import textwrap


def test_top_level_import_does_not_load_jax():
    """Importing xpcsjax must not eagerly load jax — CLI arg parsing stays instant.

    Run in a subprocess: monkey-patching `sys.modules` to remove jax in-process
    corrupts XLA bootstrapping state and SIGABRTs the next test that imports jax.
    """
    code = textwrap.dedent(
        """
        import sys
        import xpcsjax  # noqa: F401
        sys.stdout.write("jax" if "jax" in sys.modules else "no-jax")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "no-jax", (
        f"jax loaded during `import xpcsjax` — lazy-import broken (stdout={result.stdout!r})"
    )


def test_public_exports_phase4():
    """v0.1 public API symbols importable as of Phase 4 (Task 20).

    `HeterodyneModel` is a public lazy export as of Phase 6 (asserted in
    ``test_heterodyne_model_exported`` below). Keep these in sync as new
    symbols land."""
    import xpcsjax

    for name in (
        "load_xpcs_data",
        "fit_nlsq",
        "ConfigManager",
        "HomodyneModel",
        "OptimizationResult",
    ):
        assert hasattr(xpcsjax, name), f"missing public export: {name}"


def test_heterodyne_model_exported():
    """HeterodyneModel is a public lazy export as of Phase 6 (Task 27 + Task 28)."""
    import xpcsjax

    assert hasattr(xpcsjax, "HeterodyneModel"), "missing public export: HeterodyneModel"


def test_env_setup_mirrors_homodyne():
    """`import xpcsjax` must set the env vars homodyne sets at import time."""
    code = textwrap.dedent("""
        import os
        for var in ("JAX_ENABLE_X64", "XLA_FLAGS", "NLSQ_SKIP_GPU_CHECK"):
            os.environ.pop(var, None)
        import xpcsjax  # noqa: F401
        import json, sys
        sys.stdout.write(json.dumps({
            "JAX_ENABLE_X64":      os.environ.get("JAX_ENABLE_X64"),
            "XLA_FLAGS":           os.environ.get("XLA_FLAGS", ""),
            "NLSQ_SKIP_GPU_CHECK": os.environ.get("NLSQ_SKIP_GPU_CHECK"),
        }))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    env = json.loads(result.stdout)

    assert env["JAX_ENABLE_X64"] == "1", (
        f"JAX_ENABLE_X64 not set by xpcsjax import (got {env['JAX_ENABLE_X64']!r})"
    )
    assert "--xla_disable_hlo_passes=constant_folding" in env["XLA_FLAGS"], (
        f"XLA constant-folding skip not set (got {env['XLA_FLAGS']!r})"
    )
    assert "--xla_force_host_platform_device_count=4" in env["XLA_FLAGS"], (
        f"XLA device count not set (got {env['XLA_FLAGS']!r})"
    )
    assert env["NLSQ_SKIP_GPU_CHECK"] == "1", (
        f"NLSQ_SKIP_GPU_CHECK not set (got {env['NLSQ_SKIP_GPU_CHECK']!r})"
    )
