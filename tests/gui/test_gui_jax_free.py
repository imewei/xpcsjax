"""The GUI-importable surface must never pull JAX into the process.

# Probes for ipc.worker (Task 3) and ipc.handle (Task 4) are added when those modules land.
"""

import subprocess
import sys
import textwrap

import pytest


def _probe_import(module: str) -> int:
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode


def test_app_module_is_jax_free_at_import():
    """xpcsjax.gui.app must not import JAX at module level (deferred inside main())."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.app") == 0
