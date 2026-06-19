"""The GUI-importable surface must never pull JAX into the process.

# Task 3 probe: ipc.worker (spawn target)
# Task 4 probe: ipc.handle (parent)
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


def test_worker_module_is_jax_free_at_import():
    """xpcsjax.gui.ipc.worker must not import JAX at module level (deferred inside run_worker())."""
    assert _probe_import("xpcsjax.gui.ipc.worker") == 0


def test_handle_module_is_jax_free_at_import():
    """xpcsjax.gui.ipc.handle must not import JAX at module level (GUI process stays JAX-free)."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.ipc.handle") == 0


def test_fit_controller_is_jax_free_at_import():
    """xpcsjax.gui.controllers.fit_controller must not import JAX at module level."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.controllers.fit_controller") == 0


def test_main_window_is_jax_free_at_import():
    """xpcsjax.gui.views.main_window must not import JAX at module level."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.views.main_window") == 0


def test_diagnostics_module_is_jax_free():
    # stdlib + event schema only — no Qt / pyqtgraph needed.
    assert _probe_import("xpcsjax.gui.ipc.diagnostics") == 0


def test_diagnostics_panel_is_jax_free():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    assert _probe_import("xpcsjax.gui.views.diagnostics_panel") == 0


def test_project_modules_are_jax_free():
    """xpcsjax.gui.project.model must not import JAX (stdlib-only)."""
    assert _probe_import("xpcsjax.gui.project.model") == 0


def test_project_tree_model_is_jax_free():
    """xpcsjax.gui.project.tree_model must not import JAX."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.project.tree_model") == 0


def test_fit_queue_controller_is_jax_free():
    """xpcsjax.gui.controllers.fit_queue must not import JAX."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.controllers.fit_queue") == 0


def test_project_panel_is_jax_free():
    """xpcsjax.gui.views.project_panel must not import JAX."""
    pytest.importorskip("PySide6")
    assert _probe_import("xpcsjax.gui.views.project_panel") == 0
