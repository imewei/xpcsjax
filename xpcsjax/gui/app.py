"""GUI entry point — wires the workbench and runs the Qt event loop."""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpcsjax.gui.controllers.fit_queue import FitQueueController
    from xpcsjax.gui.views.main_window import MainWindow


def build_workbench() -> tuple[MainWindow, FitQueueController]:
    """Construct the controller-less MainWindow (which owns its own Project + FitQueueController).

    Returns ``(window, window._queue)`` — the queue is the single execution path.
    Worker cleanup on exit is the caller's job: ``main`` registers the ``atexit``
    hook and ``MainWindow.closeEvent`` also calls ``queue.shutdown``. Registration
    is kept OUT of here so repeated construction (e.g. in tests) cannot accumulate
    stale atexit hooks.
    Concrete types (``MainWindow`` / ``FitQueueController``) are annotated under
    ``TYPE_CHECKING`` only, so the module stays import-light at runtime.
    """
    from xpcsjax.gui.views.main_window import MainWindow

    window = MainWindow()
    return window, window._queue


def _resolve_version() -> str:
    """Best-effort version string, mirroring ``cli.args_parser._add_version_arg``."""
    try:
        import importlib.metadata as _md

        return _md.version("xpcsjax")
    except Exception:  # pragma: no cover — uninstalled / dev tree
        try:
            from xpcsjax import __version__ as version

            return version
        except Exception:
            return "unknown"


def _parse_cli_args(argv: list[str]) -> list[str]:
    """Handle ``xpcsjax-gui``'s own flags; return the leftover args for Qt.

    Recognises ``--help`` / ``--version`` (consistent with the other xpcsjax
    console scripts) and forwards everything else — e.g. ``-platform offscreen``
    — to Qt. ``--help`` / ``--version`` raise ``SystemExit`` via argparse, which
    is the correct console-script behaviour.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="xpcsjax-gui",
        description="Launch the xpcsjax analysis workbench (PySide6 GUI).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_resolve_version()}",
    )
    _, qt_extra = parser.parse_known_args(argv)
    return qt_extra


def main(argv: list[str] | None = None) -> int:
    """Launch the workbench. Returns the Qt event-loop exit code.

    Recognises ``--help`` / ``--version`` like the sibling console scripts; any
    other arguments pass through to Qt (e.g. ``xpcsjax-gui -platform offscreen``
    for a headless smoke run).
    """
    import multiprocessing

    multiprocessing.freeze_support()  # frozen-app spawn safety — must be first

    import sys

    raw = list(sys.argv[1:] if argv is None else argv)
    qt_extra = _parse_cli_args(raw)  # may SystemExit on --help / --version

    from PySide6.QtWidgets import QApplication

    from xpcsjax.gui import theme

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_extra])
    # Apply the system-aware "instrument console" theme before any window is built
    # so every widget is born styled (no first-paint flash of unstyled defaults).
    theme.apply_theme(app)
    window, queue = build_workbench()
    # Registered here (once per process), not in build_workbench, so a hard exit
    # still terminates a running worker without accumulating hooks across tests.
    atexit.register(queue.shutdown)
    window.show()
    return int(app.exec())
