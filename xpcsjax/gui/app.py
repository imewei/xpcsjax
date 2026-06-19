"""GUI entry point — wires the workbench and runs the Qt event loop."""

from __future__ import annotations

import atexit


def build_workbench() -> tuple[object, object]:
    """Construct the controller-less MainWindow (which owns its own Project + FitQueueController).

    Returns ``(window, window._queue)`` — the queue is the single execution path.
    Worker cleanup on exit is the caller's job: ``main`` registers the ``atexit``
    hook and ``MainWindow.closeEvent`` also calls ``queue.shutdown``. Registration
    is kept OUT of here so repeated construction (e.g. in tests) cannot accumulate
    stale atexit hooks.
    Return type is ``object`` to keep this module import-light; concrete types are
    ``MainWindow`` / ``FitQueueController``.
    """
    from xpcsjax.gui.views.main_window import MainWindow

    window = MainWindow()
    return window, window._queue


def main(argv: list[str] | None = None) -> int:
    """Launch the workbench. Returns the Qt event-loop exit code."""
    import multiprocessing

    multiprocessing.freeze_support()  # frozen-app spawn safety — must be first
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(argv or [])
    window, queue = build_workbench()
    # Registered here (once per process), not in build_workbench, so a hard exit
    # still terminates a running worker without accumulating hooks across tests.
    atexit.register(queue.shutdown)
    window.show()
    return int(app.exec())
