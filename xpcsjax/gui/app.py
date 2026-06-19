"""GUI entry point — wires the workbench and runs the Qt event loop."""

from __future__ import annotations

import atexit


def build_workbench() -> tuple[object, object]:
    """Construct + wire the MainWindow and FitController (no event loop).

    Returns the ``(window, controller)`` pair. Worker cleanup on exit is the
    caller's job: ``main`` registers the ``atexit`` hook and ``MainWindow.closeEvent``
    also calls ``controller.shutdown``. Registration is kept OUT of here so
    repeated construction (e.g. in tests) cannot accumulate stale atexit hooks.
    Return type is ``object`` to keep this module import-light; concrete types
    are ``MainWindow`` / ``FitController``.
    """
    from xpcsjax.gui.controllers.fit_controller import FitController
    from xpcsjax.gui.views.main_window import MainWindow

    controller = FitController()
    window = MainWindow(controller)
    return window, controller


def main(argv: list[str] | None = None) -> int:
    """Launch the workbench. Returns the Qt event-loop exit code."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(argv or [])
    window, controller = build_workbench()
    # Registered here (once per process), not in build_workbench, so a hard exit
    # still terminates a running worker without accumulating hooks across tests.
    atexit.register(controller.shutdown)
    window.show()
    return int(app.exec())
