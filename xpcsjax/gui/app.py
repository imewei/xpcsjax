"""GUI entry point.

Plan C ships a placeholder window so ``xpcsjax-gui`` launches; Plan D wires the
real panels (project sidebar, config/data/fit/results tabs, fit-monitor dock)
and the controller that drives a :class:`~xpcsjax.gui.ipc.handle.WorkerHandle`.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Launch the workbench. Returns the Qt event-loop exit code."""
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

    app = QApplication.instance() or QApplication(argv or [])
    window = QMainWindow()
    window.setWindowTitle("xpcsjax — analysis workbench (skeleton)")
    window.setCentralWidget(
        QLabel("Workbench skeleton.\nPanels + fit wiring arrive in Plan D.")
    )
    window.resize(900, 600)
    window.show()
    return int(app.exec())
