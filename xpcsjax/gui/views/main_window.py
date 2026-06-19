"""The workbench main window — a logic-free view driven by FitController.

All orchestration lives in the controller; this module only renders state and
forwards user actions (Open Config / Output Dir / Run / Cancel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.controllers.fit_controller import FitController


class MainWindow(QMainWindow):
    """Main workbench window for the single-dataset happy path."""

    def __init__(self, controller: FitController) -> None:
        super().__init__()
        self._controller = controller
        self._config_path: Path | None = None
        self._output_dir: Path | None = None

        self.setWindowTitle("xpcsjax — analysis workbench")
        self.resize(1000, 700)

        self._status = QLabel("idle")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self.setCentralWidget(self._results)
        self._build_toolbar()
        self._build_monitor_dock()
        self._connect_controller()

    # --- construction helpers -------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        self.addToolBar(bar)
        self._actions: dict[str, QAction] = {}
        for key, text, slot in [
            ("action_open_config", "Open Config", self._on_open_config),
            ("action_output_dir", "Output Dir", self._on_choose_output),
            ("action_run", "Run", self._on_run),
            ("action_cancel", "Cancel", self._controller.cancel),
        ]:
            action = QAction(text, self)
            action.setObjectName(key)
            action.triggered.connect(slot)
            bar.addAction(action)
            self._actions[key] = action

    def _build_monitor_dock(self) -> None:
        dock = QDockWidget("Fit Monitor", self)
        dock.setObjectName("dock_fit_monitor")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._status)
        layout.addWidget(self._log)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _connect_controller(self) -> None:
        self._controller.status_changed.connect(self.set_status)
        self._controller.log_received.connect(self.append_log)
        self._controller.fit_finished.connect(self.show_result)
        self._controller.fit_failed.connect(self.show_error)

    # --- view slots (driven by the controller) --------------------------------
    def set_status(self, status: str) -> None:
        """Render the current run status."""
        self._status.setText(status)

    def append_log(self, level: str, message: str) -> None:
        """Append one forwarded log line to the tail."""
        self._log.appendPlainText(f"[{level}] {message}")

    def show_result(self, summary: Any) -> None:
        """Render the finished-fit summary (a ResultSummary or None)."""
        if summary is None:
            self._results.setPlainText("Fit finished, but no result file was found.")
            return
        lines = [
            f"status:          {summary.convergence_status}",
            f"success:         {summary.success}",
            f"chi^2:           {summary.chi_squared}",
            f"reduced chi^2:   {summary.reduced_chi_squared}",
            f"quality:         {summary.quality_flag}",
            f"results dir:     {summary.result_dir}",
            "",
            "parameters:",
            *[f"  {name} = {value}" for name, value in summary.parameters.items()],
            "",
            "Publication figures (Matplotlib) were written under "
            f"{summary.result_dir}/plots — use 'Output Dir' to locate them.",
        ]
        self._results.setPlainText("\n".join(lines))

    def show_error(self, message: str) -> None:
        """Render a fit failure."""
        self._results.setPlainText(f"FIT FAILED\n\n{message}")

    # --- introspection for tests ----------------------------------------------
    def status_text(self) -> str:
        """Return the current status text (test/inspection helper)."""
        return self._status.text()

    def log_text(self) -> str:
        """Return the accumulated log text (test/inspection helper)."""
        return self._log.toPlainText()

    def result_text(self) -> str:
        """Return the current results-panel text (test/inspection helper)."""
        return self._results.toPlainText()

    # --- user actions ---------------------------------------------------------
    def _on_open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open config", "", "YAML configs (*.yaml *.yml)")
        if path:
            self._config_path = Path(path)
            self.set_status(f"config: {self._config_path.name}")

    def _on_choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self._output_dir = Path(path)

    def _on_run(self) -> None:
        if self._config_path is None:
            self.set_status("pick a config first")
            return
        out = self._output_dir or self._config_path.parent / "xpcsjax_gui_out"
        self._controller.run(self._config_path, out)

    # --- lifecycle ------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override name
        """Terminate any running worker before the window closes."""
        self._controller.shutdown()
        super().closeEvent(event)
