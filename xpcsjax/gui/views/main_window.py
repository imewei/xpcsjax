"""The workbench main window — a logic-free view driven by FitQueueController.

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
    QMessageBox,
    QPlainTextEdit,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.controllers.fit_queue import FitQueueController
from xpcsjax.gui.error_presenter import present_failure
from xpcsjax.gui.export import export_figures
from xpcsjax.gui.project.model import Project
from xpcsjax.gui.views.diagnostics_panel import BannerList, LayerStatusChips, SSRCurveWidget
from xpcsjax.gui.views.error_dialog import ErrorDialog
from xpcsjax.gui.views.plots_view import ResultPlots
from xpcsjax.gui.views.project_panel import ComparisonView, ProjectSidebar
from xpcsjax.gui.viz_bundle import load_viz_bundle


class MainWindow(QMainWindow):
    """Main workbench window — owns a Project and a FitQueueController."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = Project()
        self._queue = FitQueueController()
        self._active_run_id: str | None = None
        self._active_dataset_id: str | None = None
        self._output_dir: Path | None = None

        self.setWindowTitle("xpcsjax — analysis workbench")
        self.resize(1100, 700)

        self._status = QLabel("idle")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        # Central widget: a stacked widget with two pages.
        # Page 0: plain-text summary (shown when no interactive bundle is available).
        # Page 1: interactive ResultPlots (shown when a viz bundle loads).
        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._result_plots = ResultPlots()
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self._results)       # index 0 → text summary
        self._central_stack.addWidget(self._result_plots)  # index 1 → interactive plots
        self._central_stack.setCurrentIndex(0)
        self.setCentralWidget(self._central_stack)

        self._sidebar = ProjectSidebar()
        self._comparison = ComparisonView()

        self._build_toolbar()
        self._build_sidebar_dock()
        self._build_monitor_dock()
        self._build_comparison_dock()
        self._connect_signals()

    # --- construction helpers -------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        self.addToolBar(bar)
        self._actions: dict[str, QAction] = {}
        for key, text, slot in [
            ("action_open_config", "Open Config", self._on_open_config),
            ("action_output_dir", "Output Dir", self._on_choose_output),
            ("action_run", "Run", self._on_run),
            ("action_cancel", "Cancel", self._on_cancel),
            ("action_export_figure", "Export Figure", self._on_export_figure),
        ]:
            action = QAction(text, self)
            action.setObjectName(key)
            action.triggered.connect(slot)
            bar.addAction(action)
            self._actions[key] = action

    def _build_sidebar_dock(self) -> None:
        dock = QDockWidget("Project", self)
        dock.setObjectName("dock_project_sidebar")
        dock.setWidget(self._sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_monitor_dock(self) -> None:
        dock = QDockWidget("Fit Monitor", self)
        dock.setObjectName("dock_fit_monitor")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._status)
        self._chips = LayerStatusChips()
        self._ssr_curve = SSRCurveWidget()
        self._banners = BannerList()
        layout.addWidget(self._chips)
        layout.addWidget(self._ssr_curve)
        layout.addWidget(self._banners)
        layout.addWidget(self._log)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_comparison_dock(self) -> None:
        dock = QDockWidget("Comparison", self)
        dock.setObjectName("dock_comparison")
        dock.setWidget(self._comparison)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_signals(self) -> None:
        # Queue signals → window slots
        self._queue.run_status_changed.connect(self._on_run_status)
        self._queue.run_finished.connect(self._on_run_finished)
        self._queue.run_failed.connect(self._on_run_failed)
        self._queue.log_received.connect(self._on_log)
        self._queue.iteration_received.connect(self._on_iteration)
        self._queue.layer_status_received.connect(self._on_layer_status)
        self._queue.banner_received.connect(self._on_banner)
        # Sidebar selection signal
        self._sidebar.runs_selected.connect(self._on_runs_selected)

    # --- view slots (driven by the queue) -------------------------------------
    def set_status(self, status: str) -> None:
        """Render the current run status."""
        self._status.setText(status)

    def append_log(self, level: str, message: str) -> None:
        """Append one forwarded log line to the tail."""
        self._log.appendPlainText(f"[{level}] {message}")

    def show_result(self, summary: Any) -> None:
        """Render the finished-fit summary (a ResultSummary or None) in the text panel."""
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

    def _show_result_with_bundle(self, summary: Any, result_dir: str | None) -> None:
        """Render the result: interactive plots when a bundle exists, text otherwise.

        Parameters
        ----------
        summary:
            A ``ResultSummary`` (or ``None``) to show in the text fallback.
        result_dir:
            The run's result directory; used to locate the viz bundle.
            ``None`` forces the text-summary path.
        """
        bundle = None
        if result_dir:
            try:
                bundle = load_viz_bundle(result_dir)
            except Exception:  # pragma: no cover — defensive only
                bundle = None

        if bundle is not None:
            self._result_plots.set_bundle(bundle)
            self._central_stack.setCurrentIndex(1)  # show interactive plots
        else:
            # Fall back to (or keep) the text summary.
            self.show_result(summary)
            self._central_stack.setCurrentIndex(0)

    def show_error(self, message: str) -> None:
        """Render a fit failure."""
        self._results.setPlainText(f"FIT FAILED\n\n{message}")

    def _on_run_status(self, run_id: str, status: str) -> None:
        if status == "running":
            self._active_run_id = run_id  # this run's stream now drives the monitor
            self._ssr_curve.reset()
        self._project.set_run_status(run_id, status)
        self._sidebar.update_run(self._project, run_id)
        self.set_status(f"{run_id[:8]}: {status}")

    def _on_log(self, run_id: str, level: str, msg: str) -> None:
        if run_id == self._active_run_id:
            self.append_log(level, msg)

    def _on_iteration(self, run_id: str, n: int, ssr: float) -> None:
        if run_id == self._active_run_id:
            self._ssr_curve.add_point(n, ssr)

    def _on_layer_status(self, run_id: str, layers: object) -> None:
        if run_id == self._active_run_id:
            self._chips.set_layers(layers)

    def _on_banner(self, run_id: str, text: str, kind: str) -> None:
        if run_id == self._active_run_id:
            self._banners.add_banner(text, kind)

    def _on_run_failed(self, run_id: str, error_text: str) -> None:
        title, friendly, details = present_failure(error_text)
        ErrorDialog.show_failure(self, title, friendly, details)

    def _on_run_finished(self, run_id: str, result_path: str, summary: object) -> None:
        # _on_run_status already set the terminal status; here we attach the result.
        found = self._project.run_by_id(run_id)
        if found is not None:
            run = found[1]
            # Store the durable result_dir ALWAYS (even when load_result_summary failed,
            # so viz/export/restore still work); attach the summary when present.
            if result_path:
                run.result_dir = result_path
            run.summary = summary
            self._sidebar.update_run(self._project, run_id)
        # Show the result in the main panel for the active run.
        if run_id == self._active_run_id:
            result_dir = result_path or None
            self._show_result_with_bundle(summary, result_dir)

    def _on_runs_selected(self, run_ids: list) -> None:
        pairs = []
        for rid in run_ids[:2]:  # compare up to two
            found = self._project.run_by_id(rid)
            if found is not None:
                _, run = found
                pairs.append((rid[:8], run.summary))
        self._comparison.show_runs(pairs)

        # Update the main panel to show the most-recently selected run's result.
        if run_ids:
            first_rid = run_ids[0]
            found = self._project.run_by_id(first_rid)
            if found is not None:
                _, run = found
                self._show_result_with_bundle(run.summary, run.result_dir)

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
            dataset = self._project.add_dataset(path)
            self._sidebar.set_project(self._project)
            # Auto-select the freshly added dataset so a single-dataset Run works.
            self._active_dataset_id = dataset.dataset_id
            self.set_status(f"config: {Path(path).name}")

    def _on_choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self._output_dir = Path(path)

    def _on_run(self) -> None:
        dataset_id = self._active_dataset_id
        if dataset_id is None:
            self.set_status("pick a config first")
            return
        dataset = self._project.dataset_by_id(dataset_id)
        if dataset is None:
            self.set_status("pick a config first")
            return
        run = self._project.add_run(dataset_id)
        out_dir = self._output_dir or Path(dataset.config_path).parent / "xpcsjax_gui_out"
        self._queue.enqueue(run.run_id, dataset.config_path, str(out_dir))
        self._sidebar.set_project(self._project)

    def _on_cancel(self) -> None:
        self._queue.cancel(self._sidebar.current_run_id())

    def _on_export_figure(self) -> None:
        """Export publication figures from the selected run to a user-chosen directory."""
        run_id = self._sidebar.current_run_id()
        if run_id is None:
            self.set_status("select a run first")
            return
        found = self._project.run_by_id(run_id)
        if found is None:
            self.set_status("select a run first")
            return
        _, run = found
        if not getattr(run, "result_dir", None):
            QMessageBox.information(
                self,
                "Export Figure",
                "No result directory for this run — run the fit first.",
            )
            return

        dest = QFileDialog.getExistingDirectory(self, "Export figures to…")
        if not dest:
            return  # user cancelled

        copied = export_figures(run.result_dir, dest)
        if not copied:
            QMessageBox.information(
                self,
                "Export Figure",
                "No figures to export — this run produced none.",
            )
        else:
            QMessageBox.information(
                self,
                "Export Figure",
                f"Copied {len(copied)} figure(s) to:\n{dest}",
            )

    # --- lifecycle ------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override name
        """Terminate any running worker before the window closes."""
        self._queue.shutdown()
        super().closeEvent(event)
