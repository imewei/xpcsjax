"""The workbench main window — a logic-free view driven by FitQueueController.

All orchestration lives in the controller; this module only renders state and
forwards user actions (Open Config / Output Dir / Run / Cancel).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
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
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.controllers.fit_queue import FitQueueController
from xpcsjax.gui.error_presenter import present_failure
from xpcsjax.gui.export import export_figures
from xpcsjax.gui.project.model import Project
from xpcsjax.gui.project.persist import load_project, save_project
from xpcsjax.gui.result_loader import load_result_summary
from xpcsjax.gui.views.config_editor import ConfigEditor
from xpcsjax.gui.views.data_panel import DataPanel
from xpcsjax.gui.views.diagnostics_panel import BannerList, LayerStatusChips, SSRCurveWidget
from xpcsjax.gui.views.error_dialog import ErrorDialog
from xpcsjax.gui.views.fit_panel import FitPanel
from xpcsjax.gui.views.inspector import InspectorDock
from xpcsjax.gui.views.plots_view import ResultPlots
from xpcsjax.gui.views.project_panel import ComparisonView, ProjectSidebar
from xpcsjax.gui.viz_bundle import load_viz_bundle

if TYPE_CHECKING:
    from xpcsjax.gui.result_loader import ResultSummary


def _expand_path(path: str) -> Path:
    """Expand ``${ENV}`` and ``~`` in a stored path before an existence check.

    A no-op on plain absolute paths; keeps a config/result reference that merely
    uses a shell variable from being mis-flagged "missing" on project load.
    """
    return Path(os.path.expandvars(path)).expanduser()


class MainWindow(QMainWindow):
    """Main workbench window — owns a Project and a FitQueueController."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = Project()
        self._queue = FitQueueController()
        self._active_run_id: str | None = None
        self._active_dataset_id: str | None = None
        self._output_dir: Path | None = None
        # Temp config YAMLs (ConfigEditor → worker handoff), keyed by the synthetic
        # dataset they back. Kept alive for the whole session — a worker opens its
        # config by path and the same dataset may be re-run from the toolbar — and
        # all are unlinked on close. (Never eagerly deleted on the next Validate,
        # which would yank a file an active/pending worker still needs.)
        self._dataset_temp_paths: dict[str, str] = {}

        self.setWindowTitle("xpcsjax — analysis workbench")
        self.resize(1200, 750)

        self._status = QLabel("idle")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        # Results area: a stacked widget with two pages.
        # Page 0: plain-text summary (shown when no interactive bundle is available).
        # Page 1: interactive ResultPlots (shown when a viz bundle loads).
        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._result_plots = ResultPlots()
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self._results)  # index 0 → text summary
        self._central_stack.addWidget(self._result_plots)  # index 1 → interactive plots
        self._central_stack.setCurrentIndex(0)

        # --- Center tab widget -------------------------------------------------
        # Data / Config / Fit tabs are new; Results tab wraps the existing stack.
        self._data_panel = DataPanel()
        self._config_editor = ConfigEditor()
        self._fit_panel = FitPanel()

        self._center_tabs = QTabWidget()
        self._center_tabs.setObjectName("center_tabs")
        self._center_tabs.addTab(self._data_panel, "Data")
        self._center_tabs.addTab(self._config_editor, "Config")
        self._center_tabs.addTab(self._fit_panel, "Fit")
        # Results tab wraps the existing central_stack so all prior behavior
        # (show_result / _show_result_with_bundle / result_text) is unchanged.
        results_container = QWidget()
        rc_layout = QVBoxLayout(results_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.addWidget(self._central_stack)
        self._center_tabs.addTab(results_container, "Results")

        self.setCentralWidget(self._center_tabs)

        self._sidebar = ProjectSidebar()
        self._comparison = ComparisonView()
        self._inspector = InspectorDock()

        self._build_toolbar()
        self._build_file_menu()
        self._build_sidebar_dock()
        self._build_monitor_dock()
        self._build_comparison_dock()
        self._build_inspector_dock()
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

    def _build_file_menu(self) -> None:
        """Add a File menu with Save Project / Open Project actions."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        save_action = QAction("Save Project", self)
        save_action.setObjectName("action_save_project")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        open_action = QAction("Open Project", self)
        open_action.setObjectName("action_open_project")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

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

    def _build_inspector_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("dock_inspector")
        dock.setWidget(self._inspector)
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
        # ConfigEditor → run-launch path: validated config feeds _on_config_ready
        self._config_editor.config_ready.connect(self._on_config_ready)

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
        if status in ("starting", "running"):
            # Attach the monitor as soon as the run begins (a cold spawn streams
            # nothing until the worker is up), and reset the SSR curve once.
            if self._active_run_id != run_id:
                self._active_run_id = run_id  # this run's stream now drives the monitor
                # Reset ALL live-diagnostics views, not just the SSR curve — else
                # the previous run's banners and lit layer-chips bleed into the
                # new run's monitor.
                self._ssr_curve.reset()
                self._banners.clear()
                self._chips.set_layers({})
        self._project.set_run_status(run_id, status)
        self._sidebar.update_run(self._project, run_id)
        if status == "starting":
            # Cold-spawn pause is expected, not a hang (spec §4 F10).
            self.set_status(f"{run_id[:8]}: starting (JAX import / XLA compile may take a moment)…")
        else:
            self.set_status(f"{run_id[:8]}: {status}")

    def _on_log(self, run_id: str, level: str, msg: str) -> None:
        if run_id == self._active_run_id:
            self.append_log(level, msg)

    def _on_iteration(self, run_id: str, n: int, ssr: float) -> None:
        if run_id == self._active_run_id:
            self._ssr_curve.add_point(n, ssr)

    def _on_layer_status(self, run_id: str, layers: dict[str, bool]) -> None:
        if run_id == self._active_run_id:
            self._chips.set_layers(layers)

    def _on_banner(self, run_id: str, text: str, kind: str) -> None:
        if run_id == self._active_run_id:
            self._banners.add_banner(text, kind)

    def _on_run_failed(self, run_id: str, error_text: str) -> None:
        title, friendly, details = present_failure(error_text)
        # Identify which run failed (matters once multiple runs share the window).
        title = f"{title} (run {run_id[:8]})"
        ErrorDialog.show_failure(self, title, friendly, details)

    def _on_run_finished(self, run_id: str, result_path: str, summary: ResultSummary | None) -> None:
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
            # Mirror finished result into the inspector dock.
            self.show_inspector(summary)

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
                # Mirror the run's results into the inspector (params/uncertainties/
                # diagnostics live there). The Fit panel shows the *resolved config*,
                # which a ResultSummary does not carry — so clear it rather than
                # fabricate a misleading analysis_mode from the convergence status.
                # (Seam: recover the run's config from result_dir to repopulate it.)
                self.show_inspector(run.summary)
                self._fit_panel.clear()

    def _on_config_ready(self, cfg: dict) -> None:
        """Slot: ConfigEditor emitted config_ready — write to temp YAML + launch run.

        The validated dict is serialized to a NamedTemporaryFile (kept on disk
        until the next config_ready or window close).  The run is enqueued via
        the same ``FitQueueController`` path the toolbar "Run" button uses, so
        all monitor/sidebar wiring is preserved.

        The temp file is created with ``delete=False`` so the worker process can
        open it after this method returns. It is registered per-dataset and
        unlinked on window close — never eagerly, so an active/pending worker is
        never left pointing at a deleted config.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="xpcsjax_gui_",
            delete=False,
            encoding="utf-8",
        ) as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
            temp_path = fh.name

        # Mirror the validated config into the Fit panel (informational).
        self._fit_panel.show_settings(cfg, None)
        # Switch the center tabs to Results so the user sees the run progress.
        results_idx = next(
            (
                i
                for i in range(self._center_tabs.count())
                if self._center_tabs.tabText(i) == "Results"
            ),
            -1,
        )
        if results_idx >= 0:
            self._center_tabs.setCurrentIndex(results_idx)

        # Register a synthetic dataset for the temp config + enqueue run.
        dataset = self._project.add_dataset(temp_path)
        self._dataset_temp_paths[dataset.dataset_id] = temp_path  # unlinked on close
        self._active_dataset_id = dataset.dataset_id

        run = self._project.add_run(dataset.dataset_id)
        out_dir = self._per_run_output_dir(temp_path, run.run_id)
        run.result_dir = str(out_dir)  # durable per-run dir, recorded before enqueue
        self._queue.enqueue(run.run_id, temp_path, str(out_dir))
        # Single sidebar rebuild: the dataset AND its run are both present now, so
        # one rebuild suffices (previously rebuilt twice — once after add_dataset).
        self._sidebar.set_project(self._project)

    # --- inspector public API -------------------------------------------------

    def show_inspector(self, summary: ResultSummary | None) -> None:
        """Populate the inspector dock with *summary* (or clear on None).

        Parameters
        ----------
        summary:
            A :class:`~xpcsjax.gui.result_loader.ResultSummary` or ``None``.
        """
        self._inspector.show_summary(summary)

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

    def sidebar_dataset_count(self) -> int:
        """Return the number of datasets in the current project (test/inspection helper)."""
        return len(self._project.datasets)

    # --- project I/O slots (testable without dialogs) -------------------------

    def add_dataset(self, config_path: str) -> None:
        """Add a dataset to the project and refresh the sidebar.

        This is the callable form of the Open-Config toolbar action, extracted
        so tests and menu actions can invoke it without a file dialog.

        Parameters
        ----------
        config_path:
            Absolute path to the YAML config file.
        """
        dataset = self._project.add_dataset(config_path)
        self._sidebar.set_project(self._project)
        # Auto-select the freshly added dataset so a single-dataset Run works.
        self._active_dataset_id = dataset.dataset_id
        self.set_status(f"config: {Path(config_path).name}")

    def save_project_to(self, path: str | Path) -> None:
        """Serialize the current project to *path* (.xpcsproj).

        Parameters
        ----------
        path:
            Destination file path (created or overwritten).
        """
        save_project(self._project, path)

    def open_project_from(self, path: str | Path) -> None:
        """Deserialize a project from *path* and restore the sidebar.

        Dead-path tolerance (spec §8): every config/result reference is resolved
        *eagerly at load*. A run whose ``result_dir`` is gone (or whose summary
        will not load) is flagged ``result_missing`` and a dataset whose
        ``config_path`` is gone is flagged ``config_missing`` — both surface as a
        clearly-labelled "missing" entry in the sidebar tree, never as a deferred
        ``FileNotFoundError`` thrown far from the load call. Neither is a hard
        failure.

        Parameters
        ----------
        path:
            Path to a ``.xpcsproj`` file previously written by
            :meth:`save_project_to`.
        """
        self._project = load_project(path)
        # Resolve every reference eagerly at load (spec §8 dead-path rule).
        # Expand ${ENV}/~ first so a path that merely uses a shell variable is not
        # mis-flagged "missing" (mirrors config.data_folder_path resolution).
        for dataset in self._project.datasets:
            dataset.config_missing = (
                bool(dataset.config_path) and not _expand_path(dataset.config_path).exists()
            )
            for run in dataset.runs:
                if run.result_dir:
                    if not _expand_path(run.result_dir).exists():
                        run.summary = None
                        run.result_missing = True
                        continue
                    try:
                        run.summary = load_result_summary(str(_expand_path(run.result_dir)))
                    except Exception:  # noqa: BLE001 — never raise on open
                        run.summary = None
                        run.result_missing = True
        # A freshly-opened project has no active dataset, so a subsequent "Run"
        # would say "pick a config first". Default to the first dataset (matches
        # add_dataset's auto-select), so Run works straight after Open Project.
        self._active_dataset_id = (
            self._project.datasets[0].dataset_id if self._project.datasets else None
        )
        self._sidebar.set_project(self._project)

    # --- user actions ---------------------------------------------------------
    def _on_open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open config", "", "YAML configs (*.yaml *.yml)"
        )
        if path:
            self.add_dataset(path)

    def _on_save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "xpcsjax project (*.xpcsproj)"
        )
        if path:
            self.save_project_to(path)

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "xpcsjax project (*.xpcsproj)"
        )
        if path:
            self.open_project_from(path)

    def _on_choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self._output_dir = Path(path)

    def _per_run_output_dir(self, config_path: str, run_id: str) -> Path:
        """Return a unique output dir for one run: ``<base>/runs/<run_id>``.

        ``<base>`` is the user-chosen output directory (if set) or
        ``<config_dir>/xpcsjax_gui_out``.  Namespacing by ``run_id`` keeps each
        run's ``nlsq_result.*`` and ``plots/`` artifacts isolated, so re-running
        a dataset never overwrites a prior run's durable outputs.
        """
        base = self._output_dir or Path(config_path).parent / "xpcsjax_gui_out"
        return base / "runs" / run_id

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
        out_dir = self._per_run_output_dir(dataset.config_path, run.run_id)
        run.result_dir = str(out_dir)  # durable per-run dir, recorded before enqueue
        self._queue.enqueue(run.run_id, dataset.config_path, str(out_dir))
        self._sidebar.set_project(self._project)

    def _on_cancel(self) -> None:
        run_id = self._sidebar.current_run_id()
        if run_id is None:
            self.set_status("select a run first")
            return
        self._queue.cancel(run_id)

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
        result_dir = run.result_dir
        if not result_dir:
            QMessageBox.information(
                self,
                "Export Figure",
                "No result directory for this run — run the fit first.",
            )
            return

        dest = QFileDialog.getExistingDirectory(self, "Export figures to…")
        if not dest:
            return  # user cancelled

        copied = export_figures(result_dir, dest)
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
        """Terminate any running worker and delete temp configs before closing."""
        self._queue.shutdown()
        # Unlink every session temp config (best-effort) — these are only deleted
        # here, never mid-session, so no worker is ever left without its config.
        for temp_path in self._dataset_temp_paths.values():
            Path(temp_path).unlink(missing_ok=True)
        self._dataset_temp_paths.clear()
        super().closeEvent(event)
