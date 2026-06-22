"""The workbench main window — a logic-free view driven by FitQueueController.

All orchestration lives in the controller; this module only renders state and
forwards user actions. The workflow is config-first: the config file carries
everything the NLSQ fit needs, so there are no data/config/fit setup tabs — the
central area shows only the per-angle fitting results and residual analysis. The
File menu holds the project lifecycle (Create / Open / Save / Close Project); the
quick-access toolbar drives the flow Create Config → Edit Config → Load Config →
Run → view Results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    QWidget,
)

from xpcsjax.gui.controllers.fit_queue import FitQueueController
from xpcsjax.gui.error_presenter import present_failure
from xpcsjax.gui.export import export_figures
from xpcsjax.gui.project.model import Project
from xpcsjax.gui.project.persist import load_project, save_project
from xpcsjax.gui.result_loader import load_result_summary
from xpcsjax.gui.theme import repolish
from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog, CreateConfigDialog
from xpcsjax.gui.views.error_dialog import ErrorDialog
from xpcsjax.gui.views.inspector import InspectorDock
from xpcsjax.gui.views.main_window_support.result_presenter import ResultPresenter
from xpcsjax.gui.views.main_window_support.status_manager import StatusManager
from xpcsjax.gui.views.plots_view import PhiResultsGrid
from xpcsjax.gui.views.project_panel import ComparisonView, ProjectSidebar

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
        # The run the user is deliberately viewing (set by a sidebar click). The
        # queue-driven finish refresh must not yank the central panel/inspector away
        # from it. ``None`` means "follow the active run" (the common single-run path).
        self._viewing_run_id: str | None = None
        self._active_dataset_id: str | None = None
        # The project working directory (set by Create Project); also the default
        # base for per-run output dirs. ``_output_dir`` mirrors ``_project_dir``
        # (both set by Create Project); there is no separate override action.
        self._project_dir: Path | None = None
        self._output_dir: Path | None = None

        self.setWindowTitle("xpcsjax — analysis workbench")
        self.resize(1320, 820)
        self.setMinimumSize(960, 640)

        # Status pill: themed via the global QSS (#status_pill) with a `state`
        # dynamic property (idle/running/finished/failed) driving its colour.
        # It lives in the window status bar now that the bottom dock is log-only.
        self._status = QLabel("idle")
        self._status.setObjectName("status_pill")
        self.statusBar().addWidget(self._status)
        self._set_status_state("idle")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        # Central results area: a stacked widget with two pages.
        # Page 0: plain-text summary (shown when no interactive bundle is available).
        # Page 1: the per-phi results grid (Exp/Fitted/Residual maps + diagnostics).
        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._result_grid = PhiResultsGrid()
        self._central_stack = QStackedWidget()
        self._central_stack.setObjectName("central_results")
        self._central_stack.addWidget(self._results)  # index 0 → text summary
        self._central_stack.addWidget(self._result_grid)  # index 1 → per-phi grid
        self._central_stack.setCurrentIndex(0)
        self.setCentralWidget(self._central_stack)

        self._sidebar = ProjectSidebar()
        self._comparison = ComparisonView()
        self._inspector = InspectorDock()
        self._status_manager = StatusManager(self)
        self._result_presenter = ResultPresenter(self)

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
        bar.setMovable(False)
        self.addToolBar(bar)
        self._actions: dict[str, QAction] = {}
        # The quick-access toolbar owns every operational action. ``None`` markers
        # insert a visual separator between logical groups:
        # [config: create/edit/load] | [execute: run/cancel] | [export].
        spec: list[tuple[str, str, object] | None] = [
            ("action_create_config", "Create Config", self._on_create_config),
            ("action_edit_config", "Edit Config", self._on_edit_config),
            ("action_load_config", "Load Config", self._on_load_config),
            None,
            ("action_run", "Run", self._on_run),
            ("action_cancel", "Cancel", self._on_cancel),
            None,
            ("action_export_figure", "Export Figure", self._on_export_figure),
        ]
        for entry in spec:
            if entry is None:
                bar.addSeparator()
                continue
            key, text, slot = entry
            action = QAction(text, self)
            action.setObjectName(key)
            action.triggered.connect(slot)  # type: ignore[arg-type]
            bar.addAction(action)
            self._actions[key] = action

    def _build_file_menu(self) -> None:
        """Build the File menu: project-lifecycle actions only.

        Order: Create Project → Open Project → Save Project → Close Project. The
        operational actions (Create/Edit/Load Config, Run, Cancel, Export Figure)
        live solely on the quick-access toolbar — they are not duplicated here.
        """
        file_menu = self.menuBar().addMenu("File")

        create_project = QAction("Create Project", self)
        create_project.setObjectName("action_create_project")
        create_project.triggered.connect(self._on_create_project)
        file_menu.addAction(create_project)

        open_action = QAction("Open Project", self)
        open_action.setObjectName("action_open_project")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("Save Project", self)
        save_action.setObjectName("action_save_project")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        close_action = QAction("Close Project", self)
        close_action.setObjectName("action_close_project")
        close_action.triggered.connect(self._on_close_project)
        file_menu.addAction(close_action)

        self._actions["action_create_project"] = create_project
        self._actions["action_open_project"] = open_action
        self._actions["action_save_project"] = save_action
        self._actions["action_close_project"] = close_action

    def _build_sidebar_dock(self) -> None:
        dock = QDockWidget("Project", self)
        dock.setObjectName("dock_project_sidebar")
        dock.setWidget(self._sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_monitor_dock(self) -> None:
        """Build the bottom dock: the live fitting-process log, enlarged.

        The old SSR curve, L1–L5 layer chips, banner list, and status pill were
        removed — the bottom dock now shows only the streaming log.
        """
        dock = QDockWidget("Fitting Process", self)
        dock.setObjectName("dock_fitting_process")
        dock.setWidget(self._log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._monitor_dock = dock

    def _build_comparison_dock(self) -> None:
        dock = QDockWidget("Comparison", self)
        dock.setObjectName("dock_comparison")
        dock.setWidget(self._comparison)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._comparison_dock = dock

    def _build_inspector_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("dock_inspector")
        dock.setWidget(self._inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._inspector_dock = dock
        # Stack Inspector + Comparison as tabs in the right dock area so neither
        # is squeezed to a sliver; surface the Inspector (params/diagnostics)
        # first since it is the primary post-fit readout.
        self.tabifyDockWidget(self._comparison_dock, self._inspector_dock)
        self._inspector_dock.raise_()

    def _connect_signals(self) -> None:
        # Queue signals → window slots. The per-iteration SSR / layer-status /
        # banner streams are no longer rendered (the bottom dock is log-only), so
        # only status / finished / failed / log are wired.
        self._queue.run_status_changed.connect(self._on_run_status)
        self._queue.run_finished.connect(self._on_run_finished)
        self._queue.run_failed.connect(self._on_run_failed)
        self._queue.log_received.connect(self._on_log)
        # Sidebar selection signal
        self._sidebar.runs_selected.connect(self._on_runs_selected)

    # --- view slots (driven by the queue) -------------------------------------
    def set_status(self, status: str) -> None:
        """Render the current run status."""
        self._status_manager.set_status(status)

    def _set_status_state(self, state: str) -> None:
        """Set the status-pill colour state (``idle``/``running``/``finished``/``failed``).

        Drives the ``#status_pill[state=...]`` QSS rule. Text is untouched — the
        ``status_text`` contract (and tests reading it) stay exactly as before.
        """
        self._status.setProperty("state", state)
        repolish(self._status)

    def append_log(self, level: str, message: str) -> None:
        """Append one forwarded log line to the tail."""
        self._status_manager.append_log(level, message)

    def show_result(self, summary: Any) -> None:
        """Render the finished-fit summary (a ResultSummary or None) in the text panel."""
        self._result_presenter.show_result(summary)

    def _show_result_with_bundle(self, summary: Any, result_dir: str | None) -> None:
        """Render the result: per-phi grid when a bundle exists, text otherwise.

        Parameters
        ----------
        summary:
            A ``ResultSummary`` (or ``None``) to show in the text fallback.
        result_dir:
            The run's result directory; used to locate the viz bundle.
            ``None`` forces the text-summary path.
        """
        self._result_presenter.show_result_with_bundle(summary, result_dir)

    def show_error(self, message: str) -> None:
        """Render a fit failure."""
        self._result_presenter.show_error(message)

    def _on_run_status(self, run_id: str, status: str) -> None:
        if status in ("starting", "running"):
            # Attach the monitor as soon as the run begins (a cold spawn streams
            # nothing until the worker is up).
            if self._active_run_id != run_id:
                self._active_run_id = run_id  # this run's stream now drives the log
                # A freshly-started run takes focus: clear any prior deliberate
                # selection and the stale per-run log so the new stream starts clean.
                self._viewing_run_id = None
                self._log.clear()
        self._project.set_run_status(run_id, status)
        self._sidebar.update_run(self._project, run_id)
        if status in ("starting", "running"):
            self._set_status_state("running")
        if status == "starting":
            # Cold-spawn pause is expected, not a hang (spec §4 F10).
            self.set_status(f"{run_id[:8]}: starting (JAX import / XLA compile may take a moment)…")
        else:
            self.set_status(f"{run_id[:8]}: {status}")

    def _on_log(self, run_id: str, level: str, msg: str) -> None:
        if run_id == self._active_run_id:
            self.append_log(level, msg)

    def _on_run_failed(self, run_id: str, error_text: str) -> None:
        if run_id == self._active_run_id:
            self._set_status_state("failed")
        title, friendly, details = present_failure(error_text)
        # Identify which run failed (matters once multiple runs share the window).
        title = f"{title} (run {run_id[:8]})"
        ErrorDialog.show_failure(self, title, friendly, details)

    def _on_run_finished(
        self, run_id: str, result_path: str, summary: ResultSummary | None
    ) -> None:
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
        # Show the result in the main panel for the active run — but never yank the
        # panel/inspector away from a run the user has deliberately selected to view.
        if run_id == self._active_run_id:
            self._set_status_state("finished")
            if self._viewing_run_id is None or self._viewing_run_id == run_id:
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
                # Record the deliberate selection so a finishing active run can't
                # clobber it (see _on_run_finished).
                self._viewing_run_id = first_rid
                self._show_result_with_bundle(run.summary, run.result_dir)
                # Mirror the run's results into the inspector (params/uncertainties/
                # diagnostics live there).
                self.show_inspector(run.summary)

    # --- inspector public API -------------------------------------------------

    def show_inspector(self, summary: ResultSummary | None) -> None:
        """Populate the inspector dock with *summary* (or clear on None).

        Parameters
        ----------
        summary:
            A :class:`~xpcsjax.gui.result_loader.ResultSummary` or ``None``.
        """
        self._result_presenter.show_inspector(summary)

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

    # --- project / config I/O slots (testable without dialogs) ----------------

    def create_project(self, directory: str | Path) -> None:
        """Set the project working directory (and the default per-run output base).

        This is the callable form of the Create-Project menu action, extracted so
        tests and the menu can invoke it without a directory dialog.

        Parameters
        ----------
        directory:
            The project working directory; created if it does not exist.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._project_dir = path
        self._output_dir = path
        # Surface the project in the left sidebar: show the folder name as the
        # project name in the tree header (root paths have no name -> full path).
        self._sidebar.set_project_name(path.name or str(path))
        self.set_status(f"project: {path}")

    def create_config(
        self,
        mode: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Generate a config from the *mode* template at *output_path*.

        Thin wrapper over :func:`xpcsjax.cli.config_generator.generate_config`
        (the same function the ``xpcsjax-config`` CLI uses), imported lazily to
        keep the GUI process JAX-free at import time.

        Parameters
        ----------
        mode:
            One of the four analysis modes.
        output_path:
            Destination YAML path.
        overwrite:
            Replace an existing file at *output_path* when ``True``.
        **kwargs:
            Optional ``data_path`` / ``q`` / ``dt`` / ``time_length`` injections.

        Returns
        -------
        pathlib.Path
            The generated config path.
        """
        from xpcsjax.cli.config_generator import generate_config

        written = generate_config(mode, output_path, overwrite=overwrite, **kwargs)
        self.set_status(f"created config: {written.name} (mode={mode})")
        return written

    def add_dataset(self, config_path: str) -> None:
        """Add a dataset to the project and refresh the sidebar.

        This is the callable form of the Load-Config toolbar/menu action,
        extracted so tests and menu actions can invoke it without a file dialog.

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

    def close_project(self) -> None:
        """Reset the workbench to its empty launch state (no project loaded).

        This is the callable form of the Close-Project menu action, extracted so
        tests and the menu can invoke it without a confirmation dialog. It clears
        the in-memory project, the active selections, the project/output dirs, and
        every results surface (sidebar tree, comparison, inspector, per-phi grid,
        and the text summary), returning the central view to the summary page.

        Active/queued fits belong to the project being discarded, so they are
        cancelled first — otherwise the workers would keep running unreachable
        (terminal events can no longer attach to a project run, logs are filtered
        out because ``_active_run_id`` is cleared) while still consuming RAM and
        writing artifacts under the old output dir.
        """
        # shutdown() cancels every active worker, joins its reader, and clears
        # the pending queue while leaving the controller reusable for the next
        # fit (the same teardown closeEvent uses).
        self._queue.shutdown()
        self._project = Project()
        self._active_run_id = None
        self._viewing_run_id = None
        self._active_dataset_id = None
        self._project_dir = None
        self._output_dir = None
        self._sidebar.set_project_name(None)
        self._sidebar.set_project(self._project)
        self._comparison.show_runs([])
        self._inspector.show_summary(None)
        self._result_grid.set_bundle(None)
        self._results.clear()
        self._log.clear()  # the Fitting-Process dock belongs to the discarded project
        self._central_stack.setCurrentIndex(0)
        self._set_status_state("idle")
        self.set_status("idle")

    # --- user actions ---------------------------------------------------------
    def _on_create_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Create / choose project directory")
        if path:
            self.create_project(path)

    def _on_create_config(self) -> None:
        dialog = CreateConfigDialog(self, default_dir=self._project_dir or self._output_dir)
        if dialog.exec() != int(CreateConfigDialog.DialogCode.Accepted):
            return
        output_path = dialog.output_path()
        if not output_path:
            QMessageBox.information(self, "Create Config", "No output path was given.")
            return
        mode = dialog.selected_mode()
        try:
            kwargs = dialog.generation_kwargs()
        except ValueError as exc:
            QMessageBox.warning(self, "Create Config", f"Invalid input:\n{exc}")
            return
        try:
            self.create_config(mode, output_path, overwrite=False, **kwargs)
        except FileExistsError:
            resp = QMessageBox.question(
                self,
                "Create Config",
                f"{output_path} exists. Overwrite?",
            )
            if resp == QMessageBox.StandardButton.Yes:
                # The overwrite retry can itself fail (permission denied, disk
                # full, read-only FS) — guard it so the error surfaces as a
                # warning instead of escaping the slot through the event loop.
                try:
                    self.create_config(mode, output_path, overwrite=True, **kwargs)
                except (ValueError, FileNotFoundError, OSError) as exc:
                    QMessageBox.warning(self, "Create Config", f"Could not create config:\n{exc}")
        except (ValueError, FileNotFoundError, OSError) as exc:
            # OSError covers write failures on the initial create (FileExistsError,
            # an OSError subclass, is caught above first so its overwrite prompt
            # still runs).
            QMessageBox.warning(self, "Create Config", f"Could not create config:\n{exc}")

    def _on_edit_config(self) -> None:
        start_dir = str(self._project_dir) if self._project_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Edit config", start_dir, "YAML configs (*.yaml *.yml)"
        )
        if path:
            ConfigTextEditorDialog(path, self).exec()

    def _on_load_config(self) -> None:
        start_dir = str(self._project_dir) if self._project_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", start_dir, "YAML configs (*.yaml *.yml)"
        )
        if path:
            self.add_dataset(path)

    def _on_save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "xpcsjax project (*.xpcsproj);;All files (*)"
        )
        if path:
            self.save_project_to(path)

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "xpcsjax project (*.xpcsproj);;All files (*)"
        )
        if path:
            self.open_project_from(path)

    def _on_close_project(self) -> None:
        resp = QMessageBox.question(
            self,
            "Close Project",
            "Close the current project? Unsaved results will be cleared.",
        )
        if resp == QMessageBox.StandardButton.Yes:
            self.close_project()

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
        """Terminate any running worker before closing."""
        self._queue.shutdown()
        super().closeEvent(event)
