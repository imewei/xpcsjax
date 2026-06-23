"""Run / cancel / export-figure slot collaborator for MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QMessageBox

from xpcsjax.gui.export import export_figures

if TYPE_CHECKING:
    from xpcsjax.gui.views.main_window import MainWindow


class RunController(QObject):
    """Owns the on_run / on_cancel / on_export_figure slot bodies.

    Operates on ``MainWindow`` state; the MainWindow ``_on_*`` shims delegate
    straight through so button wiring is unchanged.

    Parameters
    ----------
    main_window : MainWindow
        The owning MainWindow instance.  Passed as Qt parent so this object's
        lifetime is tied to the window.
    """

    def __init__(self, main_window: MainWindow) -> None:
        """Initialise and attach to *main_window* as Qt parent.

        Parameters
        ----------
        main_window : MainWindow
            The owning MainWindow instance.
        """
        super().__init__(main_window)
        self._mw = main_window

    def on_run(self) -> None:
        """Enqueue a new fit run for the currently active dataset.

        Reads ``_active_dataset_id`` from the owning window, creates a new
        :class:`~xpcsjax.gui.project.model.FitRun`, assigns its per-run output
        directory, enqueues it with the fit-queue controller, and refreshes the
        sidebar.
        """
        dataset_id = self._mw._active_dataset_id
        if dataset_id is None:
            self._mw.set_status("pick a config first")
            return
        dataset = self._mw._project.dataset_by_id(dataset_id)
        if dataset is None:
            self._mw.set_status("pick a config first")
            return
        run = self._mw._project.add_run(dataset_id)
        out_dir = self._mw._per_run_output_dir(dataset.config_path, run.run_id)
        run.result_dir = str(out_dir)  # durable per-run dir, recorded before enqueue
        self._mw._queue.enqueue(run.run_id, dataset.config_path, str(out_dir))
        self._mw._sidebar.set_project(self._mw._project)

    def on_cancel(self) -> None:
        """Cancel the currently selected run in the sidebar."""
        run_id = self._mw._sidebar.current_run_id()
        if run_id is None:
            self._mw.set_status("select a run first")
            return
        self._mw._queue.cancel(run_id)

    def on_export_figure(self) -> None:
        """Export publication figures from the selected run to a user-chosen directory.

        Opens a directory chooser; copies all ``*.png`` / ``*.pdf`` files from
        the run's ``plots/`` sub-tree into the chosen destination.  Shows an
        informational dialog on success or when nothing was found.
        """
        run_id = self._mw._sidebar.current_run_id()
        if run_id is None:
            self._mw.set_status("select a run first")
            return
        found = self._mw._project.run_by_id(run_id)
        if found is None:
            self._mw.set_status("select a run first")
            return
        _, run = found
        result_dir = run.result_dir
        if not result_dir:
            QMessageBox.information(
                self._mw,
                "Export Figure",
                "No result directory for this run — run the fit first.",
            )
            return

        dest = QFileDialog.getExistingDirectory(self._mw, "Export figures to…")
        if not dest:
            return  # user cancelled

        copied = export_figures(result_dir, dest)
        if not copied:
            QMessageBox.information(
                self._mw,
                "Export Figure",
                "No figures to export — this run produced none.",
            )
        else:
            QMessageBox.information(
                self._mw,
                "Export Figure",
                f"Copied {len(copied)} figure(s) to:\n{dest}",
            )
