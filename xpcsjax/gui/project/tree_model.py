"""Mirror the in-memory Project into a QStandardItemModel for the sidebar tree."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel

from xpcsjax.gui.project.model import Dataset, FitRun, Project


def _run_label(run: FitRun) -> str:
    # Dead-path flag (spec §8): a restored run whose result_dir is gone is shown
    # clearly flagged, never silently as a normal "done" run.
    suffix = " · result missing" if run.result_missing else ""
    return f"{run.status} · {run.run_id[:8]}{suffix}"


def _dataset_label(dataset: Dataset) -> str:
    # Dead-path flag (spec §8): a dataset whose config_path is gone is flagged in place.
    return f"{dataset.label} (config missing)" if dataset.config_missing else dataset.label


def _dataset_item(dataset: Dataset) -> QStandardItem:
    item = QStandardItem(_dataset_label(dataset))
    item.setEditable(False)
    item.setData(dataset.dataset_id, Qt.ItemDataRole.UserRole)
    for run in dataset.runs:
        item.appendRow(_run_item(run))
    return item


def _run_item(run: FitRun) -> QStandardItem:
    item = QStandardItem(_run_label(run))
    item.setEditable(False)
    item.setData(run.run_id, Qt.ItemDataRole.UserRole)
    return item


_DEFAULT_HEADER = "Project"


class ProjectTreeModel(QStandardItemModel):
    """A datasets -> runs tree mirroring a :class:`Project`."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # The project name shown as the tree header (set by Create / Open Project).
        # ``None`` falls back to the generic ``"Project"`` label.
        self._project_name: str | None = None

    def _header_label(self) -> str:
        return self._project_name or _DEFAULT_HEADER

    def set_project_name(self, name: str | None) -> None:
        """Set (or clear) the project name shown as the tree header.

        Persists across :meth:`rebuild`; pass ``None`` to revert to ``"Project"``.
        """
        self._project_name = name or None
        self.setHorizontalHeaderLabels([self._header_label()])

    def rebuild(self, project: Project) -> None:
        """Replace the whole tree from ``project`` (full sync)."""
        self.clear()
        self.setHorizontalHeaderLabels([self._header_label()])
        for dataset in project.datasets:
            self.appendRow(_dataset_item(dataset))

    def update_run(self, project: Project, run_id: str) -> None:
        """Refresh the single run row's status label in place."""
        found = project.run_by_id(run_id)
        if found is None:
            return
        dataset, run = found
        for i in range(self.rowCount()):
            ds_item = self.item(i)
            if ds_item.data(Qt.ItemDataRole.UserRole) != dataset.dataset_id:
                continue
            for j in range(ds_item.rowCount()):
                run_item = ds_item.child(j)
                if run_item.data(Qt.ItemDataRole.UserRole) == run_id:
                    run_item.setText(_run_label(run))
                    return
