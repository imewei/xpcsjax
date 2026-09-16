"""Project sidebar (datasets -> runs tree) + a side-by-side comparison view."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QItemSelectionModel, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QPlainTextEdit, QTreeView, QVBoxLayout, QWidget

from xpcsjax.gui.comparison import format_comparison
from xpcsjax.gui.project.model import Project
from xpcsjax.gui.project.tree_model import ProjectTreeModel


class ProjectSidebar(QWidget):
    """A tree of datasets -> runs with multi-select."""

    runs_selected = Signal(list)  # list[str] of run_ids
    dataset_selected = Signal(str)  # dataset_id, emitted when a dataset row is selected

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = ProjectTreeModel()
        self._tree = QTreeView()
        self._tree.setModel(self._model)
        self._tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self._tree.selectionModel().selectionChanged.connect(self._on_selection)
        layout = QVBoxLayout(self)
        layout.addWidget(self._tree)

    def model(self) -> ProjectTreeModel:
        """Return the backing tree model (inspection helper)."""
        return self._model

    def set_project(self, project: Project, *, select_run_id: str | None = None) -> None:
        """Rebuild the tree from ``project``.

        ``rebuild`` fully resets the underlying ``QStandardItemModel``, which
        drops the tree's current selection. Restore it afterward — the
        explicitly requested *select_run_id* if given, else whatever run was
        selected before the rebuild — so a Run/Load-Config refresh doesn't
        silently strand Cancel/Export Figure with no selected run.
        """
        keep = select_run_id if select_run_id is not None else self.current_run_id()
        self._model.rebuild(project)
        self._tree.expandAll()
        if keep is not None:
            self._select_run(keep)

    def _select_run(self, run_id: str) -> None:
        """Select *run_id*'s row in the tree, if it still exists.

        This is a widget-state fixup after a full tree rebuild, not a user
        action — block ``selectionChanged`` so it doesn't re-fire
        ``runs_selected``/``_on_runs_selected`` and repaint the results panel
        with that run's (possibly not-yet-existing) summary, or re-pin
        ``_viewing_run_id`` to a run nobody actually clicked.
        """
        index = self._model.index_for_run(run_id)
        if index is not None:
            with QSignalBlocker(self._tree.selectionModel()):
                self._tree.selectionModel().select(
                    index,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                self._tree.setCurrentIndex(index)

    def set_project_name(self, name: str | None) -> None:
        """Show ``name`` as the sidebar header (the project name); ``None`` resets it."""
        self._model.set_project_name(name)

    def update_run(self, project: Project, run_id: str) -> None:
        """Refresh one run row."""
        self._model.update_run(project, run_id)

    def selected_run_ids(self) -> list[str]:
        """Return the run ids of selected run rows (datasets are ignored)."""
        ids: list[str] = []
        for index in self._tree.selectionModel().selectedIndexes():
            # Run rows have a parent (a dataset); dataset rows do not.
            if index.parent().isValid():
                rid = index.data(Qt.ItemDataRole.UserRole)
                if rid is not None:
                    ids.append(str(rid))
        return ids

    def current_run_id(self) -> str | None:
        """Return the focused run id, or None.

        Uses Qt's ``currentIndex()`` (the actually-focused/anchor row) rather
        than the first entry of ``selectedIndexes()`` — with ExtendedSelection
        multi-select those can differ (e.g. select run A, then Ctrl-click run B:
        both stay selected but B is current), and actions like Cancel/Export
        must act on the row the user last clicked, not model order.
        """
        ids = self.selected_run_ids()
        current = self._tree.selectionModel().currentIndex()
        if current.isValid() and current.parent().isValid():
            rid = current.data(Qt.ItemDataRole.UserRole)
            if rid is not None and str(rid) in ids:
                return str(rid)
        # No focused (and selected) run row (e.g. current is a dataset row, or
        # Qt left current on a row that's no longer selected) — fall back to
        # the first selected run, if any.
        return ids[0] if ids else None

    def selected_dataset_ids(self) -> list[str]:
        """Return the dataset ids of selected dataset (top-level) rows."""
        ids: list[str] = []
        for index in self._tree.selectionModel().selectedIndexes():
            # Dataset rows have no parent; run rows do.
            if not index.parent().isValid():
                did = index.data(Qt.ItemDataRole.UserRole)
                if did is not None:
                    ids.append(str(did))
        return ids

    def _on_selection(self, *_args: Any) -> None:
        run_ids = self.selected_run_ids()
        self.runs_selected.emit(run_ids)
        if not run_ids:
            dataset_ids = self.selected_dataset_ids()
            if len(dataset_ids) == 1:
                self.dataset_selected.emit(dataset_ids[0])


class ComparisonView(QWidget):
    """Renders one or more run summaries side by side."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._text)

    def show_runs(self, summaries: list[tuple[str, Any]]) -> None:
        """Render ``[(label, ResultSummary|None), ...]`` as a side-by-side table.

        Table layout / diff-marking is pure logic; see
        :func:`xpcsjax.gui.comparison.format_comparison` (Qt-free, unit-tested).
        """
        self._text.setPlainText(format_comparison(summaries))

    def rendered_text(self) -> str:
        """Return the rendered comparison text (inspection helper)."""
        return self._text.toPlainText()
