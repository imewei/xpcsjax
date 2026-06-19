"""Project sidebar (datasets -> runs tree) + a side-by-side comparison view."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPlainTextEdit, QTreeView, QVBoxLayout, QWidget

from xpcsjax.gui.project.model import Project
from xpcsjax.gui.project.tree_model import ProjectTreeModel


class ProjectSidebar(QWidget):
    """A tree of datasets -> runs with multi-select."""

    runs_selected = Signal(list)  # list[str] of run_ids

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

    def set_project(self, project: Project) -> None:
        """Rebuild the tree from ``project``."""
        self._model.rebuild(project)
        self._tree.expandAll()

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
        """Return the focused run id, or None."""
        ids = self.selected_run_ids()
        return ids[0] if ids else None

    def _on_selection(self, *_args: Any) -> None:
        self.runs_selected.emit(self.selected_run_ids())


class ComparisonView(QWidget):
    """Renders one or more run summaries side by side."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._text)

    def show_runs(self, summaries: list[tuple[str, Any]]) -> None:
        """Render ``[(label, ResultSummary|None), ...]`` as a comparison block."""
        blocks: list[str] = []
        for label, summary in summaries:
            if summary is None:
                blocks.append(f"=== {label} ===\n(no result)")
                continue
            lines = [
                f"=== {label} ===",
                f"status:        {summary.convergence_status}",
                f"chi^2:         {summary.chi_squared}",
                f"reduced chi^2: {summary.reduced_chi_squared}",
                f"quality:       {summary.quality_flag}",
                "parameters:",
                *[f"  {name} = {value}" for name, value in summary.parameters.items()],
            ]
            blocks.append("\n".join(lines))
        self._text.setPlainText("\n\n".join(blocks))

    def rendered_text(self) -> str:
        """Return the rendered comparison text (inspection helper)."""
        return self._text.toPlainText()
