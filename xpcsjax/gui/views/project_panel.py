"""Project sidebar (datasets -> runs tree) + a side-by-side comparison view."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import QPlainTextEdit, QTreeView, QVBoxLayout, QWidget

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
        """Select *run_id*'s row in the tree, if it still exists."""
        index = self._model.index_for_run(run_id)
        if index is not None:
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
        """Return the focused run id, or None."""
        ids = self.selected_run_ids()
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
        """Render ``[(label, ResultSummary|None), ...]`` as a real side-by-side table.

        Runs are columns, not stacked text blocks, so values line up for direct
        comparison. Any field row where the runs' values disagree is prefixed
        with ``≠`` — a "side-by-side" view that never marks what differs was not
        actually doing the job of a comparison tool.
        """
        if not summaries:
            self._text.setPlainText("")
            return

        labels = [label for label, _ in summaries]
        col_w = max(12, max(len(label) for label in labels) + 2)
        label_w = 16

        def row(field_label: str, values: list[str | None]) -> str:
            rendered = [v if v is not None else "—" for v in values]
            present = {v for v in rendered if v != "—"}
            prefix = "≠ " if len(present) > 1 else "  "
            cells = "".join(v.ljust(col_w) for v in rendered)
            return f"{prefix}{field_label.ljust(label_w - 2)}{cells}"

        def fmt(x: float | None) -> str | None:
            """Format a possibly-``None`` numeric field.

            chi_squared/reduced_chi_squared can themselves be ``None`` on an
            incomplete/older result — ``f"{None:.6g}"`` raises, so the
            summary-presence check above isn't enough on its own.
            """
            return f"{x:.6g}" if x is not None else None

        lines = [" " * label_w + "".join(label.ljust(col_w) for label in labels)]
        lines.append("-" * len(lines[0]))
        lines.append(row("status", [s.convergence_status if s else None for _, s in summaries]))
        lines.append(row("chi^2", [fmt(s.chi_squared) if s else None for _, s in summaries]))
        lines.append(
            row(
                "reduced chi^2",
                [fmt(s.reduced_chi_squared) if s else None for _, s in summaries],
            )
        )
        lines.append(row("quality", [s.quality_flag if s else None for _, s in summaries]))

        # Union of parameter names across present summaries, first-seen order.
        param_names: list[str] = []
        seen: set[str] = set()
        for _, summary in summaries:
            if summary is None:
                continue
            for name in summary.parameters:
                if name not in seen:
                    seen.add(name)
                    param_names.append(name)
        if param_names:
            lines.append("")
            lines.append("parameters:")
            for name in param_names:
                values = [
                    f"{s.parameters[name]:.6g}" if s is not None and name in s.parameters else None
                    for _, s in summaries
                ]
                lines.append(row(name, values))

        missing = [label for label, s in summaries if s is None]
        if missing:
            lines.append("")
            lines.extend(f"{label}: no result" for label in missing)

        self._text.setPlainText("\n".join(lines))

    def rendered_text(self) -> str:
        """Return the rendered comparison text (inspection helper)."""
        return self._text.toPlainText()
