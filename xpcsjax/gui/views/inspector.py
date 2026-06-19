"""Inspector dock: parameters/uncertainties table + diagnostics tree.

JAX-free: display-only widget. All data is handed in as plain Python dicts/values
via show_summary(); no numerical computation happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from xpcsjax.gui.result_loader import ResultSummary


class InspectorDock(QWidget):
    """Read-only display of a fit result's parameters and diagnostics.

    Parameters
    ----------
    parent:
        Optional parent widget.

    Notes
    -----
    - ``show_summary(summary)`` populates the parameter table and diagnostics tree.
    - ``show_summary(None)`` clears both.
    - No JAX imports; no numerical logic.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Parameters table ---
        layout.addWidget(QLabel("Parameters"))
        self._param_table = QTableWidget(0, 3)
        self._param_table.setHorizontalHeaderLabels(["Name", "Value", "± Uncertainty"])
        self._param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._param_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._param_table.setAlternatingRowColors(True)
        layout.addWidget(self._param_table)

        # --- Diagnostics tree ---
        layout.addWidget(QLabel("Diagnostics"))
        self._diag_tree = QTreeWidget()
        self._diag_tree.setHeaderLabels(["Key", "Value"])
        self._diag_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._diag_tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_summary(self, summary: ResultSummary | None) -> None:
        """Populate the inspector from *summary*, or clear when *summary* is None."""
        self._clear()
        if summary is None:
            return
        self._populate_params(summary)
        self._populate_diagnostics(summary.diagnostics)

    def param_row_count(self) -> int:
        """Return the number of rows currently in the parameter table."""
        return self._param_table.rowCount()

    def diagnostics_row_count(self) -> int:
        """Return the number of top-level items in the diagnostics tree."""
        return self._diag_tree.topLevelItemCount()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        self._param_table.setRowCount(0)
        self._diag_tree.clear()

    def _populate_params(self, summary: ResultSummary) -> None:
        rows = list(summary.parameters.items())
        self._param_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            unc = summary.uncertainties.get(name)
            self._param_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
            unc_text = f"{unc:.4g}" if unc is not None else "—"
            self._param_table.setItem(row, 2, QTableWidgetItem(unc_text))

    def _populate_diagnostics(self, diag: dict) -> None:
        """Recursively walk *diag* and build a QTreeWidget."""
        self._diag_tree.clear()
        for key, value in diag.items():
            item = self._make_tree_item(str(key), value)
            self._diag_tree.addTopLevelItem(item)
        self._diag_tree.expandAll()

    def _make_tree_item(self, key: str, value: object) -> QTreeWidgetItem:
        """Return a QTreeWidgetItem for *key*/*value*, recursing into dicts."""
        if isinstance(value, dict):
            item = QTreeWidgetItem([key, ""])
            for child_key, child_val in value.items():
                item.addChild(self._make_tree_item(str(child_key), child_val))
        else:
            item = QTreeWidgetItem([key, str(value)])
        return item
