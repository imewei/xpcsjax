"""Integration tests for the workbench surfaces (Task 5).

Asserts that MainWindow exposes:
  - A center QTabWidget with Data / Config / Fit / Results tabs.
  - An Inspector QDockWidget on the right side.
  - A ``show_inspector(summary)`` method that passes the summary to InspectorDock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QDockWidget, QTabWidget  # noqa: E402

from xpcsjax.gui.app import build_workbench  # noqa: E402
from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402


def _summary() -> ResultSummary:
    return ResultSummary(
        result_dir=Path("."),
        success=True,
        convergence_status="converged",
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        quality_flag="good",
        parameters={"D0": 1234.5, "beta": 0.9},
        uncertainties={"D0": 12.0, "beta": 0.01},
        diagnostics={"hierarchical_active": True, "regularization_active": False},
    )


def test_center_tab_widget_has_four_tabs(qtbot):
    """MainWindow must have a center QTabWidget with Data/Config/Fit/Results tabs."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    # Find the center QTabWidget
    tab_widget = window.findChild(QTabWidget, "center_tabs")
    assert tab_widget is not None, "center_tabs QTabWidget not found"

    tab_names = [tab_widget.tabText(i) for i in range(tab_widget.count())]
    assert "Data" in tab_names, f"'Data' tab not found; tabs: {tab_names}"
    assert "Config" in tab_names, f"'Config' tab not found; tabs: {tab_names}"
    assert "Fit" in tab_names, f"'Fit' tab not found; tabs: {tab_names}"
    assert "Results" in tab_names, f"'Results' tab not found; tabs: {tab_names}"


def test_inspector_dock_exists(qtbot):
    """MainWindow must have an Inspector QDockWidget."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    dock = window.findChild(QDockWidget, "dock_inspector")
    assert dock is not None, "dock_inspector QDockWidget not found"


def test_show_inspector_populates_dock(qtbot):
    """show_inspector(summary) must populate the inspector dock (param_row_count >= 1)."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    summary = _summary()
    window.show_inspector(summary)

    # The inspector's row count should reflect the summary's parameters
    assert window._inspector.param_row_count() >= 1, (
        "InspectorDock.param_row_count() should be >= 1 after show_inspector()"
    )


def test_show_inspector_clears_on_none(qtbot):
    """show_inspector(None) must clear the inspector dock."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    # First populate
    window.show_inspector(_summary())
    assert window._inspector.param_row_count() >= 1

    # Then clear
    window.show_inspector(None)
    assert window._inspector.param_row_count() == 0


def test_results_area_inside_results_tab(qtbot):
    """The existing Results area (QStackedWidget with text+plots) must be inside
    the 'Results' tab, not as the direct central widget."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    tab_widget = window.findChild(QTabWidget, "center_tabs")
    assert tab_widget is not None

    # The central widget of the window should be the tab widget itself (or a
    # container wrapping it), not the bare QStackedWidget.
    central = window.centralWidget()
    # The central widget should contain the tab widget
    assert tab_widget.isAncestorOf(window._central_stack) or (
        central is tab_widget
    ), "Results area (central_stack) should live inside the center_tabs widget"
