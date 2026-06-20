"""Integration tests for the workbench surfaces (post-redesign).

Asserts that MainWindow exposes:
  - A central results-only area (a QStackedWidget; no Data/Config/Fit tabs).
  - An Inspector QDockWidget on the right side.
  - A ``show_inspector(summary)`` method that passes the summary to InspectorDock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QDockWidget, QStackedWidget, QTabWidget  # noqa: E402

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


def test_no_data_config_fit_tabs(qtbot):
    """The redesign removed the Data/Config/Fit setup tabs entirely.

    The central area is results-only — there must be no center QTabWidget left
    holding the old setup tabs.
    """
    window, _ = build_workbench()
    qtbot.addWidget(window)

    assert window.findChild(QTabWidget, "center_tabs") is None
    # No widget in the central area should be tab-labelled Data/Config/Fit.
    for tabs in window.centralWidget().findChildren(QTabWidget):
        labels = {tabs.tabText(i) for i in range(tabs.count())}
        assert not ({"Data", "Config", "Fit"} & labels)


def test_central_area_is_results_stack(qtbot):
    """The central widget is the results QStackedWidget (text + per-phi grid)."""
    window, _ = build_workbench()
    qtbot.addWidget(window)

    central = window.centralWidget()
    assert isinstance(central, QStackedWidget)
    assert central is window._central_stack


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


def test_bottom_dock_is_log_only(qtbot):
    """The bottom 'Fitting Process' dock holds only the log (no SSR/chips/banners)."""
    from PySide6.QtWidgets import QPlainTextEdit

    window, _ = build_workbench()
    qtbot.addWidget(window)

    dock = window.findChild(QDockWidget, "dock_fitting_process")
    assert dock is not None, "dock_fitting_process QDockWidget not found"
    assert dock.widget() is window._log
    assert isinstance(dock.widget(), QPlainTextEdit)
