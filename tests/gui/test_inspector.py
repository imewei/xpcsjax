"""Tests for InspectorDock (parameters/uncertainties/diagnostics view)."""

import pytest

pytest.importorskip("PySide6")

from pathlib import Path  # noqa: E402

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402
from xpcsjax.gui.views.inspector import InspectorDock  # noqa: E402


def _summary():
    return ResultSummary(
        result_dir=Path("."),
        success=True,
        convergence_status="converged",
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        quality_flag="good",
        parameters={"D0": 1234.5},
        uncertainties={"D0": 12.0},
        diagnostics={"hierarchical_active": True},
    )


def test_inspector_renders_params_and_diagnostics(qtbot):
    w = InspectorDock()
    qtbot.addWidget(w)
    w.show_summary(_summary())
    assert w.param_row_count() == 1
    assert w.diagnostics_row_count() >= 1


def test_inspector_clears_on_none(qtbot):
    w = InspectorDock()
    qtbot.addWidget(w)
    w.show_summary(_summary())
    w.show_summary(None)
    assert w.param_row_count() == 0


def test_inspector_renders_nan_parameter_without_crash(qtbot):
    """A diverged fit's non-finite parameter (persisted as None) must render as
    a visible "NaN" row, not crash on the f"{value:.6g}" format spec and not
    silently disappear from the table (regression for the loader's
    dict[str, float | None] widening)."""
    summary = ResultSummary(
        result_dir=Path("."),
        success=False,
        convergence_status="diverged",
        chi_squared=None,
        reduced_chi_squared=None,
        quality_flag="poor",
        parameters={"D0": 1234.5, "alpha": None},
        uncertainties={"D0": 12.0, "alpha": None},
        diagnostics={},
    )
    w = InspectorDock()
    qtbot.addWidget(w)
    w.show_summary(summary)
    assert w.param_row_count() == 2
    value_cells = {
        w._param_table.item(row, 0).text(): w._param_table.item(row, 1).text()
        for row in range(w.param_row_count())
    }
    assert value_cells["alpha"] == "NaN"
    assert value_cells["D0"] == "1234.5"
