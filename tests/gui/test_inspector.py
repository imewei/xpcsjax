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
