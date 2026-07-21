"""pytest-qt tests for the project sidebar + comparison view."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.project.model import Project  # noqa: E402
from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402
from xpcsjax.gui.views.project_panel import ComparisonView, ProjectSidebar  # noqa: E402


def _summary(chi2):
    return ResultSummary(
        result_dir=".",
        success=True,
        convergence_status="converged",
        chi_squared=chi2,
        reduced_chi_squared=chi2,
        quality_flag="good",
        parameters={"D0": 100.0 + chi2},
    )


def test_sidebar_reflects_project(qtbot):
    p = Project()
    d = p.add_dataset("a.yaml", label="DS-A")
    p.add_run(d.dataset_id)
    sb = ProjectSidebar()
    qtbot.addWidget(sb)
    sb.set_project(p)
    assert sb.model().rowCount() == 1


def test_comparison_view_shows_two_runs(qtbot):
    cv = ComparisonView()
    qtbot.addWidget(cv)
    cv.show_runs([("run A", _summary(1.0)), ("run B", _summary(2.0))])
    text = cv.rendered_text()
    assert "run A" in text and "run B" in text
    assert "D0" in text
    assert "converged" in text


def test_comparison_view_tolerates_missing_summary(qtbot):
    cv = ComparisonView()
    qtbot.addWidget(cv)
    cv.show_runs([("run A", _summary(1.0)), ("run B", None)])
    assert "run B" in cv.rendered_text()  # rendered as "no result"


def test_comparison_view_tolerates_none_chi_squared(qtbot):
    """A present summary with chi_squared=None (incomplete/older result) must
    not crash the render — regression for a TypeError in f"{None:.6g}"."""
    cv = ComparisonView()
    qtbot.addWidget(cv)
    partial = ResultSummary(
        result_dir=".",
        success=False,
        convergence_status="unknown",
        chi_squared=None,
        reduced_chi_squared=None,
        quality_flag="unknown",
        parameters={},
    )
    cv.show_runs([("run A", _summary(1.0)), ("run B", partial)])
    text = cv.rendered_text()
    assert "run A" in text and "run B" in text


def test_comparison_view_marks_differing_values(qtbot):
    """A field where the two runs disagree is prefixed with the diff marker;
    a field where they agree is not."""
    cv = ComparisonView()
    qtbot.addWidget(cv)
    cv.show_runs([("run A", _summary(1.0)), ("run B", _summary(2.0))])
    lines = cv.rendered_text().splitlines()
    chi2_line = next(line for line in lines if "chi^2" in line and "reduced" not in line)
    d0_line = next(line for line in lines if "D0" in line)
    status_line = next(line for line in lines if "status" in line)
    assert chi2_line.startswith("≠")  # chi_squared differs (1.0 vs 2.0)
    assert d0_line.startswith("≠")  # D0 differs (101.0 vs 102.0)
    assert status_line.startswith(" ")  # both "converged" — not flagged
    assert not status_line.startswith("≠")


def test_sidebar_selected_run_ids_returns_run_rows_only(qtbot):
    from PySide6.QtCore import QItemSelectionModel

    p = Project()
    d = p.add_dataset("a.yaml")
    r = p.add_run(d.dataset_id)
    sb = ProjectSidebar()
    qtbot.addWidget(sb)
    sb.set_project(p)
    # Select the run row (child of the dataset row); dataset rows are excluded.
    run_index = sb.model().item(0).child(0).index()
    sb._tree.selectionModel().select(run_index, QItemSelectionModel.SelectionFlag.Select)
    assert sb.selected_run_ids() == [r.run_id]
