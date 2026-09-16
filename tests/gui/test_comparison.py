"""Plain (no Qt) unit tests for xpcsjax.gui.comparison.format_comparison.

Extracted from ComparisonView.show_runs (tests/gui/test_project_panel.py
still exercises it end-to-end through the QPlainTextEdit widget); these
tests pin the same behavior without needing a QApplication/qtbot.
"""

from xpcsjax.gui.comparison import format_comparison
from xpcsjax.gui.result_loader import ResultSummary


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


def test_empty_summaries_render_empty_string():
    assert format_comparison([]) == ""


def test_shows_two_runs():
    text = format_comparison([("run A", _summary(1.0)), ("run B", _summary(2.0))])
    assert "run A" in text and "run B" in text
    assert "D0" in text
    assert "converged" in text


def test_tolerates_missing_summary():
    text = format_comparison([("run A", _summary(1.0)), ("run B", None)])
    assert "run B: no result" in text


def test_tolerates_none_chi_squared():
    """chi_squared=None (incomplete/older result) must not crash the render --
    regression for a TypeError in f"{None:.6g}"."""
    partial = ResultSummary(
        result_dir=".",
        success=False,
        convergence_status="unknown",
        chi_squared=None,
        reduced_chi_squared=None,
        quality_flag="unknown",
        parameters={},
    )
    text = format_comparison([("run A", _summary(1.0)), ("run B", partial)])
    assert "run A" in text and "run B" in text


def test_marks_differing_values_with_diff_marker():
    text = format_comparison([("run A", _summary(1.0)), ("run B", _summary(2.0))])
    # chi^2 differs (1.0 vs 2.0) -> "≠" prefix.
    assert any(line.startswith("≠") and "chi^2" in line for line in text.splitlines())


def test_nan_parameter_renders_distinct_from_absent():
    """A reported-but-non-finite parameter renders 'NaN', not the '—' sentinel
    used for a run that never reported the parameter at all."""
    diverged = ResultSummary(
        result_dir=".",
        success=False,
        convergence_status="failed",
        chi_squared=99.0,
        reduced_chi_squared=99.0,
        quality_flag="poor",
        parameters={"D0": None},  # reported, non-finite
    )
    text = format_comparison([("run A", _summary(1.0)), ("run B", diverged)])
    assert "NaN" in text
    assert "—" not in text.split("D0")[1].split("\n")[0]  # D0 row has no "absent" cell
