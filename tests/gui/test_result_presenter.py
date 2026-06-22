"""Unit tests for ResultPresenter collaborator (show_result / show_error / show_inspector / show_result_with_bundle)."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402
from xpcsjax.gui.views.main_window import MainWindow  # noqa: E402
from xpcsjax.gui.views.main_window_support.result_presenter import (  # noqa: E402
    ResultPresenter,
)


def _window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _summary(tmp_path, label: str = "converged") -> ResultSummary:
    return ResultSummary(
        result_dir=tmp_path,
        success=True,
        convergence_status=label,
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        quality_flag="good",
        parameters={"D0": 42.0},
    )


# --- wiring / identity tests -------------------------------------------------


def test_result_presenter_is_qobject_parented_to_window(qtbot):
    """ResultPresenter is a QObject child of MainWindow."""
    win = _window(qtbot)
    assert isinstance(win._result_presenter, ResultPresenter)
    assert win._result_presenter.parent() is win


# --- show_result: text-only, does NOT switch to grid page --------------------


def test_show_result_text_only(qtbot, tmp_path):
    """show_result renders text and keeps the stack on page 0 (text-summary page)."""
    win = _window(qtbot)
    summary = _summary(tmp_path, "converged")
    win.show_result(summary)
    assert "converged" in win.result_text()
    assert win._central_stack.currentIndex() == 0


def test_show_result_contains_summary_fields(qtbot, tmp_path):
    """show_result writes chi^2, quality, and parameters into the text panel."""
    win = _window(qtbot)
    summary = _summary(tmp_path)
    win.show_result(summary)
    text = win.result_text()
    assert "42.0" in text or "D0" in text
    assert "good" in text


def test_show_result_none_summary(qtbot, tmp_path):
    """show_result with None writes the 'no result file' message."""
    win = _window(qtbot)
    win.show_result(None)
    assert "no result file" in win.result_text()
    assert win._central_stack.currentIndex() == 0


# --- show_error: text-only with FIT FAILED prefix ----------------------------


def test_show_error_renders_fit_failed(qtbot, tmp_path):
    """show_error writes 'FIT FAILED' and the message to the text panel."""
    win = _window(qtbot)
    win.show_error("boom")
    assert "FIT FAILED" in win.result_text()
    assert "boom" in win.result_text()


def test_show_error_does_not_switch_to_grid(qtbot, tmp_path):
    """show_error is text-only — stack stays on page 0."""
    win = _window(qtbot)
    win.show_error("some error")
    assert win._central_stack.currentIndex() == 0


# --- _show_result_with_bundle: grid routing via real bundle ------------------


def _write_bundle(tmp_path) -> None:
    """Write the artifact that load_viz_bundle reads: <result_dir>/plots/simulated_data/c2_fitted_data.npz."""
    import numpy as np

    sim = tmp_path / "plots" / "simulated_data"
    sim.mkdir(parents=True)
    exp = np.ones((2, 10, 10))
    model = np.ones((2, 10, 10))
    np.savez(
        sim / "c2_fitted_data.npz",
        c2_exp=exp,
        c2_fitted=model,
        residuals=exp - model,
        t1=np.arange(10.0),
        t2=np.arange(10.0),
        phi_angles=np.array([0.0, 45.0]),
    )


def test_show_result_with_bundle_uses_grid_when_bundle_found(qtbot, tmp_path):
    """_show_result_with_bundle switches to the grid page (index 1) when a bundle loads."""
    _write_bundle(tmp_path)

    win = _window(qtbot)
    summary = _summary(tmp_path)
    win._show_result_with_bundle(summary, str(tmp_path))
    assert win._central_stack.currentIndex() == 1


def test_show_result_with_bundle_falls_back_to_text_when_no_bundle(qtbot, tmp_path):
    """_show_result_with_bundle falls back to text (page 0) when result_dir has no bundle."""
    # tmp_path exists but contains no bundle file — load_viz_bundle raises.
    win = _window(qtbot)
    summary = _summary(tmp_path, "no_bundle_converged")
    win._show_result_with_bundle(summary, str(tmp_path))
    # Falls back to text page.
    assert win._central_stack.currentIndex() == 0
    assert "no_bundle_converged" in win.result_text()


def test_show_result_with_bundle_none_result_dir_falls_back(qtbot, tmp_path):
    """_show_result_with_bundle with result_dir=None forces the text fallback."""
    win = _window(qtbot)
    summary = _summary(tmp_path, "null_dir")
    win._show_result_with_bundle(summary, None)
    assert win._central_stack.currentIndex() == 0
    assert "null_dir" in win.result_text()


# --- show_inspector: delegates to the inspector dock -------------------------


def test_show_inspector_populates_inspector(qtbot, tmp_path):
    """show_inspector passes the summary to the inspector dock without error."""
    win = _window(qtbot)
    summary = _summary(tmp_path)
    # Should not raise; inspector receives the summary.
    win.show_inspector(summary)


def test_show_inspector_none_clears(qtbot):
    """show_inspector(None) clears the inspector without error."""
    win = _window(qtbot)
    win.show_inspector(None)
