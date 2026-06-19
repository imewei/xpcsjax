"""pytest-qt smoke tests for MainWindow (logic-free view)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction, QCloseEvent  # noqa: E402

from xpcsjax.gui.controllers.fit_controller import FitController  # noqa: E402
from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402


def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    ctrl = FitController()
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    return win, ctrl


def test_window_constructs_with_expected_actions(qtbot):
    win, _ = _window(qtbot)
    names = {a.objectName() for a in win.findChildren(QAction)}
    assert {"action_open_config", "action_output_dir", "action_run", "action_cancel"} <= names


def test_status_and_log_slots_update_widgets(qtbot):
    win, ctrl = _window(qtbot)
    ctrl.status_changed.emit("running")
    ctrl.log_received.emit("INFO", "hello world")
    assert "running" in win.status_text()
    assert "hello world" in win.log_text()


def test_show_result_renders_summary(qtbot, tmp_path):
    win, ctrl = _window(qtbot)
    summary = ResultSummary(
        result_dir=tmp_path, success=True, convergence_status="converged",
        chi_squared=1.5, reduced_chi_squared=1.04, quality_flag="good", parameters={"D0": 1234.5},
    )
    ctrl.fit_finished.emit(summary)
    assert "converged" in win.result_text()
    assert "1234.5" in win.result_text() or "D0" in win.result_text()


def test_close_event_calls_controller_shutdown(qtbot, monkeypatch):
    win, ctrl = _window(qtbot)
    called = {"shutdown": False}
    monkeypatch.setattr(ctrl, "shutdown", lambda: called.__setitem__("shutdown", True))
    win.closeEvent(QCloseEvent())
    assert called["shutdown"] is True
