"""pytest-qt tests for app wiring + worker cleanup."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.app import build_workbench  # noqa: E402
from xpcsjax.gui.controllers.fit_controller import FitController  # noqa: E402
from xpcsjax.gui.views.main_window import MainWindow  # noqa: E402


def test_build_workbench_returns_wired_window(qtbot):
    window, controller = build_workbench()
    qtbot.addWidget(window)
    assert isinstance(window, MainWindow)
    assert isinstance(controller, FitController)
    # The controller drives the window: a status signal updates the status line.
    controller.status_changed.emit("running")
    assert "running" in window.status_text()


def test_close_triggers_shutdown(qtbot, monkeypatch):
    from PySide6.QtGui import QCloseEvent

    window, controller = build_workbench()
    qtbot.addWidget(window)
    called = {"n": 0}
    monkeypatch.setattr(controller, "shutdown", lambda: called.__setitem__("n", called["n"] + 1))
    window.closeEvent(QCloseEvent())
    assert called["n"] == 1
