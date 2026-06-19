"""pytest-qt tests for app wiring + worker cleanup."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.app import build_workbench  # noqa: E402
from xpcsjax.gui.controllers.fit_queue import FitQueueController  # noqa: E402
from xpcsjax.gui.views.main_window import MainWindow  # noqa: E402


def test_build_workbench_returns_wired_window(qtbot):
    window, queue = build_workbench()
    qtbot.addWidget(window)
    assert isinstance(window, MainWindow)
    assert isinstance(queue, FitQueueController)
    # The queue drives the window: a run_status_changed signal updates the status line.
    run_id = "aabbccdd1234567890abcdef12345678"
    queue.run_status_changed.emit(run_id, "running")
    assert "running" in window.status_text()


def test_close_triggers_shutdown(qtbot, monkeypatch):
    from PySide6.QtGui import QCloseEvent

    window, queue = build_workbench()
    qtbot.addWidget(window)
    called = {"n": 0}
    monkeypatch.setattr(queue, "shutdown", lambda: called.__setitem__("n", called["n"] + 1))
    window.closeEvent(QCloseEvent())
    assert called["n"] == 1
