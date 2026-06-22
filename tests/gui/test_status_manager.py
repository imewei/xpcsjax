"""Unit tests for StatusManager collaborator (set_status / append_log)."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.views.main_window import MainWindow  # noqa: E402
from xpcsjax.gui.views.main_window_support.status_manager import (  # noqa: E402
    StatusManager,
)


def test_status_manager_sets_status_and_appends_log(qtbot):
    """StatusManager routes set_status/append_log through to the MainWindow widgets."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.set_status("ready")
    assert win.status_text() == "ready"  # status_text() is a METHOD, delegates via StatusManager
    win.append_log("INFO", "hello")
    assert "hello" in win.log_text()  # log_text() is a METHOD


def test_status_manager_is_qobject_parented_to_window(qtbot):
    """StatusManager is a QObject child of MainWindow."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert isinstance(win._status_manager, StatusManager)
    assert win._status_manager.parent() is win


def test_status_manager_overwrite(qtbot):
    """Calling set_status twice replaces the previous status."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.set_status("first")
    win.set_status("second")
    assert win.status_text() == "second"


def test_append_log_multiple_lines(qtbot):
    """Multiple append_log calls accumulate in the log widget."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.append_log("INFO", "line one")
    win.append_log("WARNING", "line two")
    text = win.log_text()
    assert "line one" in text
    assert "line two" in text
