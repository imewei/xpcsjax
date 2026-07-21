"""Tests for RunController.on_cancel's confirm-before-cancel guard.

Regression coverage for the PR-review finding that this dialog result was
never asserted anywhere: a bug that inverted the Yes/No check (silently
restoring the misclick-discards-a-multi-minute-fit behavior this guard exists
to prevent) would have shipped with the suite green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def _window_with_queued_run(qtbot, monkeypatch, tmp_path: Path):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)
    win.add_dataset(str(cfg))
    win._on_run()
    run = win._project.datasets[0].runs[0]
    monkeypatch.setattr(win._sidebar, "current_run_id", lambda: run.run_id)
    return win, run


def test_cancel_confirmed_yes_proceeds(qtbot, monkeypatch, tmp_path: Path) -> None:
    """Answering Yes to the confirmation dialog actually cancels the run."""
    import xpcsjax.gui.views.main_window_support.run_controller as rc

    win, run = _window_with_queued_run(qtbot, monkeypatch, tmp_path)
    monkeypatch.setattr(
        rc.QMessageBox, "question", lambda *a, **k: rc.QMessageBox.StandardButton.Yes
    )
    cancelled: dict[str, str] = {}
    monkeypatch.setattr(win._queue, "cancel", lambda run_id: cancelled.setdefault("run_id", run_id))

    win._on_cancel()

    assert cancelled.get("run_id") == run.run_id


def test_cancel_declined_no_does_not_cancel(qtbot, monkeypatch, tmp_path: Path) -> None:
    """Answering No (or dismissing) the confirmation dialog must NOT cancel the run."""
    import xpcsjax.gui.views.main_window_support.run_controller as rc

    win, _run = _window_with_queued_run(qtbot, monkeypatch, tmp_path)
    monkeypatch.setattr(
        rc.QMessageBox, "question", lambda *a, **k: rc.QMessageBox.StandardButton.No
    )
    called = {"cancel": False}
    monkeypatch.setattr(win._queue, "cancel", lambda run_id: called.__setitem__("cancel", True))

    win._on_cancel()

    assert called["cancel"] is False


def test_cancel_no_selection_shows_message_not_queue_cancel(qtbot, monkeypatch) -> None:
    """With no run selected, on_cancel must not reach the confirmation/cancel path."""
    import xpcsjax.gui.views.main_window_support.run_controller as rc
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    monkeypatch.setattr(win._sidebar, "current_run_id", lambda: None)
    monkeypatch.setattr(rc.QMessageBox, "information", lambda *a, **k: None)
    asked = {"question": False}
    monkeypatch.setattr(
        rc.QMessageBox,
        "question",
        lambda *a, **k: asked.__setitem__("question", True) or rc.QMessageBox.StandardButton.Yes,
    )

    win._on_cancel()

    assert asked["question"] is False
