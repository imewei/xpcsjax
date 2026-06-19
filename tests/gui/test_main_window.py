"""pytest-qt smoke tests for MainWindow (logic-free view, controller-less constructor)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction, QCloseEvent  # noqa: E402

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402


def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_window_constructs_with_expected_actions(qtbot):
    win = _window(qtbot)
    names = {a.objectName() for a in win.findChildren(QAction)}
    assert {"action_open_config", "action_output_dir", "action_run", "action_cancel"} <= names


def test_status_and_log_slots_update_widgets(qtbot):
    win = _window(qtbot)
    # Drive via queue signals prefixed with a run_id
    run_id = "aabbccdd1234567890abcdef12345678"
    win._queue.run_status_changed.emit(run_id, "running")
    win._queue.log_received.emit(run_id, "INFO", "hello world")
    assert "running" in win.status_text()
    assert "hello world" in win.log_text()


def test_show_result_renders_summary(qtbot, tmp_path):
    win = _window(qtbot)
    run_id = "aabbccdd1234567890abcdef12345678"
    summary = ResultSummary(
        result_dir=tmp_path, success=True, convergence_status="converged",
        chi_squared=1.5, reduced_chi_squared=1.04, quality_flag="good", parameters={"D0": 1234.5},
    )
    # Simulate a finished run: first set it running (to set _active_run_id), then finish.
    win._queue.run_status_changed.emit(run_id, "running")
    win._queue.run_finished.emit(run_id, str(tmp_path), summary)
    assert "converged" in win.result_text()
    assert "1234.5" in win.result_text() or "D0" in win.result_text()


def test_repeated_runs_get_distinct_output_dirs(qtbot, tmp_path, monkeypatch):
    """Two runs for the SAME dataset must write to distinct dirs (no overwrite)."""
    win = _window(qtbot)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg))

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win._queue,
        "enqueue",
        lambda run_id, _config_path, output_dir: captured.append((run_id, output_dir)),
    )
    win._on_run()
    win._on_run()

    assert len(captured) == 2
    (rid1, out1), (rid2, out2) = captured
    assert rid1 != rid2
    assert out1 != out2  # per-run dirs — second run does not clobber the first's artifacts
    # The unique dir is the run_id-namespaced one, recorded on the FitRun BEFORE enqueue.
    runs = win._project.datasets[0].runs
    assert runs[0].result_dir == out1 and rid1 in out1
    assert runs[1].result_dir == out2 and rid2 in out2


def test_close_event_calls_queue_shutdown(qtbot, monkeypatch):
    win = _window(qtbot)
    called = {"shutdown": False}
    monkeypatch.setattr(win._queue, "shutdown", lambda: called.__setitem__("shutdown", True))
    win.closeEvent(QCloseEvent())
    assert called["shutdown"] is True
