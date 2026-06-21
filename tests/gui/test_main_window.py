"""pytest-qt smoke tests for MainWindow (logic-free view, controller-less constructor)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402
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
    # Operational actions live on the toolbar; project lifecycle on the File menu.
    assert {
        "action_create_config",
        "action_edit_config",
        "action_load_config",
        "action_run",
        "action_cancel",
        "action_export_figure",
        "action_create_project",
        "action_open_project",
        "action_save_project",
        "action_close_project",
    } <= names
    # The Output Dir override action was removed.
    assert "action_output_dir" not in names


def test_create_project_shows_folder_name_in_sidebar(qtbot, tmp_path):
    from PySide6.QtCore import Qt

    win = _window(qtbot)
    project_dir = tmp_path / "my_xpcs_project"
    win.create_project(project_dir)

    model = win._sidebar.model()
    assert model.headerData(0, Qt.Orientation.Horizontal) == "my_xpcs_project"


def test_close_project_clears_folder_name_from_sidebar(qtbot, tmp_path):
    from PySide6.QtCore import Qt

    win = _window(qtbot)
    win.create_project(tmp_path / "proj")
    win.close_project()

    model = win._sidebar.model()
    assert model.headerData(0, Qt.Orientation.Horizontal) == "Project"


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
        result_dir=tmp_path,
        success=True,
        convergence_status="converged",
        chi_squared=1.5,
        reduced_chi_squared=1.04,
        quality_flag="good",
        parameters={"D0": 1234.5},
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


class _FakeHandle(QObject):
    """Minimal WorkerHandle stand-in: never spawns a real process."""

    event = Signal(object)

    def __init__(self, job):
        super().__init__()
        self.job = job
        self._alive = False

    def start(self):
        self._alive = True

    def cancel(self):
        self._alive = False

    def is_running(self):
        return self._alive

    def shutdown(self):
        self._alive = False


def test_close_project_stops_active_and_pending_runs(qtbot, tmp_path):
    """Closing a project must not orphan queued/active fit workers (codex review).

    ``close_project()`` used to rebuild the Project and null ``_active_run_id``
    while leaving ``FitQueueController`` untouched: the worker kept running in
    its child process, now unreachable (terminal signals could no longer attach,
    logs were filtered out), still consuming RAM and writing artifacts under the
    old output dir. Closing the project must drain the queue.
    """
    from xpcsjax.gui.controllers.fit_queue import FitQueueController
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    # Swap in a fake-handle queue so no real worker process is spawned.
    win._queue = FitQueueController(max_concurrent=1, handle_factory=_FakeHandle)
    win._queue.enqueue(
        "run-active", str(tmp_path / "a.yaml"), str(tmp_path / "runs" / "run-active")
    )
    win._queue.enqueue(
        "run-pending", str(tmp_path / "b.yaml"), str(tmp_path / "runs" / "run-pending")
    )
    assert win._queue.active_count() == 1 and win._queue.pending_count() == 1

    win.close_project()

    assert win._queue.active_count() == 0, "active worker was orphaned by close_project()"
    assert win._queue.pending_count() == 0, "pending run survived close_project()"
