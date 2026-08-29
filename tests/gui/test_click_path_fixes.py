"""Regression tests for the click-path-audit findings (state-conflict bugs).

Each test pins one CLICK-PATH-NNN finding: a handler doing its stated job
while silently skipping/undoing a teardown or sync step a sibling handler is
trusted to also do.
"""

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (  # noqa: E402
    QItemSelectionModel,  # noqa: E402
    QObject,
    Signal,
)

from xpcsjax.gui.controllers.fit_queue import FitQueueController  # noqa: E402
from xpcsjax.gui.project.persist import _SCHEMA  # noqa: E402


def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


class _FakeHandle(QObject):
    """Minimal WorkerHandle stand-in: never spawns a real process (mirrors test_main_window)."""

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


# CLICK-PATH-001: fit failure never reached the central panel (show_error dead code).
def test_run_failed_renders_error_in_central_panel(qtbot, monkeypatch):
    import xpcsjax.gui.views.main_window as mw

    # show_failure() calls box.exec() — a real modal blocks headless
    # tests forever, so stub it (matches how other GUI tests avoid QMessageBox).
    monkeypatch.setattr(mw, "show_failure", lambda *a, **k: None)

    win = _window(qtbot)
    run_id = "aabbccdd1234567890abcdef12345678"
    win._queue.run_status_changed.emit(run_id, "running")
    win._queue.run_failed.emit(run_id, "boom: traceback text")

    assert win._central_stack.currentIndex() == 0
    assert "FIT FAILED" in win.result_text()
    assert "boom" in win.result_text()


# CLICK-PATH-001 edge case: a background failure for a DIFFERENT run must not
# clobber the panel pinned to the run the user is deliberately viewing.
def test_run_failed_does_not_clobber_a_different_pinned_view(qtbot, monkeypatch):
    import xpcsjax.gui.views.main_window as mw

    monkeypatch.setattr(mw, "show_failure", lambda *a, **k: None)

    win = _window(qtbot)
    viewed_run_id = "11112222333344445555666677778888"
    failing_run_id = "aabbccdd1234567890abcdef12345678"
    win._results.setPlainText("pinned view content")
    win._viewing_run_id = viewed_run_id  # user deliberately viewing a different run
    win._active_run_id = failing_run_id  # the run that's about to fail is the active one

    win._queue.run_failed.emit(failing_run_id, "boom: unrelated run failed")

    assert win.result_text() == "pinned view content", (
        "a different run's background failure clobbered the pinned view's panel"
    )


# CLICK-PATH-002: open_project_from skipped the teardown close_project does.
def test_open_project_resets_stale_run_state(qtbot, tmp_path):
    win = _window(qtbot)
    win._queue = FitQueueController(max_concurrent=1, handle_factory=_FakeHandle)
    win._queue.enqueue("stale-run", str(tmp_path / "a.yaml"), str(tmp_path / "runs" / "stale-run"))
    win._active_run_id = "stale-run"
    win._viewing_run_id = "stale-run"
    win._log.appendPlainText("stale log line")
    assert win._queue.active_count() == 1

    project_file = tmp_path / "proj.xpcsproj"
    project_file.write_text(json.dumps({"schema": _SCHEMA, "datasets": []}), encoding="utf-8")
    win.open_project_from(project_file)

    assert win._queue.active_count() == 0, "stale worker from discarded project kept running"
    assert win._active_run_id is None
    assert win._viewing_run_id is None
    assert "stale log line" not in win.log_text()


# CLICK-PATH-002 follow-up: a malformed .xpcsproj must raise BEFORE any teardown,
# so the current project (and its in-flight run) survives intact.
def test_open_project_raises_on_malformed_file_leaves_state_untouched(qtbot, tmp_path):
    win = _window(qtbot)
    win._queue = FitQueueController(max_concurrent=1, handle_factory=_FakeHandle)
    win._queue.enqueue("stale-run", str(tmp_path / "a.yaml"), str(tmp_path / "runs" / "stale-run"))
    win._active_run_id = "stale-run"
    win._viewing_run_id = "stale-run"
    win._log.appendPlainText("stale log line")

    bad_file = tmp_path / "bad.xpcsproj"
    bad_file.write_text("not json at all", encoding="utf-8")

    with pytest.raises(ValueError):
        win.open_project_from(bad_file)

    assert win._queue.active_count() == 1, "teardown ran before the malformed load failed"
    assert win._active_run_id == "stale-run"
    assert win._viewing_run_id == "stale-run"
    assert "stale log line" in win.log_text()


# CLICK-PATH-002 follow-up: the ValueError above must surface as a warning
# dialog through the real dialog-handler path, not an unhandled traceback.
def test_on_open_project_surfaces_malformed_file_as_warning_not_crash(qtbot, tmp_path, monkeypatch):
    import xpcsjax.gui.views.main_window_support.project_dialog_handler as pdh

    win = _window(qtbot)
    bad_file = tmp_path / "bad.xpcsproj"
    bad_file.write_text("not json at all", encoding="utf-8")

    monkeypatch.setattr(
        pdh.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad_file), ""))
    )
    warned = {"called": False}
    monkeypatch.setattr(
        pdh.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warned.__setitem__("called", True)),
    )

    win._dialog_handler.on_open_project()  # must not raise

    assert warned["called"], "a malformed .xpcsproj escaped on_open_project unhandled"


# CLICK-PATH-003: sidebar tree rebuild wiped the selection on every Run click.
def test_run_reselects_new_run_in_sidebar(qtbot, tmp_path, monkeypatch):
    win = _window(qtbot)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg))
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)

    win._on_run()

    new_run_id = win._project.datasets[0].runs[-1].run_id
    assert win._sidebar.current_run_id() == new_run_id, (
        "Run cleared the sidebar selection instead of selecting the new run "
        "(breaks immediate Cancel/Export Figure)"
    )


# CLICK-PATH-003 follow-up: the reselect above must NOT cascade into
# _on_runs_selected (QSignalBlocker) -- nobody actually clicked this run.
def test_run_reselect_does_not_pin_viewing_run_id(qtbot, tmp_path, monkeypatch):
    win = _window(qtbot)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg))
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)

    win._on_run()

    assert win._viewing_run_id is None, (
        "reselecting the new run in the sidebar re-fired runs_selected and pinned "
        "_viewing_run_id, even though nobody actually clicked it"
    )


# CLICK-PATH-003 (second trigger site): Load Config used to wipe the selection too.
def test_load_config_preserves_prior_sidebar_selection(qtbot, tmp_path, monkeypatch):
    win = _window(qtbot)
    cfg1 = tmp_path / "cfg1.yaml"
    cfg1.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg1))
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)
    win._on_run()
    run_id = win._project.datasets[0].runs[-1].run_id
    assert win._sidebar.current_run_id() == run_id

    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg2))

    assert win._sidebar.current_run_id() == run_id, (
        "Load Config's sidebar refresh dropped the previously selected run"
    )


# CLICK-PATH-004: selecting a dataset row never updated _active_dataset_id.
def test_dataset_row_selection_updates_active_dataset(qtbot, tmp_path):
    win = _window(qtbot)
    cfg1 = tmp_path / "cfg1.yaml"
    cfg1.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg1))
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg2))
    first_dataset_id = win._project.datasets[0].dataset_id
    # add_dataset's own auto-select already made the SECOND dataset active —
    # select the FIRST instead, so this only passes if the click actually fires.
    assert win._active_dataset_id != first_dataset_id

    model = win._sidebar.model()
    dataset_index = model.index(0, 0)  # first dataset's row (top-level, no parent)
    win._sidebar._tree.selectionModel().select(
        dataset_index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
    )

    assert win._active_dataset_id == first_dataset_id


# CLICK-PATH-005: Load Config silently retargeted Run while a run was active/viewed.
def test_add_dataset_does_not_retarget_while_run_active(qtbot, tmp_path):
    win = _window(qtbot)
    cfg1 = tmp_path / "cfg1.yaml"
    cfg1.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg1))
    first_dataset_id = win._active_dataset_id
    win._queue.run_status_changed.emit("some-run", "running")  # sets _active_run_id

    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg2))

    assert win._active_dataset_id == first_dataset_id, (
        "Load Config retargeted Run away from the dataset with an active run"
    )


# CLICK-PATH-006: no guard against double-enqueuing a run for the same dataset.
def test_on_run_blocks_duplicate_run_for_same_dataset(qtbot, tmp_path, monkeypatch):
    import xpcsjax.gui.views.main_window_support.run_controller as rc

    # The blocked-duplicate path pops an information dialog; stub it (matches
    # test_run_controller.py's convention) so headless tests don't block on it.
    monkeypatch.setattr(rc.QMessageBox, "information", lambda *a, **k: None)

    win = _window(qtbot)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg))
    enqueued = []
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: enqueued.append(a))

    win._on_run()
    win._on_run()

    assert len(enqueued) == 1, "a second Run click queued a duplicate run for the same dataset"
    assert len(win._project.datasets[0].runs) == 1
