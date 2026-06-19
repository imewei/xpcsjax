"""Regression tests for the 2026-06-19 GUI debug-audit fixes.

Each test pins one verified bug found by the codex/agy/Claude triangulated audit
of ``xpcsjax/gui/``. Grouped by the file the fix lives in.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import numpy as np  # noqa: E402

from xpcsjax.service.config import validate_config  # noqa: E402


# ----------------------------------------------------------------------------
# config_editor.py
# ----------------------------------------------------------------------------
def test_fresh_config_editor_loads_initial_template(qtbot):
    """codex#1: a freshly-opened ConfigEditor shows a populated form (set_mode in __init__).

    Without the fix ``_template`` stays {} and a first Validate would emit a
    config with empty parameter_names/values, launching a malformed worker.
    """
    from xpcsjax.gui.views.config_editor import ConfigEditor

    w = ConfigEditor()
    qtbot.addWidget(w)
    cfg = w.current_config()
    assert cfg.get("analysis_mode")  # a real mode, not empty
    assert cfg["initial_parameters"]["parameter_names"]  # form is populated


def test_scalar_initial_parameters_does_not_crash_form(qtbot):
    """agy#2: a scalar `initial_parameters` must not raise AttributeError on raw→form."""
    from xpcsjax.gui.views.config_editor import ConfigEditor

    w = ConfigEditor()
    qtbot.addWidget(w)
    w.toggle_raw(True)
    w._raw_edit.setPlainText("analysis_mode: laminar_flow\ninitial_parameters: invalid\n")
    # Toggling raw off rebuilds the form from the malformed YAML — must not raise.
    w.toggle_raw(False)
    assert w._param_names == []  # treated as empty, no crash


def test_raw_yaml_mode_edit_is_preserved(qtbot):
    """agy#7: editing analysis_mode in raw YAML must survive the raw→form toggle."""
    from xpcsjax.gui.views.config_editor import ConfigEditor

    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("static_isotropic")
    w.toggle_raw(True)
    w._raw_edit.setPlainText(
        "analysis_mode: laminar_flow\n"
        "initial_parameters:\n  parameter_names: [D0]\n  values: [100.0]\n"
    )
    w.toggle_raw(False)
    # The combo (and therefore current_config) must reflect the raw-edited mode,
    # not silently revert to static_isotropic.
    assert w._mode_combo.currentText() == "laminar_flow"
    assert w.current_config()["analysis_mode"] == "laminar_flow"


# ----------------------------------------------------------------------------
# service/config.py
# ----------------------------------------------------------------------------
def test_validate_config_scalar_initial_parameters_reports_not_crashes():
    """agy#2 (service side): scalar initial_parameters yields an error, not AttributeError."""
    rep = validate_config({"analysis_mode": "laminar_flow", "initial_parameters": "invalid"})
    assert rep.ok is False
    assert any("mapping" in e for e in rep.errors)


# ----------------------------------------------------------------------------
# plots_view.py
# ----------------------------------------------------------------------------
def test_map_views_clear_removes_image(qtbot):
    """codex#6: clear_map() drops a displayed image so stale data is not retained."""
    from xpcsjax.gui.views.plots_view import ResidualMapView, TwoTimeMapView

    for view_cls in (TwoTimeMapView, ResidualMapView):
        v = view_cls()
        qtbot.addWidget(v)
        v.show_map(np.eye(8))
        assert v.has_image()
        v.clear_map()
        assert not v.has_image()


def test_set_bundle_none_clears_all_views(qtbot):
    """codex#6: set_bundle(None) clears every sub-view (no lingering prior-run plots)."""
    from types import SimpleNamespace

    from xpcsjax.gui.views.plots_view import ResultPlots

    rp = ResultPlots()
    qtbot.addWidget(rp)
    bundle = SimpleNamespace(
        exp_c2=np.ones((2, 8, 8)),
        residuals=np.zeros((2, 8, 8)),
        model_c2=None,
        phi_angles=None,
    )
    rp.set_bundle(bundle)
    assert rp.two_time().has_image()
    rp.set_bundle(None)
    assert not rp.two_time().has_image()


def test_residual_cleared_when_absent(qtbot):
    """codex#6: switching to a bundle with residuals=None clears the prior residual."""
    from types import SimpleNamespace

    from xpcsjax.gui.views.plots_view import ResultPlots

    rp = ResultPlots()
    qtbot.addWidget(rp)
    with_resid = SimpleNamespace(
        exp_c2=np.ones((1, 8, 8)), residuals=np.zeros((1, 8, 8)), model_c2=None, phi_angles=None
    )
    rp.set_bundle(with_resid)
    assert rp._residual.has_image()
    without_resid = SimpleNamespace(
        exp_c2=np.ones((1, 8, 8)), residuals=None, model_c2=None, phi_angles=None
    )
    rp.set_bundle(without_resid)
    assert not rp._residual.has_image()


# ----------------------------------------------------------------------------
# data_panel.py
# ----------------------------------------------------------------------------
def test_data_panel_non_hdf5_file_no_crash(qtbot, tmp_path):
    """codex#5: a non-HDF5 file surfaces a recoverable error, never an uncaught OSError."""
    from xpcsjax.gui.views.data_panel import DataPanel

    bad = tmp_path / "not_really.h5"
    bad.write_text("this is plain text, not HDF5", encoding="utf-8")
    panel = DataPanel()
    qtbot.addWidget(panel)
    panel.load(str(bad))  # must not raise
    assert panel.last_error() is not None
    assert panel.metadata_tree().topLevelItemCount() == 0
    assert not panel.preview_view().has_image()


# ----------------------------------------------------------------------------
# controllers/fit_queue.py
# ----------------------------------------------------------------------------
def test_cancel_race_finished_stays_cancelled(qtbot, tmp_path):
    """agy#4: a Finished arriving after cancel() must keep status 'cancelled', not flip to 'done'."""
    from PySide6.QtCore import QObject, Signal

    from xpcsjax.gui.controllers.fit_queue import FitQueueController
    from xpcsjax.service.events import Finished

    class _FakeHandle(QObject):
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

    q = FitQueueController(max_concurrent=1, handle_factory=_FakeHandle)
    statuses, finished = [], []
    q.run_status_changed.connect(lambda rid, st: statuses.append((rid, st)))
    q.run_finished.connect(lambda rid, path, summ: finished.append(rid))

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    q.enqueue("r1", "a.yaml", str(run_dir))
    h = q._handles["r1"]
    h._alive = False  # worker already died; its Finished is in flight, handle not yet popped
    q.cancel("r1")  # elif race branch: mark cancelled + delete output dir
    assert ("r1", "cancelled") in statuses

    h.event.emit(Finished(run_id="r1", seq=9, result_path=str(run_dir)))
    assert ("r1", "done") not in statuses  # did not flip back to done
    assert finished == []  # no run_finished for a cancelled run
    assert "r1" not in q._handles  # slot freed


# ----------------------------------------------------------------------------
# ipc/handle.py
# ----------------------------------------------------------------------------
def test_worker_handle_shutdown_reaps_process(qtbot):
    """codex#3: shutdown() joins+closes the worker process and closes the queue."""
    from tests.gui import ipc_fakes  # noqa: F401 — ensures importability of spawn target
    from xpcsjax.gui.ipc.handle import WorkerHandle
    from xpcsjax.gui.ipc.job import FitJob

    job = FitJob(run_id="r1", config_path="a.yaml", output_dir="/tmp/out")
    handle = WorkerHandle(job)
    # Patch the spawn target to a quick, importable terminal-emitter.
    import xpcsjax.gui.ipc.handle as handle_mod

    orig = handle_mod.run_worker
    handle_mod.run_worker = ipc_fakes.emit_started_then_finished
    try:
        handle.start()
        # Wait for the child to exit on its own.
        handle._proc.join(timeout=10)
        handle.shutdown()
    finally:
        handle_mod.run_worker = orig
    # After reaping, the process handle and queue are released.
    assert handle._proc is None
    assert handle._queue is None


# ----------------------------------------------------------------------------
# views/main_window.py
# ----------------------------------------------------------------------------
def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_temp_config_not_unlinked_mid_session(qtbot, tmp_path, monkeypatch):
    """codex#2/Claude#1: a second config_ready must NOT delete the first temp config.

    The first run's worker still opens it by path; eagerly unlinking it on the
    next Validate yanked the file out from under an active/pending worker.
    """
    win = _window(qtbot)
    # Stop the queue from actually spawning workers.
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)
    cfg = {
        "analysis_mode": "static_isotropic",
        "initial_parameters": {"parameter_names": ["D0"], "values": [100.0]},
    }
    win._on_config_ready(dict(cfg))
    first_paths = list(win._dataset_temp_paths.values())
    assert len(first_paths) == 1
    from pathlib import Path

    assert Path(first_paths[0]).exists()
    win._on_config_ready(dict(cfg))  # second validate
    # The first temp file must still exist (not eagerly unlinked).
    assert Path(first_paths[0]).exists()
    assert len(win._dataset_temp_paths) == 2


def test_close_event_deletes_temp_configs(qtbot, monkeypatch):
    """agy#5/Claude#6: closeEvent unlinks every session temp config."""
    from pathlib import Path

    from PySide6.QtGui import QCloseEvent

    win = _window(qtbot)
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: None)
    cfg = {
        "analysis_mode": "static_isotropic",
        "initial_parameters": {"parameter_names": ["D0"], "values": [100.0]},
    }
    win._on_config_ready(dict(cfg))
    paths = [Path(p) for p in win._dataset_temp_paths.values()]
    assert all(p.exists() for p in paths)
    win.closeEvent(QCloseEvent())
    assert all(not p.exists() for p in paths)
    assert win._dataset_temp_paths == {}


def test_open_project_sets_active_dataset(qtbot, tmp_path):
    """codex#4: after Open Project, Run works (an active dataset is established)."""
    win = _window(qtbot)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    win.add_dataset(str(cfg))
    proj_path = tmp_path / "s.xpcsproj"
    win.save_project_to(proj_path)
    # Wipe active state, then reload.
    win._active_dataset_id = None
    win.open_project_from(proj_path)
    assert win._active_dataset_id is not None  # Run would no longer say "pick a config first"


def test_expand_path_resolves_env_and_user(tmp_path, monkeypatch):
    """codex#6/agy#6: the dead-path check expands ${ENV}/~ before testing existence.

    Without expansion an absolute config/result path that embeds a shell variable
    (which survives the .xpcsproj round-trip) is mis-flagged 'missing' on load.
    """
    from xpcsjax.gui.views.main_window import _expand_path

    real = tmp_path / "sub" / "real.yaml"
    real.parent.mkdir()
    real.write_text("x", encoding="utf-8")
    monkeypatch.setenv("XPCSJAX_TEST_DIR", str(tmp_path / "sub"))
    expanded = _expand_path("${XPCSJAX_TEST_DIR}/real.yaml")
    assert expanded == real
    assert expanded.exists()
    # ~ is expanded too (no-op-safe on a plain absolute path).
    assert str(_expand_path("~/foo")).startswith(str(__import__("pathlib").Path.home()))
    assert _expand_path(str(real)) == real  # plain absolute path is unchanged


def test_banners_and_chips_cleared_on_run_switch(qtbot):
    """Claude#3: switching the active run clears the prior run's banners and lit chips."""
    win = _window(qtbot)
    r1 = "a" * 32
    r2 = "b" * 32
    win._queue.run_status_changed.emit(r1, "running")
    win._queue.banner_received.emit(r1, "ANTI-DEGENERACY: Layer 2", "info")
    win._queue.layer_status_received.emit(r1, {"L1": True, "L2": True})
    assert win._banners.count() == 1
    assert win._chips.active_layers()
    # A new run becomes active.
    win._queue.run_status_changed.emit(r2, "running")
    assert win._banners.count() == 0  # prior run's banners cleared
    assert not win._chips.active_layers()  # prior run's lit chips cleared
