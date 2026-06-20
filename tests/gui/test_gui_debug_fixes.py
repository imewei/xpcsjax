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
