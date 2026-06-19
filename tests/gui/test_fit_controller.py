"""pytest-qt tests for FitController event routing (fake in-process handle)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402
from xpcsjax.service.events import Died, Failed, Finished, LogLine, Started  # noqa: E402


class _FakeHandle(QObject):
    """In-process stand-in for WorkerHandle (no spawn, no JAX)."""

    event = Signal(object)

    def __init__(self, job):
        super().__init__()
        self.job = job
        self.started = False
        self.cancelled = False
        self.joined = False
        self._alive = False

    def start(self):
        self.started = True
        self._alive = True

    def cancel(self):
        self.cancelled = True
        self._alive = False

    def is_running(self):
        return self._alive

    def shutdown(self):
        self.joined = True
        self._alive = False


def _make_controller():
    from xpcsjax.gui.controllers.fit_controller import FitController

    return FitController(handle_factory=_FakeHandle)


def test_run_starts_handle_with_a_job(tmp_path):
    ctrl = _make_controller()
    ctrl.run("cfg.yaml", tmp_path)
    handle = ctrl._handle  # the injected fake
    assert handle.started is True
    assert handle.job.config_path == "cfg.yaml"
    assert handle.job.output_dir == str(tmp_path)
    assert handle.job.run_id  # non-empty unique id


def test_log_and_status_events_emit_signals(qtbot, tmp_path):
    ctrl = _make_controller()
    statuses, logs = [], []
    ctrl.status_changed.connect(statuses.append)
    ctrl.log_received.connect(lambda lvl, msg: logs.append((lvl, msg)))
    ctrl.run("cfg.yaml", tmp_path)
    ctrl._handle.event.emit(Started(run_id="r", seq=1, mode="m", settings_summary="s"))
    ctrl._handle.event.emit(LogLine(run_id="r", seq=2, level="INFO", msg="hello"))
    assert "running" in statuses
    assert ("INFO", "hello") in logs


def test_finished_loads_summary_and_emits(qtbot, tmp_path):
    import json

    (tmp_path / "nlsq_result.json").write_text(
        json.dumps({"metadata": {"success": True, "convergence_status": "converged",
                                 "chi_squared": 1.0, "reduced_chi_squared": 1.0, "quality_flag": "good"},
                    "parameters": {}}),
        encoding="utf-8",
    )
    ctrl = _make_controller()
    received = []
    ctrl.fit_finished.connect(received.append)
    ctrl.run("cfg.yaml", tmp_path)
    ctrl._handle.event.emit(Finished(run_id="r", seq=9, result_path=str(tmp_path)))
    assert len(received) == 1 and isinstance(received[0], ResultSummary)
    assert received[0].success is True


def test_failed_event_emits_fit_failed(qtbot, tmp_path):
    ctrl = _make_controller()
    errors = []
    ctrl.fit_failed.connect(errors.append)
    ctrl.run("cfg.yaml", tmp_path)
    ctrl._handle.event.emit(Failed(run_id="r", seq=3, traceback="boom"))
    assert any("boom" in e for e in errors)


def test_died_event_emits_oom_hint(qtbot, tmp_path):
    # A worker emits exactly ONE terminal; after it, the controller disconnects
    # the handle — so Failed and Died are exercised on separate fresh controllers.
    ctrl = _make_controller()
    errors = []
    ctrl.fit_failed.connect(errors.append)
    ctrl.run("cfg.yaml", tmp_path)
    ctrl._handle.event.emit(Died(run_id="r", seq=4, exit_code=-9, signal=9))
    assert any("out of memory" in e.lower() for e in errors)


def test_cancel_and_shutdown_call_handle(tmp_path):
    ctrl = _make_controller()
    ctrl.run("cfg.yaml", tmp_path)
    ctrl.cancel()
    assert ctrl._handle.cancelled is True
    # shutdown() joins the reader QThread even after the worker is no longer running
    ctrl.shutdown()
    assert ctrl._handle.joined is True
    FitController = type(ctrl)
    FitController(handle_factory=_FakeHandle).shutdown()  # no active handle -> no error


def test_iteration_layerstatus_banner_routing(qtbot, tmp_path):
    from xpcsjax.service.events import Banner, BannerKind, Iteration, LayerStatus

    ctrl = _make_controller()
    iters, layers, banners = [], [], []
    ctrl.iteration_received.connect(lambda n, ssr: iters.append((n, ssr)))
    ctrl.layer_status_received.connect(layers.append)
    ctrl.banner_received.connect(lambda text, kind: banners.append((text, kind)))
    ctrl.run("cfg.yaml", tmp_path)
    ctrl._handle.event.emit(Iteration(run_id="r", seq=1, n=3, ssr=12.0, chi2=1.1))
    ctrl._handle.event.emit(LayerStatus(run_id="r", seq=2, layers={"L1": True, "L2": False}, mode="laminar_flow"))
    ctrl._handle.event.emit(Banner(run_id="r", seq=3, text="ANTI-DEGENERACY: Layer 2", kind=BannerKind.INFO))
    assert iters == [(3, 12.0)]
    assert layers == [{"L1": True, "L2": False}]
    assert banners == [("ANTI-DEGENERACY: Layer 2", "info")]
