"""pytest-qt tests for the bounded-concurrency fit queue (fake handles)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

from xpcsjax.service.events import (  # noqa: E402
    Banner,
    BannerKind,
    Finished,
    Iteration,
    LayerStatus,
    LogLine,
    Started,
)


class _FakeHandle(QObject):
    event = Signal(object)

    def __init__(self, job):
        super().__init__()
        self.job = job
        self._alive = False
        self.cancelled = False
        self.joined = False

    def start(self):
        self._alive = True

    def cancel(self):
        self.cancelled = True
        self._alive = False

    def is_running(self):
        return self._alive

    def shutdown(self):
        self.joined = True
        self._alive = False

    def finish(self, result_path=""):
        # Test helper: emit a terminal event and go not-running.
        self._alive = False
        self.event.emit(Finished(run_id=self.job.run_id, seq=9, result_path=result_path))


def _queue(max_concurrent=1):
    from xpcsjax.gui.controllers.fit_queue import FitQueueController

    return FitQueueController(max_concurrent=max_concurrent, handle_factory=_FakeHandle)


def test_bounded_concurrency_runs_one_at_a_time(qtbot, tmp_path):
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(tmp_path))
    q.enqueue("r2", "b.yaml", str(tmp_path))
    assert q.active_count() == 1 and q.pending_count() == 1
    # finish r1 -> r2 starts
    q._handles["r1"].finish(str(tmp_path))
    assert q.active_count() == 1 and q.pending_count() == 0


def test_status_and_finished_signals(qtbot, tmp_path):
    q = _queue(max_concurrent=2)
    statuses, finished = [], []
    q.run_status_changed.connect(lambda rid, st: statuses.append((rid, st)))
    q.run_finished.connect(lambda rid, path, summ: finished.append((rid, path)))
    q.enqueue("r1", "a.yaml", str(tmp_path))
    # Cold-spawn UX (spec §4 F10): "starting" precedes "running" (running arrives on Started).
    assert ("r1", "starting") in statuses
    assert ("r1", "running") not in statuses  # not yet — no Started seen
    q._handles["r1"].event.emit(Started(run_id="r1", seq=1, mode="m", settings_summary="s"))
    q._handles["r1"].finish(str(tmp_path))
    assert ("r1", "running") in statuses and ("r1", "done") in statuses
    assert finished == [("r1", str(tmp_path))]  # result_path is carried through
    assert statuses.index(("r1", "starting")) < statuses.index(("r1", "running"))


def test_cancel_active_removes_partial_output_dir(qtbot, tmp_path):
    # Spec §4 cancellation: a cancelled run's partial per-run output dir
    # (``<base>/runs/<run_id>``) is removed so half-written artifacts don't linger.
    run_dir = tmp_path / "runs" / "r1"  # final component == run_id (per-run contract)
    run_dir.mkdir(parents=True)
    (run_dir / "nlsq_result.partial").write_text("half-written")
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(run_dir))
    q.cancel("r1")
    assert not run_dir.exists()  # partial output swept on cancel


def test_cancel_pending_removes_partial_output_dir(qtbot, tmp_path):
    # Same cleanup contract for a queued-but-not-started run.
    active_dir = tmp_path / "runs" / "r1"
    active_dir.mkdir(parents=True)
    pending_dir = tmp_path / "runs" / "r2"
    pending_dir.mkdir(parents=True)
    (pending_dir / "x").write_text("y")
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(active_dir))  # active
    q.enqueue("r2", "b.yaml", str(pending_dir))  # pending
    q.cancel("r2")
    assert not pending_dir.exists()


def test_cancel_does_not_delete_non_per_run_dir(qtbot, tmp_path):
    # Safety: if the output dir is NOT named after the run (a misconfigured shared
    # dir), cancel must never delete it.
    shared = tmp_path / "shared_out"  # final component != run_id
    shared.mkdir()
    (shared / "keepme").write_text("important")
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(shared))
    q.cancel("r1")
    assert shared.exists() and (shared / "keepme").exists()


def test_queue_forwards_per_run_diagnostics(qtbot, tmp_path):
    q = _queue(max_concurrent=1)
    iters, layers, banners, logs = [], [], [], []
    q.iteration_received.connect(lambda rid, n, ssr: iters.append((rid, n, ssr)))
    q.layer_status_received.connect(lambda rid, lyr: layers.append((rid, lyr)))
    q.banner_received.connect(lambda rid, txt, kind: banners.append((rid, kind)))
    q.log_received.connect(lambda rid, lvl, msg: logs.append((rid, lvl, msg)))
    q.enqueue("r1", "a.yaml", str(tmp_path))
    h = q._handles["r1"]
    h.event.emit(Iteration(run_id="r1", seq=1, n=2, ssr=9.0, chi2=9.0))
    h.event.emit(LayerStatus(run_id="r1", seq=2, layers={"L1": True}, mode="laminar_flow"))
    h.event.emit(Banner(run_id="r1", seq=3, text="ANTI-DEGENERACY: Layer 2", kind=BannerKind.INFO))
    h.event.emit(LogLine(run_id="r1", seq=4, level="INFO", msg="hello"))
    assert iters == [("r1", 2, 9.0)]
    assert layers == [("r1", {"L1": True})]
    assert banners == [("r1", "info")]
    assert logs == [("r1", "INFO", "hello")]


def test_cancel_active_frees_slot_and_starts_next(qtbot, tmp_path):
    # Cancelling an active run frees its slot SYNCHRONOUSLY (Plan C suppresses the
    # synthetic Died on interruption, so no terminal event arrives) and promotes the
    # next queued job — without ever reporting the cancel as a "killed"/OOM failure.
    q = _queue(max_concurrent=1)
    statuses, failures = [], []
    q.run_status_changed.connect(lambda rid, st: statuses.append((rid, st)))
    q.run_failed.connect(lambda rid, txt: failures.append((rid, txt)))
    q.enqueue("r1", "a.yaml", str(tmp_path))
    q.enqueue("r2", "b.yaml", str(tmp_path))  # queued behind r1
    h1 = q._handles["r1"]
    q.cancel("r1")  # synchronous: terminate + join reader + free slot (no Died arrives)
    assert h1.cancelled is True and h1.joined is True
    assert ("r1", "cancelled") in statuses
    assert ("r1", "killed") not in statuses
    assert failures == []  # cancellation is not an error -> no OOM dialog
    assert "r2" in q._handles  # the freed slot promoted the queued job
    assert q.active_count() == 1 and q.pending_count() == 0


def test_cancel_and_shutdown(qtbot, tmp_path):
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(tmp_path))
    q.enqueue("r2", "b.yaml", str(tmp_path))
    h1 = q._handles["r1"]
    q.shutdown()
    assert h1.cancelled is True
    assert h1.joined is True  # reader QThread joined on shutdown (app close / atexit)
    assert q.pending_count() == 0  # queue cleared on shutdown
    assert q._handles == {}  # dead handles dropped (no stale references after shutdown)


def test_cancel_pending_removes_before_start(qtbot, tmp_path):
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(tmp_path))  # active
    q.enqueue("r2", "b.yaml", str(tmp_path))  # pending
    q.cancel("r2")
    assert q.pending_count() == 0
    # finishing r1 must NOT start the cancelled r2
    q._handles["r1"].finish(str(tmp_path))
    assert q.active_count() == 0


def test_duplicate_enqueue_is_ignored(qtbot, tmp_path):
    q = _queue(max_concurrent=1)
    q.enqueue("r1", "a.yaml", str(tmp_path))
    q.enqueue("r1", "a.yaml", str(tmp_path))  # duplicate run_id -> ignored
    assert q.active_count() == 1 and q.pending_count() == 0
