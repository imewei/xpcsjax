"""Tests for the JAX-free IPC primitives (job / emitter / log_capture)."""

import logging
import queue as queue_mod

from xpcsjax.gui.ipc.emitter import EventEmitter
from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.ipc.log_capture import QueueLogHandler
from xpcsjax.service.events import Failed, Iteration, Started


def test_fitjob_is_frozen_and_picklable():
    import pickle

    job = FitJob(run_id="r1", config_path="cfg.yaml", phi_subset=(0.0, 45.0))
    assert pickle.loads(pickle.dumps(job)) == job


def test_emitter_stamps_run_id_and_monotonic_seq():
    q = queue_mod.Queue()
    emit = EventEmitter(q, run_id="r9")
    emit(Started(run_id="", seq=0, mode="laminar_flow", settings_summary="x"))
    emit.emit(Iteration(run_id="", seq=0, n=1, ssr=2.0, chi2=0.5))
    e1, e2 = q.get_nowait(), q.get_nowait()
    assert (e1.run_id, e1.seq) == ("r9", 1)
    assert (e2.run_id, e2.seq) == ("r9", 2)


def test_emitter_blocks_for_terminal_but_drops_telemetry_when_full():
    q = queue_mod.Queue(maxsize=1)
    emit = EventEmitter(q, run_id="r1")
    emit.emit(Iteration(run_id="", seq=0, n=1, ssr=1.0, chi2=1.0))  # first telemetry -> enqueued
    emit.emit(
        Iteration(run_id="", seq=0, n=2, ssr=1.0, chi2=1.0)
    )  # coalesced (within 20 Hz window)
    first = q.get_nowait()
    assert isinstance(first, Iteration) and first.n == 1
    # Terminal must still get through after we free a slot.
    emit.emit(Failed(run_id="", seq=0, traceback="boom"))
    assert isinstance(q.get_nowait(), Failed)


def test_log_handler_forwards_records_as_loglines():
    q = queue_mod.Queue()
    handler = QueueLogHandler(EventEmitter(q, run_id="r1"))
    log = logging.getLogger("xpcsjax.test.loghandler")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.warning("hello %s", "world")
    log.removeHandler(handler)
    ev = q.get_nowait()
    assert ev.level == "WARNING"
    assert ev.msg == "hello world"
    assert ev.run_id == "r1"
