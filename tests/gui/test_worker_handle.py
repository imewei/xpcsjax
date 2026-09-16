"""pytest-qt tests for WorkerHandle: event forwarding, Died synthesis, cancel."""

import os

import pytest

pytest.importorskip("PySide6")

from tests.gui import ipc_fakes
from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.service.events import Died, Finished, Started


def _collect(handle, qtbot, predicate, timeout=10000):
    events = []
    handle.event.connect(events.append)
    qtbot.waitUntil(lambda: any(predicate(e) for e in events), timeout=timeout)
    return events


def test_handle_forwards_events_until_finished(qtbot, monkeypatch):
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.emit_started_then_finished)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    events = _collect(h, qtbot, lambda e: isinstance(e, Finished))
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "Started" and isinstance(events[-1], Finished)


def test_handle_synthesizes_died_on_abnormal_exit(qtbot, monkeypatch):
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.exit_without_terminal)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    events = _collect(h, qtbot, lambda e: isinstance(e, Died))
    died = [e for e in events if isinstance(e, Died)]
    assert died and died[0].exit_code == 3


def test_cancel_terminates_running_worker(qtbot, monkeypatch):
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.sleep_forever)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    _collect(h, qtbot, lambda e: isinstance(e, Started))
    assert h.is_running()
    h.cancel()
    qtbot.waitUntil(lambda: not h.is_running(), timeout=10000)
    assert not h.is_running()


def test_cancel_reaps_via_event_loop_and_clears_timer(qtbot, monkeypatch):
    """A10's QTimer escalation must actually run off the Qt event loop against
    a real (fake-worker) process, not just be driven by hand via _poll_cancel()
    with backdated deadlines (as the escalation unit tests above do). Waiting
    on ``reaped`` -- emitted only from ``_finish_cancel()`` -- proves the timer
    fired for real; ``test_cancel_terminates_running_worker`` only waits on
    ``is_running()``, which flips false as soon as the child dies regardless
    of whether the QTimer poll loop ever ran.
    """
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.sleep_forever)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    _collect(h, qtbot, lambda e: isinstance(e, Started))
    assert h.is_running()

    with qtbot.waitSignal(h.reaped, timeout=10000):
        h.cancel()

    assert h._cancel_timer is None
    assert not h.is_running()


def test_cancel_and_shutdown_join_reader_thread(qtbot, monkeypatch):
    # After cancel() + shutdown() the reader QThread must be fully stopped — a
    # QThread still running at GC aborts with "QThread: Destroyed while running".
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.sleep_forever)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    _collect(h, qtbot, lambda e: isinstance(e, Started))
    reader = h._reader
    h.cancel()
    h.shutdown()
    qtbot.waitUntil(lambda: reader is not None and reader.isFinished(), timeout=10000)
    assert reader.isFinished()


def test_cancel_blocking_stops_a_real_worker(qtbot, monkeypatch):
    """_cancel_blocking() (the atexit/closeEvent teardown path) has no direct
    coverage elsewhere -- only the FakeHandle's `cancel_blocking_called` flag
    is asserted in test_fit_queue.py. Exercise the real synchronous
    terminate -> join -> (kill -> join) sequence against a real (fake-worker)
    process.
    """
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle

    monkeypatch.setattr(handle_mod, "run_worker", ipc_fakes.sleep_forever)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h.start()
    _collect(h, qtbot, lambda e: isinstance(e, Started))
    assert h.is_running()

    h._cancel_blocking()

    assert not h.is_running()
    assert h._cancel_timer is None


def test_cancel_blocking_order_terminate_join_kill_join():
    """MagicMock case pinning the exact synchronous escalation order:
    terminate() -> join(_TERMINATE_JOIN_S) -> (still alive ->) kill() ->
    join(_KILL_JOIN_S)."""
    from unittest.mock import MagicMock

    from xpcsjax.gui.ipc.handle import (
        _KILL_JOIN_S,
        _TERMINATE_JOIN_S,
        WorkerHandle,
    )

    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    fake_proc = MagicMock()
    # First is_alive() (the entry guard) and the post-terminate check both
    # report alive, forcing the kill escalation branch.
    fake_proc.is_alive.side_effect = [True, True]
    h._proc = fake_proc
    h._reader = MagicMock()
    h._queue = MagicMock()

    h._cancel_blocking()

    assert [c[0] for c in fake_proc.method_calls] == [
        "is_alive",
        "terminate",
        "join",
        "is_alive",
        "kill",
        "join",
    ]
    fake_proc.join.assert_any_call(timeout=_TERMINATE_JOIN_S)
    fake_proc.join.assert_any_call(timeout=_KILL_JOIN_S)


def test_shutdown_defers_reap_when_reader_wait_times_out():
    """shutdown() must not drop the reader ref or reap the process when the
    reader QThread fails to stop within the join deadline -- doing so aborts
    Qt ("QThread: Destroyed while thread is still running") and races the
    still-running reader's next is_alive() check against a closed Process.
    """
    from unittest.mock import MagicMock

    from xpcsjax.gui.ipc.handle import WorkerHandle

    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    fake_reader = MagicMock()
    fake_reader.wait.return_value = False  # simulates a timed-out join
    h._reader = fake_reader
    fake_proc = MagicMock()
    h._proc = fake_proc

    h.shutdown()

    fake_reader.requestInterruption.assert_called_once()
    assert h._reader is fake_reader  # ref NOT dropped
    fake_proc.join.assert_not_called()  # _reap_process() NOT reached
    fake_proc.close.assert_not_called()


def test_cancel_is_non_blocking_and_does_not_join():
    """cancel() must return without ever calling proc.join() -- the join/kill/
    give-up escalation sequence runs off a QTimer instead (A10), so a stuck
    child can never freeze the calling (UI) thread.
    """
    from unittest.mock import MagicMock

    from xpcsjax.gui.ipc.handle import WorkerHandle

    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    fake_proc = MagicMock()
    fake_proc.is_alive.return_value = True  # never reports dead
    h._proc = fake_proc

    h.cancel()

    fake_proc.terminate.assert_called_once()
    fake_proc.join.assert_not_called()
    assert h._cancel_timer is not None  # escalation continues on a timer


def test_cancel_escalates_to_sigkill_then_gives_up():
    """Driving the poll past the escalate/give-up deadlines fires SIGKILL,
    then finishes (emits `reaped`) even if the process never reports dead --
    the "give up at 7s" contract from the audit item.
    """
    import time
    from unittest.mock import MagicMock

    from xpcsjax.gui.ipc.handle import WorkerHandle

    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    fake_proc = MagicMock()
    fake_proc.is_alive.return_value = True  # never reports dead -> forces escalation
    h._proc = fake_proc
    reaped: list[bool] = []
    h.reaped.connect(lambda: reaped.append(True))

    h.cancel()
    fake_proc.kill.assert_not_called()

    # Simulate 5s elapsed -> escalate to SIGKILL.
    h._cancel_escalate_at = time.monotonic() - 1
    h._poll_cancel()
    fake_proc.kill.assert_called_once()
    assert reaped == []  # not yet -- give-up deadline hasn't passed

    # Simulate 7s elapsed -> give up (best-effort finish) even though the
    # (mocked) process still reports alive.
    h._cancel_deadline = time.monotonic() - 1
    h._poll_cancel()
    assert reaped == [True]
    assert h._cancel_timer is None  # timer stopped/cleared


@pytest.mark.skipif(
    not hasattr(os, "killpg"),
    reason="os.killpg is POSIX-only; WorkerHandle._pgid stays None on this platform",
)
def test_cancel_terminates_even_when_killpg_races_startup(monkeypatch):
    """cancel() must still call proc.terminate() when killpg races a child
    that hasn't called os.setpgrp() yet (cancel-during-startup).

    Previously terminate() was only called in the killpg `else` branch, so a
    swallowed ProcessLookupError from killpg left the worker completely
    unsignaled until the full join-timeout + SIGKILL escalation.
    """
    from unittest.mock import MagicMock

    from xpcsjax.gui.ipc.handle import WorkerHandle

    def _raise_process_lookup(*_a, **_k):
        raise ProcessLookupError("no such process")

    monkeypatch.setattr(os, "killpg", _raise_process_lookup)

    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    h._proc = MagicMock()
    h._proc.is_alive.return_value = True
    h._pgid = 12345

    h.cancel()

    h._proc.terminate.assert_called_once()


def test_reader_final_drain_recovers_terminal_after_grace(qtbot, monkeypatch):
    """A terminal event still queued when the grace deadline expires must be
    recovered by a final non-blocking drain — NOT discarded and replaced by a
    synthetic ``Died``. Regression for the grace-period drain bug: the wall-clock
    grace loop could break with a real ``Finished`` still sitting in the pipe.
    """
    import queue as _queue

    from xpcsjax.gui.ipc import handle as handle_mod

    # Force the grace deadline to expire on the first idle poll so the test is
    # deterministic and fast (no real 1 s wait).
    monkeypatch.setattr(handle_mod, "_DEATH_GRACE_S", 0.0)

    class _FakeQueue:
        """Empty during the timed loop; yields one late terminal via get_nowait."""

        def __init__(self, late):
            self._late = list(late)

        def get(self, timeout=None):
            raise _queue.Empty

        def get_nowait(self):
            if self._late:
                return self._late.pop(0)
            raise _queue.Empty

    class _FakeProc:
        exitcode = 0

        def is_alive(self):
            return False

    fin = Finished(run_id="r1", seq=1, result_path="/tmp/out")
    reader = handle_mod._ReaderThread(_FakeQueue([fin]), _FakeProc(), "r1")
    seen: list = []
    reader.event.connect(seen.append)
    reader.run()  # drive synchronously in this thread for a deterministic result

    assert any(isinstance(e, Finished) for e in seen), "late terminal event was lost"
    assert not any(isinstance(e, Died) for e in seen), "synthesized Died despite a real terminal"
