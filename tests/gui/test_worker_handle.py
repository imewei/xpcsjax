"""pytest-qt tests for WorkerHandle: event forwarding, Died synthesis, cancel."""

import pytest

pytest.importorskip("PySide6")

from tests.gui import ipc_fakes  # noqa: E402 — importable spawn targets
from xpcsjax.gui.ipc.job import FitJob  # noqa: E402
from xpcsjax.service.events import Died, Finished, Started  # noqa: E402


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


def test_cancel_terminates_even_when_killpg_races_startup(monkeypatch):
    """cancel() must still call proc.terminate() when killpg races a child
    that hasn't called os.setpgrp() yet (cancel-during-startup).

    Previously terminate() was only called in the killpg `else` branch, so a
    swallowed ProcessLookupError from killpg left the worker completely
    unsignaled until the full join-timeout + SIGKILL escalation.
    """
    import os
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
