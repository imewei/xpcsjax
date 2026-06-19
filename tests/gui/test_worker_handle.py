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
