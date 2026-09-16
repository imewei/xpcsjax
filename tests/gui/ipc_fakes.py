"""Importable spawn-target fakes for the WorkerHandle tests.

Run inside spawned children, so this stays a plain module (no pytest, no
importorskip, no qtbot) importing nothing heavy.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.service.events import Finished, Started


class FakeHandle(QObject):
    """Shared ``WorkerHandle`` test double: never spawns a real process.

    Consolidates six near-identical ``_FakeHandle`` copies that had drifted
    (two of them predated A10 and lacked ``reaped``/``_cancel_blocking``,
    so ``FitQueueController.cancel()``/``shutdown()`` against a running fake
    there would have raised ``AttributeError``). Mirrors the real contract in
    ``xpcsjax/gui/ipc/handle.py``: ``event``/``reaped`` signals, ``start``/
    ``cancel``/``is_running``/``shutdown``/``_cancel_blocking``, plus a
    ``finish()`` test helper. Call counts are tracked as plain attributes
    (``cancel_calls``, ``cancel_blocking_calls``) so per-test call-counting
    never needs a separate closure dict.
    """

    event = Signal(object)
    reaped = Signal()  # matches WorkerHandle's non-blocking-cancel contract (A10)

    def __init__(self, job: Any = None, *, alive: bool = False) -> None:
        super().__init__()
        self.job = job
        self._alive = alive
        self.joined = False
        self.cancel_calls = 0
        self.cancel_blocking_calls = 0

    def start(self) -> None:
        self._alive = True

    def cancel(self) -> None:
        # Mirror the real WorkerHandle.cancel() contract: signals are posted
        # synchronously but `reaped` fires on a LATER event-loop tick, so a
        # caller must not assume it has fired by the time cancel() returns.
        # singleShot(0) reproduces exactly that ordering in tests.
        self.cancel_calls += 1
        self._alive = False
        QTimer.singleShot(0, self.reaped.emit)

    def is_running(self) -> bool:
        return self._alive

    def shutdown(self) -> None:
        self.joined = True
        self._alive = False

    def _cancel_blocking(self) -> None:
        """Fake the atexit/closeEvent-only fully-synchronous cancel path."""
        self.cancel_blocking_calls += 1
        self._alive = False

    def finish(self, result_path: str = "") -> None:
        # Test helper: emit a terminal event and go not-running.
        self._alive = False
        self.event.emit(Finished(run_id=self.job.run_id, seq=9, result_path=result_path))


def emit_started_then_finished(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    q.put(Finished(run_id=job.run_id, seq=2, result_path="/tmp/out"))


def exit_without_terminal(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    raise SystemExit(3)  # abnormal exit, no terminal event


def sleep_forever(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    time.sleep(120)
