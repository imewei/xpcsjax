"""Owns the worker process lifecycle and bridges its events onto Qt signals.

The GUI process imports this module; it is JAX-free (``run_worker`` is imported
at module level but is itself JAX-free at import — its service imports are lazy).
"""

from __future__ import annotations

import multiprocessing
import os
import queue as _queue
import signal
import time
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.ipc.worker import run_worker
from xpcsjax.service.events import TERMINAL_EVENTS, Died

_QUEUE_MAXSIZE = 1000
_TERMINATE_JOIN_S = 5.0
_KILL_JOIN_S = 2.0
# After the worker process exits, keep draining for this long: multiprocessing.Queue
# uses a background feeder thread, so a real terminal event may not be visible the
# instant ``is_alive()`` flips to False. Only after this idle grace with no terminal
# do we synthesize ``Died`` — this prevents a spurious Died racing a late Finished.
_DEATH_GRACE_S = 1.0
# Max time (ms) to wait for the reader QThread to leave get() and finish after we
# requestInterruption() — on cancel and on shutdown. QThread.wait() takes ms.
_READER_JOIN_MS = 2000


class _ReaderThread(QThread):
    """Drains the event queue onto a Qt signal; synthesizes Died on abnormal exit."""

    # The signal is intentionally named ``event``. It shadows QThread/QObject's
    # ``event()`` handler at the Python level only — Qt's C++ event dispatch is
    # unaffected — so this is a runtime-safe, deliberate PySide pattern that mypy
    # cannot model. See the wiring in controllers/fit_queue.py.
    event = Signal(object)  # type: ignore[assignment]

    def __init__(self, event_queue: Any, proc: Any, run_id: str) -> None:
        super().__init__()
        self._queue = event_queue
        self._proc = proc
        self._run_id = run_id

    def run(self) -> None:  # noqa: D102 — QThread entry; behavior in class docstring
        terminal_seen = False
        grace_deadline: float | None = None
        # The 0.1 s get() timeout doubles as the interruption poll: WorkerHandle
        # calls requestInterruption() on cancel/shutdown, and the loop exits within
        # one poll so the reader leaves get() *before* cancel_join_thread() runs.
        while not self.isInterruptionRequested():
            try:
                ev = self._queue.get(timeout=0.1)
            except _queue.Empty:
                if self._proc.is_alive():
                    continue
                # Process exited. Keep polling for a grace window — the feeder
                # thread may still flush a final terminal event after is_alive()
                # flips. Start the clock on the first idle poll post-exit.
                if grace_deadline is None:
                    grace_deadline = time.monotonic() + _DEATH_GRACE_S
                if time.monotonic() >= grace_deadline:
                    # The feeder thread may have pushed a real terminal event into
                    # the pipe just before the process exited. Drain it now rather
                    # than synthesize a spurious Died over a lost Finished/Failed.
                    terminal_seen = self._drain_remaining()
                    break
                continue
            except (OSError, EOFError):
                # The underlying pipe broke/closed (e.g. the worker crashed hard
                # enough to corrupt the multiprocessing Queue). Treat like a
                # process-exit: fall through to the Died-synthesis below instead
                # of letting the exception kill this thread with no terminal
                # event ever emitted.
                break
            self.event.emit(ev)
            if isinstance(ev, TERMINAL_EVENTS):
                terminal_seen = True
                break

        # Synthesize Died only on genuine abnormal exit — NOT when we were
        # deliberately interrupted (cancel/shutdown already accounts for the run).
        if not terminal_seen and not self.isInterruptionRequested():
            code = self._proc.exitcode
            sig = -code if isinstance(code, int) and code < 0 else None
            self.event.emit(Died(run_id=self._run_id, seq=-1, exit_code=code, signal=sig))

    def _drain_remaining(self) -> bool:
        """Emit any still-queued events; return True if a terminal was among them.

        multiprocessing's ``Queue.put`` hands the object to a background feeder
        thread, so a terminal event the worker enqueued just before exiting can
        still be sitting in the pipe when the timed ``get`` loop gives up. This
        final non-blocking sweep recovers it instead of losing it to a synthetic
        ``Died``. Stops at the first terminal (nothing follows it).
        """
        saw_terminal = False
        while True:
            try:
                ev = self._queue.get_nowait()
            except _queue.Empty:
                break
            except (OSError, ValueError):  # queue closed / broken mid-drain
                break
            self.event.emit(ev)
            if isinstance(ev, TERMINAL_EVENTS):
                saw_terminal = True
                break
        return saw_terminal


class WorkerHandle(QObject):
    """Spawns a fit worker and re-emits its events as the ``event`` signal."""

    # Intentional Qt-signal name shadowing QObject.event() at the Python level
    # only; runtime-safe (see _ReaderThread above).
    event = Signal(object)  # type: ignore[assignment]

    def __init__(self, job: FitJob) -> None:
        super().__init__()
        self._job = job
        self._proc: Any = None
        self._queue: Any = None
        self._reader: _ReaderThread | None = None
        self._pgid: int | None = None

    def start(self) -> None:
        """Spawn the worker process and begin draining its events."""
        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue(maxsize=_QUEUE_MAXSIZE)
        self._proc = ctx.Process(target=run_worker, args=(self._job, self._queue), daemon=False)
        self._proc.start()
        # The worker calls os.setpgrp(), so its own pid is its process-group id.
        # Capture it now, while the child is alive, for a POSIX group sweep on
        # cancel — reading proc.pid only after a join() risks PID reuse.
        self._pgid = self._proc.pid if hasattr(os, "killpg") else None
        self._reader = _ReaderThread(self._queue, self._proc, self._job.run_id)
        self._reader.event.connect(self.event)
        self._reader.start()

    def is_running(self) -> bool:
        """Return True while the worker process is alive."""
        return bool(self._proc is not None and self._proc.is_alive())

    def cancel(self) -> None:
        """Hard-stop the worker: terminate → join → (group-sweep + kill).

        Process-group cleanup is **POSIX-only**: on Windows there is no
        ``setpgrp``/``killpg``, so ``terminate()``/``kill()`` stop the worker but
        any grandchildren it spawned are not swept.
        """
        proc = self._proc
        if proc is None or not proc.is_alive():
            return
        # Graceful stop of the WHOLE process group (worker + any grandchildren)
        # while the pgid is still valid and the proc is unreaped. Signalling the
        # group up-front — not just the leader — closes the leak where the worker
        # exits on SIGTERM within the grace window (so the `is_alive()` escalation
        # below never runs) yet left children behind. Doing it here, before the
        # join/reap, also avoids the PID-reuse hazard of a killpg after waitpid.
        if hasattr(os, "killpg") and self._pgid and self._pgid > 0:
            try:
                os.killpg(self._pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
                pass
        else:
            proc.terminate()
        proc.join(timeout=_TERMINATE_JOIN_S)
        if proc.is_alive():
            # Escalate BEFORE reaping: sweep the worker's process group (any
            # grandchildren) while its pgid is still valid, then hard-kill + reap.
            # Guard pid > 0 — killpg(0) would signal the GUI's OWN group.
            if hasattr(os, "killpg") and self._pgid and self._pgid > 0:
                try:
                    os.killpg(self._pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
                    pass
            proc.kill()
            proc.join(timeout=_KILL_JOIN_S)
        # Tear down the reader thread BEFORE touching the queue's feeder: calling
        # cancel_join_thread() while the reader is still blocked in queue.get() is
        # undefined behavior (it can spin or read garbage). requestInterruption()
        # + wait() so the reader has left get() first.
        if self._reader is not None:
            self._reader.requestInterruption()
            self._reader.wait(_READER_JOIN_MS)
        # The queue's feeder thread may be mid-write after a hard kill; don't block on it.
        if self._queue is not None:
            self._queue.cancel_join_thread()

    def shutdown(self) -> None:
        """Join the reader thread so Qt never destroys a still-running QThread.

        Called on normal completion and on GUI close (atexit/closeEvent, §8).
        Idempotent and safe whether the worker finished, failed, or was cancelled
        — the reader exits within one 0.1 s poll of requestInterruption().
        """
        if self._reader is not None:
            self._reader.requestInterruption()
            self._reader.wait(_READER_JOIN_MS)
            self._reader = None
        self._reap_process()

    def _reap_process(self) -> None:
        """Join + close the (already-exited) worker process and close the queue.

        On the normal terminal path the worker has already exited before its
        ``Finished``/``Failed`` reached us, so ``join`` returns immediately; this
        reaps the zombie and releases the OS process handle + queue FDs that would
        otherwise accumulate across many fits. Best-effort and idempotent.
        """
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.join(timeout=_KILL_JOIN_S)
                if not proc.is_alive():
                    proc.close()  # release the OS handle (only valid once exited)
                    self._proc = None
            except (ValueError, AssertionError):  # already closed / never started
                self._proc = None
        queue = self._queue
        if queue is not None:
            try:
                queue.close()
            except Exception:  # noqa: BLE001 — best-effort resource release
                pass
            self._queue = None
