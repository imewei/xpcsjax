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

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.ipc.worker import run_worker
from xpcsjax.service.events import TERMINAL_EVENTS, Died
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

_QUEUE_MAXSIZE = 1000
_TERMINATE_JOIN_S = 5.0
_KILL_JOIN_S = 2.0
# Non-blocking cancel() poll cadence / escalation timings (A10). Chosen so a
# well-behaved child (SIGTERM handled promptly) is reaped within one or two
# polls, while a stuck one still gets a bounded SIGKILL escalation and a
# hard give-up -- WITHOUT ever blocking the calling (UI) thread on join().
_CANCEL_POLL_MS = 250
_CANCEL_ESCALATE_S = 5.0
_CANCEL_GIVE_UP_S = 7.0
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
            except (OSError, EOFError, ValueError):
                # The underlying pipe broke/closed (e.g. the worker crashed hard
                # enough to corrupt the multiprocessing Queue). A CLOSED queue's
                # get() raises ValueError, not OSError (verified empirically) —
                # without it here, that case falls through uncaught and this
                # thread dies with no terminal event ever emitted. Treat all
                # three like a process-exit: fall through to the Died-synthesis
                # below.
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
    # Emitted once cancel()'s async join/kill/give-up sequence has concluded
    # (process confirmed dead, or the 7s give-up elapsed). NOT used for slot
    # accounting by FitQueueController (that stays synchronous, see cancel()
    # there) — only for keeping this handle alive/referenced until cleanup
    # finishes, and for optional logging.
    reaped = Signal()  # type: ignore[assignment]

    def __init__(self, job: FitJob) -> None:
        super().__init__()
        self._job = job
        self._proc: Any = None
        self._queue: Any = None
        self._reader: _ReaderThread | None = None
        self._pgid: int | None = None
        self._cancel_timer: QTimer | None = None
        self._cancel_escalate_at: float = 0.0
        self._cancel_deadline: float = 0.0
        self._cancel_escalated: bool = False

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

    def _signal_terminate(self) -> None:
        """Send SIGTERM to the process (and, on POSIX, its whole group).

        Just posts signals — never blocks. Process-group cleanup is
        **POSIX-only**: on Windows there is no ``setpgrp``/``killpg``, so
        ``terminate()``/``kill()`` stop the worker but any grandchildren it
        spawned are not swept.
        """
        # Graceful stop of the WHOLE process group (worker + any grandchildren)
        # while the pgid is still valid and the proc is unreaped. Signalling the
        # group up-front — not just the leader — closes the leak where the worker
        # exits on SIGTERM within the grace window (so the SIGKILL escalation
        # below never runs) yet left children behind. Doing it here, before the
        # join/reap, also avoids the PID-reuse hazard of a killpg after waitpid.
        if hasattr(os, "killpg") and self._pgid and self._pgid > 0:
            try:
                os.killpg(self._pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
                # killpg can race a child that hasn't called os.setpgrp() yet
                # (cancel-during-startup): fall through to terminate() below
                # instead of leaving the process unsignaled.
                pass
        # Always also signal the process directly — a swallowed killpg
        # failure above must not leave the worker unsignaled until the
        # escalation below.
        self._proc.terminate()

    def _signal_kill(self) -> None:
        """Escalate to SIGKILL (process + group). Never blocks."""
        # Guard pid > 0 — killpg(0) would signal the GUI's OWN group.
        if hasattr(os, "killpg") and self._pgid and self._pgid > 0:
            try:
                os.killpg(self._pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
                pass
        self._proc.kill()

    def _teardown_reader(self) -> None:
        """Interrupt + join the reader thread (fast: exits within ~0.1s).

        Must run BEFORE touching the queue's feeder: calling
        ``cancel_join_thread()`` while the reader is still blocked in
        ``queue.get()`` is undefined behavior (it can spin or read garbage).
        The reader's loop checks ``isInterruptionRequested()`` ahead of every
        ``get()``, so this does not block on the CHILD PROCESS dying — only on
        the reader thread noticing the flag, which is bounded by its 0.1s
        poll interval, independent of ``_READER_JOIN_MS``.
        """
        if self._reader is not None:
            self._reader.requestInterruption()
            self._reader.wait(_READER_JOIN_MS)

    def cancel(self) -> None:
        """Hard-stop the worker WITHOUT blocking the calling (UI) thread.

        Sends SIGTERM (process + POSIX group) and tears down the reader
        thread synchronously — both are fast, bounded operations, not waits
        on the child process itself. The join → SIGKILL-escalation → give-up
        sequence that used to block here for up to ~9s (audit A10) now runs
        off a ``QTimer`` poll (:data:`_CANCEL_POLL_MS` cadence, escalating at
        :data:`_CANCEL_ESCALATE_S`, giving up at :data:`_CANCEL_GIVE_UP_S`),
        emitting :attr:`reaped` once the process is confirmed dead or the
        give-up elapses.

        This requires a live Qt event loop to complete — callers with no
        running event loop (the atexit/interpreter-shutdown path) must use
        :meth:`_cancel_blocking` instead, which keeps the old fully
        synchronous behavior.
        """
        proc = self._proc
        if proc is None or not proc.is_alive():
            return
        if self._cancel_timer is not None:
            return  # a cancel is already in flight; don't start a second poll loop
        self._signal_terminate()
        self._teardown_reader()

        now = time.monotonic()
        self._cancel_escalate_at = now + _CANCEL_ESCALATE_S
        self._cancel_deadline = now + _CANCEL_GIVE_UP_S
        self._cancel_escalated = False
        self._cancel_timer = QTimer(self)
        self._cancel_timer.timeout.connect(self._poll_cancel)
        self._cancel_timer.start(_CANCEL_POLL_MS)

    def _poll_cancel(self) -> None:
        """QTimer tick: escalate to SIGKILL at 5s, finish at death or 7s give-up."""
        proc = self._proc
        if proc is None or not proc.is_alive():
            self._finish_cancel()
            return
        now = time.monotonic()
        if not self._cancel_escalated and now >= self._cancel_escalate_at:
            self._signal_kill()
            self._cancel_escalated = True
        if now >= self._cancel_deadline:
            # Best-effort give-up: the process may still technically be alive
            # (e.g. stuck in uninterruptible I/O) — _reap_process below is
            # itself best-effort/idempotent for that residual case.
            self._finish_cancel()

    def _finish_cancel(self) -> None:
        if self._cancel_timer is not None:
            self._cancel_timer.stop()
            self._cancel_timer.deleteLater()
            self._cancel_timer = None
        # The queue's feeder thread may be mid-write after a hard kill; don't block on it.
        if self._queue is not None:
            self._queue.cancel_join_thread()
        # By the time we get here the process is normally already dead (the
        # only path into _finish_cancel where it might not be is the 7s
        # give-up), so _reap_process's internal join is a fast no-op in the
        # common case; it also releases the OS process/queue handles.
        self._reap_process()
        self.reaped.emit()

    def _cancel_blocking(self) -> None:
        """Fully-synchronous hard-stop: terminate -> join -> (kill + join).

        This is cancel()'s ORIGINAL behavior, kept for the one caller that
        genuinely needs it: the atexit/interpreter-shutdown teardown path
        (``FitQueueController.shutdown``), where a ``QTimer`` would never
        fire (no Qt event loop is running by then) and a still-alive
        grandchild left behind would simply be orphaned. Every other caller
        (the interactive Cancel action) uses the non-blocking :meth:`cancel`.
        """
        # A prior non-blocking cancel() may still have a poll timer in flight
        # (e.g. the app closed within its 7s window) — stop it so it can't
        # fire after this method has already reaped everything below.
        if self._cancel_timer is not None:
            self._cancel_timer.stop()
            self._cancel_timer.deleteLater()
            self._cancel_timer = None
        proc = self._proc
        if proc is None or not proc.is_alive():
            return
        self._signal_terminate()
        proc.join(timeout=_TERMINATE_JOIN_S)
        if proc.is_alive():
            self._signal_kill()
            proc.join(timeout=_KILL_JOIN_S)
        self._teardown_reader()
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
            if not self._reader.wait(_READER_JOIN_MS):
                # Do NOT drop the last Python reference to a still-running
                # QThread here: Qt aborts the process ("QThread: Destroyed
                # while thread is still running") the next time the GC
                # collects it, and reaping the process out from under the
                # reader makes its next is_alive() check raise ValueError on
                # a closed Process, killing it with no terminal event ever
                # emitted. Leave both in place; the caller (atexit/closeEvent)
                # may call shutdown() again later.
                logger.warning(
                    "Reader thread for run_id=%s did not stop within %dms; deferring reap.",
                    self._job.run_id,
                    _READER_JOIN_MS,
                )
                return
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
