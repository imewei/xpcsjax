"""A bounded-concurrency queue of fit workers, keyed by run id.

JAX-free (the workers import JAX only in their child processes). Default
concurrency is 1 — these fits are RAM-heavy (see the project's OOM
serial-routing rule); raise ``max_concurrent`` only with headroom.
"""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from xpcsjax.gui.ipc.handle import WorkerHandle
from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.result_loader import load_result_summary
from xpcsjax.service.events import (
    Banner,
    Died,
    Failed,
    Finished,
    Iteration,
    LayerStatus,
    LogLine,
    Started,
)


class FitQueueController(QObject):
    """Runs queued fits with bounded concurrency, demultiplexed by run id.

    Re-emits ALL per-run events (diagnostics, log, failure) with the ``run_id``
    so the (multi-run) GUI can route them to the selected run's monitor — this
    is the single execution path that replaces the Plan-D ``FitController``, so
    it must forward everything that controller did, not just terminal events.
    """

    run_status_changed = Signal(str, str)  # (run_id, status)
    run_finished = Signal(str, str, object)  # (run_id, result_path, ResultSummary | None)
    run_failed = Signal(str, str)  # (run_id, error_text) — traceback / OOM hint
    log_received = Signal(str, str, str)  # (run_id, level, msg)
    iteration_received = Signal(str, int, float)  # (run_id, n, ssr)
    layer_status_received = Signal(str, object)  # (run_id, layers dict)
    banner_received = Signal(str, str, str)  # (run_id, text, kind)

    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        handle_factory: Callable[[FitJob], Any] = WorkerHandle,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._max = max(1, int(max_concurrent))
        self._handle_factory = handle_factory
        self._pending: deque[FitJob] = deque()
        self._handles: dict[str, Any] = {}  # run_id -> active handle
        self._cancelled: set[str] = set()  # run_ids cancelled by the user
        self._output_dirs: dict[str, str] = {}  # run_id -> per-run output dir (for cancel cleanup)

    def enqueue(self, run_id: str, config_path: str, output_dir: str) -> None:
        """Queue a fit; starts it immediately if a concurrency slot is free.

        Ignores a ``run_id`` that is already active or pending (defensive — run
        ids are unique UUIDs, but a duplicate would corrupt slot accounting).
        """
        if run_id in self._handles or any(j.run_id == run_id for j in self._pending):
            return
        self._output_dirs[run_id] = str(output_dir)
        self._pending.append(
            FitJob(run_id=run_id, config_path=str(config_path), output_dir=str(output_dir))
        )
        self._try_start_next()

    def active_count(self) -> int:
        """Return the number of currently-running workers."""
        return len(self._handles)

    def pending_count(self) -> int:
        """Return the number of jobs still waiting for a slot."""
        return len(self._pending)

    def _cleanup_output_dir(self, run_id: str) -> None:
        """Remove a cancelled run's partial output dir (spec §4 cancellation).

        A cancelled/killed fit can leave half-written ``nlsq_result.*`` / ``plots/``
        artifacts. We delete the per-run dir, but only when it is genuinely per-run
        — its final path component must equal ``run_id`` (matching
        ``MainWindow._per_run_output_dir``'s ``<base>/runs/<run_id>`` contract) —
        so a misconfigured shared output dir is never blown away. Best-effort.
        """
        out = self._output_dirs.pop(run_id, None)
        if not out:
            return
        path = Path(out)
        if path.name != run_id:  # safety: only delete dirs we own per-run
            return
        shutil.rmtree(path, ignore_errors=True)

    def cancel(self, run_id: str) -> None:
        """Cancel a fit — whether it is still queued or already running."""
        # Queued-but-not-started: drop it from the pending queue.
        remaining = [j for j in self._pending if j.run_id != run_id]
        if len(remaining) != len(self._pending):
            self._pending = deque(remaining)
            self._cleanup_output_dir(run_id)
            self.run_status_changed.emit(run_id, "cancelled")
            return
        # Active: terminate the worker and free the slot SYNCHRONOUSLY. Plan C's
        # WorkerHandle.cancel() interrupts the reader QThread, and _ReaderThread
        # suppresses the synthetic Died when interrupted — so NO terminal event
        # arrives to free the slot. We therefore pop the slot here; relying on a
        # Died would leave the slot permanently stuck and never start queued jobs.
        handle = self._handles.get(run_id)
        if handle is not None and handle.is_running():
            handle.cancel()  # terminate proc + interrupt/join the reader
            handle.shutdown()  # idempotent reader-QThread join (Plan C)
            try:
                handle.event.disconnect()
            except (RuntimeError, TypeError):  # already disconnected
                pass
            self._handles.pop(run_id, None)
            self._cancelled.discard(run_id)  # no Died will arrive; don't leak the marker
            self._cleanup_output_dir(run_id)  # remove the partial per-run output dir
            self.run_status_changed.emit(run_id, "cancelled")
            self._try_start_next()  # promote the next queued job into the freed slot
        elif handle is not None:
            # Race: the worker already died and a terminal event is in flight but
            # not yet processed. Mark it so the pending Died/Failed maps to
            # "cancelled" (not "killed"/OOM) in _on_event, which frees the slot.
            self._cancelled.add(run_id)
            self._cleanup_output_dir(run_id)
            self.run_status_changed.emit(run_id, "cancelled")

    def shutdown(self) -> None:
        """Cancel every active worker, join its reader thread, and clear the queue."""
        self._pending.clear()
        for handle in list(self._handles.values()):
            if handle.is_running():
                handle.cancel()  # terminate the process + tear down its reader
            handle.shutdown()  # join the reader QThread (app close / atexit)
        self._handles.clear()  # drop the now-dead handles (no stale references)

    def _try_start_next(self) -> None:
        while self._pending and len(self._handles) < self._max:
            job = self._pending.popleft()
            handle = self._handle_factory(job)
            self._handles[job.run_id] = handle
            handle.event.connect(partial(self._on_event, job.run_id))
            # Cold-spawn UX (spec §4 F10): a fresh worker re-pays the JAX import +
            # XLA cold-compile before the first `Started` arrives. Surface a
            # distinct "starting" state so a multi-second cold spawn never reads as
            # a hung app; flip to "running" only once the worker emits `Started`.
            self.run_status_changed.emit(job.run_id, "starting")
            handle.start()

    def _on_event(self, run_id: str, event: Any) -> None:
        # --- non-terminal: forward to the selected-run monitor (Plan-E/D widgets) ---
        if isinstance(event, Started):
            self.run_status_changed.emit(run_id, "running")
            return
        if isinstance(event, LogLine):
            self.log_received.emit(run_id, event.level, event.msg)
            return
        if isinstance(event, Iteration):
            self.iteration_received.emit(run_id, event.n, event.ssr)
            return
        if isinstance(event, LayerStatus):
            self.layer_status_received.emit(run_id, event.layers)
            return
        if isinstance(event, Banner):
            self.banner_received.emit(run_id, event.text, event.kind.value)
            return
        if not isinstance(event, (Finished, Failed, Died)):
            return
        # --- terminal — ignore a stale/duplicate whose slot is already freed ---
        if run_id not in self._handles:
            return
        cancelled = run_id in self._cancelled
        if isinstance(event, Finished):
            # A worker that finished just as the user cancelled stays "cancelled":
            # cancel() already deleted the per-run output dir, so emitting "done" +
            # run_finished here would flip the UI back to done and then fail to load
            # the (deleted) result — leaving an inconsistent "done · result missing".
            if cancelled:
                self.run_status_changed.emit(run_id, "cancelled")
            else:
                summary = load_result_summary(event.result_path) if event.result_path else None
                self.run_status_changed.emit(run_id, "done")
                self.run_finished.emit(run_id, event.result_path, summary)
        elif isinstance(event, Failed):
            # A user-cancelled worker may surface as Failed/Died — keep "cancelled".
            self.run_status_changed.emit(run_id, "cancelled" if cancelled else "failed")
            if not cancelled:
                self.run_failed.emit(run_id, event.traceback)
        else:  # Died (synthetic, parent-emitted)
            if cancelled:
                self.run_status_changed.emit(run_id, "cancelled")
            else:
                self.run_status_changed.emit(run_id, "killed")
                self.run_failed.emit(
                    run_id,
                    f"Worker exited abnormally (exit_code={event.exit_code}, "
                    f"signal={event.signal}) — likely out of memory.",
                )
        self._cancelled.discard(run_id)
        # Drop the output-dir bookkeeping WITHOUT deleting: a finished/failed run's
        # artifacts on disk are durable; only an explicit cancel removes them.
        self._output_dirs.pop(run_id, None)
        # Free the slot (disconnect first, matching Plan-D's handle contract) + start next.
        handle = self._handles.pop(run_id)
        try:
            handle.event.disconnect()
        except (RuntimeError, TypeError):  # already disconnected
            pass
        handle.shutdown()  # join the reader QThread before dropping the last ref —
        #                    the reader just emitted this terminal and is returning;
        #                    without the join Qt may destroy a still-running QThread.
        self._try_start_next()
