"""Drives a fit worker and translates its event stream into UI signals.

JAX-free: imports only Qt, stdlib, the JAX-free event schema, and the Plan-C
``gui.ipc`` layer (whose worker imports JAX only in the child).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from xpcsjax.gui.ipc.handle import WorkerHandle
from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.result_loader import load_result_summary
from xpcsjax.service.events import Died, Failed, Finished, LogLine, Started


class FitController(QObject):
    """Owns the active fit and bridges worker events to view-facing signals."""

    status_changed = Signal(str)  # "running" / "done" / "failed" / "killed" / "cancelled"
    log_received = Signal(str, str)  # (level, message)
    fit_finished = Signal(object)  # ResultSummary | None
    fit_failed = Signal(str)  # human-readable error text

    def __init__(
        self,
        handle_factory: Callable[[FitJob], Any] = WorkerHandle,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._handle_factory = handle_factory
        self._handle: Any = None
        self._cancelled = False  # user-cancelled the active run -> map its Died to "cancelled"

    def run(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Start a fit for ``config_path``, writing results under ``output_dir``."""
        if self.is_running():
            return  # ignore re-entrant Run while a fit is live
        self._cancelled = False
        job = FitJob(
            run_id=uuid.uuid4().hex,
            config_path=str(config_path),
            output_dir=str(output_dir),
            overrides=overrides,
        )
        self._handle = self._handle_factory(job)
        # Auto-connection: WorkerHandle.event originates from the reader QThread,
        # so Qt delivers it *queued* onto this controller's (GUI-thread) _on_event;
        # an in-process fake handle (tests) delivers it directly. Do NOT force
        # Qt.QueuedConnection — that would defer the fake's synchronous emits and
        # break the synchronous test assertions.
        self._handle.event.connect(self._on_event)
        self.status_changed.emit("running")
        self._handle.start()

    def cancel(self) -> None:
        """Request cancellation of the running fit, if any."""
        if self._handle is not None and self._handle.is_running():
            self._cancelled = True  # the resulting Died/Failed maps to "cancelled", not "killed"
            self._handle.cancel()
            self.status_changed.emit("cancelled")

    def is_running(self) -> bool:
        """Return True while a fit worker is active."""
        return bool(self._handle is not None and self._handle.is_running())

    def shutdown(self) -> None:
        """Stop any running worker and join its reader thread (app close / atexit)."""
        if self._handle is not None:
            if self._handle.is_running():
                self._handle.cancel()    # terminate the process + tear down its reader
            self._handle.shutdown()      # join the reader QThread (also for a finished run)

    def _on_event(self, event: Any) -> None:
        if isinstance(event, LogLine):
            self.log_received.emit(event.level, event.msg)
            return
        if isinstance(event, Started):
            self.status_changed.emit("running")
            return
        if isinstance(event, Finished):
            summary = load_result_summary(event.result_path) if event.result_path else None
            self.status_changed.emit("done")
            self.fit_finished.emit(summary)
        elif isinstance(event, (Failed, Died)):
            if self._cancelled:
                self.status_changed.emit("cancelled")  # user cancel surfaced as Failed/Died
            elif isinstance(event, Failed):
                self.status_changed.emit("failed")
                self.fit_failed.emit(event.traceback)
            else:  # Died (synthetic, abnormal exit)
                self.status_changed.emit("killed")
                self.fit_failed.emit(
                    f"Worker exited abnormally (exit_code={event.exit_code}, "
                    f"signal={event.signal}) — likely out of memory."
                )
        else:
            return
        # A terminal event was handled: drop this (now-defunct) handle's
        # connection so no further events route here. We do NOT null self._handle
        # — its reader QThread is only just finishing; the handle is released when
        # the next run() reassigns it, by which time that thread has ended.
        if self._handle is not None:
            try:
                self._handle.event.disconnect(self._on_event)
            except (RuntimeError, TypeError):  # already disconnected / no slot
                pass
