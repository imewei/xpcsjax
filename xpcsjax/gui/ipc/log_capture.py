"""A logging handler that forwards records to the parent as LogLine events."""

from __future__ import annotations

import logging

from xpcsjax.gui.ipc.emitter import EventEmitter
from xpcsjax.service.events import LogLine


class QueueLogHandler(logging.Handler):
    """Forward each log record to ``emitter`` as a :class:`LogLine`.

    Robust by contract: logging must never raise, so emit failures are swallowed.
    """

    def __init__(self, emitter: EventEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        """Convert ``record`` to a LogLine and enqueue it; never raises."""
        try:
            self._emitter.emit(LogLine(run_id="", seq=0, level=record.levelname, msg=record.getMessage()))
        except Exception:  # noqa: BLE001 — a logging handler must not raise
            pass


class BannerLogHandler(logging.Handler):
    """Emit a Banner for each log record recognized as an engine banner line.

    Runs alongside :class:`QueueLogHandler` (which forwards the full log tail);
    this one surfaces the prominent anti-degeneracy / escape / collapse banners
    as structured events. Never raises (a logging handler must not).
    """

    def __init__(self, emitter: EventEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        """Classify the record; emit a Banner when recognized."""
        try:
            from xpcsjax.gui.ipc.diagnostics import classify_banner

            banner = classify_banner(record.levelname, record.getMessage())
            if banner is not None:
                self._emitter.emit(banner)
        except Exception:  # noqa: BLE001 — a logging handler must not raise
            pass
