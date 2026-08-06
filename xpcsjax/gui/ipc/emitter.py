"""Stamps and enqueues FitEvents from the worker to the parent."""

from __future__ import annotations

import queue as _queue
import time
from dataclasses import replace
from typing import Any

from xpcsjax.service.events import TERMINAL_EVENTS, FitEvent, Iteration, LogLine

_TELEMETRY_HZ = 20.0  # coalesce high-rate Iteration/LogLine to ~20 Hz (spec §4)
_TELEMETRY_MIN_INTERVAL = 1.0 / _TELEMETRY_HZ


class EventEmitter:
    """Stamp each event with the run id + a monotonic seq, then enqueue it.

    Terminal events (``Finished``/``Failed``/``Died``) block until the bounded
    queue has room — they must never be lost. High-rate telemetry
    (``Iteration``/``LogLine``) is **coalesced to ~20 Hz** (keep-latest: intermediate
    samples are skipped) so a lagging reader can never be flooded; the bounded
    queue's ``put_nowait`` drop is a further backstop for any non-terminal event.
    """

    def __init__(self, event_queue: Any, run_id: str) -> None:
        self._queue = event_queue
        self._run_id = run_id
        self._seq = 0
        self._last_telemetry = 0.0

    def emit(self, event: FitEvent) -> None:
        """Stamp ``event`` and enqueue it (see class docstring for the policy)."""
        # Throttle high-rate telemetry BEFORE stamping (so seq has no gaps for
        # actually-emitted events). LayerStatus/Banner/Started are not throttled.
        # WARNING/ERROR/CRITICAL LogLines are exempt too -- they're low-rate and
        # dropping one to the 20 Hz coalesce would silently hide the exact
        # severity status_manager's coloring exists to surface.
        if isinstance(event, Iteration) or (
            isinstance(event, LogLine) and event.level not in ("WARNING", "ERROR", "CRITICAL")
        ):
            now = time.monotonic()
            if now - self._last_telemetry < _TELEMETRY_MIN_INTERVAL:
                return
            self._last_telemetry = now
        self._seq += 1
        stamped = replace(event, run_id=self._run_id, seq=self._seq)
        if isinstance(stamped, TERMINAL_EVENTS):
            self._queue.put(stamped)
            return
        try:
            self._queue.put_nowait(stamped)
        except _queue.Full:
            pass  # drop non-terminal telemetry under backpressure (final backstop)

    def __call__(self, event: FitEvent) -> None:
        """Alias so an :class:`EventEmitter` is usable as an ``on_event`` callback."""
        self.emit(event)
