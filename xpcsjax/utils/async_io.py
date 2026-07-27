"""Async I/O utilities for pipeline overlap.

Thread-based prefetching and background writing to hide I/O latency.
GIL-safe since HDF5 and numpy release the GIL during I/O.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock, Thread
from typing import Any, TypeVar

import numpy as np

from xpcsjax.utils.logging import _LOG_CONTEXT, get_logger, log_exception, log_once

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _current_run_id() -> str | None:
    """Read the active ``run_id`` from the log-context registry, if any.

    Used to scope ``log_once`` rate-limit keys per analysis run so a repeated
    failure in a teardown loop emits once per run rather than once per item.
    """
    ctx = _LOG_CONTEXT.get() or {}
    return ctx.get("run_id")


# Per-call token source for wait_all() rate-limiting. AsyncWriter typically runs
# outside any pipeline run context, so ``run_id`` is ``None``; keying log_once on
# run_id alone would collapse to a single process-global entry and suppress the
# WARNING for every later wait_all() call (and leak across tests). A fresh token
# per invocation makes the key unique per call: one WARNING per wait_all() across
# all failing futures in that call, with no cross-call suppression.
_WAIT_ALL_CALL_COUNTER = itertools.count()


class PrefetchLoader(Iterator[R]):
    """Thread-based prefetch iterator.

    Loads the next item in a background thread while the current
    item is being processed.

    Parameters
    ----------
    source : Iterator[T]
        Source items to load.
    load_fn : callable
        Transform applied to each item in background thread.
    """

    def __init__(self, source: Iterator[T], load_fn: Callable[[T], R]) -> None:
        self._source = source
        self._load_fn = load_fn
        self._prefetched: R | None = None
        self._has_prefetched = False
        self._exhausted = False
        self._thread: Thread | None = None
        self._error: Exception | None = None
        self._start_prefetch()

    def _start_prefetch(self) -> None:
        if self._exhausted:
            return

        def _load() -> None:
            try:
                # Only the source's StopIteration means "iteration complete"; keep
                # the load_fn call outside that except so its own StopIteration is
                # not misread as exhaustion.
                try:
                    item = next(self._source)
                except StopIteration:
                    self._exhausted = True
                    return
                self._prefetched = self._load_fn(item)
                self._has_prefetched = True
            except Exception as e:
                # __next__ re-raises _error, and a bare StopIteration there would
                # still silently end the caller's for-loop — re-tag it.
                err: Exception = (
                    RuntimeError(f"prefetch load_fn raised StopIteration: {e!r}")
                    if isinstance(e, StopIteration)
                    else e
                )
                log_exception(
                    logger,
                    err,
                    context={"operation": "prefetch_load"},
                    level=logging.WARNING,
                )
                self._error = err
                self._exhausted = True

        # daemon=True: prefetch is read-only; safe to abandon on exit
        self._thread = Thread(target=_load, daemon=True)
        self._thread.start()

    def __iter__(self) -> PrefetchLoader[R]:
        """Return ``self`` (this loader is its own iterator)."""
        return self

    def __next__(self) -> R:
        """Return the prefetched item and kick off the next background load.

        Joins the in-flight prefetch thread (120 s timeout), re-raising any
        error it captured, then returns the ready item.

        Raises
        ------
        StopIteration
            When the source iterator is exhausted.
        RuntimeError
            When the prefetch thread does not finish within the 120 s timeout.
        """
        if self._thread is not None:
            self._thread.join(timeout=120.0)
            if self._thread.is_alive():
                self._exhausted = True
                self._thread = None
                timeout_err = RuntimeError("Prefetch thread did not complete within 120s timeout")
                # Store so any future (invalid) call also surfaces the error
                self._error = timeout_err
                raise timeout_err
            self._thread = None

        if self._error is not None:
            raise self._error

        if self._exhausted and not self._has_prefetched:
            raise StopIteration

        # Readiness is governed solely by `_has_prefetched`; the prefetched
        # value itself may legitimately be None (load_fn is generic over R and
        # may return None), so do not use `result is not None` as a proxy.
        result = self._prefetched
        self._has_prefetched = False
        self._prefetched = None
        self._start_prefetch()
        return result  # type: ignore[return-value]


class AsyncWriter:
    """Background thread pool for result serialization.

    Parameters
    ----------
    max_workers : int
        Maximum concurrent write threads.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[None]] = []
        self._lock = Lock()
        self._shutdown = False

    def submit_npz(self, path: Path, data: dict[str, np.ndarray]) -> None:
        """Write NPZ file in background."""

        def _write() -> None:
            try:
                self._write_npz(path, data)
            except Exception as e:
                logger.error("Failed to write NPZ %s: %s", path, e, exc_info=True)
                raise

        # Check-and-submit must be atomic w.r.t. shutdown(): a concurrent
        # shutdown could otherwise flip _shutdown and tear down the executor
        # between the check and the submit, racing _executor.submit().
        with self._lock:
            if self._shutdown:
                raise RuntimeError("AsyncWriter is shut down; cannot submit new writes")
            self._futures.append(self._executor.submit(_write))

    def submit_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file in background."""

        def _write() -> None:
            try:
                self._write_json(path, data)
            except Exception as e:
                logger.error("Failed to write JSON %s: %s", path, e, exc_info=True)
                raise

        # Atomic check-and-submit under _lock (see submit_npz).
        with self._lock:
            if self._shutdown:
                raise RuntimeError("AsyncWriter is shut down; cannot submit new writes")
            self._futures.append(self._executor.submit(_write))

    def submit_task(self, fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Submit an arbitrary callable for background execution."""
        # Atomic check-and-submit under _lock (see submit_npz).
        with self._lock:
            if self._shutdown:
                raise RuntimeError("AsyncWriter is shut down; cannot submit new writes")
            self._futures.append(self._executor.submit(fn, *args, **kwargs))

    def wait_all(self, timeout: float = 60.0) -> list[BaseException]:
        """Wait for all pending writes. Returns list of errors.

        TimeoutError is not treated as a failure — the write is still
        in progress and will complete during shutdown(). Timed-out futures
        are kept in the tracking list so their eventual errors are not lost.
        """
        with self._lock:
            pending = list(self._futures)
        errors: list[BaseException] = []
        completed: list[Future[None]] = []
        # Unique token per wait_all() invocation: scopes the rate-limit so the
        # WARNING fires once per call (not once per process), with no cross-call
        # or cross-test suppression.
        call_token = next(_WAIT_ALL_CALL_COUNTER)
        # Single shared time budget across all pending futures: `timeout` bounds
        # the whole call, not each future. (Looping future.result(timeout=timeout)
        # blocked up to N*timeout, breaking shutdown()'s drain_timeout contract.)
        done, not_done = wait(pending, timeout=timeout)
        if not_done:
            logger.info(
                "%d background write(s) still in progress after %.0fs "
                "(will complete during shutdown)",
                len(not_done),
                timeout,
            )
        for future in done:
            completed.append(future)
            exc = future.exception()
            if exc is None:
                continue
            run_id = _current_run_id()
            # Key on the call token PLUS the error type+message so DISTINCT
            # failures in one wait_all() each surface once (a bare per-call key
            # collapsed N different errors into a single record, hiding all but
            # the first). Identical repeats within the call still dedup.
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{call_token}:async_writer_write_fail:{type(exc).__name__}:{exc}",
                "Background write failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            logger.debug("Background write traceback:", exc_info=True)
            errors.append(exc)
        # Remove only futures that finished (succeeded or errored); keep timed-out ones
        with self._lock:
            for f in completed:
                try:
                    self._futures.remove(f)
                except ValueError:
                    pass
        return errors

    def shutdown(self, *, drain_timeout: float = 300.0) -> None:
        """Wait for pending writes and shut down. Idempotent.

        ``drain_timeout`` bounds the cooperative :meth:`wait_all` drain before
        the executor is torn down. A write still in flight past it is kept and
        finished by ``executor.shutdown(wait=True)``; any such write that *fails*
        during that final drain is surfaced here rather than silently dropped,
        honouring the error-observation contract (``wait_all`` never re-observes
        a future it left behind on timeout).
        """
        # Flip the flag under _lock so it serializes against the submit_*
        # check-and-submit. Release before wait_all()/executor.shutdown(), which
        # re-acquire _lock — holding it across them would deadlock.
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        errors = self.wait_all(timeout=drain_timeout)
        if errors:
            logger.error("AsyncWriter.shutdown: %d background write(s) failed", len(errors))
        self._executor.shutdown(wait=True)

        # wait_all() keeps any future still mid-write at its timeout; the
        # executor.shutdown(wait=True) above has now finished those. Surface any
        # that FAILED during that final drain — wait_all() never calls result()
        # on them again, so without this their exception is silently lost.
        with self._lock:
            remaining = list(self._futures)
            self._futures.clear()
        late_failures = 0
        for future in remaining:
            try:
                # Non-blocking: every submitted future is done after
                # executor.shutdown(wait=True).
                exc = future.exception()
            except Exception:  # noqa: BLE001 - cancelled/never-ran; nothing to surface
                continue
            if exc is not None:
                late_failures += 1
                logger.warning(
                    "Background write failed during shutdown drain (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        if late_failures:
            logger.error(
                "AsyncWriter.shutdown: %d background write(s) failed during final drain",
                late_failures,
            )

    def __del__(self) -> None:
        """Warn if garbage-collected without an explicit :meth:`shutdown`.

        A live writer collected before ``shutdown()`` may drop pending
        background writes, so emit a :class:`ResourceWarning`.
        """
        if not getattr(self, "_shutdown", True):
            import warnings

            warnings.warn(
                "AsyncWriter garbage-collected without shutdown(); background writes may be lost",
                ResourceWarning,
                stacklevel=2,
            )

    @staticmethod
    def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **data)  # type: ignore[arg-type]

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        from xpcsjax.io.json_utils import json_safe

        path.parent.mkdir(parents=True, exist_ok=True)
        # Sanitize first: Python's json encoder emits the invalid JSON tokens
        # NaN/Infinity for native non-finite floats (default= never fires for
        # float). json_safe() maps NaN -> null and Inf -> "Infinity" string so
        # the output is strict-JSON parseable.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_safe(data), f, indent=2, default=str)

    def __enter__(self) -> AsyncWriter:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, draining and shutting down the pool."""
        self.shutdown()
