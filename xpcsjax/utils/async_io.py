"""Async I/O utilities for pipeline overlap.

Thread-based prefetching and background writing to hide I/O latency.
GIL-safe since HDF5 and numpy release the GIL during I/O.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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
                item = next(self._source)
                self._prefetched = self._load_fn(item)
                self._has_prefetched = True
            except StopIteration:
                self._exhausted = True
            except Exception as e:
                log_exception(
                    logger,
                    e,
                    context={"operation": "prefetch_load"},
                    level=logging.WARNING,
                )
                self._error = e
                self._exhausted = True

        # daemon=True: prefetch is read-only; safe to abandon on exit
        self._thread = Thread(target=_load, daemon=True)
        self._thread.start()

    def __iter__(self) -> PrefetchLoader[R]:
        return self

    def __next__(self) -> R:
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

        result = self._prefetched
        assert result is not None, "_has_prefetched is True but _prefetched is None"
        self._has_prefetched = False
        self._prefetched = None
        self._start_prefetch()
        return result


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

    def wait_all(self, timeout: float = 60.0) -> list[Exception]:
        """Wait for all pending writes. Returns list of errors.

        TimeoutError is not treated as a failure — the write is still
        in progress and will complete during shutdown(). Timed-out futures
        are kept in the tracking list so their eventual errors are not lost.
        """
        with self._lock:
            pending = list(self._futures)
        errors: list[Exception] = []
        completed: list[Future[None]] = []
        # Unique token per wait_all() invocation: scopes the rate-limit so the
        # WARNING fires once per call (not once per process), with no cross-call
        # or cross-test suppression.
        call_token = next(_WAIT_ALL_CALL_COUNTER)
        for future in pending:
            try:
                future.result(timeout=timeout)
                completed.append(future)
            except TimeoutError:
                logger.info(
                    "Background write still in progress after %.0fs "
                    "(will complete during shutdown)",
                    timeout,
                )
                # Do NOT mark as completed — keep in _futures so shutdown() sees it
            except Exception as e:
                run_id = _current_run_id()
                # Key on the call token PLUS the error type+message so DISTINCT
                # failures in one wait_all() each surface once (a bare per-call key
                # collapsed N different errors into a single record, hiding all but
                # the first). Identical repeats within the call still dedup.
                log_once(
                    logger,
                    logging.WARNING,
                    f"{run_id}:{call_token}:async_writer_write_fail:{type(e).__name__}:{e}",
                    "Background write failed (%s): %s",
                    type(e).__name__,
                    e,
                )
                logger.debug("Background write traceback:", exc_info=True)
                errors.append(e)
                completed.append(future)
        # Remove only futures that finished (succeeded or errored); keep timed-out ones
        with self._lock:
            for f in completed:
                try:
                    self._futures.remove(f)
                except ValueError:
                    pass
        return errors

    def shutdown(self) -> None:
        """Wait for pending writes and shut down. Idempotent."""
        # Flip the flag under _lock so it serializes against the submit_*
        # check-and-submit. Release before wait_all()/executor.shutdown(), which
        # re-acquire _lock — holding it across them would deadlock.
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        errors = self.wait_all(timeout=300.0)
        if errors:
            logger.error("AsyncWriter.shutdown: %d background write(s) failed", len(errors))
        self._executor.shutdown(wait=True)

    def __del__(self) -> None:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def __enter__(self) -> AsyncWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
