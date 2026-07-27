"""Background-write failures in :mod:`xpcsjax.utils.async_io` must be observed.

Phase-2 Task 4 of the logging overhaul. ``AsyncWriter.wait_all`` collects
background-write failures into an error list (control flow unchanged). The
decided policy is OBSERVATIONAL-ONLY: a failed write is logged at WARNING, and
because ``wait_all`` runs in a shutdown/teardown loop the WARNING is rate-limited
via ``log_once`` (keyed on the active ``run_id``) so it emits once per run rather
than once per failing future.

These tests drive write failures and assert (a) a WARNING is emitted and (b) the
shutdown-loop WARNING is emitted once across repeated failures in the same run.
"""

import logging
import time

import pytest

from xpcsjax.utils import logging as xlog
from xpcsjax.utils.async_io import AsyncWriter, PrefetchLoader


@pytest.fixture(autouse=True)
def _reset_log_once():
    """Each rate-limit assertion needs a clean log_once cache."""
    xlog.reset_log_once_cache()
    yield
    xlog.reset_log_once_cache()


def test_background_write_failure_logs_warning(caplog):
    """A failed background write surfaces a WARNING in wait_all (control intact)."""
    writer = AsyncWriter(max_workers=1)
    try:

        def _boom() -> None:
            raise RuntimeError("write boom")

        writer.submit_task(_boom)
        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            errors = writer.wait_all(timeout=10.0)

        # Control flow unchanged: the error is collected and returned.
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

        assert any(
            r.levelno == logging.WARNING and "write boom" in r.getMessage() for r in caplog.records
        ), "a failed background write must be logged at WARNING"
    finally:
        writer.shutdown()


def test_wait_all_reports_task_raised_timeout_error_not_pending(caplog):
    """A task that itself raises TimeoutError must surface as an error.

    concurrent.futures.TimeoutError is the builtin TimeoutError (3.11+) — the
    same exception Future.result(timeout=...) raises for a still-pending
    future. wait_all must disambiguate via future.done(): a DONE future that
    raised TimeoutError is a real failure, not "still in progress".
    """
    writer = AsyncWriter(max_workers=1)
    try:

        def _raises_timeout() -> None:
            raise TimeoutError("write timed out")

        writer.submit_task(_raises_timeout)
        with caplog.at_level(logging.WARNING, logger="xpcsjax"):
            errors = writer.wait_all(timeout=10.0)

        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
        assert any(
            r.levelno == logging.WARNING and "write timed out" in r.getMessage()
            for r in caplog.records
        )
        # The future must not linger forever waiting for a shutdown that will
        # never observe a new completion.
        assert len(writer._futures) == 0
    finally:
        writer.shutdown()


def test_wait_all_distinct_failures_each_logged(caplog):
    """#8: DISTINCT failures in one wait_all() must each surface a WARNING.

    Rate-limiting keyed only on the call previously collapsed every failure in a
    call to a single record, hiding all but the first when futures failed with
    different errors. The key now includes the error type+message so distinct
    failure modes are each visible (identical repeats still collapse — see the
    rate-limit test below).
    """
    writer = AsyncWriter(max_workers=2)
    try:

        def _boom_a() -> None:
            raise RuntimeError("failure alpha")

        def _boom_b() -> None:
            raise ValueError("failure beta")

        writer.submit_task(_boom_a)
        writer.submit_task(_boom_b)

        with caplog.at_level(logging.WARNING, logger="xpcsjax"):
            errors = writer.wait_all(timeout=10.0)

        assert len(errors) == 2
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Background write failed" in r.getMessage()
        ]
        joined = " ".join(r.getMessage() for r in warnings)
        assert "failure alpha" in joined and "failure beta" in joined, (
            "each distinct write failure must surface its own WARNING, not be collapsed"
        )
        assert len(warnings) == 2
    finally:
        writer.shutdown()


def test_shutdown_loop_warning_is_rate_limited_per_call(caplog):
    """A single wait_all with N>=3 failing futures emits exactly one WARNING."""
    writer = AsyncWriter(max_workers=2)
    try:

        def _boom() -> None:
            raise RuntimeError("repeated boom")

        # Submit several failing writes so wait_all iterates the failure
        # branch multiple times within the SAME call.
        for _ in range(4):
            writer.submit_task(_boom)

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            errors = writer.wait_all(timeout=10.0)

        # Control flow unchanged: every failure is still collected.
        assert len(errors) == 4

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "repeated boom" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            "the shutdown-loop write-failure WARNING must be rate-limited to "
            f"once per wait_all() call, got {len(warnings)}"
        )
    finally:
        writer.shutdown()


def test_second_wait_all_call_logs_independently(caplog):
    """A second separate wait_all() call still logs (no cross-call suppression).

    Regression guard for the None-collapsed key: keying log_once on run_id alone
    (None outside a run context) suppressed the WARNING for every later call. A
    fresh per-call token must let the second call emit its own WARNING.
    """
    writer = AsyncWriter(max_workers=1)
    try:

        def _boom() -> None:
            raise RuntimeError("call boom")

        # First call: one failing future -> one WARNING.
        writer.submit_task(_boom)
        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            first_errors = writer.wait_all(timeout=10.0)
        assert len(first_errors) == 1
        first_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "call boom" in r.getMessage()
        ]
        assert len(first_warnings) == 1

        caplog.clear()

        # Second, separate call: a new failing future must ALSO log a WARNING.
        writer.submit_task(_boom)
        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            second_errors = writer.wait_all(timeout=10.0)
        assert len(second_errors) == 1
        second_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "call boom" in r.getMessage()
        ]
        assert len(second_warnings) == 1, (
            "a second separate wait_all() call must emit its own WARNING; "
            f"got {len(second_warnings)} (cross-call suppression regression)"
        )
    finally:
        writer.shutdown()


def test_shutdown_surfaces_write_failing_after_drain_timeout(caplog):
    """A write that fails AFTER shutdown's wait_all() timeout must still be logged.

    Regression guard: ``shutdown`` runs a bounded ``wait_all`` drain, then
    ``executor.shutdown(wait=True)``. A write still in flight at the drain
    timeout is kept and finished by the executor teardown — but ``wait_all``
    never re-observes it, so a failure there used to be silently dropped,
    violating the error-observation contract. ``shutdown`` must surface it.
    """
    import threading

    writer = AsyncWriter(max_workers=1)
    release = threading.Event()

    def _blocked_then_fail() -> None:
        # Stay in flight past shutdown's (tiny) drain timeout, then fail during
        # the executor.shutdown(wait=True) drain.
        release.wait(5.0)
        raise RuntimeError("late drain boom")

    writer.submit_task(_blocked_then_fail)

    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        # Run shutdown in a worker so its wait_all(drain_timeout=...) times out
        # (the write is still blocked) and the future is kept; the worker then
        # blocks in executor.shutdown(wait=True) until we release the write.
        worker = threading.Thread(target=lambda: writer.shutdown(drain_timeout=0.05))
        worker.start()
        time.sleep(0.3)  # let wait_all time out and enter executor.shutdown(wait=True)
        release.set()  # write now raises -> drained by executor teardown
        worker.join(5.0)
        assert not worker.is_alive(), "shutdown() did not complete"

    late = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "late drain boom" in r.getMessage()
    ]
    assert len(late) == 1, (
        "a write failing during the final shutdown drain must be surfaced as a "
        f"WARNING; got {len(late)} (silent-drop regression)"
    )


def test_wait_all_timeout_is_a_shared_budget():
    """``timeout`` bounds the whole wait_all() call, not each pending future.

    Regression: the old per-future ``result(timeout=T)`` loop blocked up to
    N*T seconds for N still-pending writes, breaking shutdown()'s drain_timeout
    contract. ``wait(pending, timeout=T)`` bounds the whole call to ~T.
    """
    import threading

    release = threading.Event()
    writer = AsyncWriter(max_workers=3)
    try:
        for _ in range(3):
            writer.submit_task(lambda: release.wait(10.0))

        budget = 0.3
        start = time.monotonic()
        errors = writer.wait_all(timeout=budget)
        elapsed = time.monotonic() - start

        # All three writes are still blocked, so none completed and no errors.
        assert errors == []
        # A shared budget returns in ~budget; the N*budget bug would take >= 0.9s.
        assert elapsed < 2 * budget, (
            f"wait_all(timeout={budget}) took {elapsed:.2f}s; expected a single "
            "shared budget, not one timeout per pending future"
        )
    finally:
        release.set()
        writer.shutdown(drain_timeout=5.0)


def test_prefetch_loader_load_fn_stopiteration_is_not_exhaustion(caplog):
    """StopIteration from load_fn must surface as an error, not silent truncation."""
    loader = PrefetchLoader(iter([1, 2, 3]), lambda _item: next(iter([])))
    with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
        with pytest.raises(RuntimeError, match="StopIteration"):
            next(loader)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_prefetch_loader_source_failure_still_surfaces():
    """A non-StopIteration error from the source must still be re-raised."""

    def _bad_source():
        yield 1
        raise ValueError("source boom")

    loader = PrefetchLoader(_bad_source(), lambda item: item)
    assert next(loader) == 1
    with pytest.raises(ValueError, match="source boom"):
        next(loader)


def test_prefetch_loader_normal_exhaustion_terminates_loop():
    assert list(PrefetchLoader(iter([1, 2, 3]), lambda item: item * 2)) == [2, 4, 6]
