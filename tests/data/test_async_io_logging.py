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

import pytest

from xpcsjax.utils import logging as xlog
from xpcsjax.utils.async_io import AsyncWriter


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
    """A second independent wait_all() call still logs (no cross-call suppression).

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

        # Second, independent call: a new failing future must ALSO log a WARNING.
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
            "a second independent wait_all() call must emit its own WARNING; "
            f"got {len(second_warnings)} (cross-call suppression regression)"
        )
    finally:
        writer.shutdown()
