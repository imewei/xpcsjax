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
            r.levelno == logging.WARNING and "write boom" in r.getMessage()
            for r in caplog.records
        ), "a failed background write must be logged at WARNING"
    finally:
        writer.shutdown()


def test_shutdown_loop_warning_is_rate_limited_per_run(caplog):
    """Repeated write failures in one run emit the loop WARNING once, not N times."""
    with xlog.log_context(run_id="run-rate-limit"):
        writer = AsyncWriter(max_workers=2)
        try:

            def _boom() -> None:
                raise RuntimeError("repeated boom")

            # Submit several failing writes so wait_all iterates the failure
            # branch multiple times within the same run.
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
                f"once per run, got {len(warnings)}"
            )
        finally:
            writer.shutdown()
