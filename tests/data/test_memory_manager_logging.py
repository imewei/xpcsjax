"""Memory-manager cleanup/monitoring must be observed, not silently swallowed.

Phase-2 Task 2 of the logging overhaul. The ~47 non-fatal ``except Exception``
handlers in :mod:`xpcsjax.data.memory_manager` (cleanup / teardown / monitoring)
previously swallowed failures into a bare ``pass`` with no diagnostic. The
decided policy is OBSERVATIONAL-ONLY: a failure in a best-effort cleanup path is
now logged at DEBUG (via the Phase-1 ``logged_errors`` / ``log_once`` helpers)
while the existing control flow (the cleanup stays non-fatal, the manager keeps
working) is unchanged.

This test drives one cleanup path's failure (``_cleanup_old_pools`` raising
inside ``_handle_memory_warning``) and asserts the failure is logged at DEBUG and
does NOT escape — the manager remains usable afterwards.
"""

import logging

from xpcsjax.data.memory_manager import AdvancedMemoryManager, MemoryStats
from xpcsjax.utils import logging as xlog


def _make_manager() -> AdvancedMemoryManager:
    """Build a manager with the background pressure monitor disabled."""
    return AdvancedMemoryManager(config={"memory": {"enable_monitoring": False}})


def test_cleanup_failure_is_logged_at_debug_and_does_not_escape(caplog, monkeypatch):
    """A crash inside a best-effort cleanup path logs DEBUG and is swallowed."""
    xlog.reset_log_once_cache()
    manager = _make_manager()
    try:

        def _boom(*_args, **_kwargs):
            raise RuntimeError("cleanup boom")

        # _cleanup_old_pools is invoked from _handle_memory_warning inside a
        # best-effort guard; forcing it to raise must not escape the handler.
        monkeypatch.setattr(manager, "_cleanup_old_pools", _boom)

        stats = MemoryStats()
        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            # Must NOT raise — the cleanup failure stays non-fatal.
            manager._handle_memory_warning(stats)

        assert any(
            r.levelno == logging.DEBUG and "cleanup boom" in r.getMessage()
            for r in caplog.records
        ), "the swallowed cleanup failure must be logged at DEBUG with context"

        # The manager is still usable after the swallowed failure.
        assert isinstance(manager.get_memory_stats(), dict)
    finally:
        manager.shutdown()
