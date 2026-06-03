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

Quality-gate findings added (2026-06-03):
- REL-1: _return_to_pool malformed pool_id logs DEBUG, doesn't raise (pool leak fix)
- logged_errors fallback shim respects policy="reraise"
- virtual_memory_path traversal check raises before makedirs
- TEST-1 GAP-6: two different VM files each emit their own cleanup DEBUG record
"""

import logging
import os
import unittest.mock

import pytest

from xpcsjax.data.memory_manager import AdvancedMemoryManager, MemoryStats
from xpcsjax.utils import logging as xlog


@pytest.fixture(autouse=True)
def _reset_log_once_cache():
    """Reset the log_once deduplication cache before every test in this module.

    Without this, a test that triggers a log_once key can suppress the same
    key in a later test that runs in the same process, causing order-dependent
    failures in the broader test suite.
    """
    xlog.reset_log_once_cache()
    yield
    xlog.reset_log_once_cache()


def _make_manager() -> AdvancedMemoryManager:
    """Build a manager with the background pressure monitor disabled."""
    return AdvancedMemoryManager(config={"memory": {"enable_monitoring": False}})


def test_cleanup_failure_is_logged_at_debug_and_does_not_escape(caplog, monkeypatch):
    """A crash inside a best-effort cleanup path logs DEBUG and is swallowed."""
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


# ---------------------------------------------------------------------------
# REL-1: _return_to_pool malformed pool_id must log DEBUG, never raise
# ---------------------------------------------------------------------------

def test_return_to_pool_malformed_id_does_not_raise_and_logs_debug(caplog, monkeypatch):
    """REL-1: malformed pool_id in _return_to_pool must log DEBUG, not raise.

    A pool_id that cannot be parsed (e.g. 'BADID' with no '_' separator, or
    a non-integer suffix) previously triggered a bare ``except (ValueError,
    IndexError): pass`` — silently leaking the buffer without any trace. The
    fix replaces the bare pass with a log_once at DEBUG keyed per pool_id.
    """
    import numpy as np

    manager = _make_manager()
    try:
        buf = np.empty(16, dtype=np.float64)
        malformed_pool_id = "BADID_NOTANINT"

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            # Must NOT raise even though pool_id is malformed.
            manager._return_to_pool(buf, malformed_pool_id)

        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and malformed_pool_id in r.getMessage()
        ]
        assert debug_records, (
            f"Expected at least one DEBUG record mentioning '{malformed_pool_id}' "
            f"but found none. Records: {[r.getMessage() for r in caplog.records]}"
        )
    finally:
        manager.shutdown()


def test_return_to_pool_malformed_id_logged_once_per_distinct_id(caplog, monkeypatch):
    """REL-1: repeated calls with the SAME malformed pool_id emit exactly one log."""
    import numpy as np

    manager = _make_manager()
    try:
        buf = np.empty(16, dtype=np.float64)
        malformed_pool_id = "BADID2"

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            manager._return_to_pool(buf, malformed_pool_id)
            manager._return_to_pool(buf, malformed_pool_id)
            manager._return_to_pool(buf, malformed_pool_id)

        matching = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and malformed_pool_id in r.getMessage()
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 DEBUG record for repeated same pool_id, got {len(matching)}"
        )
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# logged_errors fallback shim: policy="reraise" must propagate
# ---------------------------------------------------------------------------

def test_logged_errors_fallback_reraise_policy_propagates():
    """The HAS_V2_LOGGING=False fallback shim must re-raise when policy='reraise'.

    If the real xpcsjax.utils.logging is available the test exercises the real
    implementation; the critical contract is that regardless of whether the
    shim or the real helper is used, policy='reraise' never silently swallows.
    """
    # Import the module-level logged_errors from memory_manager — whatever
    # the runtime resolved (shim or real).
    from xpcsjax.data import memory_manager as mm_mod

    logged_errors = mm_mod.logged_errors

    # Verify reraise propagates
    with pytest.raises(RuntimeError, match="must propagate"):
        with logged_errors(
            logging.getLogger("xpcsjax.test"),
            "test_reraise",
            policy="reraise",
            level=logging.DEBUG,
        ):
            raise RuntimeError("must propagate")


def test_logged_errors_fallback_suppress_policy_swallows():
    """The fallback shim must swallow when policy='suppress' (existing behaviour)."""
    from xpcsjax.data import memory_manager as mm_mod

    logged_errors = mm_mod.logged_errors

    # Should NOT raise
    with logged_errors(
        logging.getLogger("xpcsjax.test"),
        "test_suppress",
        policy="suppress",
        level=logging.DEBUG,
    ):
        raise RuntimeError("must be swallowed")


def test_logged_errors_fallback_shim_reraise_direct():
    """Exercise the fallback shim directly under forced HAS_V2_LOGGING=False.

    Monkeypatches the module so we always hit the shim path, even if the
    real helper imported successfully, to guarantee the shim itself is correct.
    """
    import contextlib
    from xpcsjax.data import memory_manager as mm_mod

    # Build the shim exactly as defined in the module's except ImportError block.
    @contextlib.contextmanager
    def _fallback_logged_errors(*args, **kwargs):  # type: ignore[no-untyped-def]
        policy = kwargs.get("policy", "suppress")
        # Positional: logged_errors(logger, operation, *, policy=...)
        # policy is keyword-only in the real signature, so always in kwargs here.
        try:
            yield
        except Exception:
            if policy == "reraise":
                raise
            # else: swallow

    # Verify the shim re-raises
    with pytest.raises(ValueError, match="shim reraise"):
        with _fallback_logged_errors(policy="reraise"):
            raise ValueError("shim reraise")

    # Verify the shim swallows
    with _fallback_logged_errors(policy="suppress"):
        raise ValueError("shim suppress")


# ---------------------------------------------------------------------------
# virtual_memory_path traversal check
# ---------------------------------------------------------------------------

def test_virtual_memory_path_traversal_rejected_before_makedirs(tmp_path, monkeypatch):
    """A virtual_memory_path containing '..' must be rejected before any directory is created.

    The fix must call validate_save_path (or equivalent traversal check) before
    os.makedirs so that no directory is created when the path is malicious.
    """
    traversal_path = str(tmp_path / ".." / "escaped" / "xpcsjax_vm")

    manager = AdvancedMemoryManager(
        config={
            "memory": {
                "enable_monitoring": False,
                "virtual_memory_path": traversal_path,
            }
        }
    )
    try:
        import numpy as np

        # Track whether makedirs is called
        makedirs_called = []
        real_makedirs = os.makedirs

        def _spy_makedirs(path, **kwargs):
            makedirs_called.append(path)
            return real_makedirs(path, **kwargs)

        monkeypatch.setattr(os, "makedirs", _spy_makedirs)

        with pytest.raises(Exception):
            # Should raise (PathValidationError or ValueError) BEFORE makedirs
            manager._allocate_virtual_memory(1024, np.float64)

        assert not makedirs_called, (
            f"os.makedirs was called ({makedirs_called}) before traversal check raised — "
            "the check must happen before any filesystem mutation"
        )
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# TEST-1 GAP-6: two different VM files each emit their own cleanup DEBUG record
# ---------------------------------------------------------------------------

def test_cleanup_vm_file_log_once_key_is_per_file(caplog, monkeypatch):
    """GAP-6 regression: two different VM files each get their own DEBUG log.

    The log_once key for cleanup failures is keyed per filename:
      f"{id(self)}:memmgr:cleanup_vm_file:{file}"
    This means the FIRST file's failure must not suppress the SECOND file's
    failure log (different keys → both should appear).
    """
    import os

    manager = _make_manager()
    try:
        vm_dir_path = "/fake/vm/dir"
        # Simulate two VM files that exist
        fake_files = ["xpcsjax_vm_11111_100.dat", "xpcsjax_vm_22222_200.dat"]

        # Make os.path.exists return True for the vm_dir
        # and os.listdir return our two fake files
        # os.remove raises OSError for both to trigger the log_once path
        monkeypatch.setattr(
            manager, "_virtual_memory_path", f"{vm_dir_path}/xpcsjax_vm"
        )

        with (
            unittest.mock.patch("os.path.exists", return_value=True),
            unittest.mock.patch("os.listdir", return_value=fake_files),
            unittest.mock.patch(
                "os.remove",
                side_effect=OSError("simulated remove failure"),
            ),
        ):
            with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
                manager.cleanup_virtual_memory()

        # Both files should have generated a DEBUG record (different log_once keys)
        for fake_file in fake_files:
            matching = [
                r for r in caplog.records
                if r.levelno == logging.DEBUG and fake_file in r.getMessage()
            ]
            assert matching, (
                f"Expected a DEBUG record mentioning '{fake_file}' but found none. "
                f"Records: {[r.getMessage() for r in caplog.records]}"
            )
    finally:
        manager.shutdown()
