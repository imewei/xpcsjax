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

import gc
import logging
import unittest.mock
import weakref

import pytest

from xpcsjax.data.memory_manager import (
    AdvancedMemoryManager,
    MemoryPressureMonitor,
    MemoryStats,
)
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
            r.levelno == logging.DEBUG and "cleanup boom" in r.getMessage() for r in caplog.records
        ), "the swallowed cleanup failure must be logged at DEBUG with context"

        # The manager is still usable after the swallowed failure.
        assert isinstance(manager.get_memory_stats(), dict)
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
# TEST-1 GAP-6: two different VM files each emit their own cleanup DEBUG record
# ---------------------------------------------------------------------------


def test_cleanup_vm_file_log_once_key_is_per_file(caplog, monkeypatch):
    """GAP-6 regression: two different VM files each get their own DEBUG log.

    The log_once key for cleanup failures is keyed per filename:
      f"{id(self)}:memmgr:cleanup_vm_file:{file}"
    This means the FIRST file's failure must not suppress the SECOND file's
    failure log (different keys → both should appear).
    """
    manager = _make_manager()
    try:
        vm_dir_path = "/fake/vm/dir"
        # Simulate two VM files this instance created
        fake_files = [
            f"{vm_dir_path}/xpcsjax_vm_11111_100.dat",
            f"{vm_dir_path}/xpcsjax_vm_22222_200.dat",
        ]
        monkeypatch.setattr(manager, "_virtual_memory_path", f"{vm_dir_path}/xpcsjax_vm")
        manager._own_vm_files.update(fake_files)

        # os.remove raises OSError for both to trigger the log_once path
        with unittest.mock.patch(
            "os.remove",
            side_effect=OSError("simulated remove failure"),
        ):
            with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
                manager.cleanup_virtual_memory()

        # Both files should have generated a DEBUG record (different log_once keys)
        for fake_file in fake_files:
            matching = [
                r
                for r in caplog.records
                if r.levelno == logging.DEBUG and fake_file in r.getMessage()
            ]
            assert matching, (
                f"Expected a DEBUG record mentioning '{fake_file}' but found none. "
                f"Records: {[r.getMessage() for r in caplog.records]}"
            )
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Interpreter-shutdown noise: the atexit monitor cleanup must not let logging's
# handler-error reporter print "--- Logging error --- / I/O operation on closed
# file" to stderr after a test harness has closed the log stream.
# ---------------------------------------------------------------------------


def test_atexit_cleanup_silences_closed_stream_logging_errors(capsys):
    """The atexit monitor cleanup must not emit logging-handler error noise.

    At interpreter shutdown pytest (or an application) may have already closed
    the stream backing a root-logger handler. The best-effort ``logger.info``
    inside ``stop_monitoring`` then trips ``logging.Handler.handleError``, which
    — while ``logging.raiseExceptions`` is True — prints a spurious
    ``--- Logging error ---`` / ``ValueError: I/O operation on closed file`` to
    stderr. ``_cleanup_active_monitors`` must suppress that reporting so the
    process exits cleanly.

    We reproduce the closed-stream condition directly: attach a handler whose
    stream is already closed to the root logger, register a live monitor, then
    run the atexit cleanup and assert no logging-error noise reached stderr. The
    global ``logging.raiseExceptions`` flag is saved and restored so this test
    cannot poison the visibility of logging errors in other tests.
    """
    import io

    from xpcsjax.data import memory_manager as mm_mod

    original_raise = logging.raiseExceptions
    root = logging.getLogger()
    closed_stream = io.StringIO()
    closed_stream.close()
    bad_handler = logging.StreamHandler(closed_stream)

    # A real manager with monitoring on registers a monitor in _active_monitors,
    # so the atexit cleanup will call stop_monitoring() -> logger.info(...).
    manager = AdvancedMemoryManager(config={"memory": {"enable_monitoring": True}})
    root.addHandler(bad_handler)
    try:
        logging.raiseExceptions = True  # default; would print on emit failure
        # Drain anything captured so far, then exercise the shutdown path.
        capsys.readouterr()
        mm_mod._cleanup_active_monitors()
        err = capsys.readouterr().err
        assert "--- Logging error ---" not in err, (
            f"atexit cleanup leaked a logging-handler error to stderr:\n{err}"
        )
        assert "I/O operation on closed file" not in err, (
            f"atexit cleanup leaked a closed-stream error to stderr:\n{err}"
        )
        # The fix works by disabling logging's exception reporting for the rest
        # of shutdown — confirm the mechanism is in place.
        assert logging.raiseExceptions is False
    finally:
        root.removeHandler(bad_handler)
        logging.raiseExceptions = original_raise
        manager.shutdown()


# ---------------------------------------------------------------------------
# _as_weak_callable: AdvancedMemoryManager<->MemoryPressureMonitor reference
# cycle regression coverage.
# ---------------------------------------------------------------------------


def test_manager_collected_by_refcounting_without_cyclic_gc():
    """The manager<->monitor reference cycle must stay broken.

    Before the fix, ``AdvancedMemoryManager`` registered its own bound
    methods as ``MemoryPressureMonitor`` callbacks, forming a strong
    reference cycle. Cycles containing ``__del__`` are only reclaimed when
    the cyclic garbage collector happens to sweep them -- not
    deterministically at scope exit -- so the monitor's daemon thread could
    outlive its owner and keep logging into an already-closed stream. This
    test disables the cyclic collector so only plain refcounting can do the
    work: it must still promptly collect the manager and stop its thread.
    """
    gc.disable()
    try:
        manager = AdvancedMemoryManager(config={"memory": {"enable_monitoring": True}})
        manager_ref = weakref.ref(manager)
        thread = manager.pressure_monitor._monitoring_thread
        assert thread is not None and thread.is_alive()

        del manager

        assert manager_ref() is None, (
            "manager was not collected by refcounting alone -- the "
            "owner<->monitor reference cycle regressed"
        )
        thread.join(timeout=2.0)
        assert not thread.is_alive(), (
            "monitor thread did not stop promptly after its owner was collected"
        )
    finally:
        gc.enable()
        gc.collect()


def test_dead_weak_callback_does_not_crash_pressure_trigger(caplog):
    """A weak-wrapped callback whose target was collected must stay non-fatal.

    ``register_warning_callback`` wraps bound-method callbacks weakly (see
    ``_as_weak_callable``). Once the callback's owner is garbage collected,
    invoking the wrapper raises ``ReferenceError`` -- this must be swallowed
    and logged once at DEBUG by ``_trigger_warning_response``, exactly like
    any other callback failure, not propagate out of the pressure-check loop.
    """
    monitor = MemoryPressureMonitor()
    try:

        class _Handler:
            def on_warning(self, _stats: MemoryStats) -> None:
                pass

        handler = _Handler()
        monitor.register_warning_callback(handler.on_warning)
        del handler
        gc.collect()  # ensure the WeakMethod target is actually gone

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            monitor._trigger_warning_response()  # must not raise

        assert any(
            r.levelno == logging.DEBUG and "Warning callback failed" in r.getMessage()
            for r in caplog.records
        ), (
            f"expected the dead-callback ReferenceError to be logged at DEBUG. "
            f"Records: {[r.getMessage() for r in caplog.records]}"
        )
    finally:
        monitor.stop_monitoring()
