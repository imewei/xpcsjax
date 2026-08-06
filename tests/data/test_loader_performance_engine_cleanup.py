"""Regression test: XPCSDataLoader must not leak its PerformanceEngine's
monitoring thread.

Both PerformanceEngine.shutdown() and AdvancedMemoryManager.shutdown() were
already fully implemented (thread join, executor shutdown, cache/mmap
cleanup) but nothing on XPCSDataLoader ever called them, so every loader
built with the (default-on) performance engine enabled left its
"PerformanceMonitoring" daemon thread running for the life of the process —
and everything it transitively kept alive (cache, worker pools, h5py
handles) along with it.
"""

from __future__ import annotations

from xpcsjax.data.performance_engine import PerformanceEngine
from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _bare_loader() -> XPCSDataLoader:
    """Bypass __init__ to avoid needing a YAML config / real dataset on disk."""
    return XPCSDataLoader.__new__(XPCSDataLoader)


def test_close_stops_performance_engine_monitoring_thread():
    loader = _bare_loader()
    loader.performance_engine = PerformanceEngine({})
    loader.memory_manager = None

    monitoring_thread = loader.performance_engine._monitoring_thread
    assert monitoring_thread is not None
    assert monitoring_thread.is_alive()

    loader.close()

    assert not monitoring_thread.is_alive()
    assert loader.performance_engine is None


def test_close_is_idempotent_and_safe_with_no_components():
    loader = _bare_loader()
    loader.performance_engine = None
    loader.memory_manager = None

    loader.close()
    loader.close()  # must not raise on a second call


def test_context_manager_closes_on_exit():
    loader = _bare_loader()
    loader.performance_engine = PerformanceEngine({})
    loader.memory_manager = None
    monitoring_thread = loader.performance_engine._monitoring_thread

    with loader:
        assert monitoring_thread.is_alive()

    assert not monitoring_thread.is_alive()
