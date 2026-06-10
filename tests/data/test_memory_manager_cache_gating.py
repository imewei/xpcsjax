"""Emergency cleanup must not thrash JAX recompiles under live-array pressure.

RCA (C044 ``two_component`` >=1M stratified-LS, log
``xpcsjax_two_component_20260610_110842.log``): the background
``MemoryPressureMonitor`` fired CRITICAL repeatedly *during* an active NLSQ
``trf``+``soft_l1`` solve. ``_emergency_memory_cleanup`` responded each time with
``jax.clear_caches()``, evicting the compiled ``residual_fn`` / nlsq Jacobian
executables and forcing an XLA recompile on the next solver step. Because these
fits are cold-compile-dominated, the recompile itself spiked memory + time —
a feedback loop that raised peak pressure while freeing *none* of the live
working-set arrays (every ``gc.collect()`` freed 0 objects).

The fix gates the cache clear on the same "GC is freeing nothing => memory is
live JAX/NumPy arrays" signal the warning path already trusts, plus a cooldown.
These tests pin that gating. They are pure control-flow assertions (mocked
``jax.clear_caches`` / ``gc.collect``); no real fit runs.
"""

import logging

import pytest

from xpcsjax.data import memory_manager as mm_mod
from xpcsjax.data.memory_manager import AdvancedMemoryManager


def _make_manager() -> AdvancedMemoryManager:
    """Manager with the background pressure monitor disabled (deterministic)."""
    return AdvancedMemoryManager(config={"memory": {"enable_monitoring": False}})


@pytest.mark.skipif(not mm_mod.HAS_JAX, reason="JAX required for cache-clear gating")
def test_live_array_pressure_skips_jax_cache_clear(monkeypatch, caplog):
    """When GC keeps freeing 0 (live arrays), the cache clear must be skipped."""
    manager = _make_manager()
    try:
        calls: list[int] = []
        monkeypatch.setattr(mm_mod.jax, "clear_caches", lambda: calls.append(1))
        # Live-array regime: gc.collect frees nothing.
        monkeypatch.setattr(mm_mod.gc, "collect", lambda *a, **k: 0)
        manager._consecutive_zero_gc = 3  # established "memory is live" signal

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            manager._emergency_memory_cleanup()

        assert calls == [], (
            "jax.clear_caches() must be skipped when pressure is live JAX/NumPy "
            "arrays — clearing forces recompiles without freeing the working set"
        )
        assert any(
            "Skipping jax.clear_caches()" in r.getMessage() for r in caplog.records
        ), "the skip must be logged honestly at DEBUG"
    finally:
        manager.shutdown()


@pytest.mark.skipif(not mm_mod.HAS_JAX, reason="JAX required for cache-clear gating")
def test_jax_cache_clear_respects_cooldown(monkeypatch):
    """Two emergencies inside the cooldown window trigger exactly one clear."""
    manager = _make_manager()
    try:
        calls: list[int] = []
        monkeypatch.setattr(mm_mod.jax, "clear_caches", lambda: calls.append(1))
        # Productive GC so the live-array gate does NOT apply; isolate the cooldown.
        monkeypatch.setattr(mm_mod.gc, "collect", lambda *a, **k: 5)
        manager._consecutive_zero_gc = 0
        manager._jax_cache_clear_cooldown_s = 1000.0

        manager._emergency_memory_cleanup()  # first: allowed
        manager._emergency_memory_cleanup()  # second: within cooldown -> skipped

        assert calls == [1], (
            f"expected exactly one cache clear within the cooldown window, "
            f"got {len(calls)}"
        )
    finally:
        manager.shutdown()


@pytest.mark.skipif(not mm_mod.HAS_JAX, reason="JAX required for cache-clear gating")
def test_productive_gc_after_cooldown_allows_clear(monkeypatch):
    """With productive GC and no cooldown, the clear proceeds every time."""
    manager = _make_manager()
    try:
        calls: list[int] = []
        monkeypatch.setattr(mm_mod.jax, "clear_caches", lambda: calls.append(1))
        monkeypatch.setattr(mm_mod.gc, "collect", lambda *a, **k: 5)
        manager._consecutive_zero_gc = 0
        manager._jax_cache_clear_cooldown_s = 0.0  # cooldown disabled

        manager._emergency_memory_cleanup()
        manager._emergency_memory_cleanup()

        assert calls == [1, 1], (
            "clear should proceed when GC is productive and the cooldown elapsed"
        )
    finally:
        manager.shutdown()


def test_warning_demoted_to_debug_in_live_array_regime(caplog):
    """A live-array regime logs pressure warnings at DEBUG, not WARNING.

    During a large solve the pressure is held by live JAX/NumPy arrays the
    manager cannot free, so the warning is not actionable. The monitor must log
    it calmly (DEBUG, "no action available") rather than as an alarming WARNING.
    """
    manager = _make_manager()
    try:
        monitor = manager.pressure_monitor
        manager._consecutive_zero_gc = 3  # live-array regime established
        monitor.stats.memory_pressure = 0.85
        monitor.stats.available_memory_gb = 9.3

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            monitor._trigger_warning_response()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "no action available" in r.getMessage()
        ]
        assert not warnings, "pressure warning must NOT be WARNING-level under live arrays"
        assert debugs, "expected a calm DEBUG 'no action available' record instead"
    finally:
        manager.shutdown()


def test_critical_demoted_to_warning_in_live_array_regime(caplog):
    """Critical pressure under live arrays stays visible (WARNING) but not CRITICAL."""
    manager = _make_manager()
    try:
        monitor = manager.pressure_monitor
        manager._consecutive_zero_gc = 3
        monitor.stats.memory_pressure = 0.90
        monitor.stats.available_memory_gb = 6.2

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            monitor._trigger_critical_response()

        criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        warns = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "best-effort only" in r.getMessage()
        ]
        assert not criticals, "must NOT scream CRITICAL when pressure is live arrays"
        assert warns, "critical pressure must stay visible at WARNING with honest framing"
    finally:
        manager.shutdown()


def test_normal_regime_preserves_actionable_levels(caplog):
    """Outside the live-array regime the original WARNING/CRITICAL are preserved."""
    manager = _make_manager()
    try:
        monitor = manager.pressure_monitor
        manager._consecutive_zero_gc = 0  # garbage IS reclaimable => actionable
        monitor.stats.memory_pressure = 0.92
        monitor.stats.available_memory_gb = 5.0

        with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
            monitor._trigger_warning_response()
            monitor._trigger_critical_response()

        assert any(
            r.levelno == logging.WARNING and "Memory pressure warning" in r.getMessage()
            for r in caplog.records
        ), "normal regime must keep the actionable WARNING"
        assert any(
            r.levelno == logging.CRITICAL for r in caplog.records
        ), "normal regime must keep the actionable CRITICAL"
    finally:
        manager.shutdown()


@pytest.mark.skipif(not mm_mod.HAS_JAX, reason="JAX required for cache-clear gating")
def test_emergency_gc_does_not_loop_when_unproductive(monkeypatch):
    """A first zero-result GC must short-circuit the old 3x collect loop."""
    manager = _make_manager()
    try:
        n_collects = {"n": 0}

        def _count(*_a, **_k):
            n_collects["n"] += 1
            return 0

        monkeypatch.setattr(mm_mod.gc, "collect", _count)
        monkeypatch.setattr(mm_mod.jax, "clear_caches", lambda: None)

        manager._emergency_memory_cleanup()

        assert n_collects["n"] == 1, (
            f"emergency cleanup should collect once (not 3x) when memory is live; "
            f"got {n_collects['n']} gc.collect() calls"
        )
    finally:
        manager.shutdown()
