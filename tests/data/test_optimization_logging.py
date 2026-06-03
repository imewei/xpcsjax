"""Optional-component init failures in :mod:`xpcsjax.data.optimization` must be
observed, not silently degraded.

Phase-2 Task 4 of the logging overhaul. ``AdvancedDatasetOptimizer`` degrades to
``None`` when an optional component (performance engine / memory manager) fails
to initialise. The decided policy is OBSERVATIONAL-ONLY: the failure is logged at
DEBUG (via the Phase-1 ``log_exception`` helper, carrying structured context)
while the existing control flow (the component degrades to ``None``, the
optimizer keeps working) is unchanged.

These tests drive an init failure and assert the failure is logged at DEBUG with
context and does NOT escape — the component degrades to ``None``.
"""

import logging

import xpcsjax.data.optimization as opt
from xpcsjax.utils import logging as xlog


def test_performance_engine_init_failure_logs_debug_and_degrades(caplog, monkeypatch):
    """A crash importing PerformanceEngine logs DEBUG with context and -> None."""
    xlog.reset_log_once_cache()

    # Build an optimizer instance without running __init__ so we can drive the
    # init helper in isolation.
    optimizer = opt.AdvancedDatasetOptimizer.__new__(opt.AdvancedDatasetOptimizer)
    optimizer.config = {}
    optimizer.performance_engine = "sentinel"  # type: ignore[assignment]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("engine boom")

    # The helper imports PerformanceEngine lazily; force the construction path to
    # raise a non-ImportError so the general fallback branch runs.
    monkeypatch.setattr(
        "xpcsjax.data.performance_engine.PerformanceEngine", _boom, raising=False
    )

    with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
        # Must NOT raise — the init failure stays non-fatal.
        optimizer._init_performance_engine()

    # Degrades to None.
    assert optimizer.performance_engine is None

    # The swallowed init failure is logged at DEBUG with context.
    assert any(
        r.levelno == logging.DEBUG and "engine boom" in r.getMessage()
        for r in caplog.records
    ), "performance-engine init failure must be logged at DEBUG with context"


def test_memory_manager_init_failure_logs_debug_and_degrades(caplog, monkeypatch):
    """A crash importing AdvancedMemoryManager logs DEBUG and degrades to None."""
    xlog.reset_log_once_cache()

    optimizer = opt.AdvancedDatasetOptimizer.__new__(opt.AdvancedDatasetOptimizer)
    optimizer.config = {}
    optimizer.memory_manager = "sentinel"  # type: ignore[assignment]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("memory boom")

    monkeypatch.setattr(
        "xpcsjax.data.memory_manager.AdvancedMemoryManager", _boom, raising=False
    )

    with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
        optimizer._init_memory_manager()

    assert optimizer.memory_manager is None
    assert any(
        r.levelno == logging.DEBUG and "memory boom" in r.getMessage()
        for r in caplog.records
    ), "memory-manager init failure must be logged at DEBUG with context"
