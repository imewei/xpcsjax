"""Optional-component init failures in :mod:`xpcsjax.data.optimization` must be
observed, not silently degraded.

``AdvancedDatasetOptimizer`` degrades to ``None`` when an optional component
(performance engine / memory manager) fails to initialise. The control flow is
non-fatal (the component degrades to ``None``, the optimizer keeps working), but
an *unexpected* (non-ImportError) failure must be visible at the default log
level. Quality-gate finding #3 corrected an earlier downgrade of these handlers
to DEBUG — they now log at WARNING (ImportError, the expected optional-dep case,
also logs at WARNING).

These tests drive an init failure and assert the failure is logged at WARNING
with context and does NOT escape — the component degrades to ``None``.
"""

import logging

import xpcsjax.data.optimization as opt
from xpcsjax.utils import logging as xlog


def test_performance_engine_init_failure_logs_warning_and_degrades(caplog, monkeypatch):
    """A crash importing PerformanceEngine logs at WARNING with context and -> None."""
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
    monkeypatch.setattr("xpcsjax.data.performance_engine.PerformanceEngine", _boom, raising=False)

    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        # Must NOT raise — the init failure stays non-fatal.
        optimizer._init_performance_engine()

    # Degrades to None.
    assert optimizer.performance_engine is None

    # The unexpected init failure is surfaced at WARNING with context.
    assert any(
        r.levelno == logging.WARNING and "engine boom" in r.getMessage() for r in caplog.records
    ), "performance-engine init failure must be logged at WARNING with context"


def test_memory_manager_init_failure_logs_warning_and_degrades(caplog, monkeypatch):
    """A crash importing AdvancedMemoryManager logs at WARNING and degrades to None."""
    xlog.reset_log_once_cache()

    optimizer = opt.AdvancedDatasetOptimizer.__new__(opt.AdvancedDatasetOptimizer)
    optimizer.config = {}
    optimizer.memory_manager = "sentinel"  # type: ignore[assignment]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("memory boom")

    monkeypatch.setattr("xpcsjax.data.memory_manager.AdvancedMemoryManager", _boom, raising=False)

    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        optimizer._init_memory_manager()

    assert optimizer.memory_manager is None
    assert any(
        r.levelno == logging.WARNING and "memory boom" in r.getMessage() for r in caplog.records
    ), "memory-manager init failure must be logged at WARNING with context"
