"""Logging-parity tests: two_component must emit the same setup-log narrative
as laminar_flow.

The homodyne / laminar path (``core.fit_nlsq_jax``) emits two setup blocks the
heterodyne dispatch historically skipped:

* the ``xpcsjax.device.cpu`` CPU/HPC threading-configuration banner, and
* the ``xpcsjax.optimization.nlsq.memory`` ``memory_strategy_selection`` phase
  plus the ``Adaptive memory threshold`` line.

These are emitted by best-effort guarded helpers at the single heterodyne
chokepoint (:func:`xpcsjax.optimization.nlsq._fit_nlsq_heterodyne`), mirroring
the existing ``_safe_log_heterodyne_start`` pattern. The helpers must never
raise (logging is non-critical) and must emit the parity records.
"""

import logging

from xpcsjax.optimization.nlsq import (
    _safe_configure_cpu_threading,
    _safe_log_memory_strategy,
)


def test_safe_configure_cpu_threading_emits_device_cpu_block(caplog):
    """Mirrors laminar's ``xpcsjax.device.cpu`` configuration banner."""
    with caplog.at_level(logging.INFO, logger="xpcsjax.device.cpu"):
        _safe_configure_cpu_threading()  # must not raise

    msgs = [r.getMessage() for r in caplog.records if r.name == "xpcsjax.device.cpu"]
    # The device.cpu module ships with xpcsjax, so the banner is always emitted.
    assert any("Configuring CPU optimization" in m for m in msgs), msgs


def test_safe_log_memory_strategy_emits_phase_and_threshold(caplog):
    """Mirrors laminar's ``memory_strategy_selection`` phase + threshold line."""
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.memory"):
        _safe_log_memory_strategy()  # must not raise

    text = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name == "xpcsjax.optimization.nlsq.memory"
    )
    assert "Adaptive memory threshold" in text, text
    assert "memory_strategy_selection" in text, text


def test_helpers_never_raise(monkeypatch):
    """Logging must never break a fit even when the optional deps are absent."""
    # Force the lazy imports inside the helpers to fail; helpers must swallow it.
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if "device.cpu" in name or name.endswith(".memory") or "utils.logging" in name:
            raise ImportError(f"forced failure for {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    _safe_configure_cpu_threading()  # no raise
    _safe_log_memory_strategy()  # no raise
