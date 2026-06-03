"""Phase-2 observational-logging tests for the device probe fallbacks.

These assert that when a hardware/JAX-detection probe raises, the
``device/config.py`` detection path:

1. logs the failure (WARNING) WITH structured context (an ``operation`` key),
2. does NOT propagate the exception, and
3. returns the unchanged default fallback (``platform="cpu"``, ``num_devices=1``).

The logging conversion is observational-only: control flow (the default-return
fallback) must be byte-identical to the pre-conversion behavior.
"""

from __future__ import annotations

import logging

import pytest

from xpcsjax.device import config as device_config


def test_jax_probe_failure_logs_with_context_and_returns_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising JAX-backend probe is logged with context; CPU default returned."""

    # Force the JAX-device detection block to raise via the outer ``except``.
    # ``jax.extend.backend.get_backend`` raising a RuntimeError (not
    # ImportError/AttributeError) bypasses the legacy fallback and lands in the
    # broad probe-fallback handler under test.
    import jax.extend.backend as jax_backend

    def _boom() -> object:
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(jax_backend, "get_backend", _boom)

    with caplog.at_level(logging.DEBUG, logger="xpcsjax"):
        hw = device_config.detect_hardware()

    # (1)+(3): fallback default unchanged, no exception propagated.
    assert hw.platform == "cpu"
    assert hw.num_devices == 1

    # (2): the failure was logged at WARNING with structured context naming the
    # operation and carrying the original exception text.
    records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "synthetic probe failure" in r.getMessage()
    ]
    assert records, "expected a WARNING log for the JAX probe failure"
    joined = "\n".join(r.getMessage() for r in records)
    assert "operation" in joined
    assert "synthetic probe failure" in joined
