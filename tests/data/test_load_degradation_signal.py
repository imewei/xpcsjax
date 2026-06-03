"""DATA-1: degraded-fallback paths must leave a detectable signal.

The quality-gate audit (failure-hunter F3/F4) flagged that when angle filtering
or the preprocessing pipeline crashes, the loader silently substitutes the
optimizer's input (all angles / raw un-preprocessed data) with only a WARNING
and no signal on the result — a caller gating on exceptions cannot tell a
crash-fallback from an intended no-op.

The fix records every degraded fallback on ``loader.load_degradations`` (and
logs at ERROR, same severity as the failure), so the degradation is
programmatically detectable downstream.
"""
from __future__ import annotations

import logging

from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _bare_loader() -> XPCSDataLoader:
    """An instance with __init__ bypassed — we only exercise the helper."""
    inst = object.__new__(XPCSDataLoader)
    inst.load_degradations = []
    return inst


def test_record_degradation_appends_and_logs_error(caplog) -> None:
    loader = _bare_loader()
    with caplog.at_level(logging.ERROR, logger="xpcsjax.data.xpcs_loader"):
        loader._record_degradation("filtering crashed: boom")

    assert loader.load_degradations == ["filtering crashed: boom"]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_record_degradation_accumulates() -> None:
    loader = _bare_loader()
    loader._record_degradation("a")
    loader._record_degradation("b")
    assert loader.load_degradations == ["a", "b"]
