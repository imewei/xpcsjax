"""Defensive-logging contract tests for the logging primitives.

Logging is OBSERVATIONAL ONLY: a logging failure must never escape and must
never change numerical results or control flow. These tests pin that contract
for ``log_exception``, ``log_phase`` and ``log_quantile_scaling``.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

import xpcsjax.utils.logging as lm
from xpcsjax.optimization.nlsq import heterodyne_logging as hl


def test_log_exception_never_raises_on_bad_context():
    logger = MagicMock(spec=logging.Logger)

    class Boom:
        def __repr__(self):
            raise RuntimeError("repr blew up")

    lm.log_exception(logger, ValueError("x"), context={"bad": Boom()})
    assert logger.error.called or logger.log.called


def test_log_quantile_scaling_never_raises_on_empty():
    hl.log_quantile_scaling(np.array([]), np.array([]))  # must not raise


def test_log_phase_never_raises_when_memory_probe_fails(monkeypatch):
    monkeypatch.setattr(
        lm, "_get_memory_gb", lambda: (_ for _ in ()).throw(RuntimeError("probe"))
    )
    with lm.log_phase("p", track_memory=True):
        pass  # must not raise


def test_set_and_log_context_inject_run_id():
    import logging

    import xpcsjax.utils.logging as lm
    tok = lm.set_log_context(run_id="r1", mode="laminar_flow")
    try:
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        lm.ContextFilter().filter(rec)
        assert rec.run_id == "r1" and rec.mode == "laminar_flow"
    finally:
        lm.reset_log_context(tok)


def test_log_context_restores_on_exit():
    import logging

    import xpcsjax.utils.logging as lm
    with lm.log_context(run_id="outer"):
        with lm.log_context(run_id="inner"):
            rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
            lm.ContextFilter().filter(rec)
            assert rec.run_id == "inner"
        rec2 = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        lm.ContextFilter().filter(rec2)
        assert rec2.run_id == "outer"


def test_log_once_emits_once_per_key(caplog):
    import logging

    import xpcsjax.utils.logging as lm
    lm.reset_log_once_cache()
    lg = logging.getLogger("once")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            lm.log_once(lg, logging.WARNING, "k1", "flood %d", 1)
    assert sum("flood" in r.getMessage() for r in caplog.records) == 1


def test_logged_errors_reraise_propagates_original_and_logs(caplog):
    import logging

    import xpcsjax.utils.logging as lm
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            with lm.logged_errors(
                logging.getLogger("t"), "do_thing", policy="reraise", q=0.1
            ):
                raise ValueError("boom")
    assert any(
        "do_thing" in str(r.__dict__) or "do_thing" in r.getMessage()
        for r in caplog.records
    )


def test_logged_errors_suppress_swallows():
    import logging

    import xpcsjax.utils.logging as lm
    with lm.logged_errors(logging.getLogger("t"), "op", policy="suppress"):
        raise RuntimeError("ignored")  # reaching the next line == suppressed
