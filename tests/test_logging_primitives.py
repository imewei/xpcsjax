"""Defensive-logging contract tests for the logging primitives.

Logging is OBSERVATIONAL ONLY: a logging failure must never escape and must
never change numerical results or control flow. These tests pin that contract
for ``log_exception``, ``log_phase`` and ``log_quantile_scaling``.
"""

import json
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


def test_log_operation_logging_failure_does_not_mask_original_exception():
    """If the failure-path ``logger.log`` itself raises, the caller's ORIGINAL
    exception must still propagate unchanged (logging is observational-only)."""
    logger = MagicMock(spec=logging.Logger)

    def _log(level, *args, **kwargs):
        # Only the failure-path (ERROR) log explodes; entry/success logs pass.
        if level == logging.ERROR:
            raise RuntimeError("logging backend exploded")

    logger.log.side_effect = _log

    with pytest.raises(ValueError, match="original failure"):
        with lm.log_operation("op", logger=logger):
            raise ValueError("original failure")


def test_log_phase_never_raises_when_memory_probe_fails(monkeypatch):
    monkeypatch.setattr(lm, "_get_memory_gb", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
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
            with lm.logged_errors(logging.getLogger("t"), "do_thing", policy="reraise", q=0.1):
                raise ValueError("boom")
    assert any(
        "do_thing" in str(r.__dict__) or "do_thing" in r.getMessage() for r in caplog.records
    )


def test_logged_errors_suppress_swallows():
    import logging

    import xpcsjax.utils.logging as lm

    with lm.logged_errors(logging.getLogger("t"), "op", policy="suppress"):
        raise RuntimeError("ignored")  # reaching the next line == suppressed


def test_json_formatter_schema_and_jsonsafe():
    import logging

    import xpcsjax.utils.logging as lm

    fmt = lm.JSONFormatter()
    rec = logging.LogRecord("lg", logging.INFO, __file__, 10, "hello", None, None)
    rec.run_id = "r1"
    rec.context = {"q": np.float64(0.1), "nan": float("nan")}
    out = json.loads(fmt.format(rec))
    assert out["message"] == "hello" and out["level"] == "INFO" and out["run_id"] == "r1"
    assert out["context"]["q"] == 0.1 and out["context"]["nan"] is None
    assert out["schema_version"] >= 1


def test_json_formatter_redacts_secrets():
    import logging

    import xpcsjax.utils.logging as lm

    fmt = lm.JSONFormatter()
    rec = logging.LogRecord("lg", logging.INFO, __file__, 1, "m", None, None)
    rec.context = {"API_KEY": "sk-123", "data_path": "/home/u/x.h5"}
    out = json.loads(fmt.format(rec))
    assert out["context"]["API_KEY"] == "***REDACTED***"
    assert out["context"]["data_path"] == "/home/u/x.h5"  # paths NOT redacted


def test_phaselogger_banner_widths_and_never_raises(caplog):
    import logging

    import xpcsjax.utils.logging as lm

    pl = lm.PhaseLogger(logging.getLogger("p"))
    with caplog.at_level(logging.INFO):
        pl.banner("OPTIMIZATION RESULTS", width=80)
        pl.banner("ANTI-DEGENERACY", width=60)
    lines = [r.getMessage() for r in caplog.records]
    assert any(len(line) == 80 for line in lines)
    assert any(len(line) == 60 for line in lines)
    pl.field("contrast", None)  # malformed value must not raise


def test_json_formatter_handles_empty_exc_info_tuple():
    import logging

    import xpcsjax.utils.logging as lm

    fmt = lm.JSONFormatter()
    rec = logging.LogRecord("lg", logging.INFO, __file__, 1, "m", None, None)
    rec.exc_info = (None, None, None)  # stdlib-truthy but empty
    out = json.loads(fmt.format(rec))  # must not raise
    assert "exc_type" not in out


def test_json_formatter_never_raises_on_bad_format_args():
    import logging

    import xpcsjax.utils.logging as lm

    fmt = lm.JSONFormatter()
    rec = logging.LogRecord("lg", logging.INFO, __file__, 1, "val=%s %s", ("only_one",), None)
    out = json.loads(fmt.format(rec))  # must not raise, must be valid JSON
    assert out["schema_version"] >= 1


def test_json_formatter_redaction_does_not_overredact():
    import logging

    import xpcsjax.utils.logging as lm

    fmt = lm.JSONFormatter()
    rec = logging.LogRecord("lg", logging.INFO, __file__, 1, "m", None, None)
    rec.context = {
        "API_KEY": "sk-1",
        "SECRET": "s",
        "sort_key": "phi",
        "data_path": "/home/u/x.h5",
    }
    out = json.loads(fmt.format(rec))
    assert out["context"]["API_KEY"] == "***REDACTED***"
    assert out["context"]["SECRET"] == "***REDACTED***"
    assert out["context"]["sort_key"] == "phi"  # benign ...key NOT redacted
    assert out["context"]["data_path"] == "/home/u/x.h5"  # paths NOT redacted


def test_json_safe_recursion_guard_truncates_deep_nesting():
    import xpcsjax.utils.logging as lm

    deep: dict = {}
    cur = deep
    for _ in range(100):
        nxt: dict = {}
        cur["child"] = nxt
        cur = nxt
    result = lm._json_safe(deep)  # must not raise; must terminate via depth guard

    # Walk down: beyond depth 20 the guard returns a repr string, not a dict.
    node: object = result
    depth = 0
    while isinstance(node, dict) and "child" in node:
        node = node["child"]
        depth += 1
    assert isinstance(node, str)  # truncated to repr
    assert depth <= 21
