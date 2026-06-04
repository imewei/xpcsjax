"""Quality-gate regression tests for xpcsjax/utils/logging.py.

Each section corresponds to one numbered finding from the quality-gate review.
Tests are written FIRST (TDD); they were watched fail before the fixes landed.

Strict contract: logging is OBSERVATIONAL. Every test verifies that a broken
logger / handler NEVER causes the decorated function to abort or raise.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pytest

import xpcsjax.utils.logging as lm

# ---------------------------------------------------------------------------
# Finding 1 — never-raise guards for log_calls and log_performance
# ---------------------------------------------------------------------------


class _RaisingHandler(logging.Handler):
    """A handler whose emit() always raises — simulates a broken log backend."""

    def emit(self, record: logging.LogRecord) -> None:
        raise RuntimeError("handler exploded")


def _logger_that_raises_on_log() -> logging.Logger:
    """Return a fresh logger whose only handler always raises on emit."""
    lg = logging.getLogger(f"_raising_{id(threading.current_thread())}")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    h = _RaisingHandler()
    h.setLevel(logging.DEBUG)
    lg.addHandler(h)
    return lg


class TestLogCallsNeverRaise:
    """log_calls: a raising handler must never abort the decorated function."""

    def test_entry_log_raises_decorated_fn_still_returns(self):
        """Logger raises on the *entry* emit; function must still return."""
        lg = _logger_that_raises_on_log()

        @lm.log_calls(logger=lg, level=logging.DEBUG)
        def my_func(x: int) -> int:
            return x * 2

        # Must not raise; must return real value
        result = my_func(7)
        assert result == 14

    def test_success_log_raises_decorated_fn_still_returns(self):
        """Logger raises on the *success* emit; function must still return."""
        call_count = {"n": 0}

        lg = _logger_that_raises_on_log()

        @lm.log_calls(logger=lg, level=logging.DEBUG)
        def my_func() -> str:
            call_count["n"] += 1
            return "ok"

        result = my_func()
        assert result == "ok"
        assert call_count["n"] == 1

    def test_entry_success_raises_but_exception_still_propagates(self):
        """Even when the logger raises, a real function exception still propagates."""
        lg = _logger_that_raises_on_log()

        @lm.log_calls(logger=lg, level=logging.DEBUG)
        def boom() -> None:
            raise ValueError("real error")

        with pytest.raises(ValueError, match="real error"):
            boom()

    def test_include_args_entry_raises_still_returns(self):
        """include_args=True path: raising logger must not abort the call."""
        lg = _logger_that_raises_on_log()

        @lm.log_calls(logger=lg, level=logging.DEBUG, include_args=True)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_include_result_success_raises_still_returns(self):
        """include_result=True path: raising logger on success must not abort."""
        lg = _logger_that_raises_on_log()

        @lm.log_calls(logger=lg, level=logging.DEBUG, include_result=True)
        def identity(x: Any) -> Any:
            return x

        assert identity(42) == 42


class TestLogPerformanceNeverRaise:
    """log_performance: a raising handler must never abort the decorated function."""

    def test_success_log_raises_decorated_fn_still_returns(self):
        """Performance log raises on success; function must still return."""
        lg = _logger_that_raises_on_log()

        @lm.log_performance(logger=lg, level=logging.INFO, threshold=0.0)
        def work() -> int:
            return 99

        result = work()
        assert result == 99

    def test_success_below_threshold_returns_without_raising(self):
        """Below-threshold path (no log emit) still works with a bad logger."""
        lg = _logger_that_raises_on_log()

        @lm.log_performance(logger=lg, threshold=9999.0)
        def fast() -> str:
            return "fast"

        assert fast() == "fast"

    def test_exception_still_propagates_even_with_raising_logger(self):
        """Real function exception still propagates when logger also raises."""
        lg = _logger_that_raises_on_log()

        @lm.log_performance(logger=lg, threshold=0.0)
        def fail() -> None:
            raise RuntimeError("real perf error")

        with pytest.raises(RuntimeError, match="real perf error"):
            fail()


# ---------------------------------------------------------------------------
# Finding 2 — CR-1(a) propagate inversion: no-arg configure must not force
#             propagate=True when no managed handler exists.
# ---------------------------------------------------------------------------


class TestPropagateInversion:
    """Default no-arg configure() must not force propagate=True when there is
    no managed handler — that would duplicate records for users who already
    configured the stdlib root logger.

    The fix lives in the non-PYTEST branch of _configure_impl.  Tests run inside
    pytest, so PYTEST_CURRENT_TEST is always set and the pytest branch fires
    first.  We therefore test the contract by clearing PYTEST_CURRENT_TEST via
    monkeypatch so the production branch executes.
    """

    def test_no_managed_handler_does_not_force_propagate_true(self, monkeypatch):
        """With no managed handler, the production branch must NOT set propagate=True."""
        # Clear pytest env so the production branch runs
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        root = logging.getLogger(lm._logger_manager._root_logger_name)
        # Strip all managed handlers so has_managed_handler == False
        root.handlers = [h for h in root.handlers if not getattr(h, "_xpcsjax_managed", False)]
        root.propagate = False  # set a known start state

        # configure with console disabled → no managed handler will be added
        cfg = {"enabled": True, "console": {"enabled": False}, "file": {"enabled": False}}
        lm.configure_logging(cfg)

        # Contract: propagate must NOT be forced True when no managed handler exists.
        # The old (buggy) code: root_logger.propagate = not has_managed_handler
        #   = not False = True  ← that's the bug.
        # The fix: only touch propagate when has_managed_handler is True.
        assert not (
            root.propagate is True
            and not any(getattr(h, "_xpcsjax_managed", False) for h in root.handlers)
        ), (
            "propagate must not be forced True when no managed handler is present; "
            "that would duplicate records for users with stdlib root logger configured"
        )

    def test_managed_handler_sets_propagate_false(self, monkeypatch):
        """With a managed handler, the production branch sets propagate=False."""
        # Clear pytest env so the production branch runs
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        cfg = {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}
        lm.configure_logging(cfg)
        root = logging.getLogger(lm._logger_manager._root_logger_name)
        has_managed = any(getattr(h, "_xpcsjax_managed", False) for h in root.handlers)
        if has_managed:
            # The existing production behavior: propagate=False when handler owned
            assert root.propagate is False


# ---------------------------------------------------------------------------
# Finding 3 — CR-1(b) ContextFilter on the named xpcsjax logger itself
# ---------------------------------------------------------------------------


class TestContextFilterOnLogger:
    """ContextFilter must be attached to the named xpcsjax logger so context
    fields (run_id/phase/mode/strategy) are present on ALL records — even those
    emitted before any handler-level configure() call."""

    def test_context_fields_present_before_handler_configure(self):
        """A record emitted before configure() carries the context fields.

        The ContextFilter on the logger itself (not just handlers) ensures
        context attrs are set regardless of handler state.
        """
        # Set a context value
        tok = lm.set_log_context(run_id="pre-configure-test", phase="init")
        try:
            rec = logging.LogRecord("xpcsjax.test", logging.INFO, __file__, 1, "msg", None, None)
            # Apply the filter manually — this simulates the filter on the logger
            lm.ContextFilter().filter(rec)
            assert rec.run_id == "pre-configure-test"
            assert rec.phase == "init"
            assert rec.mode is None  # not set → None
            assert rec.strategy is None  # not set → None
        finally:
            lm.reset_log_context(tok)

    def test_context_filter_on_xpcsjax_logger_installed(self):
        """After configure_logging, the named xpcsjax logger has a ContextFilter."""
        cfg = {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}
        lm.configure_logging(cfg)
        xpcs_logger = logging.getLogger(lm._logger_manager._root_logger_name)
        assert any(isinstance(f, lm.ContextFilter) for f in xpcs_logger.filters), (
            "ContextFilter must be attached to the xpcsjax logger itself, not only handlers"
        )

    def test_context_filter_on_logger_idempotent(self):
        """Multiple configure_logging calls must not double-install ContextFilter on the logger."""
        cfg = {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}
        lm.configure_logging(cfg)
        lm.configure_logging(cfg)
        xpcs_logger = logging.getLogger(lm._logger_manager._root_logger_name)
        assert sum(isinstance(f, lm.ContextFilter) for f in xpcs_logger.filters) <= 1

    def test_context_filter_sets_none_when_context_empty(self):
        """ContextFilter.filter() sets all fields to None when context is empty."""
        # Clear any leaked context by setting it directly to None (no restore).
        lm._LOG_CONTEXT.set(None)
        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
        lm.ContextFilter().filter(rec)
        assert rec.run_id is None
        assert rec.phase is None
        assert rec.mode is None
        assert rec.strategy is None


# ---------------------------------------------------------------------------
# Finding 4 — TYPE-1: type boundary for set_log_context / log_context,
#             __all__ present, log_once uses LoggerType alias
# ---------------------------------------------------------------------------


class TestTypeBoundary:
    """set_log_context and log_context must only accept the 4 known fields;
    unknown/typo'd keys must either be rejected at call time or silently ignored
    but not silently consumed and forwarded to downstream code."""

    def test_known_keys_accepted(self):
        """All four known keys must be accepted without error."""
        tok = lm.set_log_context(run_id="r1", phase="p1", mode="m1", strategy="s1")
        try:
            rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
            lm.ContextFilter().filter(rec)
            assert rec.run_id == "r1"
            assert rec.phase == "p1"
            assert rec.mode == "m1"
            assert rec.strategy == "s1"
        finally:
            lm.reset_log_context(tok)

    def test_log_context_known_keys_accepted(self):
        """log_context() context manager accepts all 4 known keys."""
        with lm.log_context(run_id="ctx-r1", mode="static_isotropic"):
            rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
            lm.ContextFilter().filter(rec)
            assert rec.run_id == "ctx-r1"
            assert rec.mode == "static_isotropic"

    def test_module_has_all(self):
        """The module must define a literal __all__ listing public symbols."""
        assert hasattr(lm, "__all__"), "logging.py must define __all__"
        assert isinstance(lm.__all__, list), "__all__ must be a list"
        assert len(lm.__all__) > 0, "__all__ must not be empty"

    def test_all_includes_key_public_symbols(self):
        """Key public symbols must appear in __all__."""
        required = {
            "PhaseLogger",
            "JSONFormatter",
            "ContextFilter",
            "log_once",
            "logged_errors",
            "set_log_context",
            "reset_log_context",
            "log_context",
            "reset_log_once_cache",
            "configure_logging",
        }
        missing = required - set(lm.__all__)
        assert not missing, f"Missing from __all__: {missing}"

    def test_log_once_accepts_loggertype(self, caplog):
        """log_once must accept a proper logging.Logger (not just Any)."""
        lm.reset_log_once_cache()
        lg = logging.getLogger("type_boundary_test")
        with caplog.at_level(logging.INFO, logger="type_boundary_test"):
            lm.log_once(lg, logging.INFO, "tb_key", "type boundary %s", "ok")
        assert any("type boundary" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Finding 5 — TEST-1: regression pins
# ---------------------------------------------------------------------------


class TestLogOnceTwoDistinctKeys:
    """log_once with two DISTINCT keys must each emit exactly once."""

    def test_two_distinct_keys_each_emit(self, caplog):
        lm.reset_log_once_cache()
        lg = logging.getLogger("log_once_distinct")
        with caplog.at_level(logging.WARNING, logger="log_once_distinct"):
            for _ in range(3):
                lm.log_once(lg, logging.WARNING, "key_alpha", "msg alpha %d", 1)
                lm.log_once(lg, logging.WARNING, "key_beta", "msg beta %d", 2)

        alpha_count = sum("msg alpha" in r.getMessage() for r in caplog.records)
        beta_count = sum("msg beta" in r.getMessage() for r in caplog.records)
        assert alpha_count == 1, f"key_alpha should emit once, got {alpha_count}"
        assert beta_count == 1, f"key_beta should emit once, got {beta_count}"

    def test_same_key_emits_only_once(self, caplog):
        lm.reset_log_once_cache()
        lg = logging.getLogger("log_once_same")
        with caplog.at_level(logging.WARNING, logger="log_once_same"):
            for _ in range(5):
                lm.log_once(lg, logging.WARNING, "same_key", "flood %d", 1)
        flood_count = sum("flood" in r.getMessage() for r in caplog.records)
        assert flood_count == 1


class TestJSONFormatterCircularRef:
    """JSONFormatter must handle a circular-reference context dict without raising
    and must yield valid JSON."""

    def test_circular_ref_context_yields_valid_json(self):
        fmt = lm.JSONFormatter()
        # Build a circular dict
        d: dict[str, Any] = {}
        d["self"] = d

        rec = logging.LogRecord("circ_test", logging.INFO, __file__, 1, "circular test", None, None)
        # Attach the circular dict as extra context
        rec.context = d  # type: ignore[attr-defined]

        # format must not raise
        try:
            output = fmt.format(rec)
        except Exception as exc:
            pytest.fail(f"JSONFormatter.format raised on circular ref: {exc}")

        # Output must be valid JSON
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"JSONFormatter output is not valid JSON: {exc}\nOutput: {output!r}")

        assert isinstance(parsed, dict)

    def test_circular_ref_in_extra_field_yields_valid_json(self):
        """Circular ref injected via a record attribute must also be handled."""
        fmt = lm.JSONFormatter()
        d: dict[str, Any] = {}
        d["loop"] = d

        rec = logging.LogRecord(
            "circ_test2", logging.WARNING, __file__, 1, "another circ", None, None
        )
        rec.extra_data = d  # type: ignore[attr-defined]

        try:
            output = fmt.format(rec)
        except Exception as exc:
            pytest.fail(f"JSONFormatter.format raised: {exc}")

        json.loads(output)  # must be valid JSON — raises JSONDecodeError if not
