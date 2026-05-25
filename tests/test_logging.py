"""Tests for xpcsjax.utils.logging.

This module mutates process-global logging state (handlers on the ``xpcsjax``
logger, external-library logger levels), so an autouse fixture snapshots and
restores that state around every test. Logger output is captured with
``MagicMock`` loggers instead of real handlers, keeping assertions precise and
side-effect-free.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xpcsjax.utils import logging as lm


@pytest.fixture(autouse=True)
def _isolate_logging():
    """Restore the xpcsjax logger's handlers/level and manager state after each test."""
    lg = logging.getLogger("xpcsjax")
    saved_handlers = lg.handlers[:]
    saved_level = lg.level
    mgr = lm._logger_manager
    saved_configured = mgr._configured
    yield
    for h in lg.handlers[:]:
        if h not in saved_handlers:
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
    for h in saved_handlers:
        if h not in lg.handlers:
            lg.addHandler(h)
    lg.setLevel(saved_level)
    mgr._configured = saved_configured


# ---------------------------------------------------------------------------
# _resolve_level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (None, None),
        (logging.DEBUG, logging.DEBUG),
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("warning", logging.WARNING),
        ("nonsense", logging.INFO),  # unknown -> INFO fallback
    ],
)
def test_resolve_level(level: object, expected: int | None) -> None:
    assert lm._resolve_level(level) == expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _ColorFormatter
# ---------------------------------------------------------------------------


def _record(level: int = logging.INFO, msg: str = "hi") -> logging.LogRecord:
    return logging.LogRecord("n", level, "path", 1, msg, None, None)


def test_color_formatter_applies_and_restores_color() -> None:
    fmt = lm._ColorFormatter("%(levelname)s|%(message)s", None, use_color=True)
    rec = _record(logging.INFO)
    out = fmt.format(rec)
    assert "\033[32m" in out and "\033[0m" in out  # green + reset
    assert rec.levelname == "INFO"  # original restored in finally


def test_color_formatter_no_color() -> None:
    fmt = lm._ColorFormatter("%(levelname)s|%(message)s", None, use_color=False)
    out = fmt.format(_record(logging.WARNING, "w"))
    assert "\033[" not in out
    assert out == "WARNING|w"


# ---------------------------------------------------------------------------
# _ContextAdapter
# ---------------------------------------------------------------------------


def test_context_adapter_no_extra_passthrough() -> None:
    adapter = lm._ContextAdapter(logging.getLogger("t"), {})
    msg, kwargs = adapter.process("hello", {})
    assert msg == "hello"


def test_context_adapter_prefixes_and_filters() -> None:
    adapter = lm._ContextAdapter(
        logging.getLogger("t"), {"run": "abc", "empty": "", "none": None, "n": 5}
    )
    msg, _ = adapter.process("go", {})
    assert msg.startswith("[")
    assert "run=abc" in msg and "n=5" in msg
    assert "empty=" not in msg and "none=" not in msg


# ---------------------------------------------------------------------------
# LogConfiguration
# ---------------------------------------------------------------------------


def test_log_configuration_defaults() -> None:
    cfg = lm.LogConfiguration()
    assert cfg.console_level == "INFO"
    assert cfg.file_enabled is True
    assert cfg.module_overrides == {}


def test_log_configuration_from_dict() -> None:
    cfg = lm.LogConfiguration.from_dict(
        {"console_level": "DEBUG", "file_enabled": False, "module_overrides": {"jax": "ERROR"}}
    )
    assert cfg.console_level == "DEBUG"
    assert cfg.file_enabled is False
    assert cfg.module_overrides == {"jax": "ERROR"}


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected_level", "expected_fmt"),
    [
        (False, False, "INFO", "simple"),
        (True, False, "DEBUG", "detailed"),
        (False, True, "ERROR", "simple"),
    ],
)
def test_log_configuration_from_cli_args(
    verbose: bool, quiet: bool, expected_level: str, expected_fmt: str
) -> None:
    cfg = lm.LogConfiguration.from_cli_args(verbose=verbose, quiet=quiet)
    assert cfg.console_level == expected_level
    assert cfg.console_format == expected_fmt


def test_log_configuration_apply_no_file() -> None:
    cfg = lm.LogConfiguration(file_enabled=False)
    assert cfg.apply() is None


def test_log_configuration_apply_with_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    cfg = lm.LogConfiguration(file_enabled=True, file_path=log_file)
    out = cfg.apply()
    assert out == log_file


# ---------------------------------------------------------------------------
# _PhaseRecord
# ---------------------------------------------------------------------------


def test_phase_record_duration() -> None:
    assert lm._PhaseRecord("x").duration is None
    assert lm._PhaseRecord("x", start_time=1.0, end_time=3.5).duration == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# AnalysisSummaryLogger
# ---------------------------------------------------------------------------


def test_summary_logger_records_and_logs() -> None:
    summary = lm.AnalysisSummaryLogger(run_id="r1", analysis_mode="laminar_flow")
    summary.start_phase("loading")
    summary.end_phase("loading", memory_peak_gb=2.1)
    summary.end_phase("never_started")  # no-op branch
    summary.record_metric("chi2", 1.234)
    summary.add_output_file("/tmp/out.npz")
    summary.set_convergence_status("converged")
    summary.increment_warning_count()
    summary.increment_error_count()
    summary.set_config_summary(
        optimizer="nlsq", n_params=7, n_data_points=2_000_000, n_phi_angles=4,
        data_file="d.h5", extra="z",
    )

    mock_logger = MagicMock()
    summary.log_summary(mock_logger)
    text = mock_logger.info.call_args[0][0]
    assert "ANALYSIS SUMMARY" in text
    assert "Run ID: r1" in text
    assert "converged" in text
    assert "2,000,000" in text  # int > 1000 gets thousands separators
    assert "loading:" in text
    assert "chi2:" in text
    assert "out.npz" in text
    assert "Warnings: 1, Errors: 1" in text


def test_summary_logger_as_dict_sanitizes_nan() -> None:
    summary = lm.AnalysisSummaryLogger(run_id="r2", analysis_mode="static_isotropic")
    summary.start_phase("p")
    summary.end_phase("p")
    summary.record_metric("bad", math.nan)
    summary.record_metric("good", 3.0)
    d = summary.as_dict()
    assert d["run_id"] == "r2"
    assert d["metrics"]["good"] == 3.0
    assert d["metrics"]["bad"] is None  # NaN sanitized for JSON
    assert "p" in d["phases"]


# ---------------------------------------------------------------------------
# MinimalLogger
# ---------------------------------------------------------------------------


def test_minimal_logger_is_singleton() -> None:
    assert lm.MinimalLogger() is lm.MinimalLogger()
    assert lm.MinimalLogger() is lm._logger_manager


def test_build_formatter_simple_vs_detailed() -> None:
    simple = lm.MinimalLogger._build_formatter("simple")
    detailed = lm.MinimalLogger._build_formatter("detailed")
    assert simple._fmt == lm.DEFAULT_FORMAT_SIMPLE
    assert detailed._fmt == lm.DEFAULT_FORMAT_DETAILED


def test_configure_creates_rotating_file_handler(tmp_path: Path) -> None:
    mgr = lm.MinimalLogger()
    out = mgr.configure(
        console_level="INFO",
        file_path=tmp_path / "a.log",
        file_level="DEBUG",
        max_size_mb=10,
        module_levels={"somemod": "ERROR"},
        force=True,
    )
    assert out == tmp_path / "a.log"
    root = logging.getLogger("xpcsjax")
    from logging.handlers import RotatingFileHandler

    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    assert logging.getLogger("somemod").level == logging.ERROR


def test_configure_plain_file_handler_when_no_rotation(tmp_path: Path) -> None:
    mgr = lm.MinimalLogger()
    mgr.configure(file_path=tmp_path / "b.log", file_level="DEBUG", max_size_mb=0, force=True)
    root = logging.getLogger("xpcsjax")
    from logging.handlers import RotatingFileHandler

    file_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler)
        and not isinstance(h, RotatingFileHandler)
    ]
    assert file_handlers


def test_configure_handles_mkdir_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: Path, *a: object, **k: object) -> None:
        raise OSError("no permission")

    monkeypatch.setattr(Path, "mkdir", _boom)
    mgr = lm.MinimalLogger()
    out = mgr.configure(file_path=tmp_path / "sub" / "c.log", force=True)
    # mkdir failure -> file logging disabled, returns None (console still works).
    assert out is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("__main__", "xpcsjax.main"),
        ("mymod", "xpcsjax.mymod"),
        ("xpcsjax.optimization", "xpcsjax.optimization"),
        ("xpcsjax", "xpcsjax"),
        ("a.b", "xpcsjax.a.b"),
    ],
)
def test_get_logger_name_normalization(name: str, expected: str) -> None:
    assert lm._logger_manager.get_logger(name).name == expected


def test_configure_from_dict_disabled() -> None:
    assert lm._logger_manager.configure_from_dict(None) is None
    assert lm._logger_manager.configure_from_dict({"enabled": False}) is None


def test_configure_from_dict_with_file_run_id(tmp_path: Path) -> None:
    cfg = {
        "enabled": True,
        "level": "INFO",
        "console": {"enabled": True, "format": "simple"},
        "file": {
            "enabled": True,
            "path": str(tmp_path),
            "filename": "run_{run_id}.log",
        },
    }
    out = lm._logger_manager.configure_from_dict(cfg, run_id="JOB7")
    assert out is not None
    assert out.name == "run_JOB7.log"


def test_configure_from_dict_filename_without_run_id_placeholder(tmp_path: Path) -> None:
    cfg = {
        "enabled": True,
        "console": {"enabled": True},
        "file": {"enabled": True, "path": str(tmp_path), "filename": "analysis.log"},
    }
    out = lm._logger_manager.configure_from_dict(cfg, run_id="X1")
    assert out is not None
    assert out.name == "analysis_X1.log"


def test_configure_from_dict_quiet_and_verbose(tmp_path: Path) -> None:
    base = {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}
    # quiet wins on console level; just assert it configures without error.
    assert lm._logger_manager.configure_from_dict(base, quiet=True) is None
    assert lm._logger_manager.configure_from_dict(base, verbose=True) is None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_configure_logging_delegates(tmp_path: Path) -> None:
    out = lm.configure_logging(
        {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}
    )
    assert out is None


def test_get_logger_infers_name_and_wraps_context() -> None:
    plain = lm.get_logger()
    assert isinstance(plain, logging.Logger)
    ctx = lm.get_logger(__name__, context={"run": "z"})
    assert isinstance(ctx, lm._ContextAdapter)


def test_with_context_plain_logger() -> None:
    base = logging.getLogger("xpcsjax.test")
    adapter = lm.with_context(base, run="a", skip=None)
    assert isinstance(adapter, lm._ContextAdapter)
    assert adapter.extra is not None
    assert "run" in adapter.extra and "skip" not in adapter.extra


def test_with_context_merges_nested_context() -> None:
    base = logging.getLogger("xpcsjax.test")
    outer = lm.with_context(base, run="a", mode="laminar")
    inner = lm.with_context(outer, mode="static", shard=5)
    # Inner overrides outer on conflict; both keys present; no nested adapters.
    assert inner.logger is base
    assert inner.extra is not None
    assert inner.extra["mode"] == "static"
    assert inner.extra["run"] == "a"
    assert inner.extra["shard"] == 5


def test_with_context_extracts_base_from_plain_adapter() -> None:
    base = logging.getLogger("xpcsjax.test")
    plain_adapter = logging.LoggerAdapter(base, {})
    out = lm.with_context(plain_adapter, k="v")
    assert isinstance(out, lm._ContextAdapter)
    assert out.logger is base


def test_get_memory_gb_returns_float_on_linux() -> None:
    mem = lm._get_memory_gb()
    # On Linux/macOS resource is available; should be a positive float.
    assert mem is None or (isinstance(mem, float) and mem > 0)


# ---------------------------------------------------------------------------
# log_phase
# ---------------------------------------------------------------------------


def test_log_phase_logs_start_and_completion() -> None:
    mock = MagicMock(spec=logging.Logger)
    with lm.log_phase("opt", logger=mock, track_memory=True) as phase:
        pass
    assert phase.name == "opt"
    assert phase.duration >= 0.0
    # Two log calls: start + completion.
    assert mock.log.call_count >= 2
    # With memory tracking the completion message includes "peak memory".
    assert any("peak memory" in c.args[1] for c in mock.log.call_args_list)


def test_log_phase_threshold_suppresses_logs() -> None:
    mock = MagicMock(spec=logging.Logger)
    # Large threshold -> start not logged, fast body -> completion not logged.
    with lm.log_phase("fast", logger=mock, threshold_s=1000.0):
        pass
    assert mock.log.call_count == 0


# ---------------------------------------------------------------------------
# log_exception
# ---------------------------------------------------------------------------


def test_log_exception_with_traceback_and_context() -> None:
    mock = MagicMock(spec=logging.Logger)
    try:
        raise ValueError("boom")
    except ValueError as e:
        lm.log_exception(mock, e, context={"iteration": 3})
    msg = mock.log.call_args[0][1]
    assert "ValueError: boom" in msg
    assert "Context: iteration=3" in msg
    assert "Traceback" in msg
    assert " in " in msg  # location info extracted from traceback


def test_log_exception_without_traceback_or_context() -> None:
    mock = MagicMock(spec=logging.Logger)
    lm.log_exception(mock, ValueError("x"), include_traceback=False)
    msg = mock.log.call_args[0][1]
    assert "ValueError: x" in msg
    assert "Traceback" not in msg


# ---------------------------------------------------------------------------
# log_calls
# ---------------------------------------------------------------------------


def test_log_calls_logs_entry_exit_with_args_and_result() -> None:
    mock = MagicMock(spec=logging.Logger)
    mock.isEnabledFor.return_value = True

    @lm.log_calls(logger=mock, include_args=True, include_result=True)
    def add(a: int, b: int = 2) -> int:
        return a + b

    assert add(1, b=3) == 4
    messages = [c.args[1] % c.args[2:] if len(c.args) > 2 else c.args[1] for c in mock.log.call_args_list]
    joined = " ".join(str(m) for m in messages)
    assert "Calling" in joined
    assert "Completed" in joined


def test_log_calls_skips_when_level_disabled() -> None:
    mock = MagicMock(spec=logging.Logger)
    mock.isEnabledFor.return_value = False

    @lm.log_calls(logger=mock)
    def f() -> int:
        return 1

    assert f() == 1
    assert mock.log.call_count == 0  # nothing logged when level disabled


def test_log_calls_logs_and_reraises_exception() -> None:
    mock = MagicMock(spec=logging.Logger)
    mock.isEnabledFor.return_value = True

    @lm.log_calls(logger=mock)
    def boom() -> None:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        boom()
    # The error path logs at ERROR level.
    assert any(c.args[0] == logging.ERROR for c in mock.log.call_args_list)


# ---------------------------------------------------------------------------
# log_performance
# ---------------------------------------------------------------------------


def test_log_performance_logs_above_threshold() -> None:
    mock = MagicMock(spec=logging.Logger)

    @lm.log_performance(logger=mock, threshold=0.0)
    def quick() -> str:
        return "ok"

    assert quick() == "ok"
    assert any("Performance" in c.args[1] for c in mock.log.call_args_list)


def test_log_performance_logs_and_reraises() -> None:
    mock = MagicMock(spec=logging.Logger)

    @lm.log_performance(logger=mock, threshold=0.0)
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()
    assert any(c.args[0] == logging.ERROR for c in mock.log.call_args_list)


# ---------------------------------------------------------------------------
# log_operation
# ---------------------------------------------------------------------------


def test_log_operation_success() -> None:
    mock = MagicMock(spec=logging.Logger)
    with lm.log_operation("save", logger=mock):
        pass
    msgs = " ".join(str(c.args[1]) for c in mock.log.call_args_list)
    assert "Starting operation" in msgs
    assert "Completed operation" in msgs


def test_log_operation_failure_reraises() -> None:
    mock = MagicMock(spec=logging.Logger)
    with pytest.raises(ValueError, match="bad"), lm.log_operation("save", logger=mock):
        raise ValueError("bad")
    assert any(c.args[0] == logging.ERROR for c in mock.log.call_args_list)
