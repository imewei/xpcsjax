"""Structured logging utilities for the xpcsjax package.

Provides a lightweight but flexible logging system: contextual log prefixes,
configurable console and rotating file handlers, and helpers for performance
monitoring.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import re
import threading
import time
import traceback
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

if TYPE_CHECKING:
    from xpcsjax.config.parameter_registry import AnalysisMode

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Type alias for logger types
LoggerType = logging.Logger | logging.LoggerAdapter[logging.Logger]

DEFAULT_FORMAT_DETAILED = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_FORMAT_SIMPLE = "%(levelname)-8s | %(message)s"


def _resolve_level(level: str | int | None) -> int | None:
    """Convert string/int log level to logging level constant."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


class _ColorFormatter(logging.Formatter):
    """Optional ANSI color formatter for console logging."""

    COLOR_MAP = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str | None, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color and original_levelname in self.COLOR_MAP:
            record.levelname = (
                f"{self.COLOR_MAP[original_levelname]}{original_levelname}{self.RESET}"
            )
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


# Redact true secrets only: TOKEN/SECRET/PASSWORD/API_KEY (case-insensitive,
# optional API[_-] prefix), plus the uppercase ``_KEY`` env-var convention
# (e.g. STRIPE_KEY). A lowercase ``...key`` suffix is deliberately NOT redacted
# so benign keys (sort_key, lookup_key) survive for error tracking.
_REDACT_KEY_CI = re.compile(r"(TOKEN|SECRET|PASSWORD|API[_-]?KEY)$", re.IGNORECASE)
_REDACT_KEY_UPPER = re.compile(r"_KEY$")


def _is_secret_key(key: str) -> bool:
    """Return True for keys naming a credential that must be redacted."""
    return bool(_REDACT_KEY_CI.search(key) or _REDACT_KEY_UPPER.search(key))
_JSON_SCHEMA_VERSION = 1
_JSON_SAFE_MAX_DEPTH = 20


def _json_safe(obj: Any, _depth: int = 0) -> Any:
    import math

    # Recursion guard: deeply nested structures degrade to a bounded repr rather
    # than escaping as a RecursionError (logging is observational only).
    if _depth > _JSON_SAFE_MAX_DEPTH:
        return repr(obj)[:500]

    if isinstance(obj, dict):
        return {
            k: (
                "***REDACTED***"
                if _is_secret_key(str(k))
                else _json_safe(v, _depth + 1)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)[:500]


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter with a stable schema and key redaction.

    Emits one JSON object per record carrying a fixed set of top-level fields
    (``timestamp``, ``level``, ``logger``, ``message``, ``schema_version`` plus
    structured fields ``event``/``phase``/``mode``/``strategy``/``run_id``/
    ``operation``). The optional ``record.context`` mapping is passed through
    :func:`_json_safe`, which coerces numpy scalars, nulls out non-finite
    floats, and redacts secret-looking keys (TOKEN/SECRET/PASSWORD/API_KEY and
    any ``*_KEY``, but not benign ``*key`` such as ``sort_key``); filesystem
    paths are intentionally NOT redacted. The whole body is exception-guarded so
    the formatter never raises.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Observational only: a formatter that raises is fatal to the logging
        # machinery, so the entire body is guarded and degrades to a minimal,
        # always-valid JSON record on any failure (e.g. getMessage() raising on
        # malformed %-args, or an exotic context value).
        try:
            out: dict[str, Any] = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "schema_version": _JSON_SCHEMA_VERSION,
            }
            for f in ("event", "phase", "mode", "strategy", "run_id", "operation"):
                out[f] = getattr(record, f, None)
            if out["event"] is None:
                out["event"] = out["message"]
            # ``record.exc_info`` is the stdlib triple; the placeholder
            # ``(None, None, None)`` is truthy but carries no exception, so guard
            # on the type slot before dereferencing ``__name__``.
            if record.exc_info and record.exc_info[0] is not None:
                out["exc_type"] = record.exc_info[0].__name__
                out["exc_message"] = str(record.exc_info[1])
                out["traceback"] = self.formatException(record.exc_info)
            ctx = getattr(record, "context", None)
            out["context"] = _json_safe(ctx) if ctx is not None else None
            return json.dumps(out, default=lambda o: repr(o)[:500])
        except Exception:  # noqa: BLE001 - a formatter must never raise
            return json.dumps(
                {
                    "level": getattr(record, "levelname", "UNKNOWN"),
                    "logger": getattr(record, "name", "unknown"),
                    "message": "<format failed>",
                    "schema_version": _JSON_SCHEMA_VERSION,
                }
            )


class PhaseLogger:
    """Mode-agnostic named-phase/banner logger; all methods exception-safe.

    A thin, observational-only wrapper around a stdlib logger that emits
    fixed-width banners and indented ``name: value`` fields. Every method
    swallows exceptions so logging never escapes to or changes control flow at
    the call site.
    """

    def __init__(self, logger: logging.Logger):
        self._log = logger

    def banner(self, title: str, *, width: int = 80, level: int = logging.INFO) -> None:
        try:
            self._log.log(level, "=" * width)
            self._log.log(level, title)
            self._log.log(level, "=" * width)
        except Exception:  # noqa: BLE001
            pass

    def field(self, name: str, value: Any) -> None:
        try:
            self._log.info("  %s: %s", name, value)
        except Exception:  # noqa: BLE001
            pass


class _ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with structured context."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        if not self.extra:
            return msg, kwargs

        context_parts = [
            f"{key}={value}"
            for key, value in self.extra.items()
            if value is not None and value != ""
        ]
        if context_parts:
            msg = f"[{' '.join(context_parts)}] {msg}"
        return msg, kwargs


@dataclass
class LogConfiguration:
    """Programmatic logging configuration.

    Alternative to configure_logging() for programmatic control over
    logging settings.

    Attributes:
        console_level: Console log level (default "INFO").
        console_format: Console format ("simple" or "detailed").
        console_colors: Enable ANSI colors in console (default False).
        file_enabled: Enable file logging (default True).
        file_path: Log file path (None = auto-generate).
        file_level: File log level (default "DEBUG").
        file_format: File format ("simple" or "detailed").
        file_rotation_mb: Max file size before rotation (default 10).
        file_backup_count: Number of backup files to keep (default 5).
        module_overrides: Per-module log level overrides.

    Example:
        >>> config = LogConfiguration.from_cli_args(verbose=True, log_file="analysis.log")
        >>> config.apply()

        >>> config = LogConfiguration(
        ...     console_level="INFO",
        ...     file_level="DEBUG",
        ...     module_overrides={"jax": "WARNING", "xpcsjax.optimization": "DEBUG"}
        ... )
        >>> config.apply()
    """

    console_level: str = "INFO"
    console_format: str = "simple"
    console_colors: bool = False
    file_enabled: bool = True
    file_path: str | Path | None = None
    file_level: str = "DEBUG"
    file_format: str = "detailed"
    file_rotation_mb: int = 10
    file_backup_count: int = 5
    module_overrides: dict[str, str] = field(default_factory=dict)

    def apply(self) -> Path | None:
        """Apply this configuration to the logging system.

        Returns:
            Path to log file if file logging is enabled, None otherwise.
        """
        # Suppress external library logging by default
        default_suppressions = {
            "jax": "WARNING",
            "numpy": "WARNING",
            "numba": "WARNING",
            "h5py": "WARNING",
        }

        # Merge default suppressions with user overrides (user overrides win)
        merged_overrides = {**default_suppressions, **self.module_overrides}

        # Determine file path
        file_path = None
        if self.file_enabled:
            if self.file_path is not None:
                file_path = Path(self.file_path)
            else:
                # Auto-generate timestamped log file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = Path(f"xpcsjax_{timestamp}.log")

        return _logger_manager.configure(
            level="DEBUG",  # Root level should be lowest to allow filtering
            console_level=self.console_level,
            console_format=self.console_format,
            console_colors=self.console_colors,
            file_path=file_path,
            file_level=self.file_level if self.file_enabled else None,
            file_format=self.file_format,
            max_size_mb=self.file_rotation_mb,
            backup_count=self.file_backup_count,
            module_levels=merged_overrides,
            force=True,
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LogConfiguration:
        """Create configuration from dictionary.

        Args:
            config: Dictionary with configuration values.

        Returns:
            LogConfiguration instance.
        """
        return cls(
            console_level=config.get("console_level", "INFO"),
            console_format=config.get("console_format", "simple"),
            console_colors=config.get("console_colors", False),
            file_enabled=config.get("file_enabled", True),
            file_path=config.get("file_path"),
            file_level=config.get("file_level", "DEBUG"),
            file_format=config.get("file_format", "detailed"),
            file_rotation_mb=config.get("file_rotation_mb", 10),
            file_backup_count=config.get("file_backup_count", 5),
            module_overrides=config.get("module_overrides", {}),
        )

    @classmethod
    def from_cli_args(
        cls,
        verbose: bool = False,
        quiet: bool = False,
        log_file: str | None = None,
    ) -> LogConfiguration:
        """Create configuration from CLI flags.

        Args:
            verbose: Enable DEBUG level console logging.
            quiet: Enable ERROR-only console logging.
            log_file: Path to log file (None = auto-generate if file logging enabled).

        Returns:
            LogConfiguration instance.
        """
        # Determine console level from flags
        if quiet:
            console_level = "ERROR"
        elif verbose:
            console_level = "DEBUG"
        else:
            console_level = "INFO"

        return cls(
            console_level=console_level,
            console_format="detailed" if verbose else "simple",
            console_colors=False,
            file_enabled=True,
            file_path=log_file,
            file_level="DEBUG",
            file_format="detailed",
        )


@dataclass
class _PhaseRecord:
    """Internal record for phase timing."""

    name: str
    start_time: float | None = None
    end_time: float | None = None
    memory_peak_gb: float | None = None

    @property
    def duration(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


class AnalysisSummaryLogger:
    """Structured logging for analysis completion summaries.

    Tracks phase timings, metrics, output files, and convergence status
    for logging a structured summary at analysis completion.

    Example:
        >>> summary = AnalysisSummaryLogger(run_id="analysis_001", analysis_mode="laminar_flow")
        >>> summary.start_phase("loading")
        >>> data = load_data(config)
        >>> summary.end_phase("loading", memory_peak_gb=2.1)
        >>> summary.record_metric("chi_squared", result.chi_squared)
        >>> summary.set_convergence_status("converged")
        >>> summary.log_summary(logger)
    """

    def __init__(self, run_id: str, analysis_mode: AnalysisMode) -> None:
        """Initialize summary logger for an analysis run.

        Args:
            run_id: Unique identifier for this analysis run.
            analysis_mode: Analysis mode (e.g., "static_isotropic", "laminar_flow").
        """
        self.run_id = run_id
        self.analysis_mode = analysis_mode
        self._phases: dict[str, _PhaseRecord] = {}
        self._metrics: dict[str, float] = {}
        self._output_files: list[Path] = []
        self._convergence_status: str | None = None
        self._start_time = time.perf_counter()
        self._warning_count = 0
        self._error_count = 0
        # T054: Configuration summary for logging
        self._config_summary: dict[str, Any] = {}

    def start_phase(self, name: str) -> None:
        """Mark phase start for timing.

        Args:
            name: Phase name (e.g., "loading", "optimization").
        """
        self._phases[name] = _PhaseRecord(name=name, start_time=time.perf_counter())

    def end_phase(self, name: str, memory_peak_gb: float | None = None) -> None:
        """Mark phase completion.

        Args:
            name: Phase name that was started.
            memory_peak_gb: Optional peak memory usage during phase.
        """
        if name in self._phases:
            self._phases[name].end_time = time.perf_counter()
            self._phases[name].memory_peak_gb = memory_peak_gb

    def record_metric(self, name: str, value: float) -> None:
        """Record a named metric (e.g., chi_squared).

        Args:
            name: Metric name.
            value: Metric value.
        """
        self._metrics[name] = value

    def add_output_file(self, path: Path | str) -> None:
        """Record an output file path.

        Args:
            path: Path to output file.
        """
        self._output_files.append(Path(path))

    def set_convergence_status(self, status: str) -> None:
        """Set final convergence status.

        Args:
            status: Convergence status (e.g., "converged", "max_iter", "failed").
        """
        self._convergence_status = status

    def increment_warning_count(self) -> None:
        """Increment warning counter."""
        self._warning_count += 1

    def increment_error_count(self) -> None:
        """Increment error counter."""
        self._error_count += 1

    def set_config_summary(
        self,
        optimizer: str | None = None,
        n_params: int | None = None,
        n_data_points: int | None = None,
        n_phi_angles: int | None = None,
        data_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """T054: Set configuration summary for logging.

        Args:
            optimizer: Optimizer used (e.g., "nlsq").
            n_params: Number of parameters being optimized.
            n_data_points: Total number of data points.
            n_phi_angles: Number of phi angles.
            data_file: Path to data file.
            **kwargs: Additional key-value pairs to include.
        """
        if optimizer is not None:
            self._config_summary["optimizer"] = optimizer
        if n_params is not None:
            self._config_summary["n_params"] = n_params
        if n_data_points is not None:
            self._config_summary["n_data_points"] = n_data_points
        if n_phi_angles is not None:
            self._config_summary["n_phi_angles"] = n_phi_angles
        if data_file is not None:
            self._config_summary["data_file"] = data_file
        # Add any additional kwargs
        self._config_summary.update(kwargs)

    def log_summary(self, logger: logging.Logger | logging.LoggerAdapter) -> None:
        """Log the complete analysis summary.

        Args:
            logger: Logger to use for output.
        """
        total_runtime = time.perf_counter() - self._start_time

        # Build summary message
        lines = [
            "=" * 60,
            "ANALYSIS SUMMARY",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Mode: {self.analysis_mode}",
            f"Status: {self._convergence_status or 'unknown'}",
            f"Total runtime: {total_runtime:.2f}s",
        ]

        # T054: Add configuration summary
        if self._config_summary:
            lines.append("")
            lines.append("Configuration:")
            for key, value in self._config_summary.items():
                if isinstance(value, int) and value > 1000:
                    lines.append(f"  {key}: {value:,}")
                else:
                    lines.append(f"  {key}: {value}")

        # Add phase timings
        if self._phases:
            lines.append("")
            lines.append("Phase Timings:")
            for name, record in self._phases.items():
                duration = record.duration
                if duration is not None:
                    mem_str = (
                        f" (peak: {record.memory_peak_gb:.1f} GB)"
                        if record.memory_peak_gb is not None
                        else ""
                    )
                    lines.append(f"  {name}: {duration:.2f}s{mem_str}")

        # Add metrics
        if self._metrics:
            lines.append("")
            lines.append("Metrics:")
            for name, value in self._metrics.items():
                lines.append(f"  {name}: {value:.6g}")

        # Add output files
        if self._output_files:
            lines.append("")
            lines.append("Output files:")
            for path in self._output_files:
                lines.append(f"  {path}")

        # Add warning/error counts
        if self._warning_count > 0 or self._error_count > 0:
            lines.append("")
            lines.append(
                f"Warnings: {self._warning_count}, Errors: {self._error_count}"
            )

        lines.append("=" * 60)

        logger.info("\n".join(lines))

    def as_dict(self) -> dict[str, Any]:
        """Export summary as dictionary for JSON serialization.

        Returns:
            Dictionary with all summary data.
        """
        total_runtime = time.perf_counter() - self._start_time

        phases_dict = {}
        for name, record in self._phases.items():
            phases_dict[name] = {
                "duration_s": record.duration,
                "memory_peak_gb": record.memory_peak_gb,
            }

        # Sanitize metrics so NaN/Inf floats (which can occur in degenerate
        # runs) do not cause json.dump to fail or produce invalid JSON output.
        # Import lazily to avoid circular dependency (logging ← json_utils).
        try:
            from xpcsjax.io.json_utils import json_safe as _json_safe
        except ImportError:

            def _json_safe(value: Any) -> Any:
                """Minimal fallback when io module unavailable."""
                import math

                if isinstance(value, dict):
                    return {k: _json_safe(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_json_safe(v) for v in value]
                if isinstance(value, float):
                    if math.isnan(value):
                        return None
                    if math.isinf(value):
                        return str(value)
                return value

        return {
            "run_id": self.run_id,
            "analysis_mode": self.analysis_mode,
            "convergence_status": self._convergence_status,
            "total_runtime_s": total_runtime,
            "config_summary": self._config_summary,  # T054
            "phases": phases_dict,
            "metrics": _json_safe(self._metrics),
            "output_files": [str(p) for p in self._output_files],
            "warning_count": self._warning_count,
            "error_count": self._error_count,
        }


class MinimalLogger:
    """Configurable logger manager for the xpcsjax package.

    Thread-safe singleton for managing xpcsjax logging configuration.
    """

    _instance: MinimalLogger | None = None
    _initialized: bool
    _configured: bool
    _root_logger_name: str
    _lock: threading.Lock

    def __new__(cls) -> MinimalLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._lock = threading.Lock()
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._configured = False
        self._root_logger_name = "xpcsjax"
        self._initialized = True

    @staticmethod
    def _build_formatter(
        format_name: str = "detailed",
        use_color: bool = False,
    ) -> logging.Formatter:
        fmt = (
            DEFAULT_FORMAT_SIMPLE
            if format_name == "simple"
            else DEFAULT_FORMAT_DETAILED
        )
        return _ColorFormatter(
            fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S", use_color=use_color
        )

    def _clear_managed_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            if getattr(handler, "_xpcsjax_managed", False):
                logger.removeHandler(handler)
                handler.close()

    def configure(
        self,
        level: str | int = "INFO",
        *,
        console_level: str | int | None = None,
        console_format: str = "detailed",
        console_colors: bool = False,
        file_path: str | Path | None = None,
        file_level: str | int | None = None,
        file_format: str = "detailed",
        max_size_mb: int = 10,
        backup_count: int = 5,
        module_levels: Mapping[str, str | int] | None = None,
        force: bool = False,
        json_format: str | None = None,
        quiet: bool = False,
    ) -> Path | None:
        """Configure xpcsjax logging.

        Thread-safe configuration of the logging system.
        Returns the file path if a file handler is created.

        ``json_format`` (the resolved YAML ``format`` hint) and ``quiet`` are
        threaded explicitly into the locked ``_configure_impl`` so the Phase-1b
        format/env-DEBUG wiring reads them as locals — no transient singleton
        state that two concurrent callers could clobber.
        """
        with self._lock:
            return self._configure_impl(
                level=level,
                console_level=console_level,
                console_format=console_format,
                console_colors=console_colors,
                file_path=file_path,
                file_level=file_level,
                file_format=file_format,
                max_size_mb=max_size_mb,
                backup_count=backup_count,
                module_levels=module_levels,
                force=force,
                json_format=json_format,
                quiet=quiet,
            )

    def _configure_impl(
        self,
        level: str | int = "INFO",
        *,
        console_level: str | int | None = None,
        console_format: str = "detailed",
        console_colors: bool = False,
        file_path: str | Path | None = None,
        file_level: str | int | None = None,
        file_format: str = "detailed",
        max_size_mb: int = 10,
        backup_count: int = 5,
        module_levels: Mapping[str, str | int] | None = None,
        force: bool = False,
        json_format: str | None = None,
        quiet: bool = False,
    ) -> Path | None:
        """Internal implementation of configure (called under lock)."""
        root_logger = logging.getLogger(self._root_logger_name)

        if force:
            self._clear_managed_handlers(root_logger)

        root_level_candidates = [_resolve_level(level)]
        if console_level is not None:
            root_level_candidates.append(_resolve_level(console_level))
        if file_level is not None:
            root_level_candidates.append(_resolve_level(file_level))
        root_level = min(lvl for lvl in root_level_candidates if lvl is not None)
        root_logger.setLevel(root_level)

        # Console handler — only reuse an existing managed handler to avoid duplicating
        # output when called multiple times (e.g., configure_from_dict with force=True).
        console_handler: logging.Handler | None = None
        for handler in root_logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and getattr(handler, "_xpcsjax_managed", False)
            ):
                console_handler = handler
                break

        if console_level is not None:
            if console_handler is None:
                console_handler = logging.StreamHandler()
                console_handler._xpcsjax_managed = True  # type: ignore[attr-defined]
                root_logger.addHandler(console_handler)
            console_handler.setLevel(_resolve_level(console_level) or root_level)
            console_handler.setFormatter(
                self._build_formatter(console_format, use_color=console_colors)
            )

        # File handler
        created_file: Path | None = None
        if file_path:
            file_path = Path(file_path)
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                warn_logger = logging.getLogger(self._root_logger_name)
                warn_logger.warning(
                    "Cannot create log directory %s: %s. File logging disabled.",
                    file_path.parent,
                    e,
                )
                file_path = None  # Skip file handler, continue with console-only
            if file_path is not None:
                created_file = file_path

                max_bytes = int(max_size_mb * 1024 * 1024)
                if max_bytes > 0:
                    file_handler: logging.Handler = RotatingFileHandler(
                        file_path,
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                    )
                else:
                    file_handler = logging.FileHandler(file_path)
                file_handler._xpcsjax_managed = True  # type: ignore[attr-defined]
                file_handler.setLevel(_resolve_level(file_level) or root_level)
                file_fmt = (
                    DEFAULT_FORMAT_SIMPLE
                    if file_format == "simple"
                    else DEFAULT_FORMAT_DETAILED
                )
                file_handler.setFormatter(
                    logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S")
                )
                root_logger.addHandler(file_handler)

        # Default suppression for external libraries (FR-005)
        # These are applied first, then user overrides can override them
        default_suppressions = {
            "jax": "WARNING",
            "numpy": "WARNING",
            "numba": "WARNING",
            "h5py": "WARNING",
        }
        for lib_name, lib_level in default_suppressions.items():
            lib_logger = logging.getLogger(lib_name)
            # Only set if not already configured by user
            if lib_logger.level == logging.NOTSET:
                lib_logger.setLevel(_resolve_level(lib_level) or logging.WARNING)

        # Module-specific overrides (user overrides win over defaults)
        if module_levels:
            for module_name, module_level in module_levels.items():
                logging.getLogger(module_name).setLevel(
                    _resolve_level(module_level) or root_level
                )

        import os
        current_test = os.environ.get("PYTEST_CURRENT_TEST", "")

        has_managed_handler = any(
            getattr(handler, "_xpcsjax_managed", False)
            for handler in root_logger.handlers
        )

        if current_test and "disables_propagation" not in current_test:
            root_logger.propagate = True
        else:
            root_logger.propagate = not has_managed_handler

        # Phase 1b wiring: env/YAML format selection, context filter install,
        # and env DEBUG override. ``root_logger`` and ``console_handler`` are
        # already in scope here (the single chokepoint all configure paths flow
        # through); do not re-fetch. The format hint and quiet flag arrive as
        # explicit locals (``json_format``/``quiet``) — no transient singleton
        # state — so concurrent callers cannot clobber each other.
        # ``XPCSJAX_LOG_FORMAT=json`` (or a YAML ``format: json``) swaps every
        # managed handler to JSON output; env wins over YAML.
        fmt = os.environ.get("XPCSJAX_LOG_FORMAT") or json_format
        if fmt == "json":
            json_fmt = JSONFormatter()
            for h in root_logger.handlers:
                if getattr(h, "_xpcsjax_managed", False):
                    h.setFormatter(json_fmt)
        for h in root_logger.handlers:
            if not getattr(h, "_xpcsjax_managed", False):
                continue
            if not any(isinstance(f, ContextFilter) for f in h.filters):
                h.addFilter(ContextFilter())
        # quiet must still win; apply env DEBUG only when not quiet, AFTER the
        # quiet/verbose level handling resolved upstream in configure_from_dict.
        # Lowering only the root logger is a no-op for console output — the
        # console handler was pinned to the (higher) console level above and
        # would drop DEBUG records — so lower the handler level too.
        if not quiet and os.environ.get("XPCSJAX_DEBUG") == "1":
            root_logger.setLevel(logging.DEBUG)
            if console_handler is not None:
                console_handler.setLevel(logging.DEBUG)

        self._configured = True
        return created_file

    def configure_from_dict(
        self,
        logging_config: Mapping[str, Any] | None,
        *,
        verbose: bool = False,
        quiet: bool = False,
        output_dir: Path | str | None = None,
        run_id: str | None = None,
    ) -> Path | None:
        """Configure logging from a `logging:` config section."""
        if not logging_config or not logging_config.get("enabled", True):
            return None

        level = logging_config.get("level", "INFO")

        console_cfg: Mapping[str, Any] = logging_config.get("console", {}) or {}
        file_cfg: Mapping[str, Any] = logging_config.get("file", {}) or {}

        console_enabled = console_cfg.get("enabled", True)
        console_level: str | int | None = (
            console_cfg.get("level", level) if console_enabled else None
        )
        if console_enabled:
            if quiet:
                console_level = "ERROR"
            elif verbose:
                console_level = "DEBUG"

        file_path: Path | None = None
        if file_cfg.get("enabled", False):
            if "path" in file_cfg:
                base_dir = Path(file_cfg.get("path", "./logs/"))
                if not base_dir.is_absolute():
                    base_dir = base_dir.resolve()
            else:
                base_dir = Path(output_dir) / "logs" if output_dir else Path("./logs")
                base_dir = base_dir.resolve()
            # A configured filename is honored as-is unless it contains a
            # placeholder, in which case per-run uniqueness is opt-in:
            #   ``{run_id}``                -> the run id (timestamp fallback)
            #   ``<timestamp>`` / ``{timestamp}`` -> current YYYYmmdd_HHMMSS
            # With no placeholder the filename is used verbatim (the
            # RotatingFileHandler handles size-based rotation/backups). When no
            # filename is configured at all, auto-generate a timestamped name.
            configured_filename = file_cfg.get("filename")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_suffix = run_id or timestamp
            if configured_filename is None:
                filename = f"xpcsjax_analysis_{run_suffix}.log"
            else:
                filename = (
                    configured_filename.replace("{run_id}", run_suffix)
                    .replace("<timestamp>", timestamp)
                    .replace("{timestamp}", timestamp)
                )
            file_path = base_dir / filename

        # Phase 1b: seed the context-local run_id (surfaced by ContextFilter)
        # and thread the YAML format hint + quiet flag down to _configure_impl
        # as EXPLICIT parameters (read as locals under the configure() lock),
        # which performs the format/filter/env-DEBUG wiring with the configured
        # root logger already in scope. Env XPCSJAX_LOG_FORMAT wins over YAML.
        if run_id is not None:
            set_log_context(run_id=run_id)
        return self.configure(
            level=level,
            console_level=console_level,
            console_format=console_cfg.get("format", "detailed"),
            console_colors=bool(console_cfg.get("colors", False)),
            file_path=file_path,
            file_level=file_cfg.get("level", "DEBUG"),
            file_format=file_cfg.get("format", "detailed"),
            max_size_mb=int(file_cfg.get("max_size_mb", 10)),
            backup_count=int(file_cfg.get("backup_count", 5)),
            module_levels=logging_config.get("modules"),
            force=True,
            json_format=logging_config.get("format"),
            quiet=quiet,
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with hierarchical naming."""
        if not name.startswith(self._root_logger_name):
            if name == "__main__":
                full_name = f"{self._root_logger_name}.main"
            elif "." in name and name.startswith("xpcsjax"):
                full_name = name
            else:
                full_name = f"{self._root_logger_name}.{name}"
        else:
            full_name = name

        if not self._configured:
            self.configure()

        return logging.getLogger(full_name)


# Global logger manager instance
_logger_manager = MinimalLogger()


def configure_logging(
    logging_config: Mapping[str, Any] | None,
    *,
    verbose: bool = False,
    quiet: bool = False,
    output_dir: Path | str | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Public helper to configure logging from config + CLI flags."""
    return _logger_manager.configure_from_dict(
        logging_config,
        verbose=verbose,
        quiet=quiet,
        output_dir=output_dir,
        run_id=run_id,
    )


def get_logger(
    name: str | None = None,
    *,
    context: Mapping[str, Any] | None = None,
) -> logging.Logger | logging.LoggerAdapter[logging.Logger]:
    """Get a logger instance with automatic naming and optional context."""
    if name is None:
        frame = inspect.currentframe()
        try:
            if frame is not None and frame.f_back is not None:
                name = frame.f_back.f_globals.get("__name__", "unknown")
        finally:
            del frame

    base_logger = _logger_manager.get_logger(name or "unknown")
    if context:
        return _ContextAdapter(base_logger, dict(context))
    return base_logger


def with_context(
    logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    **context: Any,
) -> logging.LoggerAdapter[logging.Logger]:
    """Create a contextual logger with key-value prefixes.

    Context is formatted as [key=value][key2=value2] message.
    Nested calls merge contexts (inner overrides outer on key conflicts).
    Thread-safe for use in multiprocessing.

    Args:
        logger: Base logger or existing contextual adapter to wrap.
        **context: Key-value pairs to include as prefix.

    Returns:
        A logger adapter that prefixes all messages with context.

    Example:
        >>> logger = get_logger(__name__)
        >>> ctx_logger = with_context(logger, run_id="abc123", mode="laminar_flow")
        >>> ctx_logger.info("Starting analysis")
        # Output: [run_id=abc123 mode=laminar_flow] Starting analysis

        >>> # Nested context
        >>> shard_logger = with_context(ctx_logger, shard=5)
        >>> shard_logger.info("Processing shard")
        # Output: [run_id=abc123 mode=laminar_flow shard=5] Processing shard
    """
    # Filter out None values from new context
    new_context = {k: v for k, v in context.items() if v is not None}

    # If wrapping an existing _ContextAdapter, merge contexts (inner overrides outer)
    if isinstance(logger, _ContextAdapter):
        merged_context = dict(logger.extra) if logger.extra else {}
        merged_context.update(new_context)
        # Get the underlying logger to avoid nested adapters
        base_logger = logger.logger
        return _ContextAdapter(base_logger, merged_context)

    # If wrapping a LoggerAdapter (not our _ContextAdapter), extract base logger
    if isinstance(logger, logging.LoggerAdapter):
        base_logger = logger.logger
        return _ContextAdapter(base_logger, new_context)

    # Wrapping a plain Logger
    return _ContextAdapter(logger, new_context)


@dataclass
class PhaseContext:
    """Context object returned by log_phase() with timing and memory info."""

    name: str
    duration: float = 0.0
    memory_peak_gb: float | None = None
    memory_delta_gb: float | None = None


def _get_memory_gb() -> float | None:
    """Get current process memory usage in GB, or None if unavailable.

    Prefers the stdlib ``resource`` module (POSIX). On Windows, where
    ``resource`` does not exist, falls back to ``psutil`` so memory diagnostics
    stay available cross-platform.
    """
    try:
        import resource

        # Get max resident set size (in KB on Linux, bytes on macOS)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in bytes on macOS, KB on Linux
        import sys

        scale = (1024**3) if sys.platform == "darwin" else (1024**2)
        return rusage.ru_maxrss / scale
    except (ImportError, AttributeError):
        pass
    # Windows (no ``resource``): use psutil's RSS if it is installed.
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**3)
    except Exception:
        return None


@contextmanager
def log_phase(
    name: str,
    logger: LoggerType | None = None,
    level: int = logging.INFO,
    track_memory: bool = False,
    threshold_s: float = 0.0,
) -> Generator[PhaseContext, None, None]:
    """Context manager for phase-level timing with optional memory tracking.

    Args:
        name: Phase name for logging.
        logger: Logger to use. If None, uses module logger.
        level: Log level for phase messages.
        track_memory: Track memory usage during phase.
        threshold_s: Only log if duration > threshold (0 = always log).

    Yields:
        PhaseContext with name, duration, memory_peak_gb, memory_delta_gb.
        Duration and memory values are populated after the context exits.

    Example:
        >>> with log_phase("optimization", track_memory=True) as phase:
        ...     result = run_optimization(data)
        >>> print(f"Took {phase.duration:.1f}s")
        # Logs: Phase 'optimization' completed in 45.3s (peak memory: 12.4 GB)
    """
    resolved_logger = get_logger() if logger is None else logger
    context = PhaseContext(name=name)

    # Track initial memory if requested. Memory probing is best-effort: a probe
    # failure must degrade to a memory-less phase, never escape the context.
    memory_start: float | None = None
    if track_memory:
        try:
            memory_start = _get_memory_gb()
        except Exception:
            memory_start = None

    # Log phase start (only if no threshold or threshold is 0)
    if threshold_s <= 0:
        resolved_logger.log(level, "Phase '%s' started", name)

    start_time = time.perf_counter()

    try:
        yield context
    finally:
        # Calculate duration
        context.duration = time.perf_counter() - start_time

        # Track memory if requested (best-effort; a probe failure here must not
        # escape the finally block and is degraded to a memory-less message).
        if track_memory:
            try:
                memory_end = _get_memory_gb()
            except Exception:
                memory_end = None
            if memory_end is not None:
                context.memory_peak_gb = memory_end
                if memory_start is not None:
                    context.memory_delta_gb = memory_end - memory_start

        # Log phase completion if duration exceeds threshold. Emission is
        # observational only: a logging failure must never escape the context.
        try:
            if context.duration >= threshold_s:
                if context.memory_peak_gb is not None:
                    resolved_logger.log(
                        level,
                        "Phase '%s' completed in %.2fs (peak memory: %.1f GB)",
                        name,
                        context.duration,
                        context.memory_peak_gb,
                    )
                else:
                    resolved_logger.log(
                        level,
                        "Phase '%s' completed in %.2fs",
                        name,
                        context.duration,
                    )
        except Exception:
            pass


def log_exception(
    logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    exc: BaseException,
    context: dict[str, Any] | None = None,
    level: int = logging.ERROR,
    include_traceback: bool = True,
) -> None:
    """Log an exception with full context for debugging.

    Extracts module, function, and line number from exception traceback.
    Formats context as key-value pairs in the message.

    Args:
        logger: Logger to use.
        exc: Exception to log.
        context: Additional context (e.g., parameter values).
        level: Log level (default ERROR).
        include_traceback: Include full traceback (default True).

    Example:
        >>> try:
        ...     result = compute_jacobian(params)
        ... except ValueError as e:
        ...     log_exception(logger, e, context={
        ...         "iteration": 45,
        ...         "params": params.tolist()[:5]
        ...     })
        ...     raise
        # Logs:
        # ERROR | xpcsjax.optimization.nlsq.core | Exception in compute_jacobian:
        # ValueError: invalid value
        # Context: iteration=45, params=[1.2e-11, 0.85, ...]
        # Traceback (most recent call last):
        #   ...
    """
    # Logging is observational only: a failure while formatting/emitting the
    # diagnostic (e.g. a context value whose __repr__ raises) must never escape
    # and must never change control flow at the call site.
    try:
        # Extract location info from traceback
        tb = exc.__traceback__
        location_info = ""
        if tb is not None:
            # Walk to the innermost frame where the exception occurred
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            func_name = frame.f_code.co_name
            line_no = tb.tb_lineno
            module_name = frame.f_globals.get("__name__", "unknown")
            location_info = f" in {module_name}.{func_name}:{line_no}"

        # Build the message
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        msg_parts = [f"Exception{location_info}: {exc_type}: {exc_msg}"]

        # Add context if provided
        if context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in context.items())
            msg_parts.append(f"Context: {context_str}")

        # Add traceback if requested
        if include_traceback:
            tb_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            msg_parts.append(f"Traceback:\n{tb_str}")

        logger.log(level, "\n".join(msg_parts))
    except Exception:
        # Degrade to a minimal record; the fallback is itself guarded so a
        # second failure (e.g. exc.__repr__ raising) is swallowed too.
        try:
            logger.error("log_exception failed while logging %r", exc)
        except Exception:
            pass


def log_calls(
    logger: LoggerType | None = None,
    level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        include_args: Whether to log function arguments.
        include_result: Whether to log function return value.
    """
    resolved_logger: LoggerType | None = logger

    def decorator(func: F) -> F:
        nonlocal resolved_logger
        if resolved_logger is None:
            resolved_logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            assert resolved_logger is not None  # For type narrowing

            # Guard: skip all formatting if log level is not enabled
            func_name = f"{func.__module__}.{func.__qualname__}"
            log_enabled = resolved_logger.isEnabledFor(level)

            # Log function entry
            if log_enabled:
                if include_args:
                    args_str = ", ".join([repr(arg) for arg in args])
                    kwargs_str = ", ".join(
                        [f"{k}={repr(v)}" for k, v in kwargs.items()]
                    )
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    resolved_logger.log(level, "Calling %s(%s)", func_name, all_args)
                else:
                    resolved_logger.log(level, "Calling %s", func_name)

            try:
                result = func(*args, **kwargs)

                # Log function exit
                if log_enabled:
                    if include_result:
                        resolved_logger.log(
                            level, "Completed %s -> %r", func_name, result
                        )
                    else:
                        resolved_logger.log(level, "Completed %s", func_name)

                return result

            except Exception as e:
                try:
                    resolved_logger.log(
                        logging.ERROR, "Exception in %s: %s", func_name, e
                    )
                except Exception:  # noqa: BLE001 - logging must not mask original
                    pass
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_performance(
    logger: LoggerType | None = None,
    level: int = logging.INFO,
    threshold: float = 0.1,
) -> Callable[[F], F]:
    """Decorator to log function performance.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        threshold: Minimum duration (seconds) to log.
    """
    resolved_logger: LoggerType | None = logger

    def decorator(func: F) -> F:
        nonlocal resolved_logger
        if resolved_logger is None:
            resolved_logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            func_name = f"{func.__module__}.{func.__qualname__}"
            assert resolved_logger is not None  # For type narrowing

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                if duration >= threshold:
                    resolved_logger.log(
                        level,
                        "Performance: %s completed in %.3fs",
                        func_name,
                        duration,
                    )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                try:
                    resolved_logger.log(
                        logging.ERROR,
                        "Performance: %s failed after %.3fs: %s",
                        func_name,
                        duration,
                        e,
                    )
                except Exception:  # noqa: BLE001 - logging must not mask original
                    pass
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def log_operation(
    operation_name: str,
    logger: LoggerType | None = None,
    level: int = logging.INFO,
) -> Generator[LoggerType, None, None]:
    """Context manager for logging operations.

    Args:
        operation_name: Name of the operation.
        logger: Logger to use. If None, creates one for caller's module.
        level: Logging level to use.
    """
    resolved_logger = get_logger() if logger is None else logger

    resolved_logger.log(level, "Starting operation: %s", operation_name)
    start_time = time.perf_counter()

    try:
        yield resolved_logger
        duration = time.perf_counter() - start_time
        resolved_logger.log(
            level, "Completed operation: %s in %.3fs", operation_name, duration
        )
    except Exception as e:
        duration = time.perf_counter() - start_time
        try:
            resolved_logger.log(
                logging.ERROR,
                "Failed operation: %s after %.3fs: %s",
                operation_name,
                duration,
                e,
            )
        except Exception:  # noqa: BLE001 - logging must not mask original
            pass
        raise


_LOG_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "xpcsjax_log_context", default=None
)
_CONTEXT_FIELDS = ("run_id", "phase", "mode", "strategy")


def set_log_context(**fields: Any) -> contextvars.Token:
    """Set context-local log fields, returning a token for restoration.

    Passing a field with value ``None`` removes it from the context. The
    returned token can be passed to :func:`reset_log_context` to restore the
    prior context (e.g. on scope exit).
    """
    cur = dict(_LOG_CONTEXT.get() or {})
    for k, v in fields.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    return _LOG_CONTEXT.set(cur)


def reset_log_context(token: contextvars.Token) -> None:
    """Restore the log context to the state captured by ``token``."""
    _LOG_CONTEXT.reset(token)


@contextmanager
def log_context(**fields: Any) -> Generator[None, None, None]:
    """Context manager that sets log context fields for the enclosed scope.

    The prior context is restored on exit, so nested ``log_context`` blocks
    stack and unwind correctly.
    """
    token = set_log_context(**fields)
    try:
        yield
    finally:
        reset_log_context(token)


class ContextFilter(logging.Filter):
    """Logging filter that injects context-local fields onto each record.

    Fields named in ``_CONTEXT_FIELDS`` are read from the context-local
    registry and attached to the record (without clobbering fields already
    set on the record). Always returns ``True`` so no record is dropped.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get() or {}
        for f in _CONTEXT_FIELDS:
            if not hasattr(record, f):
                setattr(record, f, ctx.get(f))
        return True


_LOG_ONCE_SEEN: set[str] = set()
_LOG_ONCE_LOCK = threading.Lock()


def _should_log_once(key: str) -> bool:
    """Return True the first time ``key`` is seen, False thereafter.

    Thread-safe keyed de-duplication primitive shared by :func:`log_once` and
    :func:`logged_errors`.
    """
    with _LOG_ONCE_LOCK:
        if key in _LOG_ONCE_SEEN:
            return False
        _LOG_ONCE_SEEN.add(key)
        return True


def reset_log_once_cache() -> None:
    """Clear the keyed de-dup cache (primarily for tests)."""
    with _LOG_ONCE_LOCK:
        _LOG_ONCE_SEEN.clear()


def log_once(logger: Any, level: int, key: str, msg: str, *args: Any) -> None:
    """Emit a log record at most once per ``key``.

    Observational only: a failure while emitting the record is swallowed and
    never escapes to the call site.
    """
    if _should_log_once(key):
        try:
            logger.log(level, msg, *args)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def logged_errors(
    logger: LoggerType,
    operation: str,
    *,
    policy: Literal["reraise", "suppress"],
    level: int = logging.ERROR,
    once_key: str | None = None,
    **context: Any,
) -> Generator[None, None, None]:
    """Log exceptions raised in the enclosed scope with structured context.

    On exception, emits a contextual diagnostic via :func:`log_exception`
    (de-duplicated by ``once_key`` when provided), then applies ``policy``:
    ``"reraise"`` re-raises the original exception, ``"suppress"`` swallows it.
    The diagnostic emission itself is guarded so logging never masks or
    replaces the original exception.
    """
    try:
        yield
    except Exception as exc:
        try:
            if once_key is None or _should_log_once(once_key):
                log_exception(
                    logger,
                    exc,
                    context={"operation": operation, **context},
                    level=level,
                )
        except Exception:  # noqa: BLE001 - logging must not mask the original
            pass
        if policy == "reraise":
            raise


# Configure default logging on import
_logger_manager.configure()
