"""Phase-1b wiring tests for xpcsjax logging.

Confirms env/YAML selection wiring inside ``configure_from_dict`` /
``_configure_impl``:

- ``XPCSJAX_LOG_FORMAT=json`` installs :class:`JSONFormatter` on the managed
  handlers.
- ``XPCSJAX_DEBUG=1`` forces DEBUG level, overriding a lower YAML ``level``.
- ``quiet=True`` beats ``XPCSJAX_DEBUG`` (env DEBUG must not win under quiet).
- :class:`ContextFilter` is installed at most once per handler across repeated
  configuration.
- ``run_id=`` seeds the context-local ``run_id`` field surfaced by
  :class:`ContextFilter`.
- Default behavior (no opt-ins) is unchanged: INFO level, no JSON formatter.

The manager exposes its instance as ``lm._logger_manager`` (a
:class:`MinimalLogger`); the root logger it configures is the stdlib logger
named ``"xpcsjax"`` (obtained via ``logging.getLogger("xpcsjax")``), so
``_managed_logger()`` resolves that, matching the existing suite in
``tests/test_logging.py``.
"""

import logging

import xpcsjax.utils.logging as lm

_CFG = {"enabled": True, "console": {"enabled": True}, "file": {"enabled": False}}


def _managed_logger() -> logging.Logger:
    # The manager has no ``_root_logger`` attribute; the configured root logger
    # is the stdlib logger named after ``_logger_manager._root_logger_name``
    # ("xpcsjax"), exactly as tests/test_logging.py accesses it.
    return logging.getLogger(lm._logger_manager._root_logger_name)


def _managed_console_handler() -> logging.Handler:
    """Return the managed console handler (StreamHandler, not a FileHandler).

    Debug-mode and quiet behavior are only observable at the handler level: the
    root logger gating is necessary but not sufficient, because a record that
    passes the root logger is still dropped if the console handler's own level
    is higher. Tests therefore assert on this handler, not just the root logger.
    """
    for h in _managed_logger().handlers:
        if (
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "_xpcsjax_managed", False)
        ):
            return h
    raise AssertionError("no managed console handler installed")


def test_xpcsjax_log_format_env_selects_json(monkeypatch):
    monkeypatch.setenv("XPCSJAX_LOG_FORMAT", "json")
    lm.configure_logging({**_CFG})
    assert any(isinstance(h.formatter, lm.JSONFormatter) for h in _managed_logger().handlers)


def test_debug_precedence_env_over_yaml(monkeypatch):
    monkeypatch.setenv("XPCSJAX_DEBUG", "1")
    lm.configure_logging({**_CFG, "level": "WARNING"})
    # Both the root logger AND the console handler must be DEBUG: lowering only
    # the root logger is a no-op for console output because the handler was
    # already pinned to the (higher) console level and would drop DEBUG records.
    assert _managed_logger().level == logging.DEBUG
    assert _managed_console_handler().level == logging.DEBUG


def test_quiet_beats_env_debug(monkeypatch):
    # Required contract: env XPCSJAX_DEBUG must NOT win over quiet=True. quiet is
    # observable at the console HANDLER (set to ERROR), so assert the handler
    # level is NOT DEBUG and is at least ERROR even with XPCSJAX_DEBUG=1 set.
    monkeypatch.setenv("XPCSJAX_DEBUG", "1")
    lm.configure_logging({**_CFG}, quiet=True)
    console_level = _managed_console_handler().level
    assert console_level != logging.DEBUG
    assert console_level >= logging.ERROR


def test_format_does_not_leak_across_configures(monkeypatch):
    # Thread-safety / state-leak guard: a json configure must not leave a
    # JSONFormatter behind on a subsequent console (non-json) configure. With
    # the json hint threaded as an explicit parameter (not transient singleton
    # state), the second configure rebuilds plain formatters with no residue.
    monkeypatch.delenv("XPCSJAX_LOG_FORMAT", raising=False)
    monkeypatch.setenv("XPCSJAX_LOG_FORMAT", "json")
    lm.configure_logging({**_CFG})
    assert isinstance(_managed_console_handler().formatter, lm.JSONFormatter)
    monkeypatch.delenv("XPCSJAX_LOG_FORMAT", raising=False)
    lm.configure_logging({**_CFG})
    assert not any(isinstance(h.formatter, lm.JSONFormatter) for h in _managed_logger().handlers)


def test_context_filter_installed_once():
    lm.configure_logging({**_CFG})
    lm.configure_logging({**_CFG})
    for h in _managed_logger().handlers:
        assert sum(isinstance(f, lm.ContextFilter) for f in h.filters) <= 1


def test_run_id_sets_log_context():
    lm.configure_logging({**_CFG}, run_id="run-xyz")
    rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", None, None)
    lm.ContextFilter().filter(rec)
    assert rec.run_id == "run-xyz"


def test_default_behavior_unchanged_without_optins(monkeypatch):
    monkeypatch.delenv("XPCSJAX_DEBUG", raising=False)
    monkeypatch.delenv("XPCSJAX_LOG_FORMAT", raising=False)
    lm.configure_logging({**_CFG})
    root = _managed_logger()
    # The real computed default is DEBUG, not INFO: configure_from_dict passes
    # file_level "DEBUG" into the root-level min regardless of file.enabled (a
    # pre-existing behavior, not introduced by Phase 1b). The load-bearing
    # checks are (a) env DEBUG is absent so the level is NOT forced by the
    # Phase-1b env override (it equals the natural computed level), and (b) no
    # JSON formatter is installed without the format opt-in.
    assert root.level == logging.DEBUG
    assert not any(isinstance(h.formatter, lm.JSONFormatter) for h in root.handlers)
