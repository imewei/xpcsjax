"""Regression test for config-driven file logging on the CLI path.

Root cause this pins: ``dispatch_command`` loaded the config but never passed
its ``logging:`` section to ``configure_logging`` — the only call site
(``cli/main.py``) hardcoded ``logging_config=None``, and because
``configure_from_dict`` short-circuits on falsy input the whole thing was a
silent no-op. A config with ``logging.file.enabled: true`` therefore produced
no log file. This test drives ``dispatch_command`` and asserts a file is
actually written.

The test mutates process-global logging state (handlers on the ``xpcsjax``
logger), so an autouse fixture snapshots and restores it around the test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xpcsjax.cli import commands
from xpcsjax.utils.logging import get_logger


@pytest.fixture(autouse=True)
def _isolate_logging() -> Any:
    """Restore the xpcsjax logger's handlers/level after the test."""
    lg = logging.getLogger("xpcsjax")
    saved_handlers = lg.handlers[:]
    saved_level = lg.level
    mgr = commands.configure_logging.__globals__["_logger_manager"]
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


def test_dispatch_command_writes_log_file_when_config_enables_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config with ``logging.file.enabled: true`` produces a log file when
    routed through ``dispatch_command`` (the bug: it produced none)."""
    log_dir = tmp_path / "logs"
    logging_cfg = {
        "enabled": True,
        "level": "INFO",
        "console": {"enabled": True, "level": "INFO"},
        "file": {
            "enabled": True,
            "level": "DEBUG",
            "path": str(log_dir),
            "filename": "xpcsjax_regression.log",
            "max_size_mb": 10,
            "backup_count": 5,
        },
    }
    cfg_manager = SimpleNamespace(config={"logging": logging_cfg})

    # Avoid touching disk for config + skip the heavy plot/fit branches; we only
    # care that logging is configured from the config before dispatch branches.
    monkeypatch.setattr(commands, "load_and_merge_config", lambda *a, **k: cfg_manager)

    def _fake_standalone(_args: Any, _cfg: Any) -> int:
        get_logger("xpcsjax.test.wiring").info("dispatch-regression-marker")
        return 0

    monkeypatch.setattr(commands, "_dispatch_standalone_plot", _fake_standalone)

    args: Any = SimpleNamespace(
        config=None,
        verbose=False,
        quiet=False,
        output=None,
        plot_simulated_data=True,
        plot_experimental_data=False,
    )

    rc = commands.dispatch_command(args)
    logging.shutdown()

    assert rc == 0
    log_files = list(log_dir.glob("*.log"))
    assert log_files, "no log file written despite logging.file.enabled=true"
    assert "dispatch-regression-marker" in log_files[0].read_text()


def test_dispatch_command_writes_no_file_when_config_disables_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default template (``file.enabled: false``) must not write a file."""
    log_dir = tmp_path / "logs"
    logging_cfg = {
        "enabled": True,
        "level": "INFO",
        "console": {"enabled": True},
        "file": {"enabled": False, "path": str(log_dir)},
    }
    cfg_manager = SimpleNamespace(config={"logging": logging_cfg})
    monkeypatch.setattr(commands, "load_and_merge_config", lambda *a, **k: cfg_manager)
    monkeypatch.setattr(commands, "_dispatch_standalone_plot", lambda *a, **k: 0)

    args: Any = SimpleNamespace(
        config=None,
        verbose=False,
        quiet=False,
        output=None,
        plot_simulated_data=True,
        plot_experimental_data=False,
    )

    rc = commands.dispatch_command(args)
    logging.shutdown()

    assert rc == 0
    assert not log_dir.exists() or not list(log_dir.glob("*.log"))
