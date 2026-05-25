"""Coverage for ConfigManager construction error paths (audit finding #17).

Exercises the previously-uncovered ``load_config`` raises and the
``_validate_config`` unknown-mode branch.
"""

from __future__ import annotations

import logging

import pytest

from xpcsjax.config.manager import ConfigManager


def test_none_config_path_falls_back_to_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # load_config() raises internally on a None path but catches it and falls back
    # to a default configuration (logged as an error) rather than propagating.
    with caplog.at_level(logging.ERROR):
        cm = ConfigManager(config_file=None)  # type: ignore[arg-type]
    assert cm.config is not None  # default config populated
    assert any("none" in rec.message.lower() for rec in caplog.records)


def test_missing_config_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        ConfigManager(config_file="/nonexistent/path/does_not_exist.yaml")


def test_unknown_analysis_mode_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    # The override path validates without touching the filesystem; an unknown
    # analysis_mode is a soft warning (not a hard raise).
    with caplog.at_level(logging.WARNING):
        ConfigManager(config_override={"analysis_mode": "bogus_mode"})
    assert any("analysis_mode" in rec.message.lower() for rec in caplog.records)
