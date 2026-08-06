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


# --- Fix 3: quoted-string max_iterations/tolerance must not TypeError (and
# must not silently discard the parsed config) ------------------------------


def test_quoted_max_iterations_does_not_raise_or_discard_config() -> None:
    mgr = ConfigManager(config_override={"optimization": {"nlsq": {"max_iterations": "10000"}}})
    # No TypeError from the unguarded `max_iter > 50000` comparison, and the
    # real config survives (not silently replaced by _get_default_config()).
    assert mgr.config["optimization"]["nlsq"]["max_iterations"] == "10000"


def test_quoted_tolerance_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        mgr = ConfigManager(config_override={"optimization": {"nlsq": {"tolerance": "1e-8"}}})
    assert mgr.config["optimization"]["nlsq"]["tolerance"] == "1e-8"
    assert any("tolerance=1e-8 is not numeric" in rec.message for rec in caplog.records)


# --- Fix 4: laminar_flow anti-degeneracy warning must fire regardless of
# analysis_mode case/synonym -------------------------------------------------


def test_laminar_flow_warning_fires_on_uppercase_mode(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        ConfigManager(
            config_override={
                "analysis_mode": "LAMINAR_FLOW",
                "optimization": {"nlsq": {"anti_degeneracy": {"hierarchical": {"enable": False}}}},
            }
        )
    assert any("gradient cancellation" in rec.message for rec in caplog.records)


# --- config_validation_error flag: load_config() keeps the parsed config on a
# validation failure but must still surface a caller-visible signal (previously
# only a logger.error line) -------------------------------------------------


def test_validation_error_is_exposed_and_config_kept(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("analysis_mode: static_isotropic\nfoo: bar\n")

    def _boom(self) -> None:
        raise ValueError("simulated validation failure")

    monkeypatch.setattr(ConfigManager, "_validate_config", _boom)
    with caplog.at_level(logging.ERROR):
        mgr = ConfigManager(config_file=str(config_file))

    assert isinstance(mgr.config_validation_error, ValueError)
    assert "simulated validation failure" in str(mgr.config_validation_error)
    # The parsed config must survive — not silently replaced by defaults.
    assert mgr.config.get("foo") == "bar"
    assert any("Configuration validation error" in rec.message for rec in caplog.records)


def test_validation_error_flag_is_none_on_success(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("analysis_mode: static_isotropic\n")
    mgr = ConfigManager(config_file=str(config_file))
    assert mgr.config_validation_error is None
