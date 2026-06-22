"""Regression tests for config_handling.py error-path hardening.

Tests the three defensive guards introduced in Task 7:

* L108: ``load_and_merge_config`` names the file in its load-failure message.
* L149: ``apply_cli_overrides`` tolerates a config-manager double without
  ``_normalize_analysis_mode``.
* L158: ``apply_cli_overrides`` logs a warning when ``config['output']`` is
  not a mapping before silently resetting it to ``{}``.
"""

from __future__ import annotations

import argparse
import logging

import pytest

from xpcsjax.cli import config_handling


def test_load_failure_names_the_file(tmp_path):
    # Signatures (verified): load_and_merge_config(yaml_path, cli_args);
    # ConfigManager(str(yaml_path)) raises on a bad file. The wrap must name the
    # path for load errors that don't already (e.g. malformed YAML). Type is NOT
    # contracted (no existing test pins it).
    bad = tmp_path / "broken.yaml"
    bad.write_text("not: [valid: yaml", encoding="utf-8")  # malformed
    with pytest.raises(Exception) as exc:
        config_handling.load_and_merge_config(bad, argparse.Namespace())
    assert str(bad) in str(exc.value)  # error names which config failed


def test_normalize_gate_tolerates_object_without_method():
    # apply_cli_overrides(config_manager, args) reads config_manager.config and,
    # when args.mode is set, calls config_manager._normalize_analysis_mode().
    # A config-manager-shaped double WITHOUT that method must not crash the
    # override (the defensive gate, formerly `except AttributeError: pass`).
    class _NoNormalize:
        config = {"analysis_mode": "static_anisotropic"}

    config_handling.apply_cli_overrides(
        _NoNormalize(), argparse.Namespace(mode="static_isotropic", output=None)
    )  # no exception == gate works


def test_non_dict_output_block_is_logged(caplog):
    # When config['output'] is not a mapping, the reset must be logged (not silent).
    class _BadConfig:
        config = {"output": "not_a_dict"}

    with caplog.at_level(logging.WARNING):
        config_handling.apply_cli_overrides(
            _BadConfig(), argparse.Namespace(mode=None, output="/tmp/out")
        )
    assert any("output" in r.message for r in caplog.records)
