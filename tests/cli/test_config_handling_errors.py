"""Regression tests for config_handling.py error-path hardening.

* L108: ``load_and_merge_config`` names the file in its load-failure message.

The other two guards this file used to test (a config-manager double without
``_normalize_analysis_mode``, and a logged warning for a non-dict
``config['output']``) lived in ``config_handling.apply_cli_overrides``'s OWN
mode/output-directory handling. That handling was deleted (audit B8):
``load_and_merge_config`` now applies mode/output-dir overrides via
``xpcsjax.service.config.load_config`` -- the SAME function the GUI uses --
so ``apply_cli_overrides`` here only applies ``--initial-*`` parameter
overrides. Two notes on what changed:

* The ``_normalize_analysis_mode`` tolerance guard is gone entirely, matching
  ``service/config.py``'s own comment that it deliberately does NOT guard
  that call (its only real caller always constructs a genuine ConfigManager).
* The non-dict-``output`` case is still handled safely in
  ``service/config.py`` (it resets to ``{}``), but SILENTLY -- that
  implementation is marked ``# pragma: no cover — defensive`` there rather
  than logging a warning. This is a minor, known behavior regression from
  the old CLI-side warning; left to the config/service owner to decide
  whether to add logging there.
"""

from __future__ import annotations

import argparse

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
