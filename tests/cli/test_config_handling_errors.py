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
from xpcsjax.config import ConfigManager


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


def _write_static_isotropic_config(tmp_path) -> str:
    cfg = tmp_path / "static_isotropic.yaml"
    cfg.write_text(
        """
analysis_mode: "static_isotropic"
analyzer_parameters:
  dt: 1.0
  start_frame: 1
  end_frame: 10
  scattering:
    wavevector_q: 0.01
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "dummy.hdf"
"""
    )
    return str(cfg)


def test_initial_override_resolver_failure_raises_value_error(tmp_path, monkeypatch):
    # A11 (xpcsjax/cli/config_handling.py:261): a resolver failure while
    # applying --initial-* overrides must abort the run with a ValueError
    # naming the failure, not silently drop the override and fall back to
    # YAML/registry defaults.
    config_manager = ConfigManager(_write_static_isotropic_config(tmp_path))

    def _boom():
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(config_manager, "get_active_parameters", _boom)
    args = argparse.Namespace(initial_D0=1234.5)

    with pytest.raises(ValueError, match="cannot apply --initial"):
        config_handling.apply_cli_overrides(config_manager, args)


def test_initial_override_lands_in_initial_parameters_values(tmp_path):
    # Green companion: a well-formed --initial-D0 override must be written
    # into the canonical config["initial_parameters"]["values"] block that
    # ConfigManager.get_initial_parameters() reads back from.
    config_manager = ConfigManager(_write_static_isotropic_config(tmp_path))
    args = argparse.Namespace(initial_D0=4242.0)

    config_handling.apply_cli_overrides(config_manager, args)

    names = config_manager.config["initial_parameters"]["parameter_names"]
    values = config_manager.config["initial_parameters"]["values"]
    assert values[names.index("D0")] == pytest.approx(4242.0)
    assert config_manager.get_initial_parameters()["D0"] == pytest.approx(4242.0)
