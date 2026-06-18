"""Tests for the argparse-free config loader (xpcsjax/service/config.py)."""

import subprocess
import sys
import textwrap

import xpcsjax.service.config as svc_config


def _probe_import(module: str) -> int:
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode


def test_config_service_is_jax_free():
    # Depends on the Plan-1A fix (parameter_manager.py lazy core.physics import):
    # importing xpcsjax.service.config triggers xpcsjax/config/__init__.py, which
    # is JAX-free only after 1A lands. 1A is a hard prerequisite of this plan, so
    # this assertion holds when B2 runs; it would (correctly) fail on a pre-1A tree.
    assert _probe_import("xpcsjax.service.config") == 0


def test_load_config_applies_mode_and_output(monkeypatch):
    # Stub ConfigManager so the test does not require any on-disk YAML.
    made = {}

    class _FakeCM:
        def __init__(self, path):
            made["path"] = path
            self.config = {"analysis_mode": "static_isotropic"}

        def _normalize_analysis_mode(self):
            made["normalized"] = True

    monkeypatch.setattr(svc_config, "ConfigManager", _FakeCM)

    cm = svc_config.load_config("cfg.yaml", mode="laminar_flow", output_dir="/tmp/out")
    assert made["path"] == "cfg.yaml"
    assert cm.config["analysis_mode"] == "laminar_flow"
    assert made.get("normalized") is True
    assert cm.config["output"]["directory"] == "/tmp/out"


def test_load_config_no_overrides_leaves_config(monkeypatch):
    class _FakeCM:
        def __init__(self, path):
            self.config = {"analysis_mode": "two_component"}

        def _normalize_analysis_mode(self):
            pass

    monkeypatch.setattr(svc_config, "ConfigManager", _FakeCM)
    cm = svc_config.load_config("cfg.yaml")
    assert cm.config == {"analysis_mode": "two_component"}
