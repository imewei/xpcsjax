"""Regression tests for unified CLI output-directory resolution.

Adversarial-review finding (medium): ``resolve_plots_dir`` only honored the
legacy ``output_settings.output_dir`` key, so a normal run driven by a shipped
template (canonical ``output.directory`` schema) saved JSON/NPZ results under
the configured output tree while plots were written to the process cwd. These
tests pin both paths to the single shared resolver so they cannot diverge
again.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xpcsjax.cli.config_handling import resolve_output_dir
from xpcsjax.cli.plot_dispatch import resolve_plots_dir


def _args(output: object = None) -> Any:
    return SimpleNamespace(output=output)


def _cfg(config: dict | None) -> Any:
    """Minimal ConfigManager stand-in — the resolver only reads ``.config``."""
    return SimpleNamespace(config=config)


# --- resolve_output_dir: schema coverage -----------------------------------


def test_template_schema_directory_with_base(tmp_path: Path) -> None:
    cfgmgr = _cfg({"output": {"base_directory": str(tmp_path), "directory": "results"}})
    assert resolve_output_dir(_args(), cfgmgr) == tmp_path / "results"


def test_template_schema_directory_without_base() -> None:
    cfgmgr = _cfg({"output": {"directory": "out"}})
    assert resolve_output_dir(_args(), cfgmgr) == Path("out")


def test_legacy_output_settings_fallback() -> None:
    cfgmgr = _cfg({"output_settings": {"output_dir": "legacy_out"}})
    assert resolve_output_dir(_args(), cfgmgr) == Path("legacy_out")


def test_cli_override_wins_over_config(tmp_path: Path) -> None:
    cfgmgr = _cfg({"output": {"directory": "results"}})
    assert resolve_output_dir(_args(output=tmp_path / "cli"), cfgmgr) == tmp_path / "cli"


def test_none_when_unconfigured() -> None:
    assert resolve_output_dir(_args(), _cfg({})) is None
    assert resolve_output_dir(_args(), _cfg(None)) is None
    assert resolve_output_dir(_args(), None) is None


# --- the core regression: plots and results share one root -----------------


def test_plots_dir_agrees_with_result_dir_template_schema(tmp_path: Path) -> None:
    """With a template-style config and no ``--output``, plots land under the
    same configured root as results (``<root>/plots``)."""
    cfgmgr = _cfg({"output": {"base_directory": str(tmp_path), "directory": "results"}})
    args = _args()  # no CLI override — the path that previously broke

    result_root = resolve_output_dir(args, cfgmgr)
    plots_dir = resolve_plots_dir(args, cfgmgr)

    assert result_root is not None
    assert result_root == tmp_path / "results"
    assert plots_dir == result_root / "plots"
    assert plots_dir.parent == result_root
    assert plots_dir.is_dir()


def test_plots_dir_honors_cli_override(tmp_path: Path) -> None:
    cfgmgr = _cfg({"output": {"directory": "results"}})
    args = _args(output=tmp_path / "cli_out")
    plots_dir = resolve_plots_dir(args, cfgmgr)
    assert plots_dir == tmp_path / "cli_out" / "plots"
    assert plots_dir.is_dir()
