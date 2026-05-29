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


def test_post_fit_plots_write_under_plots_subdir(tmp_path: Path, monkeypatch: Any) -> None:
    """``_generate_post_fit_plots`` must hand the ``plots/`` subdir to
    ``generate_nlsq_plots`` — not strip it back to the root. Previously it passed
    ``plots_dir.parent``, scattering heatmaps/residuals/simulated_data into the
    output root while the dispatcher logged "Plots written to <root>/plots".
    """
    import xpcsjax.viz as viz
    from xpcsjax.cli import plot_dispatch

    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> None:
        captured["output_dir"] = kwargs["output_dir"]

    monkeypatch.setattr(viz, "generate_nlsq_plots", _capture)

    plots_dir = tmp_path / "results" / "plots"
    plots_dir.mkdir(parents=True)
    # Minimal ConfigManager stand-in: only get_model/get_config are used.
    cfgmgr = SimpleNamespace(
        get_model=lambda: object(),
        get_config=lambda: {},
    )

    out = plot_dispatch._generate_post_fit_plots(
        args=_args(),
        config_manager=cfgmgr,
        data={},
        result=SimpleNamespace(),
        plots_dir=plots_dir,
    )

    # The artifact dump must land under <root>/plots, matching the logged path
    # and every other plot path — not the root (plots_dir.parent).
    assert captured["output_dir"] == plots_dir
    assert captured["output_dir"] != plots_dir.parent
    # Log-accuracy prevention: the helper reports the directory it actually
    # wrote to, so dispatch_plots' "Plots written to …" message derives from
    # reality rather than the pre-computed plots_dir.
    assert out == plots_dir


def test_post_fit_plots_return_none_on_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """When generate_nlsq_plots raises, _generate_post_fit_plots must report
    that nothing was written (return None) so dispatch_plots does not log a
    location that received no files."""
    import xpcsjax.viz as viz
    from xpcsjax.cli import plot_dispatch

    def _boom(**_kwargs: Any) -> None:
        raise RuntimeError("plotting blew up")

    monkeypatch.setattr(viz, "generate_nlsq_plots", _boom)

    plots_dir = tmp_path / "results" / "plots"
    plots_dir.mkdir(parents=True)
    cfgmgr = SimpleNamespace(get_model=lambda: object(), get_config=lambda: {})

    out = plot_dispatch._generate_post_fit_plots(
        args=_args(),
        config_manager=cfgmgr,
        data={},
        result=SimpleNamespace(),
        plots_dir=plots_dir,
    )
    assert out is None
