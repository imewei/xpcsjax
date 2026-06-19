"""Tests for the post-fit plotting service (xpcsjax/service/plots.py)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import xpcsjax.service.plots as svc_plots


def _cm():
    return SimpleNamespace(
        get_model=lambda: object(), get_config=lambda: {"analysis_mode": "laminar_flow"}
    )


def test_generate_plots_forwards_to_viz_and_forces_agg(monkeypatch):
    import matplotlib

    calls = {}
    monkeypatch.setattr(
        svc_plots,
        "_generate_nlsq_plots",
        lambda **kw: calls.update(kw),
    )

    out = svc_plots.generate_plots(
        MagicMock(), {"c2_exp": None}, _cm(), Path("/tmp/plots"), use_datashader=True, parallel=True
    )
    assert out == Path("/tmp/plots")
    assert matplotlib.get_backend().lower() == "agg"
    assert calls["use_datashader"] is True
    assert calls["parallel"] is True
    assert calls["output_dir"] == Path("/tmp/plots")


def test_generate_plots_returns_none_when_get_model_fails(monkeypatch):
    monkeypatch.setattr(svc_plots, "_generate_nlsq_plots", lambda **kw: None)
    cm = SimpleNamespace(
        get_model=lambda: (_ for _ in ()).throw(RuntimeError("no model")),
        get_config=lambda: {},
    )
    assert svc_plots.generate_plots(MagicMock(), {}, cm, Path("/tmp/p")) is None


def test_generate_plots_returns_none_when_render_fails(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("render failed")

    monkeypatch.setattr(svc_plots, "_generate_nlsq_plots", _boom)
    assert svc_plots.generate_plots(MagicMock(), {}, _cm(), Path("/tmp/p")) is None
