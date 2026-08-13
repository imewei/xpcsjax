"""run_worker emits LayerStatus (post-fit) + Banner (from engine log lines)."""

import logging
import queue as queue_mod
import sys
import types

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.service.events import Banner, LayerStatus


def _install_fake_services(monkeypatch):
    cfg = types.ModuleType("xpcsjax.service.config")
    cfg.load_config = lambda path, **_kw: types.SimpleNamespace(
        config={"analysis_mode": "laminar_flow"}
    )
    data = types.ModuleType("xpcsjax.service.data")
    data.load_dataset = lambda cm, **_kw: {"c2_exp": None}
    fit = types.ModuleType("xpcsjax.service.fit")
    fit.FitOverrides = lambda **kw: dict(kw)

    def _run_fit(cm, d, *, overrides=None, run_id="", on_event=None):
        # Emit a banner the way the engine logs it, then return a result with diagnostics.
        logging.getLogger("xpcsjax.fake").info("ANTI-DEGENERACY: Layer 3 - Adaptive Regularization")
        return types.SimpleNamespace(
            nlsq_diagnostics={
                "hierarchical_active": False,
                "regularization_active": True,
                "gradient_monitor": {"collapse_detected": False},
                "shear_weighting": "laminar_flow",
            }
        )

    fit.run_fit = _run_fit
    persist = types.ModuleType("xpcsjax.service.persist")
    persist.save_results = lambda *a, **k: None
    persist.merge_fitted_c2 = lambda *a, **k: False
    plots = types.ModuleType("xpcsjax.service.plots")
    plots.generate_plots = lambda *a, **k: None
    for name, mod in [
        ("xpcsjax.service.config", cfg),
        ("xpcsjax.service.data", data),
        ("xpcsjax.service.fit", fit),
        ("xpcsjax.service.persist", persist),
        ("xpcsjax.service.plots", plots),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue_mod.Empty:
            return out


def test_worker_emits_layerstatus_and_banner(monkeypatch, tmp_path):
    _install_fake_services(monkeypatch)
    from xpcsjax.gui.ipc.worker import run_worker

    q = queue_mod.Queue()
    run_worker(FitJob(run_id="r1", config_path="c.yaml", output_dir=str(tmp_path)), q)
    events = _drain(q)

    layer_events = [e for e in events if isinstance(e, LayerStatus)]
    assert len(layer_events) == 1
    assert layer_events[0].layers == {"L1": True, "L2": False, "L3": True, "L4": True, "L5": True}

    banners = [e for e in events if isinstance(e, Banner)]
    assert any("Layer 3" in b.text for b in banners)
