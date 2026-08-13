"""Headless tests for run_worker — fake service modules keep JAX out."""

import queue as queue_mod
import sys
import types

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.service.events import Failed, Finished, Iteration, Started


def _install_fake_services(monkeypatch, *, fit_raises=False):
    """Inject JAX-free fake service modules into sys.modules."""
    cfg = types.ModuleType("xpcsjax.service.config")
    cfg.load_config = lambda path, **_kw: types.SimpleNamespace(
        config={"analysis_mode": "laminar_flow"}
    )

    data = types.ModuleType("xpcsjax.service.data")
    data.load_dataset = lambda cm, **_kw: {"c2_exp": None}

    fit = types.ModuleType("xpcsjax.service.fit")
    fit.FitOverrides = lambda **kw: dict(kw)

    def _run_fit(cm, d, *, overrides=None, run_id="", on_event=None):
        if on_event is not None:
            on_event(Started(run_id="", seq=0, mode="laminar_flow", settings_summary="x"))
            on_event(Iteration(run_id="", seq=0, n=1, ssr=10.0, chi2=10.0))
        if fit_raises:
            raise RuntimeError("fit blew up")
        return object()

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


def test_run_worker_happy_path_emits_started_then_finished(monkeypatch, tmp_path):
    _install_fake_services(monkeypatch)
    from xpcsjax.gui.ipc.worker import run_worker

    q = queue_mod.Queue()
    run_worker(FitJob(run_id="r1", config_path="c.yaml", output_dir=str(tmp_path)), q)

    events = _drain(q)
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "Started"
    assert kinds[-1] == "Finished"
    assert isinstance(events[-1], Finished)
    assert any(isinstance(e, Iteration) for e in events), "expected at least one Iteration event"
    assert [e.seq for e in events] == sorted(e.seq for e in events)  # monotonic
    assert all(e.run_id == "r1" for e in events)


def test_run_worker_emits_failed_on_exception(monkeypatch, tmp_path):
    _install_fake_services(monkeypatch, fit_raises=True)
    from xpcsjax.gui.ipc.worker import run_worker

    q = queue_mod.Queue()
    run_worker(FitJob(run_id="r1", config_path="c.yaml", output_dir=str(tmp_path)), q)

    events = _drain(q)
    assert isinstance(events[-1], Failed)
    assert "fit blew up" in events[-1].traceback
