"""Tests for the argparse-free run_fit core (xpcsjax/service/fit.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import xpcsjax.service.fit as svc_fit
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.service.events import Started


def _cm(mode: str = "laminar_flow") -> SimpleNamespace:
    return SimpleNamespace(config={"analysis_mode": mode})


def test_run_fit_returns_result_and_emits_started(monkeypatch):
    # MagicMock(spec=OptimizationResult) passes isinstance(result, OptimizationResult).
    fake = MagicMock(spec=OptimizationResult)
    monkeypatch.setattr(svc_fit, "fit_nlsq", lambda data, cm, *, on_iteration=None, **_kw: fake)

    events = []
    out = svc_fit.run_fit(_cm(), {"c2_exp": None}, run_id="r1", on_event=events.append)

    assert out is fake
    started = [e for e in events if isinstance(e, Started)]
    assert len(started) == 1
    assert started[0].run_id == "r1"
    assert started[0].mode == "laminar_flow"


def test_run_fit_unwraps_best_when_not_optimization_result(monkeypatch):
    best = MagicMock(spec=OptimizationResult)
    wrapper = SimpleNamespace(best=best)  # e.g. a MultiStartResult — not an OptimizationResult
    monkeypatch.setattr(svc_fit, "fit_nlsq", lambda data, cm: wrapper)

    out = svc_fit.run_fit(_cm("static_isotropic"), {}, on_event=None)
    assert out is best


def test_run_fit_raises_on_unexpected_return_type(monkeypatch):
    monkeypatch.setattr(svc_fit, "fit_nlsq", lambda data, cm: SimpleNamespace(best=None))
    with pytest.raises(TypeError, match="unexpected type"):
        svc_fit.run_fit(_cm(), {})


def test_run_fit_applies_overrides(monkeypatch):
    monkeypatch.setattr(svc_fit, "fit_nlsq", lambda data, cm: MagicMock(spec=OptimizationResult))
    cm = _cm()
    svc_fit.run_fit(cm, {}, overrides=svc_fit.FitOverrides(multistart=True))
    assert cm.config["optimization"]["nlsq"]["multi_start"]["enable"] is True


def test_run_fit_threads_on_iteration_into_iteration_events(monkeypatch):
    from unittest.mock import MagicMock

    import xpcsjax.service.fit as svc_fit
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.events import Iteration

    def _fake_fit(data, cm, *, on_iteration=None, **_kw):
        if on_iteration is not None:
            on_iteration(1, 100.0)
            on_iteration(2, 50.0)
        return MagicMock(spec=OptimizationResult)

    monkeypatch.setattr(svc_fit, "fit_nlsq", _fake_fit)
    events = []
    svc_fit.run_fit(_cm(), {}, run_id="r", on_event=events.append)
    iters = [e for e in events if isinstance(e, Iteration)]
    assert [(e.n, e.ssr) for e in iters] == [(1, 100.0), (2, 50.0)]
