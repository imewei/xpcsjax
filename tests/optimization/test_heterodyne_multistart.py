"""Tests for heterodyne joint multistart wiring (Phase 1)."""
from __future__ import annotations

import numpy as np

import xpcsjax.optimization.nlsq.heterodyne_multistart as hm
from xpcsjax.optimization.nlsq.heterodyne_multistart import build_multistart_config
from xpcsjax.optimization.nlsq.multistart import (
    MultiStartResult,
    SingleStartResult,
)


class _StubParamManager:
    varying_names = ["D0_sample", "alpha_sample"]

    def __init__(self) -> None:
        self._initial = np.array([1000.0, 1.0])
        self.updated_with: dict[str, float] | None = None

    def get_bounds(self):
        return (np.array([100.0, -2.0]), np.array([50000.0, 2.0]))

    def get_initial_values(self):
        return self._initial.copy()

    def update_values(self, params):
        self.updated_with = dict(params)


class _StubModel:
    def __init__(self) -> None:
        self.param_manager = _StubParamManager()


class _StubResult:
    """Minimal OptimizationResult stand-in."""

    def __init__(self, params, chi2, success=True):
        self.parameters = np.asarray(params)
        self.chi_squared = chi2
        self.reduced_chi_squared = chi2
        self.success = success
        self.message = "ok"
        self.nlsq_diagnostics: dict = {}


def test_fit_nlsq_multistart_heterodyne_runs_and_annotates(monkeypatch):
    model = _StubModel()
    c2 = np.ones((2, 4, 4))
    phi = np.array([0.0, 90.0])

    captured: dict = {}

    def _fake_run_multistart(data, bounds, config, single_fit_func, cost_func=None, custom_starts=None):
        captured["bounds"] = bounds
        captured["config"] = config
        captured["custom_starts"] = custom_starts
        captured["data_keys"] = set(data)
        sr = single_fit_func(data, np.array([2000.0, 0.5]))
        captured["worker_result"] = sr
        best = SingleStartResult(
            start_idx=3,
            initial_params=np.array([2000.0, 0.5]),
            final_params=np.array([2000.0, 0.5, 0.18, 1.19]),
            chi_squared=1.5,
            reduced_chi_squared=1.5,
            success=True,
            message="best",
        )
        return MultiStartResult(
            best=best,
            all_results=[best],
            config=config,
            strategy_used="full",
            n_unique_basins=1,
            degeneracy_detected=False,
        )

    def _fake_fit_nlsq_multi_phi(m, c2_in, phi_in, cfg, w):
        return _StubResult([2000.0, 0.5, 0.18, 1.19], chi2=1.5)

    monkeypatch.setattr(hm, "run_multistart_nlsq", _fake_run_multistart)
    monkeypatch.setattr(hm, "fit_nlsq_multi_phi", _fake_fit_nlsq_multi_phi)

    ms_cfg = hm.build_multistart_config({"enable": True, "n_starts": 5})
    out = hm.fit_nlsq_multistart_heterodyne(model, c2, phi, nlsq_cfg=object(), weights=None, ms_cfg=ms_cfg)

    assert captured["bounds"].shape == (2, 2)
    assert captured["custom_starts"] == [[1000.0, 1.0]]
    assert "c2_exp" in captured["data_keys"]
    assert isinstance(captured["worker_result"], SingleStartResult)
    assert captured["worker_result"].chi_squared == 1.5
    # authoritative final re-fit happened from the winning start [2000.0, 0.5]
    assert model.param_manager.updated_with == {"D0_sample": 2000.0, "alpha_sample": 0.5}
    assert out.nlsq_diagnostics["multistart"]["n_starts"] == 5
    assert out.nlsq_diagnostics["multistart"]["best_start_idx"] == 3
    assert out.nlsq_diagnostics["multistart"]["n_unique_basins"] == 1


def test_build_multistart_config_reads_nested_keys() -> None:
    ms_dict = {
        "enable": True,
        "n_starts": 7,
        "seed": 99,
        "sampling_strategy": "latin_hypercube",
        "n_workers": 4,
        "use_screening": False,
        "screen_keep_fraction": 0.3,
        "refine_top_k": 2,
        "refinement_ftol": 1e-10,
        "degeneracy_threshold": 0.25,
    }
    cfg = build_multistart_config(ms_dict)
    assert cfg.enable is True
    assert cfg.n_starts == 7
    assert cfg.seed == 99
    assert cfg.use_screening is False
    assert cfg.screen_keep_fraction == 0.3
    assert cfg.refine_top_k == 2
    assert cfg.degeneracy_threshold == 0.25
    # Heterodyne worker closes over a JAX model -> not process-picklable.
    # n_workers MUST be clamped to 1 (sequential) regardless of config.
    assert cfg.n_workers == 1


def test_build_multistart_config_defaults_on_empty() -> None:
    cfg = build_multistart_config({})
    assert cfg.enable is False
    assert cfg.n_starts == 10
    assert cfg.seed == 42
    assert cfg.n_workers == 1
