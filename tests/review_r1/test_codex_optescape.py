"""Red tests for Codex review shard ``optescape`` (round 1)."""

from __future__ import annotations

import numpy as np

# optescape-B1 reclassified DESIGN by the lead (fix adds L2 sweeps -> numerics change); test removed.


def test_optescape_b2():
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

    x = np.arange(1.0, 5.0)
    wrapper = CMAESWrapper(
        CMAESWrapperConfig(refinement_workflow="standard", refinement_max_nfev=1)
    )
    result = wrapper._run_nlsq_refinement(
        lambda x, a: a * x, x, 2 * x, np.array([1.0]), (np.array([0.0]), np.array([4.0]))
    )
    assert not result["success"], result


# optescape-B3 reclassified DESIGN in round 2 (Codex R2-A5): the `streaming` refinement workflow
# is non-functional as a whole (the backend rejects the TRF kwargs AND any sigma, which every
# homodyne CMA-ES call passes). It fails loudly and keeps the CMA-ES point; the maintainer should
# remove the option or implement it. The round-1 half-fix was reverted. Test removed.


def test_optescape_b4(monkeypatch):
    import nlsq.global_optimization as backend

    from xpcsjax.optimization.nlsq import cmaes_wrapper as cw

    seen = {}

    class Capture:
        def __init__(self, config):
            seen["max_generations"] = config.max_generations

        def fit(self, **kwargs):
            return {"popt": kwargs["p0"]}

    monkeypatch.setattr(cw, "CMAES_AVAILABLE", True)
    monkeypatch.setattr(backend, "CMAESOptimizer", Capture)
    wrapper = cw.CMAESWrapper(
        cw.CMAESWrapperConfig(
            max_generations=3, auto_memory=False, refine_with_nlsq=False, restart_strategy="none"
        )
    )
    x = np.arange(3.0)
    upper = np.array([1.0, 1e-5])
    try:
        wrapper.fit(lambda x, a, b: a * x + b, x, x, upper / 2, (np.zeros(2), upper))
    except Exception:
        pass
    assert seen["max_generations"] == 3, seen


def test_optescape_b6():
    from tests.optimization.test_fixed_parameters_integration import (
        _ALL_PHYSICAL_NAMES,
        _config,
        _physical_index,
        _synthetic_data,
    )
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.optimization.nlsq.core import fit_nlsq_multistart

    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}},
    )
    result = fit_nlsq_multistart(data, ConfigManager(config_override=config))
    for start in result.all_results:
        init = np.asarray(start.initial_params)
        # core._SingleFitWorker's contract: [2 scaling | 7 laminar physical], fixed slot restored.
        assert len(init) == 2 + len(_ALL_PHYSICAL_NAMES), len(init)
        assert init[_physical_index(len(init), _ALL_PHYSICAL_NAMES, "D_offset")] == 37.5


def test_optescape_b8():
    from xpcsjax.optimization.nlsq.config import NLSQConfig

    cfg = NLSQConfig.from_dict({"anti_degeneracy": {"per_angle_mode": "averaged"}})
    assert cfg.is_valid(), cfg.validate()
