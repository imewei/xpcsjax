"""Regression test for the heterodyne config unwrap in ``_fit_nlsq_heterodyne``.

YAMLs nest solver settings under ``optimization.nlsq``. ``NLSQConfig.from_dict``
expects those keys at the top level, so the dispatch must unwrap before
constructing the config — otherwise every nested setting (``max_iterations``,
``enable_cmaes``, recovery thresholds, …) is silently ignored and the solver
runs with defaults while users believe their tuning is active.
"""

from __future__ import annotations

import types

import numpy as np
import pytest


class _StubConfigManager:
    """Minimal ConfigManager replacement holding only ``self.config``."""

    def __init__(self, cfg: dict) -> None:
        self.config = cfg


@pytest.fixture()
def captured_nlsq(monkeypatch):
    """Patch HeterodyneModel + fit_nlsq_multi_phi to capture the NLSQConfig."""
    captured: dict = {}

    from xpcsjax.optimization import nlsq as nlsq_pkg

    class _StubModel:
        t = np.zeros(4, dtype=np.float64)

        @classmethod
        def from_config(cls, _cfg):
            return cls()

        def sync_time_axis(self, _t):
            pass

    def _stub_fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights):
        captured["nlsq_cfg"] = nlsq_cfg
        return []

    fake_model_mod = types.ModuleType("xpcsjax.core.heterodyne_model_stateful")
    fake_model_mod.HeterodyneModel = _StubModel  # type: ignore[attr-defined]
    fake_core_mod = types.ModuleType("xpcsjax.optimization.nlsq.heterodyne_core")
    fake_core_mod.fit_nlsq_multi_phi = _stub_fit_nlsq_multi_phi  # type: ignore[attr-defined]

    monkeypatch.setitem(
        __import__("sys").modules,
        "xpcsjax.core.heterodyne_model_stateful",
        fake_model_mod,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "xpcsjax.optimization.nlsq.heterodyne_core",
        fake_core_mod,
    )

    return captured, nlsq_pkg


def _fake_data() -> dict:
    return {
        "c2_exp": np.ones((1, 4, 4), dtype=np.float64),
        "phi_angles_list": np.array([0.0], dtype=np.float64),
    }


def test_nested_optimization_nlsq_is_applied(captured_nlsq):
    """Nested ``optimization.nlsq.*`` settings must reach NLSQConfig."""
    captured, nlsq_pkg = captured_nlsq

    config = _StubConfigManager(
        {
            "analysis_mode": "two_component",
            "optimization": {
                "nlsq": {
                    "max_iterations": 7,
                    "enable_cmaes": True,
                    "tolerance": 1e-9,
                },
            },
        }
    )

    nlsq_pkg.fit_nlsq(_fake_data(), config)

    cfg = captured["nlsq_cfg"]
    assert cfg.max_iterations == 7
    assert cfg.enable_cmaes is True
    assert cfg.tolerance == pytest.approx(1e-9)
    assert cfg.analysis_mode == "two_component"


def test_flat_legacy_config_still_works(captured_nlsq):
    """Already-flat dicts (legacy/tests) must still parse correctly."""
    captured, nlsq_pkg = captured_nlsq

    config = _StubConfigManager(
        {
            "analysis_mode": "two_component",
            "max_iterations": 11,
            "enable_cmaes": True,
        }
    )

    nlsq_pkg.fit_nlsq(_fake_data(), config)

    cfg = captured["nlsq_cfg"]
    assert cfg.max_iterations == 11
    assert cfg.enable_cmaes is True
    assert cfg.analysis_mode == "two_component"


def test_analysis_mode_only_in_nested_section(captured_nlsq):
    """``analysis_mode`` placed only in the nested NLSQ section must reach NLSQConfig.

    Mirrors ``fit_nlsq``'s routing fallback, which also accepts ``analysis_mode``
    inside ``optimization.nlsq``.
    """
    captured, nlsq_pkg = captured_nlsq

    config = _StubConfigManager(
        {
            "optimization": {
                "nlsq": {
                    "analysis_mode": "two_component",
                    "max_iterations": 5,
                },
            },
        }
    )

    nlsq_pkg.fit_nlsq(_fake_data(), config)

    cfg = captured["nlsq_cfg"]
    assert cfg.analysis_mode == "two_component"
    assert cfg.max_iterations == 5
