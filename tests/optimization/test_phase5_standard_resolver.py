"""Phase 5 — laminar standard in-memory honors the per-angle resolver."""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.per_angle_mode import resolve_per_angle_mode


def _laminar_cfg(per_angle_mode: str | None, n_t: int = 8):
    """laminar_flow config; per_angle_mode set under optimization.nlsq.anti_degeneracy."""
    from xpcsjax.config import ConfigManager

    ad: dict = {"enable": False}
    if per_angle_mode is not None:
        ad = {"enable": True, "per_angle_mode": per_angle_mode, "constant_scaling_threshold": 3}
    cfg = {
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1, "start_frame": 1, "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0",
                                "beta", "gamma_dot_t_offset", "phi0"],
            "values": [1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0],
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "laminar_flow", "max_iterations": 30, "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": ad,
            },
            "stratification": {"enabled": False},
        },
    }
    return ConfigManager(config_override=cfg)


def _fit(per_angle_mode, n_phi):
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_t = 8
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    cfg = _laminar_cfg(per_angle_mode, n_t)
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0), dtype=np.float64)
    c2 = c2 + np.random.default_rng(7).normal(0.0, 5e-4, size=c2.shape)
    data = {"phi_angles_list": phi, "c2_exp": c2, "t1": t, "t2": t,
            "wavevector_q_list": np.array([0.0237], dtype=np.float64)}
    return fit_nlsq(data, cfg)


def test_resolver_truthtable_used_by_standard_path():
    # auto @ n_phi>=3 -> averaged ; auto @ n_phi<3 -> individual
    assert resolve_per_angle_mode("auto", 4, 3) == "averaged"
    assert resolve_per_angle_mode("auto", 2, 3) == "individual"


def test_standard_path_stamps_resolved_mode_individual():
    res = _fit("individual", n_phi=4)
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "individual"


def test_standard_path_stamps_resolved_mode_averaged():
    res = _fit("auto", n_phi=4)  # auto -> averaged at n_phi=4
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "averaged"
