"""Phase 5 dual-gate (b): default auto->averaged is no-worse-SSR vs explicit individual.

Directionality (spec Risk 2): averaged is MORE constrained (2 scaling DOF vs 2*n_phi),
so on the same data SSR can only stay equal or DEGRADE. "No worse" = the degradation
stays within the parity threshold; it is the INTENDED default change, not a regression.
"""

from __future__ import annotations

import numpy as np

_NO_WORSE_REL = 1e-3  # ~1e-3 no-worse contract (CLAUDE.md two_component engine-unification)


def _fit(mode, n_phi, n_t=10, seed=11):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    ad = {"enable": True, "per_angle_mode": mode, "constant_scaling_threshold": 3}
    cfg = ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": n_t,
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
            },
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "values": true.tolist(),
            },
            "optimization": {
                "method": "nlsq",
                "nlsq": {
                    "analysis_mode": "laminar_flow",
                    "max_iterations": 80,
                    "loss": "linear",
                    "cmaes": {"enable": False, "auto_select": False},
                    "multi_start": {"enable": False},
                    "anti_degeneracy": ad,
                },
                "stratification": {"enabled": False},
            },
        }
    )
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0))
    c2 = c2 + np.random.default_rng(seed).normal(0.0, 5e-4, size=c2.shape)
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),
    }
    return fit_nlsq(data, cfg)


def test_synthetic_default_averaged_no_worse_than_individual():
    """CI-safe: averaged SSR must not exceed individual SSR by more than the threshold."""
    res_ind = _fit("individual", n_phi=4)
    res_avg = _fit("auto", n_phi=4)  # -> averaged
    ssr_ind = float(res_ind.chi_squared)
    ssr_avg = float(res_avg.chi_squared)
    # averaged is more-constrained: degrade-or-equal, within ~1e-3 relative band.
    assert ssr_avg <= ssr_ind * (1.0 + _NO_WORSE_REL) + 1e-12, (
        f"averaged SSR {ssr_avg} worse than individual {ssr_ind} beyond no-worse band"
    )
    # sanity: averaged actually ran averaged
    assert dict(res_avg.nlsq_diagnostics or {}).get("per_angle_mode") == "averaged"
