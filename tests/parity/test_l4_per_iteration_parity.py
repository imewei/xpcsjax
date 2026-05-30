import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi


def test_heterodyne_l4_is_per_iteration_block():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                                "enable_gradient_monitoring": True})
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    gm = res.nlsq_diagnostics["gradient_monitor"]
    assert gm["mechanism"] in ("per_iteration_gradient_ratio", "post_solve_fallback")
    assert "collapse_detected" in gm and "max_gradient_ratio" in gm
    if gm["mechanism"] == "per_iteration_gradient_ratio":
        assert gm["n_observations"] >= 2


def test_heterodyne_l4_is_diagnostic_only():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    on = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                               "enable_gradient_monitoring": True})
    off = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                                "enable_gradient_monitoring": False})
    r_on = fit_nlsq_multi_phi(model, c2, phi, on, weights=None)
    r_off = fit_nlsq_multi_phi(model, c2, phi, off, weights=None)
    assert np.array_equal(np.asarray(r_on.parameters), np.asarray(r_off.parameters))
    assert r_on.chi_squared == r_off.chi_squared
