"""End-to-end guard for the unsorted-phi family (Codex review round 1).

Per-angle quantities (scaling, sigma, theory) are stored by SORTED phi while data
rows arrive in loader order. A real fit must therefore give the same answer whether
the caller presents the angles ascending or reversed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
from xpcsjax.config import ConfigManager
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

pytestmark = [pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian:UserWarning")]


@pytest.mark.parametrize("per_angle_mode", ["individual", "auto"])
def test_homodyne_fit_is_invariant_to_phi_row_order(per_angle_mode: str) -> None:
    data = _synthetic_data("laminar_flow")
    # Make the angles genuinely distinguishable: per-angle sigma always, and a
    # per-angle baseline where the model can represent one (individual scaling).
    if per_angle_mode == "individual":
        data["g2"] = data["g2"] + np.array([0.0, 0.05, 0.10])[:, None, None]
    data["sigma"] = data["sigma"] * np.array([1.0, 2.0, 3.0])[:, None, None]
    reversed_data = {**data, **{k: data[k][::-1].copy() for k in ("phi", "g2", "sigma")}}
    extra = {"optimization": {"nlsq": {"anti_degeneracy": {"per_angle_mode": per_angle_mode}}}}

    def fit(d: dict) -> object:
        cfg = ConfigManager(config_override=_config("laminar_flow", extra_top=extra))
        return fit_nlsq_jax(d, cfg, use_adapter=False)

    ascending, reversed_ = fit(data), fit(reversed_data)
    # Homodyne reports per-angle parameters in sorted-phi order for either input.
    np.testing.assert_allclose(reversed_.chi_squared, ascending.chi_squared, rtol=1e-6)
    np.testing.assert_allclose(reversed_.parameters, ascending.parameters, rtol=1e-4, atol=1e-8)


def test_heterodyne_individual_fit_follows_input_angle_order() -> None:
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    c2 = c2 + np.array([0.0, 0.03, 0.06])[:, None, None]  # distinct per-angle offsets
    cfg = NLSQConfig(per_angle_mode="individual")
    ascending = fit_two_component_via_engine(model, c2, phi, cfg)
    model_r, _, _ = make_synthetic_two_component(n_phi=3, n_t=12)
    reversed_ = fit_two_component_via_engine(model_r, c2[::-1].copy(), phi[::-1].copy(), cfg)

    n = len(phi)
    np.testing.assert_allclose(reversed_.chi_squared, ascending.chi_squared, rtol=1e-9)
    # Heterodyne reports the scaling head in the CALLER's angle order.
    np.testing.assert_allclose(reversed_.parameters[:n], ascending.parameters[:n][::-1])
    np.testing.assert_allclose(
        reversed_.parameters[n : 2 * n], ascending.parameters[n : 2 * n][::-1]
    )
    np.testing.assert_allclose(reversed_.parameters[2 * n :], ascending.parameters[2 * n :])
    np.testing.assert_allclose(reversed_.nlsq_diagnostics["phi_angles"], phi[::-1])
