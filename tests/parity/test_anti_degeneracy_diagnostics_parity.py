"""Cross-mode parity for the symmetric top-level anti-degeneracy activation keys.

Both ``two_component`` (heterodyne) and ``laminar_flow`` (homodyne) must surface
the SAME activation key set at the top level of ``nlsq_diagnostics``:
``hierarchical_active`` / ``regularization_active`` / ``shear_weighting``. The
laminar in-memory path runs no L2/L3 and exposes no shear-weighter diagnostics,
so it reports both inactive and a laminar-appropriate L5 marker (NOT the
heterodyne sentinel ``"not_applicable_heterodyne"``).

The laminar fixture used here is ``_build_laminar_fit`` from
``tests/optimization/test_l4_callback_observational.py`` (returns
``(fit_nlsq, data, cfg)``); the planned ``make_synthetic_laminar_flow`` helper
does not exist in this tree, so the real in-memory STANDARD curve_fit fixture is
reused and gradient-monitoring is enabled on it explicitly.
"""
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from tests.optimization.test_l4_callback_observational import _build_laminar_fit
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

_ACTIVATION = {"hierarchical_active", "regularization_active", "shear_weighting"}


def _laminar_diagnostics():
    fit_nlsq, data, cfg = _build_laminar_fit()
    cfg.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {
        "enable": True
    }
    return fit_nlsq(data, cfg).nlsq_diagnostics


def test_both_modes_emit_symmetric_activation_keys():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    het = fit_nlsq_multi_phi(
        model,
        c2,
        phi,
        NLSQConfig.from_dict(
            {
                "analysis_mode": "two_component",
                "per_angle_mode": "auto",
                "enable_gradient_monitoring": True,
            }
        ),
        weights=None,
    ).nlsq_diagnostics

    lam = _laminar_diagnostics()

    assert _ACTIVATION <= set(het)
    assert _ACTIVATION <= set(lam)
    assert (set(het) & _ACTIVATION) == (set(lam) & _ACTIVATION) == _ACTIVATION


def test_laminar_shear_weighting_is_not_heterodyne_sentinel():
    lam = _laminar_diagnostics()
    assert lam["shear_weighting"] != "not_applicable_heterodyne"
    assert lam["hierarchical_active"] is False
    assert lam["regularization_active"] is False
