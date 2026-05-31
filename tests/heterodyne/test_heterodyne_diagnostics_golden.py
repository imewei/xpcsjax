"""Golden test for heterodyne anti-degeneracy diagnostics emission.

Both modes now ALWAYS emit the three activation keys
(``hierarchical_active`` / ``regularization_active`` / ``shear_weighting``) at the
top level of ``nlsq_diagnostics``, routed through the shared
``assemble_anti_degeneracy_diagnostics``. This is a deliberate diagnostics-only
change: disabled paths now surface ``hierarchical_active=False`` /
``regularization_active=False`` rather than omitting the keys. It is fit-safe —
fit baselines do not compare the ``nlsq_diagnostics`` dict.

Field-name note: the plan draft used ``per_angle_mode="averaged"`` and
``regularization_mode="relative"``; neither is a settable value on the real
``NLSQConfig`` (``"averaged"`` is an *effective* mode resolved from ``"auto"``
when ``n_phi >= constant_scaling_threshold``; valid regularization modes are
``none`` / ``tikhonov`` / ``adaptive``). The test drives the averaged path via
``per_angle_mode="auto"`` (n_phi=3 >= threshold 3) and uses
``regularization_mode="adaptive"``.
"""
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

_ACTIVATION = ("hierarchical_active", "regularization_active", "shear_weighting")


def _diag(per_angle_mode="auto", **flags):
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": per_angle_mode, **flags}
    )
    return fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).nlsq_diagnostics


def test_enabled_path_emits_activation_keys_true():
    d = _diag(
        enable_hierarchical=True,
        regularization_mode="adaptive",
        enable_gradient_monitoring=True,
    )
    for k in (*_ACTIVATION, "gradient_monitor"):
        assert k in d, f"missing core key {k!r}"
    assert d["shear_weighting"] == "not_applicable_heterodyne"
    assert d["hierarchical_active"] is True
    assert d["regularization_active"] is True


def test_disabled_path_now_emits_activation_keys_false():
    d = _diag(
        enable_hierarchical=False,
        regularization_mode="none",
        enable_gradient_monitoring=False,
    )
    for k in _ACTIVATION:
        assert k in d, f"disabled path must still emit {k!r}"
    assert d["hierarchical_active"] is False
    assert d["regularization_active"] is False
    assert d["shear_weighting"] == "not_applicable_heterodyne"
    # gradient_monitor is still conditional (omitted, never None, when L4 is off).
    assert "gradient_monitor" not in d


def test_detail_keys_preserved_when_enabled():
    d = _diag(enable_hierarchical=True, regularization_mode="adaptive")
    # L2 detail fragment (full two-stage in averaged mode).
    assert d["hierarchical_active"] is True
    assert d["hierarchical_stages"] == 2
    assert d["hierarchical_scope"] == "full_two_stage"
    assert "hierarchical_stage1_chi2" in d
    assert "hierarchical_stage2_chi2" in d
    # L3 detail fragment.
    assert d["regularization_active"] is True
    assert d.get("regularization_mode") == "adaptive"
    assert "regularization_lambda_applied" in d
    assert "regularization_penalty_count" in d
    assert "regularization_data_residual_ssr" in d
    assert "regularization_total_ssr_with_penalty" in d
    assert d["regularization_scope"] == "full_residual_augmentation"


def test_disabled_path_omits_layer_detail_keys():
    """Only the 3 activation flags are unconditional; per-layer DETAIL keys
    (hierarchical_stages, regularization_mode, ...) are NOT fabricated when the
    layer did not run."""
    d = _diag(
        enable_hierarchical=False,
        regularization_mode="none",
        enable_gradient_monitoring=False,
    )
    assert "hierarchical_stages" not in d
    assert "hierarchical_scope" not in d
    assert "regularization_mode" not in d
    assert "regularization_scope" not in d


def test_fourier_path_always_emits_activation_keys():
    d = _diag(per_angle_mode="fourier")
    for k in _ACTIVATION:
        assert k in d, f"fourier disabled path must still emit {k!r}"
    assert d["hierarchical_active"] is False
    assert d["regularization_active"] is False


def test_constant_path_always_emits_activation_keys():
    d = _diag(per_angle_mode="constant")
    for k in _ACTIVATION:
        assert k in d, f"constant disabled path must still emit {k!r}"
    assert d["hierarchical_active"] is False
    assert d["regularization_active"] is False


def test_individual_path_always_emits_activation_keys():
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    d = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).nlsq_diagnostics
    for k in _ACTIVATION:
        assert k in d, f"individual disabled path must still emit {k!r}"
    assert d["hierarchical_active"] is False
    assert d["regularization_active"] is False
