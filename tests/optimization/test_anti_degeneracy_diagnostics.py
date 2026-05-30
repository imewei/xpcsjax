from xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics import (
    CORE_KEYS,
    assemble_anti_degeneracy_diagnostics,
)


def test_core_activation_keys_always_present():
    b = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=False,
        regularization_active=False,
        shear_weighting="not_applicable_heterodyne",
    )
    assert {"hierarchical_active", "regularization_active", "shear_weighting"} <= set(b)
    assert b["hierarchical_active"] is False
    assert b["regularization_active"] is False
    assert b["shear_weighting"] == "not_applicable_heterodyne"
    assert "gradient_monitor" not in b  # omitted -> absent (not None)


def test_gradient_monitor_included_when_provided():
    gm = {"mechanism": "per_iteration_gradient_ratio", "collapse_detected": False}
    b = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=True,
        regularization_active=True,
        shear_weighting="not_applicable_heterodyne",
        gradient_monitor=gm,
    )
    assert b["gradient_monitor"] is gm


def test_layer_detail_merged_verbatim():
    b = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=True,
        regularization_active=True,
        shear_weighting="not_applicable_heterodyne",
        hierarchical_stages=2,
        hierarchical_stage1_chi2=1.23,
        regularization_mode="relative",
    )
    assert b["hierarchical_stages"] == 2
    assert b["hierarchical_stage1_chi2"] == 1.23
    assert b["regularization_mode"] == "relative"


def test_bool_coercion_and_determinism():
    b1 = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=1, regularization_active=0, shear_weighting={"active": True}
    )
    assert b1["hierarchical_active"] is True
    assert b1["regularization_active"] is False
    b2 = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=1, regularization_active=0, shear_weighting={"active": True}
    )
    assert b1 == b2


def test_core_keys_constant_matches_contract():
    assert CORE_KEYS == (
        "hierarchical_active",
        "regularization_active",
        "shear_weighting",
        "gradient_monitor",
    )
