"""Tests for the JAX-free layer-status map + banner classifier."""

from xpcsjax.gui.ipc.diagnostics import classify_banner, layer_status_from_diagnostics
from xpcsjax.service.events import Banner, BannerKind


def test_layer_status_maps_diagnostics_keys():
    diag = {
        "hierarchical_active": True,
        "regularization_active": False,
        "gradient_monitor": {"collapse_detected": False},
        "shear_weighting": "laminar_flow",
    }
    status = layer_status_from_diagnostics(diag)
    assert status == {"L1": True, "L2": True, "L3": False, "L4": True, "L5": True}


def test_layer_status_l5_inactive_sentinels_are_false():
    for sentinel in ("laminar_flow_inactive", "not_applicable_heterodyne", None, ""):
        status = layer_status_from_diagnostics({"shear_weighting": sentinel})
        assert status["L5"] is False
    # L1 is always active; missing keys default to inactive.
    assert layer_status_from_diagnostics({})["L1"] is True
    assert layer_status_from_diagnostics(None) == {
        "L1": True,
        "L2": False,
        "L3": False,
        "L4": False,
        "L5": False,
    }


def test_classify_banner_recognizes_engine_prefixes():
    b = classify_banner("INFO", "ANTI-DEGENERACY: Layer 2 - Hierarchical Optimization")
    assert isinstance(b, Banner) and b.kind is BannerKind.INFO and "Layer 2" in b.text

    # Streaming + heterodyne paths emit the "DEFENSE" family — must also classify (verbatim
    # emissions from hybrid_streaming.py and heterodyne_logging.py)
    assert (
        classify_banner("INFO", "ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization").kind
        is BannerKind.INFO
    )
    assert (
        classify_banner(
            "INFO", "ANTI-DEGENERACY DEFENSE [AS EXECUTED] (heterodyne two_component)"
        ).kind
        is BannerKind.INFO
    )

    assert (
        classify_banner("INFO", "[CMA-ES] Global search phase starting...").kind
        is BannerKind.CMAES_ESCAPE
    )
    assert (
        classify_banner("WARNING", "GRADIENT COLLAPSE DETECTED at iteration 7!").kind
        is BannerKind.GRADIENT_COLLAPSE
    )


def test_classify_banner_ignores_ordinary_log_lines():
    assert classify_banner("INFO", "Loading XPCS data ...") is None
    assert classify_banner("INFO", "NLSQ analysis complete") is None
