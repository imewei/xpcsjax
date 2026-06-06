"""Parity tests: heterodyne ≥1M stratified-LS anti-degeneracy controller wiring."""

from __future__ import annotations

import inspect
import logging

import numpy as np

from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as _hsl
from xpcsjax.optimization.nlsq.anti_degeneracy_controller import AntiDegeneracyController


def _ad_config_dict() -> dict:
    # Raw nested YAML 'anti_degeneracy' block (same shape the loader passes).
    return {
        "enable": True,
        "per_angle_mode": "auto",
        "constant_scaling_threshold": 3,
        "hierarchical": {"enable": True, "max_outer_iterations": 5},
        "regularization": {"enable": True, "mode": "relative", "lambda": 1.0},
        "gradient_monitoring": {"enable": True, "ratio_threshold": 0.01},
    }


def test_from_config_initializes_for_two_component():
    """analysis_mode='two_component' + is_laminar_flow=False must initialize L2/L3/L4."""
    phi = np.deg2rad(np.array([0.0, 60.0, 120.0], dtype=np.float64))
    ctrl = AntiDegeneracyController.from_config(
        config_dict=_ad_config_dict(),
        n_phi=3,
        phi_angles=phi,
        n_physical=14,
        per_angle_scaling=True,
        is_laminar_flow=False,
        analysis_mode="two_component",
    )
    assert ctrl.is_enabled is True
    assert ctrl.use_hierarchical is True          # L2 component built
    assert ctrl.hierarchical is not None
    assert ctrl.regularizer is not None           # L3 component built
    assert ctrl.monitor is not None               # L4 component built
    assert ctrl.use_shear_weighting is False      # L5 gated off for two_component
    assert ctrl.shear_weighter is None


def test_from_config_static_homodyne_still_skips_init():
    """Static homodyne (is_laminar_flow=False, non-two_component) must NOT initialize."""
    phi = np.deg2rad(np.array([0.0, 90.0], dtype=np.float64))
    ctrl = AntiDegeneracyController.from_config(
        config_dict=_ad_config_dict(),
        n_phi=2,
        phi_angles=phi,
        n_physical=3,
        per_angle_scaling=True,
        is_laminar_flow=False,
        analysis_mode="static_anisotropic",
    )
    assert ctrl.hierarchical is None
    assert ctrl.regularizer is None
    assert ctrl.monitor is None


def test_from_config_initializes_for_heterodyne_synonym():
    """analysis_mode='heterodyne' (synonym for two_component) must also initialize."""
    phi = np.deg2rad(np.array([0.0, 60.0, 120.0], dtype=np.float64))
    ctrl = AntiDegeneracyController.from_config(
        config_dict=_ad_config_dict(),
        n_phi=3,
        phi_angles=phi,
        n_physical=14,
        per_angle_scaling=True,
        is_laminar_flow=False,
        analysis_mode="heterodyne",
    )
    assert ctrl.hierarchical is not None
    assert ctrl.use_shear_weighting is False


def test_from_config_laminar_still_initializes():
    """laminar_flow path is unchanged: still initializes."""
    phi = np.deg2rad(np.array([0.0, 60.0, 120.0], dtype=np.float64))
    ctrl = AntiDegeneracyController.from_config(
        config_dict=_ad_config_dict(),
        n_phi=3,
        phi_angles=phi,
        n_physical=7,
        per_angle_scaling=True,
        is_laminar_flow=True,
        analysis_mode="laminar_flow",
    )
    assert ctrl.is_enabled is True
    assert ctrl.hierarchical is not None


def test_driver_accepts_anti_degeneracy_dict_param():
    """The driver must accept an optional anti_degeneracy_dict keyword (default None)."""
    sig = inspect.signature(_hsl.fit_heterodyne_stratified_least_squares)
    assert "anti_degeneracy_dict" in sig.parameters
    assert sig.parameters["anti_degeneracy_dict"].default is None


def test_emit_parity_banners_logs_layer_setup(caplog):
    """The driver's controller instantiation emits the laminar-style Layer 2/3/4 banners."""
    phi_deg = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.anti_degeneracy_controller"):
        ctrl = _hsl._emit_anti_degeneracy_parity_banners(
            anti_degeneracy_dict=_ad_config_dict(),
            phi_deg=phi_deg,
            n_physical=14,
        )
    text = caplog.text
    assert "Layer 2 - Hierarchical Optimization" in text
    assert "Layer 3 - Adaptive Regularization" in text
    assert "Layer 4 - Gradient Collapse Monitor" in text
    # L5 must NOT be announced for heterodyne (gated off).
    assert "Layer 5" not in text
    assert ctrl is not None and ctrl.use_shear_weighting is False


def test_emit_parity_banners_best_effort_on_none():
    """No anti_degeneracy dict -> returns None, emits nothing, never raises."""
    phi_deg = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    assert _hsl._emit_anti_degeneracy_parity_banners(
        anti_degeneracy_dict=None, phi_deg=phi_deg, n_physical=14
    ) is None


def test_l2_banner_suppressed_when_hierarchical_disabled(caplog):
    """hierarchical.enable=False must suppress the Layer 2 banner (L2 IS gated)."""
    cfg = _ad_config_dict()
    cfg["hierarchical"] = {"enable": False}
    phi_deg = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.anti_degeneracy_controller"):
        _hsl._emit_anti_degeneracy_parity_banners(
            anti_degeneracy_dict=cfg, phi_deg=phi_deg, n_physical=14
        )
    assert "Layer 2 - Hierarchical Optimization" not in caplog.text


def test_l4_banner_suppressed_when_gradient_monitoring_disabled(caplog):
    """gradient_monitoring.enable=False must suppress the Layer 4 banner (L4 IS gated)."""
    cfg = _ad_config_dict()
    cfg["gradient_monitoring"] = {"enable": False}
    phi_deg = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.anti_degeneracy_controller"):
        _hsl._emit_anti_degeneracy_parity_banners(
            anti_degeneracy_dict=cfg, phi_deg=phi_deg, n_physical=14
        )
    assert "Layer 4 - Gradient Collapse Monitor" not in caplog.text


def test_l3_banner_always_on_even_when_regularization_disabled(caplog):
    """L3 is NOT gated by a regularization.enable field — it stays on under master enable.

    Pins the controller's current (laminar-shared) behavior: there is no
    ``regularization.enable`` key, so the Layer 3 banner emits under master
    ``enable`` regardless. Matching laminar; documented, not changed.
    """
    cfg = _ad_config_dict()
    cfg["regularization"] = {"enable": False, "mode": "relative"}  # 'enable' ignored by design
    phi_deg = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.anti_degeneracy_controller"):
        _hsl._emit_anti_degeneracy_parity_banners(
            anti_degeneracy_dict=cfg, phi_deg=phi_deg, n_physical=14
        )
    assert "Layer 3 - Adaptive Regularization" in caplog.text
