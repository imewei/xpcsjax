"""Unit tests for the shared per-angle-mode banner formatter.

Covers ``log_effective_per_angle_mode`` (direct) and
``log_effective_mode_from_controller`` (controller convenience wrapper).
"""

from __future__ import annotations

import logging

import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_logging import (
    log_effective_mode_from_controller,
    log_effective_per_angle_mode,
)

_LOGGER = "xpcsjax.test.banner"


def test_averaged_banner_text(caplog):
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_per_angle_mode(
            log, mode="averaged", n_phi=3, n_physics=7, n_scaling=2, threshold=3
        )
    text = caplog.text
    assert "ANTI-DEGENERACY: Effective per-angle mode 'averaged'" in text
    assert "Reason: n_phi (3) >= constant_scaling_threshold (3)" in text
    assert "Parameters: 7 physical + 2 averaged scaling = 9 total" in text


def test_individual_banner_uses_dynamic_less_than(caplog):
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_per_angle_mode(
            log, mode="individual", n_phi=2, n_physics=7, n_scaling=4, threshold=3
        )
    text = caplog.text
    assert "Reason: n_phi (2) < constant_scaling_threshold (3)" in text
    assert "Parameters: 7 physical + 4 per-angle scaling = 11 total" in text


def test_fourier_banner_text(caplog):
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_per_angle_mode(
            log, mode="fourier", n_phi=6, n_physics=7, n_scaling=5, threshold=3
        )
    assert "Parameters: 7 physical + 5 Fourier coeffs = 12 total" in caplog.text


def test_constant_banner_has_no_zero_scaling(caplog):
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_per_angle_mode(
            log, mode="constant", n_phi=3, n_physics=7, n_scaling=0
        )
    text = caplog.text
    assert "Parameters: 7 physical only (per-angle scaling fixed from quantiles)" in text
    assert "0 fixed scaling" not in text
    assert "= 7 total" not in text


def test_threshold_omitted_skips_reason_line(caplog):
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_per_angle_mode(
            log, mode="averaged", n_phi=3, n_physics=7, n_scaling=2, threshold=None
        )
    assert "Reason:" not in caplog.text


def test_record_uses_caller_logger_name(caplog):
    log = logging.getLogger("xpcsjax.optimization.nlsq.some_path")
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.some_path"):
        log_effective_per_angle_mode(
            log, mode="averaged", n_phi=3, n_physics=7, n_scaling=2, threshold=3
        )
    names = {r.name for r in caplog.records if "Effective per-angle mode" in r.message}
    assert names == {"xpcsjax.optimization.nlsq.some_path"}


def _laminar_controller(n_phi=3, mode="auto"):
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

    phi = np.deg2rad(np.linspace(0.0, 120.0, n_phi, endpoint=False))
    return AntiDegeneracyController.from_config(
        config_dict={
            "enable": True,
            "per_angle_mode": mode,
            "constant_scaling_threshold": 3,
            "hierarchical": {"enable": True, "max_outer_iterations": 5},
            "regularization": {"enable": True, "mode": "relative", "lambda": 1.0},
            "gradient_monitoring": {"enable": True, "ratio_threshold": 0.01},
        },
        n_phi=n_phi,
        phi_angles=phi,
        n_physical=7,
        per_angle_scaling=True,
        is_laminar_flow=True,
        analysis_mode="laminar_flow",
    )


def test_from_controller_maps_auto_averaged(caplog):
    ctrl = _laminar_controller(n_phi=3, mode="auto")
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_mode_from_controller(log, ctrl)
    text = caplog.text
    assert "ANTI-DEGENERACY: Effective per-angle mode 'averaged'" in text
    assert "Parameters: 7 physical + 2 averaged scaling = 9 total" in text


def test_from_controller_constant_has_no_reason_line(caplog):
    ctrl = _laminar_controller(n_phi=3, mode="constant")
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_mode_from_controller(log, ctrl)
    text = caplog.text
    assert "ANTI-DEGENERACY: Effective per-angle mode 'constant'" in text
    assert "physical only (per-angle scaling fixed from quantiles)" in text
    assert "Reason:" not in text


def test_from_controller_disabled_is_noop(caplog):
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

    phi = np.deg2rad(np.linspace(0.0, 120.0, 3, endpoint=False))
    ctrl = AntiDegeneracyController.from_config(
        config_dict={"enable": False, "per_angle_mode": "auto"},
        n_phi=3,
        phi_angles=phi,
        n_physical=7,
        per_angle_scaling=True,
        is_laminar_flow=True,
        analysis_mode="laminar_flow",
    )
    log = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_effective_mode_from_controller(log, ctrl)
    assert "Effective per-angle mode" not in caplog.text


def test_compute_fixed_per_angle_scaling_emits_neutral_banner(caplog):
    """The shared quantile helper is reused by the auto_averaged path, so its
    banner must NOT claim 'CONSTANT MODE' (root cause of the laminar
    auto_averaged log contradiction)."""
    ctrl = _laminar_controller(n_phi=3, mode="auto")  # -> auto_averaged, use_constant=True

    class _D:
        g2_flat = np.tile(np.linspace(1.0, 1.3, 40), 3)
        phi_flat = np.repeat([0.0, 60.0, 120.0], 40)
        t1_flat = np.tile(np.arange(40, dtype=float), 3)
        t2_flat = np.tile(np.arange(40, dtype=float) + 1.0, 3)

    cl = "xpcsjax.optimization.nlsq.anti_degeneracy_controller"
    with caplog.at_level(logging.INFO, logger=cl):
        ctrl.compute_fixed_per_angle_scaling(
            stratified_data=_D(), contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5)
        )
    text = caplog.text
    assert "CONSTANT MODE" not in text
    assert "Estimating per-angle scaling from quantiles" in text
