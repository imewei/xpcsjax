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
    """The shared quantile helper is reused by the averaged path, so its
    banner must NOT claim 'CONSTANT MODE' (root cause of the laminar
    averaged log contradiction)."""
    ctrl = _laminar_controller(n_phi=3, mode="auto")  # -> averaged, use_constant=True

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


# ---------------------------------------------------------------------------
# Tripwire: the fit_nlsq_cmaes summary line must reflect the *resolved* mode.
#
# `AntiDegeneracyController.use_constant` is an umbrella over fixed-constant and
# averaged scaling, so the laminar CMA-ES summary used to hard-code "constant
# mode with fixed per-angle scaling" even for an averaged run (the C020 log
# contradiction: header banners say 'averaged', the summary says 'constant').
# The label now flows through the single-source-of-truth helper
# `_anti_degeneracy_mode_summary`; these guard against vocabulary drift.
# ---------------------------------------------------------------------------


def test_broadcast_mode_label_resolves_word():
    from xpcsjax.optimization.nlsq.core import _broadcast_scaling_mode_label

    assert _broadcast_scaling_mode_label(True) == "averaged"
    assert _broadcast_scaling_mode_label(False) == "constant"


def test_summary_helper_averaged_has_no_constant_or_fixed_wording():
    from xpcsjax.optimization.nlsq.core import _anti_degeneracy_mode_summary

    msg = _anti_degeneracy_mode_summary(
        use_averaged_scaling=True, n_optimized=9, n_total=13
    )
    assert "averaged mode" in msg
    assert "optimized broadcast scaling" in msg
    assert "9 optimized -> 13 total params" in msg
    # The drift this tripwire exists to catch:
    assert "constant mode" not in msg
    assert "fixed" not in msg


def test_summary_helper_constant_keeps_fixed_wording():
    from xpcsjax.optimization.nlsq.core import _anti_degeneracy_mode_summary

    msg = _anti_degeneracy_mode_summary(
        use_averaged_scaling=False, n_optimized=7, n_total=13
    )
    assert "constant mode with fixed per-angle scaling" in msg
    assert "7 optimized -> 13 total params" in msg
    assert "averaged" not in msg


def test_fit_nlsq_cmaes_summary_routes_through_helper():
    """Pin the wiring: the summary label must be produced by the helper, never
    re-inlined as an f-string that could drift back to unconditional 'constant
    mode' wording on the averaged path."""
    import inspect

    from xpcsjax.optimization.nlsq import core

    src = inspect.getsource(core.fit_nlsq_cmaes)
    assert "_anti_degeneracy_mode_summary(" in src
    # No raw summary f-string should survive alongside the helper call.
    assert "Anti-degeneracy: constant mode with fixed per-angle scaling" not in src
    # The expansion line's mode word is folded into the same shared helper, so
    # neither call site may hard-code the "averaged"/"constant" ternary again.
    assert "_broadcast_scaling_mode_label(use_averaged_scaling)" in src
    assert '"averaged" if use_averaged_scaling else "constant"' not in src
