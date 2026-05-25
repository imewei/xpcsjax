"""Tests for xpcsjax.optimization.nlsq.gradient_monitor.

The monitor is a pure state machine over gradient vectors: it tracks the ratio
``norm(grad_physical) / norm(grad_per_angle)`` and trips after N consecutive
sub-threshold iterations. Tests drive that machine with hand-built gradient
vectors so every branch (OK / WARNING / COLLAPSE, re-arm after recovery,
watched-parameter collapse, best-param tracking) is deterministic.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq import gradient_monitor as gm

# ---------------------------------------------------------------------------
# GradientMonitorConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = gm.GradientMonitorConfig()
    assert cfg.enable is True
    assert cfg.ratio_threshold == 0.01
    assert cfg.consecutive_triggers == 5
    assert cfg.response_mode == "hierarchical"


def test_config_from_dict_full() -> None:
    cfg = gm.GradientMonitorConfig.from_dict(
        {
            "enable": False,
            "ratio_threshold": 0.05,
            "consecutive_triggers": 2,
            "response": "abort",
            "lambda_multiplier_on_collapse": 5.0,
            "watch_parameters": [3, 4],
            "watch_threshold": 1e-6,
        }
    )
    assert cfg.enable is False
    assert cfg.ratio_threshold == 0.05
    assert cfg.consecutive_triggers == 2
    assert cfg.response_mode == "abort"
    assert cfg.watch_parameters == [3, 4]
    assert cfg.watch_threshold == 1e-6


def test_config_from_dict_watch_parameters_int_coerced_to_list() -> None:
    cfg = gm.GradientMonitorConfig.from_dict({"watch_parameters": 7})
    assert cfg.watch_parameters == [7]


def test_config_from_dict_watch_parameters_none() -> None:
    cfg = gm.GradientMonitorConfig.from_dict({})
    assert cfg.watch_parameters is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _monitor(**cfg_kw: object) -> gm.GradientCollapseMonitor:
    cfg = gm.GradientMonitorConfig(**cfg_kw)  # type: ignore[arg-type]
    return gm.GradientCollapseMonitor(
        cfg, physical_indices=[2, 3], per_angle_indices=[0, 1]
    )


# Physical grad tiny, per-angle grad large -> ratio << threshold.
_COLLAPSE_GRAD = np.array([1.0, 1.0, 1e-6, 1e-6])
# Balanced gradient -> ratio ~ 1 > threshold.
_HEALTHY_GRAD = np.array([1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# check() state machine
# ---------------------------------------------------------------------------


def test_check_disabled_returns_ok() -> None:
    mon = _monitor(enable=False)
    assert mon.check(_COLLAPSE_GRAD, 0) == "OK"


def test_check_skips_off_interval() -> None:
    mon = _monitor(check_interval=5)
    # Iteration 1 is not a multiple of 5 -> skipped.
    assert mon.check(_COLLAPSE_GRAD, 1) == "OK"
    assert len(mon.history) == 0


def test_check_healthy_gradient_is_ok() -> None:
    mon = _monitor(ratio_threshold=0.01)
    assert mon.check(_HEALTHY_GRAD, 0) == "OK"
    assert mon.consecutive_count == 0


def test_check_collapse_after_consecutive_triggers() -> None:
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=3)
    assert mon.check(_COLLAPSE_GRAD, 0) == "WARNING"  # count 1
    assert mon.check(_COLLAPSE_GRAD, 1) == "WARNING"  # count 2
    assert mon.check(_COLLAPSE_GRAD, 2) == "COLLAPSE_DETECTED"  # count 3
    assert mon.collapse_detected is True
    assert len(mon.collapse_events) == 1


def test_check_rearms_after_recovery() -> None:
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=2)
    mon.check(_COLLAPSE_GRAD, 0)
    mon.check(_COLLAPSE_GRAD, 1)  # COLLAPSE_DETECTED
    assert mon.collapse_detected is True
    # A healthy gradient re-arms (resets) the detector.
    assert mon.check(_HEALTHY_GRAD, 2) == "OK"
    assert mon.collapse_detected is False
    assert mon.consecutive_count == 0


def test_check_tracks_best_params() -> None:
    mon = _monitor()
    mon.check(_HEALTHY_GRAD, 0, params=np.array([1.0, 2.0, 3.0, 4.0]), loss=10.0)
    assert mon.best_loss == 10.0
    mon.check(_HEALTHY_GRAD, 1, params=np.array([5.0, 6.0, 7.0, 8.0]), loss=3.0)
    assert mon.best_loss == 3.0
    assert mon.best_params is not None
    np.testing.assert_array_equal(mon.best_params, [5.0, 6.0, 7.0, 8.0])
    # A worse loss does not overwrite the best.
    mon.check(_HEALTHY_GRAD, 2, params=np.zeros(4), loss=99.0)
    assert mon.best_loss == 3.0


def test_check_watched_parameter_collapse() -> None:
    cfg = gm.GradientMonitorConfig(
        ratio_threshold=0.01,
        consecutive_triggers=100,  # keep ratio-based path from firing
        watch_parameters=[3],
        watch_threshold=1e-8,
        watch_consecutive_triggers=2,
        watch_min_iteration=0,
    )
    mon = gm.GradientCollapseMonitor(cfg, physical_indices=[2, 3], per_angle_indices=[0, 1])
    # Healthy ratio but watched param[3] gradient ~ 0.
    grad = np.array([1.0, 1.0, 1.0, 1e-10])
    mon.check(grad, 0)
    mon.check(grad, 1)  # second consecutive -> confirmed
    assert mon._watch_collapse_detected[3] is True


# ---------------------------------------------------------------------------
# get_response / compute_reset_params / reset
# ---------------------------------------------------------------------------


def test_get_response_none_without_collapse() -> None:
    assert _monitor().get_response() is None


def test_get_response_dict_after_collapse() -> None:
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=1, response_mode="reset")
    mon.check(_COLLAPSE_GRAD, 0)
    resp = mon.get_response()
    assert resp is not None
    assert resp["mode"] == "reset"
    assert "best_params" in resp
    assert "collapse_events" in resp


def test_compute_reset_params_resets_to_mean() -> None:
    cfg = gm.GradientMonitorConfig()
    mon = gm.GradientCollapseMonitor(
        cfg, physical_indices=[4], per_angle_indices=[0, 1, 2, 3]
    )
    params = np.array([0.2, 0.4, 1.0, 1.2, 5.0])  # contrast[0:2], offset[2:4], physical
    out = mon.compute_reset_params(params, n_phi=2)
    np.testing.assert_allclose(out[:2], 0.3)  # contrast mean
    np.testing.assert_allclose(out[2:4], 1.1)  # offset mean
    assert out[4] == 5.0  # physical untouched


def test_reset_clears_state() -> None:
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=1)
    mon.check(_COLLAPSE_GRAD, 0)
    assert mon.collapse_detected is True
    mon.reset()
    assert mon.collapse_detected is False
    assert mon.consecutive_count == 0
    assert len(mon.history) == 0
    assert mon.best_params is None


# ---------------------------------------------------------------------------
# diagnostics + summary
# ---------------------------------------------------------------------------


def test_get_diagnostics_empty() -> None:
    diag = _monitor().get_diagnostics()
    assert diag == {"enabled": True, "n_checks": 0}


def test_get_diagnostics_with_history_and_watch() -> None:
    cfg = gm.GradientMonitorConfig(watch_parameters=[3], watch_min_iteration=0)
    mon = gm.GradientCollapseMonitor(cfg, physical_indices=[2, 3], per_angle_indices=[0, 1])
    mon.check(_HEALTHY_GRAD, 0)
    mon.check(_COLLAPSE_GRAD, 1)
    diag = mon.get_diagnostics()
    assert diag["n_checks"] == 2
    assert "min_ratio" in diag and "mean_ratio" in diag
    assert diag["watch_parameters"] == [3]
    assert 3 in diag["watch_consecutive_counts"]


def test_log_summary_variants() -> None:
    # Disabled monitor.
    _monitor(enable=False).log_summary()
    # Enabled, no checks.
    _monitor().log_summary()
    # Enabled with a collapse.
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=1)
    mon.check(_COLLAPSE_GRAD, 0)
    mon.log_summary()
    # Enabled, healthy (no collapse).
    healthy = _monitor()
    healthy.check(_HEALTHY_GRAD, 0)
    healthy.log_summary()


# ---------------------------------------------------------------------------
# create_gradient_function_with_monitoring
# ---------------------------------------------------------------------------


def test_gradient_function_wrapper_records_and_increments() -> None:
    mon = _monitor(ratio_threshold=0.01, consecutive_triggers=2)

    def grad_fn(_p: np.ndarray) -> np.ndarray:
        return _COLLAPSE_GRAD

    wrapped = gm.create_gradient_function_with_monitoring(grad_fn, mon)
    g0 = wrapped(np.zeros(4))
    np.testing.assert_array_equal(g0, _COLLAPSE_GRAD)
    wrapped(np.zeros(4))
    # Two checks recorded; collapse confirmed at the second.
    assert len(mon.history) == 2
    assert mon.collapse_detected is True
