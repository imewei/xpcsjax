import numpy as np

from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
    build_gradient_collapse_callback,
    gradient_monitor_diagnostics,
)


def _monitor(threshold=0.01, triggers=3):
    cfg = GradientMonitorConfig(
        ratio_threshold=threshold, consecutive_triggers=triggers, check_interval=1
    )
    return GradientCollapseMonitor(cfg, physical_indices=[0, 1], per_angle_indices=[2, 3])


def test_callback_feeds_monitor_per_iteration():
    mon = _monitor()

    def grad_fn(p):  # physical tiny -> ratio<<thresh -> collapse
        return np.array([1e-9, 1e-9, 1.0, 1.0])

    cb = build_gradient_collapse_callback(mon, grad_fn)
    for it in range(5):
        assert cb(it, 1.0, np.zeros(4)) is None
    assert mon.collapse_detected is True
    assert len(mon.history) == 5


def test_callback_swallows_grad_fn_errors():
    mon = _monitor()
    def boom(p):
        raise RuntimeError("grad failed")
    cb = build_gradient_collapse_callback(mon, boom)
    assert cb(0, 1.0, np.zeros(4)) is None
    assert len(mon.history) == 0


def test_diagnostics_block_shape_and_mechanism():
    mon = _monitor()

    def grad_fn(p):
        return np.array([1.0, 1.0, 1.0, 1.0])

    cb = build_gradient_collapse_callback(mon, grad_fn)
    for it in range(4):
        cb(it, 1.0, np.zeros(4))
    d = gradient_monitor_diagnostics(mon)
    assert set(d) >= {"collapse_detected", "trigger_count", "min_gradient_ratio",
                      "max_gradient_ratio", "n_observations", "ratio_threshold",
                      "consecutive_triggers", "mechanism"}
    assert d["n_observations"] == 4
    assert d["mechanism"] == "per_iteration_gradient_ratio"
    assert np.isfinite(d["max_gradient_ratio"])


def test_diagnostics_empty_history_is_fallback():
    mon = _monitor()
    d = gradient_monitor_diagnostics(mon)
    assert d["n_observations"] == 0
    assert d["mechanism"] == "post_solve_fallback"


def test_zero_or_nan_denominator_yields_inf_ratio():
    # per-angle (scaling) block norm is exactly 0 -> collapsed scaling block
    # (the opposite degeneracy end). Ratio must be inf and ALSO count as a
    # collapse via the dual-ended trigger predicate.
    mon = _monitor(threshold=0.01, triggers=1)

    def grad_fn(p):  # per_angle_indices=[2,3] both zero -> denom == 0
        return np.array([1.0, 1.0, 0.0, 0.0])

    cb = build_gradient_collapse_callback(mon, grad_fn)
    cb(0, 1.0, np.zeros(4))

    assert np.isinf(mon.history[-1]["ratio"])
    assert mon.collapse_detected is True
