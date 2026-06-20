"""Regression test for the L4 gradient-collapse monitor index layout.

``_build_l4_callback`` partitions the joint vector into physical vs per-angle
(scaling) index sets. Heterodyne has TWO joint layouts:

- averaged path: PHYSICS-FIRST ``[physics | contrast, offset]``
- ``_fit_joint_multi_phi``: canonical SCALING-FIRST ``[scaling_head | physics]``

A prior bug hard-coded the physics-first partition for both, so on the
scaling-first path the monitor labeled the first ``n_physics`` *scaling*
parameters as "physical" — meaningless gradient ratios in the L4 diagnostics
(L4 is observation-only, so the SOLVE was unaffected, but the diagnostics lied).
"""

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_core import _build_l4_callback


class _StubParamManager:
    def __init__(self, n_varying: int):
        self.n_varying = n_varying


class _StubModel:
    def __init__(self, n_varying: int):
        self.param_manager = _StubParamManager(n_varying)


class _StubConfig:
    enable_gradient_monitoring = True
    gradient_ratio_threshold = 0.1
    gradient_consecutive_triggers = 3


def _make(n_physics, n_scaling, scaling_first):
    total = n_scaling + n_physics
    x0 = np.zeros(total)
    monitor, _callback = _build_l4_callback(
        _StubModel(n_physics),
        x0,
        lambda p: np.asarray(p),  # joint_residual_fn; unused at construction
        _StubConfig(),
        scaling_first=scaling_first,
    )
    return monitor, total


def test_scaling_first_layout_partitions_physics_as_tail():
    n_physics, n_phi = 7, 4
    n_scaling = 2 * n_phi  # individual scaling-first head
    monitor, total = _make(n_physics, n_scaling, scaling_first=True)
    assert monitor.physical_indices.tolist() == list(range(n_scaling, total))
    assert monitor.per_angle_indices.tolist() == list(range(0, n_scaling))


def test_physics_first_layout_partitions_physics_as_head():
    n_physics, n_scaling = 7, 2  # averaged path: [physics | contrast, offset]
    monitor, total = _make(n_physics, n_scaling, scaling_first=False)
    assert monitor.physical_indices.tolist() == list(range(n_physics))
    assert monitor.per_angle_indices.tolist() == list(range(n_physics, total))


def test_disabled_monitoring_returns_none():
    class _Off(_StubConfig):
        enable_gradient_monitoring = False

    monitor, callback = _build_l4_callback(
        _StubModel(7), np.zeros(9), lambda p: p, _Off(), scaling_first=True
    )
    assert monitor is None and callback is None
