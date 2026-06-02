"""Characterization tests for PerAngleScaling pack/unpack helpers.

Quality-gate gap: ``get_varying_values`` / ``update_from_varying`` /
``get_for_angle`` — which feed the per-angle scaling layouts used by every
escape and streaming path — had no direct tests. These pin the
optimizer<->scaling round-trip and the constant-mode propagation.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.core.heterodyne_scaling_utils import PerAngleScaling, ScalingConfig


def test_individual_mode_varying_roundtrip_is_identity():
    scaling = PerAngleScaling.from_config(ScalingConfig(n_angles=3, mode="individual"))
    # individual ⇒ all 2*3 params vary: [c0,c1,c2, o0,o1,o2]
    v = np.array([0.30, 0.35, 0.40, 1.00, 1.01, 1.02])
    scaling.update_from_varying(v)
    assert np.array_equal(scaling.get_varying_values(), v)
    assert scaling.get_for_angle(1) == (0.35, 1.01)


def test_constant_mode_propagates_first_angle_to_all():
    scaling = PerAngleScaling.from_config(ScalingConfig(n_angles=4, mode="constant"))
    # constant ⇒ only angle 0 varies: 1 contrast + 1 offset
    scaling.update_from_varying(np.array([0.42, 1.05]))
    assert np.array_equal(scaling.get_varying_values(), np.array([0.42, 1.05]))
    for a in range(4):
        assert scaling.get_for_angle(a) == (0.42, 1.05)
