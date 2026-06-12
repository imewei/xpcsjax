# tests/parity/test_heterodyne_scaling_first_layout.py
"""Phase 1+2 — heterodyne scaling-first joint layout helpers."""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_core import (
    _joint_param_names_scaling_first,
)


def test_individual_names_scaling_first():
    names = _joint_param_names_scaling_first(
        mode="individual", physics_names=["D0", "alpha"], n_phi=3
    )
    # scaling head [c0,c1,c2,o0,o1,o2] then physics tail [D0, alpha]
    assert names == [
        "contrast_0", "contrast_1", "contrast_2",
        "offset_0", "offset_1", "offset_2",
        "D0", "alpha",
    ]


def test_averaged_names_scaling_first():
    names = _joint_param_names_scaling_first(
        mode="averaged", physics_names=["D0", "alpha"], n_phi=5
    )
    assert names == ["contrast_avg", "offset_avg", "D0", "alpha"]


def test_constant_names_physics_only():
    names = _joint_param_names_scaling_first(
        mode="constant", physics_names=["D0", "alpha"], n_phi=4
    )
    assert names == ["D0", "alpha"]


def test_rejects_fourier():
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        _joint_param_names_scaling_first(
            mode="fourier", physics_names=["D0"], n_phi=4
        )


def test_split_individual_scaling_first():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [c0,c1, o0,o1, D0,alpha,beta]  (n_phi=2, n_physics=3)
    x = np.array([0.1, 0.2, 1.0, 1.1, 5.0, 6.0, 7.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="individual", n_phi=2, n_physics=3
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(contrast, [0.1, 0.2])
    np.testing.assert_array_equal(offset, [1.0, 1.1])


def test_split_averaged_broadcasts():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [c_avg, o_avg, D0, alpha]  (n_phi=4, n_physics=2)
    x = np.array([0.3, 1.2, 5.0, 6.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="averaged", n_phi=4, n_physics=2
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0])
    np.testing.assert_array_equal(contrast, [0.3, 0.3, 0.3, 0.3])
    np.testing.assert_array_equal(offset, [1.2, 1.2, 1.2, 1.2])


def test_split_constant_uses_frozen():
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _split_scaling_first_joint,
    )
    # [D0, alpha] only; frozen scaling supplied
    x = np.array([5.0, 6.0])
    physics, contrast, offset = _split_scaling_first_joint(
        x, mode="constant", n_phi=3, n_physics=2,
        frozen_contrast=np.array([0.4, 0.5, 0.6]),
        frozen_offset=np.array([1.4, 1.5, 1.6]),
    )
    np.testing.assert_array_equal(physics, [5.0, 6.0])
    np.testing.assert_array_equal(contrast, [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(offset, [1.4, 1.5, 1.6])
