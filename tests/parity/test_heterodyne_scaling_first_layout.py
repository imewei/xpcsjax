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
