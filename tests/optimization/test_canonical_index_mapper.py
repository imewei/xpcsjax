# tests/optimization/test_canonical_index_mapper.py
"""Phase-0 unit tests for the scaling-first canonical layout authority (spec §4 Seam 2).

Canonical optimizer vector is scaling-first: [scaling_head | physics].
vector_length MUST equal n_physics + n_optimized(mode, n_phi) for every
(mode, n_phi, n_physics) combination.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper
from xpcsjax.optimization.nlsq.per_angle_mode import n_optimized

# Homodyne laminar_flow uses 7 physical params; heterodyne two_component uses 14.
PHYSICS_CASES = [7, 14]
N_PHI_SWEEP = [1, 2, 3, 4, 10, 23]
MODES = ["constant", "averaged", "individual"]


@pytest.mark.parametrize("n_physics", PHYSICS_CASES)
@pytest.mark.parametrize("n_phi", N_PHI_SWEEP)
@pytest.mark.parametrize("mode", MODES)
def test_vector_length_property(mode, n_phi, n_physics):
    m = ParameterIndexMapper.canonical(mode=mode, n_phi=n_phi, n_physics=n_physics)
    assert m.vector_length == n_physics + n_optimized(mode, n_phi)
    assert m.mode == mode
    assert m.n_phi == n_phi
    assert m.n_physics == n_physics
    assert m.n_optimized == n_optimized(mode, n_phi)


@pytest.mark.parametrize("n_physics", PHYSICS_CASES)
@pytest.mark.parametrize("n_phi", N_PHI_SWEEP)
@pytest.mark.parametrize("mode", MODES)
def test_blocks_partition_the_vector_scaling_first(mode, n_phi, n_physics):
    m = ParameterIndexMapper.canonical(mode=mode, n_phi=n_phi, n_physics=n_physics)
    n_opt = n_optimized(mode, n_phi)
    # scaling_block is the HEAD slice; physics_block is the TAIL slice.
    assert m.scaling_block == slice(0, n_opt)
    assert m.physics_block == slice(n_opt, n_opt + n_physics)
    # The two blocks tile [0, vector_length) with no gap/overlap.
    vec = np.arange(m.vector_length)
    head = vec[m.scaling_block]
    tail = vec[m.physics_block]
    assert head.size == n_opt
    assert tail.size == n_physics
    np.testing.assert_array_equal(np.concatenate([head, tail]), vec)


def test_constant_mode_is_frozen_with_empty_scaling_head():
    m = ParameterIndexMapper.canonical(mode="constant", n_phi=23, n_physics=14)
    assert m.freeze is True
    assert m.n_optimized == 0
    assert m.scaling_block == slice(0, 0)
    assert m.physics_block == slice(0, 14)
    assert m.vector_length == 14


def test_averaged_and_individual_are_not_frozen():
    avg = ParameterIndexMapper.canonical(mode="averaged", n_phi=23, n_physics=7)
    ind = ParameterIndexMapper.canonical(mode="individual", n_phi=23, n_physics=7)
    assert avg.freeze is False
    assert ind.freeze is False
    assert avg.n_optimized == 2
    assert ind.n_optimized == 46


@pytest.mark.parametrize(
    ("mode", "n_phi", "expected"),
    [
        # group_indices for L3: [(contrast_start, contrast_end), (offset_start, offset_end)]
        # within the scaling head. constant has no optimized scaling -> empty list.
        ("constant", 23, []),
        ("averaged", 23, [(0, 1), (1, 2)]),
        ("individual", 23, [(0, 23), (23, 46)]),
        ("individual", 3, [(0, 3), (3, 6)]),
    ],
)
def test_group_indices(mode, n_phi, expected):
    m = ParameterIndexMapper.canonical(mode=mode, n_phi=n_phi, n_physics=7)
    assert m.group_indices == expected


def test_canonical_rejects_unresolved_mode():
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        ParameterIndexMapper.canonical(mode="auto", n_phi=10, n_physics=7)
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        ParameterIndexMapper.canonical(mode="nonsense", n_phi=10, n_physics=7)


def test_legacy_fourier_constructor_is_untouched():
    # Phase-0 guard: the legacy (n_phi, n_physical, use_constant) surface still works
    # and still reports its own physics-LAST layout (regression tripwire for Risk 2).
    legacy = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=True)
    assert legacy.total_params == 9
    assert legacy.get_group_indices() == [(0, 1), (1, 2)]
    assert legacy.mode_name == "constant"
