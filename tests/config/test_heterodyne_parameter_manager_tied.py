"""Tests for ParameterManager tied-parameter support."""

from __future__ import annotations

import numpy as np

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
from xpcsjax.config.heterodyne_parameter_space import ParameterSpace


def _manager_with_tie() -> ParameterManager:
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"tied_parameters": {"D0_ref": "D0_sample"}},
    }
    space = ParameterSpace.from_config(config)
    return ParameterManager(space)


def test_tied_idx_pairs_empty_when_untied():
    space = ParameterSpace.from_config({"analysis_mode": "two_component"})
    pm = ParameterManager(space)
    assert pm.tied_idx_pairs == []


def test_tied_idx_pairs_maps_names_to_physics_indices():
    pm = _manager_with_tie()
    # D0_ref is index 0, D0_sample is index 3 in ALL_PARAM_NAMES.
    assert pm.tied_idx_pairs == [(0, 3)]


def test_expand_varying_to_full_mirrors_tied_child():
    pm = _manager_with_tie()
    varying = pm.get_initial_values()
    full = pm.expand_varying_to_full(varying)
    assert full[0] == full[3]  # D0_ref == D0_sample


def test_expand_varying_to_full_untied_matches_previous_behavior():
    space = ParameterSpace.from_config({"analysis_mode": "two_component"})
    pm = ParameterManager(space)
    varying = pm.get_initial_values()
    full = pm.expand_varying_to_full(varying)
    assert full.shape == (14,)
    assert np.all(np.isfinite(full))


def test_expand_varying_to_full_tied_value_updates_with_parent():
    """Changing the free variable that backs the parent must change the
    reported child too -- proves the mirror reads the CURRENT parent value,
    not a frozen snapshot."""
    pm = _manager_with_tie()
    varying = pm.get_initial_values().copy()
    d0_sample_pos = pm.varying_names.index("D0_sample")
    varying[d0_sample_pos] = 99999.0
    full = pm.expand_varying_to_full(varying)
    assert full[0] == 99999.0  # D0_ref mirrors the NEW D0_sample value
    assert full[3] == 99999.0
