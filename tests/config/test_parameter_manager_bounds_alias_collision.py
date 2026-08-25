"""Homodyne ParameterManager._load_config_bounds must reject a
parameter_space.bounds list that specifies both an alias and its canonical
name (e.g. ``gamma_dot_0`` and ``gamma_dot_t0``, both PARAMETER_NAME_MAPPING
entries for the same physics parameter) -- not silently merge the two
entries' min/max/value into one Frankenstein bound dict combining fields
from two logically distinct user-supplied entries with no diagnostic that a
collision occurred.

Same bug class the heterodyne tied_parameters fix in
test_heterodyne_tied_parameters.py closes; found via the same review-pr
silent-failure-hunter pass, in the homodyne bounds-loading sibling.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_space import ParameterSpace


def test_bounds_alias_canonical_collision_rejected():
    config = {
        "analysis_mode": "laminar_flow",
        "parameter_space": {
            "bounds": [
                {"name": "gamma_dot_0", "min": 0.0, "max": 1.0},
                {"name": "gamma_dot_t0", "min": 0.0, "max": 2.0},
            ],
        },
    }
    with pytest.raises(ValueError, match="alias and its canonical name"):
        ParameterManager(config_dict=config)


def test_bounds_alias_canonical_collision_rejected_via_parameter_space():
    # ParameterSpace.from_config constructs a ParameterManager internally on
    # the same config, so the collision must surface here too, not just via
    # direct ParameterManager construction.
    config = {
        "analysis_mode": "laminar_flow",
        "parameter_space": {
            "bounds": [
                {"name": "phi_0", "min": -90.0, "max": 90.0},
                {"name": "phi0", "min": -45.0, "max": 45.0},
            ],
        },
    }
    with pytest.raises(ValueError, match="alias and its canonical name"):
        ParameterSpace.from_config(config)


def test_bounds_no_collision_still_loads_normally():
    config = {
        "analysis_mode": "laminar_flow",
        "parameter_space": {
            "bounds": [
                {"name": "gamma_dot_t0", "min": 0.0, "max": 5.0},
            ],
        },
    }
    pm = ParameterManager(config_dict=config)
    assert pm._default_bounds["gamma_dot_t0"]["min"] == 0.0
    assert pm._default_bounds["gamma_dot_t0"]["max"] == 5.0
