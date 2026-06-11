"""Regression: heterodyne ``parameter_space.bounds`` list overrides are honored.

Guards the 2026-06-11 C044 regression. The ``beta``->``v_beta`` registry alias
(commit 45e879d) narrowed the velocity-exponent default window to [0, 2], which
clamped C044's physical warm-start ``v_beta≈-0.43`` (decelerating creep flow) to
~0 and drove the fit into a degenerate basin (flat thin-diagonal C2, no velocity
fan). The C044 config *does* request ``v_beta∈[-2, 2]`` explicitly — but via the
``parameter_space.bounds`` LIST format, which ``ParameterSpace.from_config`` did
not parse (only ``parameters.{group}`` grouped and ``initial_parameters`` flat).
``_apply_parameter_space_bounds`` closes that gap with homodyne parity.

Template/alias names (``v_beta``, ``phi0_het``) must translate to the canonical
kernel names (``beta``, ``phi0``) so the overrides land on the right entry.
"""

import numpy as np

from xpcsjax.config.heterodyne_parameter_space import (
    ALL_PARAM_NAMES_WITH_SCALING,
    ParameterSpace,
)


def _resolved_bounds(space, name):
    names = list(ALL_PARAM_NAMES_WITH_SCALING)
    lo, hi = space.get_bounds_arrays()
    i = names.index(name)
    return float(lo[i]), float(hi[i])


def test_parameter_space_bounds_override_v_beta_negative():
    """``parameter_space.bounds`` with the template name ``v_beta`` overrides the
    registry default [0, 2] to the config's [-2, 2] (and the warm-start is
    feasible)."""
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "parameter_names": ["v_beta"],
            "values": [-0.4315],
        },
        "parameter_space": {
            "bounds": [
                {"name": "v_beta", "min": -2.0, "max": 2.0},
            ],
        },
    }
    space = ParameterSpace.from_config(config)
    # Canonical kernel name is "beta"; template "v_beta" must map onto it.
    lo, hi = _resolved_bounds(space, "beta")
    assert (lo, hi) == (-2.0, 2.0), f"v_beta bounds not honored: got [{lo}, {hi}]"
    # The negative warm-start must be inside the resolved window.
    assert lo <= space.values["beta"] <= hi
    assert np.isclose(space.values["beta"], -0.4315)


def test_parameter_space_bounds_default_unchanged_without_override():
    """Absent an explicit override, ``beta`` keeps the conservative registry
    default [0, 2] (the engine-route single-angle basin relies on it)."""
    space = ParameterSpace.from_config({"analysis_mode": "two_component"})
    assert _resolved_bounds(space, "beta") == (0.0, 2.0)


def test_parameter_space_bounds_override_phi0_het_translation():
    """The ``phi0_het`` template name also translates onto canonical ``phi0``."""
    config = {
        "parameter_space": {
            "bounds": [
                {"name": "phi0_het", "min": -45.0, "max": 45.0},
            ],
        },
    }
    space = ParameterSpace.from_config(config)
    assert _resolved_bounds(space, "phi0") == (-45.0, 45.0)


def test_parameter_space_bounds_unknown_name_is_skipped():
    """An unrecognised bound name is skipped, not fatal (defensive parity)."""
    config = {
        "parameter_space": {
            "bounds": [
                {"name": "not_a_parameter", "min": 0.0, "max": 1.0},
                {"name": "v0", "min": 10.0, "max": 5000.0},
            ],
        },
    }
    space = ParameterSpace.from_config(config)
    assert _resolved_bounds(space, "v0") == (10.0, 5000.0)
