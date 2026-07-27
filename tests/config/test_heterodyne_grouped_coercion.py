"""Regression test for grouped-format value/bounds coercion (audit C10).

The grouped `parameters.{group}.{param}` branch of ParameterSpace.from_config
stored value/min/max without float() coercion, so YAML string scalars (e.g.
a quoted "458.3") propagated as str into space.values / space.bounds, unlike
the flat/list-format paths which coerce.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.heterodyne_parameter_space import ParameterSpace


def test_grouped_format_coerces_string_scalars_to_float():
    config = {
        "parameters": {
            "velocity": {
                "v0": {"value": "458.3", "min": "0.0", "max": "1000.0", "vary": True},
            }
        }
    }
    space = ParameterSpace.from_config(config)

    assert isinstance(space.values["v0"], float)
    assert space.values["v0"] == 458.3
    lo, hi = space.bounds["v0"]
    assert isinstance(lo, float) and isinstance(hi, float)
    assert (lo, hi) == (0.0, 1000.0)


# --- Fix 1 regression: non-finite YAML numerics rejected at the coercion
# boundary instead of silently reaching the live NLSQ x0/bounds arrays -------


def test_grouped_format_rejects_nan_value():
    config = {"parameters": {"velocity": {"v0": {"value": "nan"}}}}
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)


def test_grouped_format_rejects_inf_bounds():
    config = {"parameters": {"velocity": {"v0": {"min": "-inf", "max": "1000.0"}}}}
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)


def test_parameter_space_bounds_list_rejects_nan():
    config = {"parameter_space": {"bounds": [{"name": "v_beta", "min": "nan", "max": "2.0"}]}}
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)


def test_parameter_space_bounds_list_rejects_inf_value():
    config = {
        "parameter_space": {
            "bounds": [{"name": "v_beta", "min": "-2.0", "max": "2.0", "value": "inf"}]
        }
    }
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)


def test_grouped_format_rejects_inverted_bounds():
    config = {"parameters": {"velocity": {"v0": {"min": "1000.0", "max": "0.0"}}}}
    with pytest.raises(ValueError, match="exceeds max"):
        ParameterSpace.from_config(config)


def test_parameter_space_bounds_list_rejects_inverted_bounds():
    config = {"parameter_space": {"bounds": [{"name": "v_beta", "min": "2.0", "max": "-2.0"}]}}
    with pytest.raises(ValueError, match="exceeds max"):
        ParameterSpace.from_config(config)


def test_equal_bounds_allowed():
    """min == max is a fixed parameter, not an error (mirrors the registry)."""
    config = {"parameters": {"velocity": {"v0": {"min": "50.0", "max": "50.0"}}}}
    assert ParameterSpace.from_config(config).bounds["v0"] == (50.0, 50.0)


def test_initial_parameters_flat_format_rejects_nan():
    config = {
        "initial_parameters": {
            "parameter_names": ["v0"],
            "values": ["nan"],
        }
    }
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)
