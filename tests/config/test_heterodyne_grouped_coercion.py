"""Regression test for grouped-format value/bounds coercion (audit C10).

The grouped `parameters.{group}.{param}` branch of ParameterSpace.from_config
stored value/min/max without float() coercion, so YAML string scalars (e.g.
a quoted "458.3") propagated as str into space.values / space.bounds, unlike
the flat/list-format paths which coerce.
"""

from __future__ import annotations

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
