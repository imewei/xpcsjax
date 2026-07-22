"""Regression test for Fix 2: homodyne ParameterSpace.from_config's public
bounds loader must reject non-finite YAML numerics (mirrors the heterodyne
ParameterSpace and the already-guarded ParameterManager._load_config_bounds).
"""

from __future__ import annotations

import pytest

from xpcsjax.config.parameter_space import ParameterSpace


def test_from_config_rejects_nan_bound():
    config = {
        "analysis_mode": "static_isotropic",
        "parameter_space": {
            "model": "static_isotropic",
            "bounds": [{"name": "D0", "min": "nan", "max": "1e5"}],
        },
    }
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)


def test_from_config_rejects_inf_bound():
    config = {
        "analysis_mode": "static_isotropic",
        "parameter_space": {
            "model": "static_isotropic",
            "bounds": [{"name": "D0", "min": "100.0", "max": "inf"}],
        },
    }
    with pytest.raises(ValueError, match="non-finite"):
        ParameterSpace.from_config(config)
