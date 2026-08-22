"""Heterodyne fixed_parameters: vary=False + value write, wins over EVERY
overlay including parameter_space.bounds and grouped parameters, including
scaling names."""

import pytest

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager


def _config(fixed_parameters, **extra_initial):
    initial = {"fixed_parameters": fixed_parameters, **extra_initial}
    return {"analysis_mode": "two_component", "initial_parameters": initial}


def test_fixed_parameter_value_is_honored_not_flat_list_value():
    config = _config(
        fixed_parameters={"D0_ref": 999.0}, parameter_names=["D0_ref"], values=[10000.0]
    )
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 999.0
    assert pm.space.vary["D0_ref"] is False
    assert "D0_ref" not in pm.varying_names


def test_fixed_contrast_is_honored():
    pm = ParameterManager.from_config(_config(fixed_parameters={"contrast": 0.42}))
    assert pm.space.values["contrast"] == 0.42
    assert pm.space.vary["contrast"] is False


def test_fixed_offset_is_honored():
    pm = ParameterManager.from_config(_config(fixed_parameters={"offset": 1.05}))
    assert pm.space.values["offset"] == 1.05
    assert pm.space.vary["offset"] is False


def test_fixed_parameters_applies_without_flat_parameter_names():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"fixed_parameters": {"D0_ref": 7.0}},
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 7.0
    assert pm.space.vary["D0_ref"] is False


def test_fixed_wins_over_active_on_conflict():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "active_parameters": ["D0_ref"],
            "fixed_parameters": {"D0_ref": 3.0},
        },
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.vary["D0_ref"] is False
    assert pm.space.values["D0_ref"] == 3.0


def test_fixed_wins_over_later_parameter_space_bounds_overlay():
    """v2-review-identified gap: fixed_parameters must win even against
    overlays that run AFTER it in ParameterSpace.from_config()'s call order."""
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"fixed_parameters": {"D0_ref": 5.0}},
        "parameter_space": {"bounds": [{"name": "D0_ref", "vary": True, "min": 0.0, "max": 1e5}]},
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.vary["D0_ref"] is False
    assert pm.space.values["D0_ref"] == 5.0


def test_tied_child_also_fixed_raises():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "tied_parameters": {"D0_ref": "D0_sample"},
            "fixed_parameters": {"D0_ref": 5.0},
        },
    }
    with pytest.raises(ValueError, match="tied_parameters.*D0_ref.*fixed_parameters"):
        ParameterManager.from_config(config)


def test_tied_parent_also_fixed_raises():
    """The parent-side mirror of the child check -- antigravity round-4 finding.
    space.vary is not a reliable signal here because _apply_fixed_parameters
    hasn't run yet when _apply_tied_parameters validates; must read the raw
    fixed_parameters config dict directly."""
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "tied_parameters": {"D0_sample": "D0_ref"},
            "fixed_parameters": {"D0_ref": 5.0},
        },
    }
    with pytest.raises(ValueError, match="tied_parameters.*D0_ref.*fixed_parameters"):
        ParameterManager.from_config(config)
