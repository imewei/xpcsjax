"""Heterodyne parameter registry entries — verbatim from heterodyne docs.

Source: https://heterodyne.readthedocs.io/en/latest/configuration/options.html

Note: two parameter renames to avoid collisions with homodyne:
- heterodyne docs' `beta` (velocity exponent) → `v_beta` (matches v0/v_offset prefix)
- heterodyne docs' `phi0` → `phi0_het` (renamed to avoid colliding with
  homodyne's `phi0`; both share the same unit and bounds: degrees, [-10, 10])
"""
import pytest

from xpcsjax.config.parameter_registry import get_param_names, get_registry

EXPECTED_HETERODYNE = {
    "D0_ref":          {"default": 1e4,  "bounds": (0.0,  1e6),     "log_space": True},
    "alpha_ref":       {"default": 0.0,  "bounds": (-2.0, 2.0),     "log_space": False},
    "D_offset_ref":    {"default": 0.0,  "bounds": (-1e4, 1e4),     "log_space": False},
    "D0_sample":       {"default": 1e4,  "bounds": (0.0,  1e6),     "log_space": True},
    "alpha_sample":    {"default": 0.0,  "bounds": (-2.0, 2.0),     "log_space": False},
    "D_offset_sample": {"default": 0.0,  "bounds": (-1e4, 1e4),     "log_space": False},
    "v0":              {"default": 1e3,  "bounds": (0.0,  1e6),     "log_space": True},
    "v_beta":          {"default": 1.0,  "bounds": (0.0,  2.0),     "log_space": False},
    "v_offset":        {"default": 0.0,  "bounds": (-100.0, 100.0), "log_space": False},
    "f0":              {"default": 0.5,  "bounds": (0.0,  1.0),     "log_space": False},
    "f1":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "log_space": False},
    "f2":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "log_space": False},
    "f3":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "log_space": False},
    "phi0_het":        {"default": 0.0,  "bounds": (-10.0, 10.0),   "log_space": False},
}


@pytest.mark.parametrize("name", list(EXPECTED_HETERODYNE))
def test_heterodyne_param_specs(name):
    registry = get_registry()
    info = registry.get_param_info(name)
    expected = EXPECTED_HETERODYNE[name]
    assert info.default == expected["default"], f"{name}: default drift"
    assert (info.lower_bound, info.upper_bound) == expected["bounds"], f"{name}: bounds drift"
    assert info.log_space == expected["log_space"], f"{name}: log_space drift"
    assert info.is_physical, f"{name}: must be is_physical=True"


def test_heterodyne_mode_lists_14_params():
    names = get_param_names("two_component")
    assert len(names) == 14, f"expected 14 heterodyne params, got {len(names)}: {names}"
    assert set(names) == set(EXPECTED_HETERODYNE), f"name mismatch: {names}"


def test_heterodyne_synonym_normalize():
    """'heterodyne' should normalize to 'two_component'."""
    names = get_param_names("heterodyne")
    assert "D0_ref" in names
