"""Tests for resolve_optimized_physical_parameters and mask-based strip/restore
(fixed/active physical-parameter resolution for homodyne)."""

import numpy as np
import pytest

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    resolve_optimized_physical_parameters,
    restore_by_mask_jax,
    restore_by_mask_numpy,
    strip_by_mask,
)

PHYSICAL_NAMES_LAMINAR = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]


def _base_arrays():
    values = np.array([8000.0, -1.2, 50.0, 0.01, 0.1, 0.0, 0.0])
    lower = np.array([100.0, -2.0, -1e5, 1e-6, -2.0, -0.1, -10.0])
    upper = np.array([1e5, 2.0, 1e5, 0.5, 2.0, 0.1, 10.0])
    return values, lower, upper


def test_no_config_is_all_free_and_byte_identical():
    pm = ParameterManager({}, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper
    )
    assert isinstance(resolved, ResolvedPhysicalParameters)
    np.testing.assert_array_equal(resolved.free_mask, np.ones(7, dtype=bool))
    np.testing.assert_array_equal(resolved.values_full, values)


def test_fixed_parameter_excluded_from_free_mask_and_value_substituted():
    """The critical v1 regression: the resolved value must be the CONFIGURED
    fixed value, not whatever the flat initial-values array happened to have --
    use DIFFERENT numbers so a bug that leaves values_full unchanged is caught."""
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"D_offset": 12.5}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()  # values[2] (D_offset) == 50.0, NOT 12.5
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper
    )
    d_offset_idx = PHYSICAL_NAMES_LAMINAR.index("D_offset")
    assert resolved.free_mask[d_offset_idx] == False  # noqa: E712
    assert resolved.free_mask.sum() == 6
    assert resolved.values_full[d_offset_idx] == 12.5  # NOT 50.0 -- this is the v1 bug
    assert resolved.values_full[0] == values[0]


def test_scaling_name_in_fixed_parameters_raises():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"contrast": 0.5}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="contrast"):
        resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)


def test_typo_in_active_parameters_raises():
    """A typo'd active_parameters entry must be rejected, not silently freeze
    the intended parameter (asymmetry with fixed_parameters' unknown_fixed
    check -- see resolve_optimized_physical_parameters docstring)."""
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"active_parameters": ["D0", "D_offset_typo"]},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="D_offset_typo"):
        resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)


def test_valid_active_parameters_still_works():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"active_parameters": ["D0", "D_offset"]},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper
    )
    np.testing.assert_array_equal(
        resolved.free_mask, [True, False, True, False, False, False, False]
    )


def test_all_physical_fixed_raises_by_default():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|all.*fixed"):
        resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)


def test_all_physical_fixed_tolerated_when_allowed():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm,
        PHYSICAL_NAMES_LAMINAR,
        values,
        lower,
        upper,
        allow_all_fixed=True,
    )
    assert resolved.free_mask.sum() == 0


def test_strip_by_mask():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    np.testing.assert_array_equal(strip_by_mask(values, mask), [1.0, 3.0])


def test_restore_by_mask_numpy_round_trip():
    full_values = np.array([1.0, 99.0, 3.0, 99.0])  # positions 1,3 hold the FIXED values
    mask = np.array([True, False, True, False])
    free_result = np.array([10.0, 30.0])
    restored = restore_by_mask_numpy(free_result, full_values, mask)
    np.testing.assert_array_equal(restored, [10.0, 99.0, 30.0, 99.0])


def test_restore_by_mask_jax_matches_numpy_and_is_traceable():
    import jax

    full_values = np.array([1.0, 99.0, 3.0, 99.0])
    mask = np.array([True, False, True, False])
    free_result = np.array([10.0, 30.0])

    @jax.jit
    def traced(free):
        return restore_by_mask_jax(free, full_values, mask)

    result = np.asarray(traced(free_result))
    np.testing.assert_allclose(result, [10.0, 99.0, 30.0, 99.0])


def test_relocated_strip_and_restore_unchanged():
    """sequential.py's own bounds-equality helpers, relocated but not altered."""
    from xpcsjax.optimization.nlsq.parameter_utils import (
        restore_fixed_parameters,
        strip_fixed_parameters,
    )

    p = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.0, 0.0])
    hi = np.array([5.0, 2.0, 5.0])
    free, free_lo, free_hi, mask = strip_fixed_parameters(p, lo, hi)
    np.testing.assert_array_equal(free, [1.0, 3.0])
    np.testing.assert_array_equal(mask, [True, False, True])
    restored = restore_fixed_parameters(np.array([9.0, 8.0]), p, mask)
    np.testing.assert_array_equal(restored, [9.0, 2.0, 8.0])
