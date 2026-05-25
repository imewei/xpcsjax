"""Coverage for the heterodyne physics-constraint validators (audit finding #5).

These validators had ZERO test imports despite being the heterodyne input gate
that reached per-angle-mode parity in Phase 6. Pure, deterministic checks.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.config.heterodyne_physics_validators import (
    ConstraintSeverity,
    validate_correlation_inputs,
    validate_cross_parameter_constraints,
    validate_single_parameter,
    validate_time_integral_safety,
)


def test_single_parameter_flags_negative_diffusion() -> None:
    violations = validate_single_parameter("D0_ref", -5.0)
    assert len(violations) >= 1
    assert any(v.severity is ConstraintSeverity.ERROR for v in violations)


def test_single_parameter_valid_value_no_violations() -> None:
    assert validate_single_parameter("D0_ref", 1.0) == []


def test_single_parameter_min_severity_filters_warnings() -> None:
    near_zero = 5e-13  # non-negative but below 1e-12 -> WARNING-level rule
    assert len(validate_single_parameter("D0_ref", near_zero)) >= 1
    # Raising the floor to ERROR filters the warning out.
    assert (
        validate_single_parameter(
            "D0_ref", near_zero, min_severity=ConstraintSeverity.ERROR
        )
        == []
    )


def test_cross_parameter_fraction_sum_exceeds_unity() -> None:
    violations = validate_cross_parameter_constraints({"f0": 0.7, "f3": 0.5})
    assert any(v.severity is ConstraintSeverity.ERROR for v in violations)


def test_cross_parameter_valid_fractions() -> None:
    assert validate_cross_parameter_constraints({"f0": 0.2, "f3": 0.3}) == []


def test_time_integral_negative_alpha_requires_positive_tmin() -> None:
    result = validate_time_integral_safety(alpha=-0.5, t_min=0.0, t_max=1.0)
    assert not result  # __bool__ -> is_valid
    assert result.errors


def test_time_integral_well_posed_is_valid() -> None:
    assert validate_time_integral_safety(alpha=0.5, t_min=1e-3, t_max=1.0)


def test_correlation_inputs_shape_mismatch_is_error() -> None:
    t1 = np.linspace(1.0, 5.0, 5)
    t2 = np.linspace(1.0, 5.0, 5)
    result = validate_correlation_inputs(t1, t2, np.ones((5, 4)))
    assert not result.is_valid


def test_correlation_inputs_nan_is_error() -> None:
    t1 = t2 = np.linspace(1.0, 5.0, 4)
    c2 = np.ones((4, 4))
    c2[0, 0] = np.nan
    result = validate_correlation_inputs(t1, t2, c2)
    assert not result.is_valid
    assert any("NaN" in e for e in result.errors)


def test_correlation_inputs_non_monotonic_time_is_error() -> None:
    t1 = np.array([1.0, 0.5, 2.0, 3.0])  # not strictly increasing
    t2 = np.linspace(1.0, 4.0, 4)
    result = validate_correlation_inputs(t1, t2, np.ones((4, 4)))
    assert not result.is_valid


def test_correlation_inputs_clean_is_valid() -> None:
    t1 = t2 = np.linspace(1.0, 10.0, 6)
    result = validate_correlation_inputs(t1, t2, np.ones((6, 6)))
    assert result.is_valid
