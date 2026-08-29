"""Coverage for the homodyne physics-constraint validators (audit finding).

xpcsjax/config/physics_validators.py's cross-parameter and aggregate
validators had zero test coverage despite being production code reached
from ParameterManager.validate_all_parameters and exported via
xpcsjax.config.__init__ -- unlike their heterodyne sibling in
test_heterodyne_physics_validators.py, which is tested.
"""

from __future__ import annotations

from xpcsjax.config.physics_validators import (
    ConstraintSeverity,
    validate_all_parameters,
    validate_cross_parameter_constraints,
)


def test_d_offset_overfitting_flagged_above_half_d0() -> None:
    violations = validate_cross_parameter_constraints(
        {"D0": 1000.0, "alpha": 1.0, "D_offset": 600.0}, min_severity=ConstraintSeverity.INFO
    )
    assert any(v.param == "D_offset" and v.severity is ConstraintSeverity.INFO for v in violations)


def test_d_offset_within_bound_no_violation() -> None:
    violations = validate_cross_parameter_constraints(
        {"D0": 1000.0, "alpha": 1.0, "D_offset": 100.0}
    )
    assert violations == []


def test_cross_parameter_missing_keys_no_violation() -> None:
    # D_offset overfitting check requires D0, alpha, and D_offset together.
    assert validate_cross_parameter_constraints({"D0": 1000.0}) == []


def test_cross_parameter_min_severity_filters_info() -> None:
    params = {"D0": 1000.0, "alpha": 1.0, "D_offset": 600.0}
    assert validate_cross_parameter_constraints(params, min_severity=ConstraintSeverity.INFO)
    assert (
        validate_cross_parameter_constraints(params, min_severity=ConstraintSeverity.WARNING) == []
    )


def test_validate_all_parameters_aggregates_single_and_cross() -> None:
    # D0 = -5 trips a single-parameter rule; D_offset > 0.5*D0 needs D0 > 0
    # so only the single-parameter violation should fire here.
    violations = validate_all_parameters(
        {"D0": -5.0, "alpha": 1.0, "D_offset": 0.0}, min_severity=ConstraintSeverity.ERROR
    )
    assert any(v.param == "D0" for v in violations)


def test_validate_all_parameters_no_violations_for_sane_values() -> None:
    assert validate_all_parameters({"D0": 1000.0, "alpha": 1.0, "D_offset": 0.0}) == []
