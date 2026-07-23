"""Regression test: validate_by_rules' phi_range rule must accept wrapped
ranges (min >= max, crossing the -180/180 boundary), matching the production
phi-range validator in config.py which already passes allow_wrapped=True.
"""

from xpcsjax.data.validators import validate_by_rules


def test_validate_by_rules_accepts_wrapped_phi_range():
    config = {"data_filtering": {"phi_range": {"min": 170.0, "max": -170.0}}}
    errors = validate_by_rules(config, "data_filtering")
    assert errors == [], f"wrapped phi_range should be valid, got: {errors}"


def test_validate_by_rules_still_rejects_out_of_bounds_phi_range():
    config = {"data_filtering": {"phi_range": {"min": -400.0, "max": 400.0}}}
    errors = validate_by_rules(config, "data_filtering")
    assert any("phi_range" in e for e in errors)
