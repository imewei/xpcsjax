"""Coverage tests for `xpcsjax.optimization.nlsq.validation`.

Closes the /double-check Phase 6 gap: ``validation.py`` ships ~750 lines of
fit-quality / input / result validators but is not imported by any other
test, so future regressions in the band thresholds, NaN-rejection, or
covariance checks would not trip a CI gate. This file pins the public
contract — especially the ``<=`` boundary semantics that were the original
parity gap (xpcsjax v0.1 had ``<2 / <10`` vs homodyne ``<=2 / <=5``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.validation import (
    FitQualityConfig,
    FitQualityReport,
    InputValidator,
    ResultValidator,
    classify_fit_quality,
    validate_array_dimensions,
    validate_bounds_consistency,
    validate_covariance,
    validate_fit_quality,
    validate_no_nan_inf,
    validate_optimized_params,
    validate_result_consistency,
)

# ---------------------------------------------------------------------------
# classify_fit_quality — bands MUST be `<=` (not `<`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduced_chi2,expected",
    [
        # Interior
        (0.5, "good"),
        (1.0, "good"),
        (3.0, "acceptable"),
        (7.0, "poor"),
        (1e9, "poor"),
        # Boundary (the homodyne-parity fix — `<=` not `<`)
        (2.0, "good"),
        (5.0, "acceptable"),
        # Edge values
        (0.0, "good"),
        # Special inputs
        (None, "unknown"),
        (float("nan"), "unknown"),
        (float("inf"), "unknown"),
        (float("-inf"), "unknown"),
    ],
)
def test_classify_fit_quality_bands(reduced_chi2, expected):
    assert classify_fit_quality(reduced_chi2) == expected


def test_classify_fit_quality_respects_custom_thresholds():
    cfg = FitQualityConfig(chi2_good_threshold=1.0, chi2_acceptable_threshold=3.0)
    assert classify_fit_quality(1.0, cfg) == "good"
    assert classify_fit_quality(1.5, cfg) == "acceptable"
    assert classify_fit_quality(3.0, cfg) == "acceptable"
    assert classify_fit_quality(3.001, cfg) == "poor"


# ---------------------------------------------------------------------------
# FitQualityConfig defaults — these are the homodyne parity contract
# ---------------------------------------------------------------------------


def test_fit_quality_config_defaults_match_homodyne():
    cfg = FitQualityConfig()
    assert cfg.chi2_good_threshold == 2.0
    assert cfg.chi2_acceptable_threshold == 5.0
    assert cfg.reduced_chi_squared_threshold == 10.0
    assert cfg.min_parameter_significance == 2.0
    assert cfg.max_condition_number == 1e12
    assert cfg.enable is True


def test_fit_quality_config_from_dict_falls_back_on_missing_keys():
    cfg = FitQualityConfig.from_validation_config({"chi2_good_threshold": 1.5})
    assert cfg.chi2_good_threshold == 1.5
    assert cfg.chi2_acceptable_threshold == 5.0  # default preserved


def test_fit_quality_config_from_none_returns_defaults():
    cfg = FitQualityConfig.from_validation_config(None)
    assert cfg.chi2_good_threshold == 2.0


# ---------------------------------------------------------------------------
# validate_fit_quality — duck-typed `result`, returns FitQualityReport
# ---------------------------------------------------------------------------


def _make_result(**kwargs):
    """Build a duck-typed OptimizationResult-like object."""
    defaults = {
        "reduced_chi_squared": None,
        "parameters": None,
        "uncertainties": None,
        "covariance": None,
        "device_info": {},
        "convergence_status": "",
        "sigma_is_default": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_validate_fit_quality_returns_report_with_expected_fields():
    report = validate_fit_quality(_make_result(reduced_chi_squared=1.0))
    assert isinstance(report, FitQualityReport)
    assert isinstance(report.passed, bool)
    assert isinstance(report.warnings, list)
    assert isinstance(report.checks_performed, dict)


def test_validate_fit_quality_disabled_short_circuits():
    cfg = FitQualityConfig(enable=False)
    report = validate_fit_quality(_make_result(reduced_chi_squared=1e9), config=cfg)
    assert report.passed is True
    assert report.warnings == []
    assert report.checks_performed == {"enabled": False}


def test_validate_fit_quality_warns_on_high_reduced_chi2():
    report = validate_fit_quality(_make_result(reduced_chi_squared=100.0))
    assert report.passed is False
    assert any("Reduced chi-squared" in w for w in report.warnings)


def test_validate_fit_quality_report_to_dict_keys():
    report = FitQualityReport(passed=False, warnings=["w"], checks_performed={"k": True})
    out = report.to_dict()
    assert set(out.keys()) == {
        "quality_validation_passed",
        "quality_warnings",
        "quality_checks",
    }


# ---------------------------------------------------------------------------
# Input validators — array dims, NaN/Inf, bounds consistency
# ---------------------------------------------------------------------------


def test_validate_array_dimensions_accepts_matching_shapes():
    assert validate_array_dimensions(np.zeros(5), np.zeros(5)) is True


def test_validate_array_dimensions_rejects_empty_and_mismatched():
    assert validate_array_dimensions(np.zeros(0), np.zeros(5)) is False
    assert validate_array_dimensions(np.zeros(5), np.zeros(0)) is False
    assert validate_array_dimensions(np.zeros(5), np.zeros(6)) is False


def test_validate_no_nan_inf_accepts_finite():
    assert validate_no_nan_inf(np.array([1.0, 2.0, 3.0]), "x") is True


def test_validate_no_nan_inf_rejects_nan_and_inf():
    assert validate_no_nan_inf(np.array([1.0, np.nan, 3.0]), "x") is False
    assert validate_no_nan_inf(np.array([1.0, np.inf, 3.0]), "x") is False


def test_validate_bounds_consistency_accepts_sorted_bounds():
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 2.0])
    assert validate_bounds_consistency((lo, hi), np.array([0.5, 1.0])) is True


def test_validate_bounds_consistency_rejects_inverted_or_misshaped():
    # Inverted bounds (lower > upper)
    assert (
        validate_bounds_consistency(
            (np.array([2.0]), np.array([1.0])), np.array([1.5])
        )
        is False
    )
    # Length mismatch
    assert (
        validate_bounds_consistency(
            (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            np.array([0.5, 0.5, 0.5]),
        )
        is False
    )


def test_input_validator_strict_raises_on_bad_input():
    iv = InputValidator(strict_mode=True)
    with pytest.raises(ValueError, match="Input validation failed"):
        iv.validate_all(
            xdata=np.array([1.0, np.nan, 3.0]),
            ydata=np.array([1.0, 2.0, 3.0]),
            initial_params=np.array([0.5]),
            bounds=None,
        )


def test_input_validator_non_strict_returns_false_and_records_errors():
    iv = InputValidator(strict_mode=False)
    ok = iv.validate_all(
        xdata=np.zeros(0),
        ydata=np.zeros(3),
        initial_params=np.array([0.5]),
        bounds=None,
    )
    assert ok is False
    assert len(iv.validation_errors) >= 1


def test_input_validator_passes_on_clean_input():
    iv = InputValidator(strict_mode=True)
    ok = iv.validate_all(
        xdata=np.array([1.0, 2.0, 3.0]),
        ydata=np.array([0.1, 0.2, 0.3]),
        initial_params=np.array([0.5]),
        bounds=(np.array([0.0]), np.array([1.0])),
    )
    assert ok is True


# ---------------------------------------------------------------------------
# Result validators — params, covariance, consistency
# ---------------------------------------------------------------------------


def test_validate_optimized_params_accepts_in_bounds():
    assert (
        validate_optimized_params(
            np.array([0.5, 1.0]), (np.array([0.0, 0.0]), np.array([1.0, 2.0]))
        )
        is True
    )


def test_validate_optimized_params_rejects_non_finite():
    assert validate_optimized_params(np.array([1.0, np.nan]), bounds=None) is False


def test_validate_optimized_params_rejects_out_of_bounds():
    assert (
        validate_optimized_params(
            np.array([1.5]), (np.array([0.0]), np.array([1.0]))
        )
        is False
    )


def test_validate_covariance_accepts_symmetric_finite_positive_diag():
    cov = np.array([[1.0, 0.1], [0.1, 2.0]])
    assert validate_covariance(cov, n_params=2) is True


def test_validate_covariance_rejects_wrong_shape():
    cov = np.zeros((2, 3))
    assert validate_covariance(cov, n_params=2) is False


def test_validate_covariance_rejects_non_symmetric():
    cov = np.array([[1.0, 0.5], [0.0, 2.0]])
    assert validate_covariance(cov, n_params=2) is False


def test_validate_covariance_rejects_negative_diagonal():
    cov = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert validate_covariance(cov, n_params=2) is False


def test_validate_result_consistency_accepts_reasonable_inputs():
    assert validate_result_consistency(np.array([0.1, 0.2]), chi_squared=1.5) is True


def test_validate_result_consistency_rejects_negative_chi_squared():
    assert validate_result_consistency(np.array([0.1]), chi_squared=-1.0) is False


def test_validate_result_consistency_rejects_non_finite_chi_squared():
    assert (
        validate_result_consistency(np.array([0.1]), chi_squared=float("nan")) is False
    )


def test_validate_result_consistency_rejects_empty_or_nan_params():
    # Empty
    assert validate_result_consistency(np.array([]), chi_squared=1.0) is False
    # NaN in params
    assert (
        validate_result_consistency(np.array([1.0, np.nan]), chi_squared=1.0) is False
    )


def test_result_validator_records_warnings_for_bad_covariance():
    rv = ResultValidator(strict_mode=False)
    ok = rv.validate_all(
        params=np.array([0.5]),
        covariance=np.array([[-1.0]]),  # negative diagonal
        bounds=None,
        chi_squared=1.0,
    )
    assert ok is False
    assert any("Covariance" in w for w in rv.validation_warnings)
