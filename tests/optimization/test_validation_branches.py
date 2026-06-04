"""Branch-coverage complement for xpcsjax.optimization.nlsq.validation.

``test_validation.py`` pins the public band/NaN/covariance contract. This file
closes the remaining uncovered branches: the deep ``validate_fit_quality``
checks (parameter significance, covariance condition number, CMA-ES restarts,
physical-parameter bounds, convergence status), the bound/param helpers, and
the strict/non-strict validator dispatch paths. Pure logic over duck-typed
results, so SimpleNamespace fakes suffice.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import validation as v


def _result(**kw: Any) -> Any:
    return SimpleNamespace(**kw)


# ---------------------------------------------------------------------------
# helpers: _classify_parameter_status / _is_physical_param
# ---------------------------------------------------------------------------


def test_classify_parameter_status_all_three_states() -> None:
    values = np.array([0.0, 5.0, 2.5])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 5.0, 5.0])
    assert v._classify_parameter_status(values, lower, upper) == [
        "at_lower_bound",
        "at_upper_bound",
        "active",
    ]


def test_is_physical_param() -> None:
    from xpcsjax.config.parameter_registry import ParameterRegistry

    scaling = ParameterRegistry().scaling_names
    assert v._is_physical_param("D0") is True
    assert v._is_physical_param("gamma_dot_t0") is True
    s = next(iter(scaling))
    assert v._is_physical_param(f"{s}[0]") is False
    assert v._is_physical_param(s) is False


# ---------------------------------------------------------------------------
# validate_fit_quality — deep branches
# ---------------------------------------------------------------------------


def test_fit_quality_acceptable_band_logged() -> None:
    report = v.validate_fit_quality(_result(reduced_chi_squared=4.0))
    assert report.checks_performed["chi2_quality"] is True
    assert report.passed is True


def test_fit_quality_high_chi2_default_sigma_warning() -> None:
    report = v.validate_fit_quality(_result(reduced_chi_squared=50.0, sigma_is_default=True))
    assert report.passed is False
    assert any("sigma was not provided" in w for w in report.warnings)


def test_fit_quality_parameter_significance_flagged() -> None:
    report = v.validate_fit_quality(
        _result(
            reduced_chi_squared=1.0,
            parameters=np.array([10.0, 0.001]),
            uncertainties=np.array([1.0, 1.0]),
        )
    )
    assert report.checks_performed["parameter_significance"] is False


def test_fit_quality_parameter_significance_all_pass() -> None:
    report = v.validate_fit_quality(
        _result(
            reduced_chi_squared=1.0,
            parameters=np.array([10.0, 20.0]),
            uncertainties=np.array([1.0, 1.0]),
        )
    )
    assert report.checks_performed["parameter_significance"] is True


def test_fit_quality_condition_number_too_high() -> None:
    pcov = np.array([[1.0, 0.0], [0.0, 1e-20]])
    report = v.validate_fit_quality(_result(reduced_chi_squared=1.0, covariance=pcov))
    assert report.checks_performed["condition_number"] is False
    assert report.passed is False


def test_fit_quality_condition_number_ok_via_pcov_fallback() -> None:
    # No 'covariance' attribute -> the 'pcov' fallback is used.
    report = v.validate_fit_quality(_result(reduced_chi_squared=1.0, pcov=np.eye(2)))
    assert report.checks_performed["condition_number"] is True


def test_fit_quality_cmaes_max_restarts_warns() -> None:
    report = v.validate_fit_quality(
        _result(
            reduced_chi_squared=1.0,
            device_info={"convergence_reason": "max_restarts"},
        )
    )
    assert report.checks_performed["cmaes_convergence"] is False
    assert report.passed is False


def test_fit_quality_cmaes_other_reason_passes() -> None:
    report = v.validate_fit_quality(
        _result(reduced_chi_squared=1.0, device_info={"convergence_reason": "ftol"})
    )
    assert report.checks_performed["cmaes_convergence"] is True


def test_fit_quality_physical_param_at_bounds_warns() -> None:
    # D0 (physical) sits on its lower bound -> warning; scaling params are skipped.
    params = np.array([0.3, 1.0, 0.0])  # contrast, offset, D0
    bounds = (np.array([0.0, 0.5, 0.0]), np.array([1.0, 1.5, 1.0]))
    report = v.validate_fit_quality(
        _result(reduced_chi_squared=1.0, parameters=params),
        bounds=bounds,
        param_labels=["contrast[0]", "offset[0]", "D0"],
    )
    assert report.checks_performed["physical_bounds"] is False
    assert any("D0" in w for w in report.warnings)


def test_fit_quality_only_scaling_at_bounds_passes() -> None:
    # Only the scaling param (contrast) is at a bound -> no physical warning.
    params = np.array([0.0, 1.0, 0.5])
    bounds = (np.array([0.0, 0.5, 0.0]), np.array([1.0, 1.5, 1.0]))
    report = v.validate_fit_quality(
        _result(reduced_chi_squared=1.0, parameters=params),
        bounds=bounds,
        param_labels=["contrast[0]", "offset[0]", "D0"],
    )
    assert report.checks_performed["physical_bounds"] is True


def test_fit_quality_convergence_failed() -> None:
    report = v.validate_fit_quality(_result(reduced_chi_squared=1.0, convergence_status="max_iter"))
    assert report.checks_performed["convergence_status"] is False
    assert report.passed is False


def test_fit_quality_convergence_ok() -> None:
    report = v.validate_fit_quality(
        _result(reduced_chi_squared=1.0, convergence_status="converged")
    )
    assert report.checks_performed["convergence_status"] is True


# ---------------------------------------------------------------------------
# input-validation helpers (uncovered branches)
# ---------------------------------------------------------------------------


def test_validate_no_nan_inf_with_iteration_and_context() -> None:
    out = v.validate_no_nan_inf(
        np.array([1.0, np.nan]), "arr", iteration=7, context={"phase": "opt"}
    )
    assert out is False


def test_validate_bounds_consistency_upper_length_mismatch() -> None:
    # Lower length OK, upper length wrong -> exercises the upper-mismatch branch.
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0])
    assert v.validate_bounds_consistency((lower, upper), np.array([0.5, 0.5])) is False


def test_validate_initial_params_within_bounds_direct() -> None:
    assert v._validate_initial_params_within_bounds(np.array([1.0]), None) is True
    bounds = (np.array([0.0]), np.array([1.0]))
    assert v._validate_initial_params_within_bounds(np.array([0.5]), bounds) is True
    assert v._validate_initial_params_within_bounds(np.array([-1.0]), bounds) is False
    assert v._validate_initial_params_within_bounds(np.array([2.0]), bounds) is False


def test_input_validator_dimension_mismatch_recorded() -> None:
    iv = v.InputValidator(strict_mode=False)
    ok = iv.validate_all(np.arange(5.0), np.arange(4.0), np.array([0.5]), None)
    assert ok is False
    assert any("dimension mismatch" in e.lower() for e in iv.validation_errors)


def test_input_validator_bounds_inconsistent_and_out_of_range() -> None:
    iv = v.InputValidator(strict_mode=False)
    p = np.array([5.0])
    bounds = (np.array([0.0]), np.array([1.0]))  # p outside [0, 1]
    ok = iv.validate_all(np.arange(4.0), np.arange(4.0), p, bounds)
    assert ok is False
    assert iv.validation_errors


def test_input_validator_errors_property_is_copy() -> None:
    iv = v.InputValidator(strict_mode=False)
    iv.validate_all(np.zeros(0), np.zeros(3), np.array([0.5]), None)
    assert iv.validation_errors is not iv._validation_errors


# ---------------------------------------------------------------------------
# result-validation helpers (uncovered branches)
# ---------------------------------------------------------------------------


def test_validate_optimized_params_below_lower() -> None:
    bounds = (np.array([0.0]), np.array([1.0]))
    assert v.validate_optimized_params(np.array([-5.0]), bounds) is False


def test_validate_covariance_rejects_non_finite() -> None:
    cov = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert v.validate_covariance(cov, 2) is False


def test_validate_result_consistency_low_and_high_still_pass() -> None:
    params = np.array([1.0])
    # Suspiciously low and very high both log but return True.
    assert v.validate_result_consistency(params, 1e-16) is True
    assert v.validate_result_consistency(params, 1e11) is True


def test_result_validator_happy_path_true() -> None:
    rv = v.ResultValidator(strict_mode=False)
    ok = rv.validate_all(
        np.array([1.0, 2.0]),
        np.eye(2),
        (np.array([0.0, 0.0]), np.array([10.0, 10.0])),
        chi_squared=1.0,
    )
    assert ok is True
    assert rv.validation_warnings == []


def test_result_validator_strict_raises_out_of_bounds() -> None:
    rv = v.ResultValidator(strict_mode=True)
    with pytest.raises(ValueError, match="Result validation failed"):
        rv.validate_all(np.array([5.0]), None, (np.array([0.0]), np.array([1.0])))


def test_result_validator_consistency_failure_recorded() -> None:
    rv = v.ResultValidator(strict_mode=False)
    ok = rv.validate_all(np.array([1.0]), None, None, chi_squared=float("nan"))
    assert ok is False
    assert any("consistency" in w.lower() for w in rv.validation_warnings)


def test_result_validator_warnings_property_is_copy() -> None:
    rv = v.ResultValidator(strict_mode=False)
    rv.validate_all(np.array([5.0]), None, (np.array([0.0]), np.array([1.0])))
    assert rv.validation_warnings is not rv._validation_warnings
