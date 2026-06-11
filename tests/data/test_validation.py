"""Tests for xpcsjax.data.validation (M-2: closes the largest coverage gap).

``validate_xpcs_data`` is the data-integrity contract (CLAUDE.md Principle 2:
"No silent data loss"). These cover the public report-building paths: clean
data, non-finite detection, structural/shape errors, and the quality score.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data.validation import DataQualityReport, ValidationIssue, validate_xpcs_data


def _clean_data() -> dict[str, np.ndarray]:
    """A minimal, internally consistent XPCS data dict."""
    n_t = 8
    t = np.linspace(0.0, 1.0, n_t)
    c2 = np.ones((2, n_t, n_t))  # 2 angle matrices, square, non-negative
    return {
        "wavevector_q_list": np.array([0.01, 0.02]),
        "phi_angles_list": np.array([0.0, 45.0]),
        "t1": t,
        "t2": t,
        "c2_exp": c2,
    }


def test_clean_data_is_valid() -> None:
    report = validate_xpcs_data(_clean_data())
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_validation_level_none_skips_checks() -> None:
    bad = _clean_data()
    bad["c2_exp"] = np.full((2, 8, 8), np.nan)
    report = validate_xpcs_data(bad, validation_level="none")
    # Disabled validation must not flag anything.
    assert report.is_valid is True
    assert report.total_issues == 0


def test_nonfinite_in_correlation_is_error() -> None:
    data = _clean_data()
    data["c2_exp"][0, 0, 0] = np.nan
    report = validate_xpcs_data(data)
    assert report.is_valid is False
    assert any(i.severity == "error" and "non-finite" in i.message.lower() for i in report.errors)


def test_missing_required_key_is_error() -> None:
    data = _clean_data()
    del data["c2_exp"]
    report = validate_xpcs_data(data)
    assert report.is_valid is False
    assert len(report.errors) >= 1


def test_t1_t2_shape_mismatch_is_reported() -> None:
    data = _clean_data()
    data["t2"] = np.linspace(0.0, 1.0, 9)  # different length from t1 (8)
    report = validate_xpcs_data(data)
    assert any("shape" in i.message.lower() for i in report.errors)


def test_non_positive_q_is_error() -> None:
    data = _clean_data()
    data["wavevector_q_list"] = np.array([0.0, 0.02])  # 0 is non-physical
    report = validate_xpcs_data(data)
    assert report.is_valid is False
    assert any(i.parameter == "wavevector_q_list" for i in report.errors)


def test_quality_score_drops_with_issues() -> None:
    clean = validate_xpcs_data(_clean_data())
    dirty_data = _clean_data()
    dirty_data["c2_exp"][0, 1, 1] = np.nan
    dirty = validate_xpcs_data(dirty_data)
    assert clean.quality_score >= dirty.quality_score


def _two_time_matrix(
    n_t: int, beta: float, tau: float, diag_spike: float | None
) -> np.ndarray:
    """Build a symmetric two-time correlation matrix.

    ``g2(t1, t2) = 1 + beta * exp(-|t1 - t2| / tau)`` (Siegert form). When
    ``diag_spike`` is given, the exact tau=0 main diagonal is overwritten with
    that value to mimic the self-correlation / shot-noise spike present in raw
    two-time XPCS data (which is excluded from analysis).
    """
    idx = np.arange(n_t)
    lag = np.abs(idx[:, None] - idx[None, :])
    matrix = 1.0 + beta * np.exp(-lag / tau)
    if diag_spike is not None:
        np.fill_diagonal(matrix, diag_spike)
    return matrix


def _correlation_warnings(report: DataQualityReport) -> list[ValidationIssue]:
    """Warnings emitted by the near-zero-lag correlation-matrix check."""
    return [i for i in report.warnings if "near-zero-lag correlation" in i.message]


def test_tau0_self_correlation_spike_is_not_flagged() -> None:
    """Regression: the excluded tau=0 self-correlation spike must not fail QC.

    The exact main diagonal of raw two-time XPCS C2 carries an uncorrelated
    shot-noise spike (commonly ~2.4, well above the Siegert ceiling g2(0) <= 2).
    The physical near-zero-lag g2 (first off-diagonal) here is ~1.29 and is
    Siegert-valid, so full validation must emit zero near-zero-lag warnings even
    though matrix[k, k] = 2.4 > 2.0 (which the old diagonal[0] check flagged).
    """
    n_t = 12
    t = np.linspace(0.0, 1.0, n_t)
    # diag spike 2.4 > 2.0; lag-1 g2 = 1 + 0.3*exp(-1/50) ~ 1.294 (valid).
    matrix = _two_time_matrix(n_t, beta=0.3, tau=50.0, diag_spike=2.4)
    assert matrix[0, 0] > 2.0 and 0.5 <= matrix[0, 1] <= 2.0  # guards the setup
    data = _clean_data()
    data["c2_exp"] = np.stack([matrix, matrix])
    data["t1"] = t
    data["t2"] = t

    report = validate_xpcs_data(data, validation_level="full")
    assert _correlation_warnings(report) == []


def test_overnormalized_near_zero_lag_is_still_flagged() -> None:
    """The lag-aware check must still catch genuinely over-normalized data.

    When even the first off-diagonal (near-zero-lag) g2 exceeds the Siegert
    ceiling of 2.0, the matrix is genuinely mis-normalized and must be flagged.
    """
    n_t = 12
    t = np.linspace(0.0, 1.0, n_t)
    # beta=1.5 is non-physical: lag-1 g2 = 1 + 1.5*exp(-1/50) ~ 2.47 > 2.0.
    matrix = _two_time_matrix(n_t, beta=1.5, tau=50.0, diag_spike=None)
    assert matrix[0, 1] > 2.0  # guards the setup
    data = _clean_data()
    data["c2_exp"] = np.stack([matrix, matrix])
    data["t1"] = t
    data["t2"] = t

    report = validate_xpcs_data(data, validation_level="full")
    assert len(_correlation_warnings(report)) >= 1


def test_report_add_issue_flips_is_valid_on_error() -> None:
    report = DataQualityReport(is_valid=True, validation_level="basic", total_issues=0)
    report.add_issue(ValidationIssue(severity="error", category="test", message="boom"))
    assert report.is_valid is False


def test_report_warning_does_not_invalidate() -> None:
    report = DataQualityReport(is_valid=True, validation_level="basic", total_issues=0)
    report.add_issue(ValidationIssue(severity="warning", category="test", message="heads up"))
    assert report.is_valid is True
    assert report.total_issues == 1


@pytest.mark.parametrize("level", ["basic", "full"])
def test_levels_run_without_crashing_on_clean_data(level: str) -> None:
    report = validate_xpcs_data(_clean_data(), validation_level=level)
    assert isinstance(report, DataQualityReport)
