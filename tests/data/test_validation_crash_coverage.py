"""F8 TEST-1 GAP-3: crash-logging regression tests for the four missing validator crash sites.

Mirrors ``test_array_shapes_crash_logs_error_and_invalidates_report`` from
``test_validation_integrity_logging.py`` for the other four ``except`` blocks:

  - ``_validate_physics_parameters``
  - ``_validate_correlation_matrices``
  - ``_validate_statistical_properties``  (first ``except``, line ~582)
  - ``_compute_data_statistics``           (second ``except``, line ~638)

Policy (matches the array-shapes test):
  - An unexpected exception inside a validator is logged at ERROR.
  - The report is invalidated (``is_valid is False``).
  - An error-severity issue is added to ``report.errors``.
  - The validator does NOT re-raise — loading remains non-fatal.

The injected exception type is RuntimeError (same type used by the existing
array-shapes test), which is within the narrowed ``except`` tuple after F8
is applied: ``(ValueError, TypeError, KeyError, IndexError, RuntimeError)``.
"""

from __future__ import annotations

import logging

import numpy as np

from xpcsjax.data import validation as v

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _fresh_report() -> v.DataQualityReport:
    return v.DataQualityReport(
        is_valid=True,
        validation_level="basic",
        total_issues=0,
    )


def _raise_boom(*_args, **_kwargs):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# _validate_physics_parameters
# ---------------------------------------------------------------------------


def test_physics_parameters_crash_logs_error_and_invalidates_report(caplog, monkeypatch) -> None:
    """A crash inside ``_validate_physics_parameters`` must log ERROR and fail the report."""
    report = _fresh_report()

    # Force the validator body to raise by monkeypatching np.asarray used
    # inside the try block.  The physics validator calls np.asarray internally.
    monkeypatch.setattr(np, "asarray", _raise_boom)

    with caplog.at_level(logging.ERROR):
        v._validate_physics_parameters({"wavevector_q_list": object()}, {}, report)

    assert any(r.levelno == logging.ERROR for r in caplog.records), (
        "validator crash must be logged at ERROR"
    )
    assert report.is_valid is False, "a crashed validator must invalidate the report"
    assert any(issue.severity == "error" for issue in report.errors), (
        "the crash must be recorded as an error-severity issue"
    )


# ---------------------------------------------------------------------------
# _validate_correlation_matrices
# ---------------------------------------------------------------------------


def test_correlation_matrices_crash_logs_error_and_invalidates_report(caplog, monkeypatch) -> None:
    """A crash inside ``_validate_correlation_matrices`` must log ERROR and fail the report."""
    report = _fresh_report()

    monkeypatch.setattr(np, "asarray", _raise_boom)

    with caplog.at_level(logging.ERROR):
        v._validate_correlation_matrices({"c2_exp": object()}, report)

    assert any(r.levelno == logging.ERROR for r in caplog.records), (
        "validator crash must be logged at ERROR"
    )
    assert report.is_valid is False, "a crashed validator must invalidate the report"
    assert any(issue.severity == "error" for issue in report.errors), (
        "the crash must be recorded as an error-severity issue"
    )


# ---------------------------------------------------------------------------
# _validate_statistical_properties  (first except block, ~line 582)
# ---------------------------------------------------------------------------


def test_statistical_properties_crash_logs_error_and_invalidates_report(
    caplog, monkeypatch
) -> None:
    """A crash inside ``_validate_statistical_properties`` must log ERROR and fail the report."""
    report = _fresh_report()

    monkeypatch.setattr(np, "asarray", _raise_boom)

    with caplog.at_level(logging.ERROR):
        v._validate_statistical_properties({"c2_exp": object()}, report)

    assert any(r.levelno == logging.ERROR for r in caplog.records), (
        "validator crash must be logged at ERROR"
    )
    assert report.is_valid is False, "a crashed validator must invalidate the report"
    assert any(issue.severity == "error" for issue in report.errors), (
        "the crash must be recorded as an error-severity issue"
    )


# ---------------------------------------------------------------------------
# _compute_data_statistics  (second except block, ~line 638)
# ---------------------------------------------------------------------------


def test_compute_data_statistics_crash_logs_error_and_invalidates_report(
    caplog, monkeypatch
) -> None:
    """A crash inside ``_compute_data_statistics`` must log ERROR and fail the report."""
    report = _fresh_report()

    # Pass a list value so the isinstance check passes and np.asarray is called,
    # which is then monkeypatched to raise.
    monkeypatch.setattr(np, "asarray", _raise_boom)

    with caplog.at_level(logging.ERROR):
        v._compute_data_statistics({"c2_exp": [1.0, 2.0, 3.0]}, report)

    assert any(r.levelno == logging.ERROR for r in caplog.records), (
        "validator crash must be logged at ERROR"
    )
    assert report.is_valid is False, "a crashed validator must invalidate the report"
    assert any(issue.severity == "error" for issue in report.errors), (
        "the crash must be recorded as an error-severity issue"
    )
