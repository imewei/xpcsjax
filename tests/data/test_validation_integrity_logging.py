"""Data-integrity regression: unexpected validator-body crashes must be loud.

Phase-2 Task 1 of the logging overhaul. The five ``except Exception`` handlers
in :mod:`xpcsjax.data.validation` previously swallowed unexpected validator-body
exceptions into the report as a mere ``warning`` (or, in
``_compute_data_statistics``, only a silent ``logger.warning``) — a data-integrity
bug, because a crashed validator would still leave ``report.is_valid is True``.

The decided policy is: log at ERROR with context AND mark the report failed
(``add_issue(severity="error", ...)`` sets ``is_valid = False``). Loading stays
non-fatal — the validators do NOT re-raise.
"""

import logging

import numpy as np

from xpcsjax.data import validation as v


def _raise_boom(*_args, **_kwargs):
    raise RuntimeError("boom")


def test_array_shapes_crash_logs_error_and_invalidates_report(caplog, monkeypatch):
    """A crash inside ``_validate_array_shapes`` must log ERROR and fail the report."""
    report = v.DataQualityReport(
        is_valid=True,
        validation_level="basic",
        total_issues=0,
    )
    # Force the validator body to raise an UNEXPECTED exception.
    monkeypatch.setattr(np, "asarray", _raise_boom)

    with caplog.at_level(logging.ERROR):
        v._validate_array_shapes({"c2_exp": object()}, report)

    assert any(r.levelno == logging.ERROR for r in caplog.records), (
        "validator crash must be logged at ERROR"
    )
    assert report.is_valid is False, "a crashed validator must invalidate the report"
    assert any(issue.severity == "error" for issue in report.errors), (
        "the crash must be recorded as an error-severity issue"
    )
