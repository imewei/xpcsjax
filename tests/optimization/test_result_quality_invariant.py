"""Result invariant: a 'good' quality flag requires a finite reduced chi-squared.

Defense-in-depth for the data-integrity finding that non-finite fit objectives
could surface as a confidently-"good" scientific result. ``OptimizationResult``
already forbids ``convergence_status='converged'`` with non-finite parameters;
this extends the same "illegal states are unrepresentable" guarantee to the
quality flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.results import OptimizationResult


def _kwargs(**overrides):
    base = dict(
        parameters=np.array([1.0, 2.0]),
        uncertainties=np.array([0.1, 0.1]),
        covariance=np.eye(2),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=3,
        execution_time=0.01,
        device_info={"device": "cpu"},
    )
    base.update(overrides)
    return base


def test_good_flag_with_nonfinite_reduced_chi2_is_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite reduced_chi_squared"):
            OptimizationResult(**_kwargs(reduced_chi_squared=bad, quality_flag="good"))


def test_nonfinite_reduced_chi2_allowed_when_not_good():
    # A failed/poor result is permitted to carry a non-finite objective.
    res = OptimizationResult(
        **_kwargs(
            reduced_chi_squared=float("nan"),
            convergence_status="failed",
            quality_flag="poor",
        )
    )
    assert res.quality_flag == "poor"


def test_good_flag_with_finite_reduced_chi2_is_fine():
    res = OptimizationResult(**_kwargs(reduced_chi_squared=1.5, quality_flag="good"))
    assert res.quality_flag == "good"
