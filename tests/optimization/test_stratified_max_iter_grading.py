"""R2 (status-grading parity): laminar stratified-LS must grade a max_nfev-limited
solve with a good reduced chi-squared as ``max_iter`` (graded on its real chi^2),
not a blanket ``failed`` — mirroring ``heterodyne_result_builder`` (the heterodyne
path already does this; laminar's ``_create_fit_result`` did not).

These tests pin the centralized grading in ``NLSQWrapper._create_fit_result`` so
both the stratified-LS and hybrid-streaming call sites inherit it.

The relabel is numerics-safe: parameters / chi^2 / covariance are unchanged; only
``convergence_status`` is upgraded from ``failed`` to ``max_iter`` when the solver
hit its function-evaluation budget yet produced a finite reduced chi^2.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper


def _wrapper() -> NLSQWrapper:
    return NLSQWrapper(["p0", "p1"], max_retries=0)


def _good_chi2_inputs() -> dict:
    # 100 data points, residuals ~0.1 -> chi^2 = 1.0, dof = 98 -> reduced ~0.01 (good)
    return dict(
        popt=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        residuals=np.full(100, 0.1),
        n_data=100,
        iterations=1000,
        execution_time=0.0,
    )


def test_max_nfev_with_good_chi2_grades_max_iter_via_status() -> None:
    """status==0 (SciPy max_nfev code) + finite reduced chi^2 -> max_iter, not failed."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=0,
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "max_iter"
    # quality is graded on the real reduced chi^2 (good here), never forced poor
    assert result.quality_flag in ("good", "marginal")


def test_max_nfev_with_good_chi2_grades_max_iter_via_reason_string() -> None:
    """When no status code is threaded, the SciPy message string also triggers it."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        convergence_reason="The maximum number of function evaluations is exceeded.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "max_iter"


def test_genuine_failure_stays_failed() -> None:
    """A non-budget failure (e.g. status=-1, no max_nfev reason) is not upgraded."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=-1,
        convergence_reason="Improper input parameters.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "failed"


def test_converged_status_is_never_downgraded() -> None:
    """A converged solve stays converged regardless of the threaded reason/status."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="converged",
        solver_status=0,
        convergence_reason="The maximum number of function evaluations is exceeded.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "converged"


def test_grading_is_numerics_safe() -> None:
    """The relabel must not perturb parameters / chi^2 / covariance."""
    w = _wrapper()
    inputs = _good_chi2_inputs()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=0,
        **inputs,
    )
    np.testing.assert_array_equal(result.parameters, inputs["popt"])
    assert result.chi_squared == float(np.sum(inputs["residuals"] ** 2))
