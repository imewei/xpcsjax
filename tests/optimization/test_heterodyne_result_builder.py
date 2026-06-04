"""Tests for the heterodyne NLSQ result layer.

Covers the ``NLSQResult`` dataclass (derived properties, correlation matrix,
validation warnings, summary) and the ``heterodyne_result_builder`` dispatch
that normalizes scipy / array / dict / tuple / object optimizer outputs into a
consistent ``NLSQResult`` (plus covariance from the Gauss-Newton Jacobian and
the status-code reason map).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from xpcsjax.optimization.nlsq import heterodyne_result_builder as rb
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult

# ---------------------------------------------------------------------------
# NLSQResult dataclass
# ---------------------------------------------------------------------------


def _result(**kw: object) -> NLSQResult:
    base: dict = {
        "parameters": np.array([1.0, 2.0]),
        "parameter_names": ["a", "b"],
        "success": True,
        "message": "ok",
    }
    base.update(kw)
    return NLSQResult(**base)


def test_n_params_and_params_dict() -> None:
    r = _result()
    assert r.n_params == 2
    assert r.params_dict == {"a": 1.0, "b": 2.0}


def test_get_param_and_missing() -> None:
    r = _result()
    assert r.get_param("b") == 2.0
    with pytest.raises(KeyError, match="not found"):
        r.get_param("missing")


def test_get_uncertainty() -> None:
    r = _result(uncertainties=np.array([0.1, 0.2]))
    assert r.get_uncertainty("a") == pytest.approx(0.1)
    assert r.get_uncertainty("missing") is None
    assert _result().get_uncertainty("a") is None  # no uncertainties


def test_correlation_matrix() -> None:
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    corr = _result(covariance=cov).get_correlation_matrix()
    assert corr is not None
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(1.0 / (2.0 * 3.0))
    assert _result().get_correlation_matrix() is None  # no covariance


def test_validate_warnings() -> None:
    # Failed fit + poor chi2 + huge uncertainty + high correlation.
    cov = np.array([[1.0, 0.99 * 1.0 * 5.0], [0.99 * 1.0 * 5.0, 25.0]])
    r = _result(
        success=False,
        message="diverged",
        reduced_chi_squared=5.0,
        uncertainties=np.array([10.0, 0.1]),
        covariance=cov,
    )
    warnings = r.validate()
    assert any("failed" in w.lower() for w in warnings)
    assert any("Poor fit" in w for w in warnings)
    assert any("Large uncertainty" in w for w in warnings)
    assert any("Highly correlated" in w for w in warnings)


def test_validate_overfit_warning() -> None:
    r = _result(reduced_chi_squared=0.2)
    assert any("overfit" in w.lower() for w in r.validate())


def test_summary_contains_params_and_stats() -> None:
    r = _result(
        uncertainties=np.array([0.1, 0.2]),
        final_cost=1.5,
        reduced_chi_squared=1.1,
        n_iterations=7,
        wall_time_seconds=2.0,
    )
    text = r.summary()
    assert "NLSQ Fit Result" in text
    assert "Reduced" in text
    assert "Wall time" in text


def test_summary_without_uncertainties() -> None:
    text = _result().summary()
    assert "a" in text  # parameter listed even without uncertainties


# ---------------------------------------------------------------------------
# build_result_from_scipy
# ---------------------------------------------------------------------------


def test_build_from_scipy_with_jacobian() -> None:
    opt = OptimizeResult(
        x=np.array([1.0, 2.0, 3.0]),
        fun=np.array([0.1, 0.1, 0.1]),
        jac=np.eye(3),
        status=1,
        nit=5,
        nfev=12,
        message="converged",
    )
    res = rb.build_result_from_scipy(opt, ["a", "b", "c"], n_data=100)
    assert res.success is True
    assert res.covariance is not None
    assert res.uncertainties is not None
    assert res.n_iterations == 5
    assert res.convergence_reason == "gtol convergence (gradient sufficiently small)"
    assert res.reduced_chi_squared == pytest.approx(0.03 / 97)


def test_build_from_scipy_without_jacobian() -> None:
    opt = OptimizeResult(x=np.array([1.0]), fun=np.array([0.0]), status=0, message="maxfev")
    res = rb.build_result_from_scipy(opt, ["a"], n_data=10)
    assert res.covariance is None
    assert res.uncertainties is None
    assert res.success is False  # status 0 -> not > 0


# ---------------------------------------------------------------------------
# build_result_from_arrays
# ---------------------------------------------------------------------------


def test_build_from_arrays_with_and_without_jacobian() -> None:
    params = np.array([1.0, 2.0])
    residuals = np.array([0.5, 0.5, 0.5])
    res = rb.build_result_from_arrays(params, ["a", "b"], residuals, n_data=50)
    assert res.covariance is None
    assert res.final_cost == pytest.approx(0.75)
    assert res.reduced_chi_squared == pytest.approx(0.75 / 48)

    res2 = rb.build_result_from_arrays(
        params, ["a", "b"], residuals, n_data=50, jacobian=np.ones((3, 2))
    )
    assert res2.covariance is not None


# ---------------------------------------------------------------------------
# build_result_from_nlsq — format dispatch
# ---------------------------------------------------------------------------


def test_build_from_nlsq_dict() -> None:
    res = rb.build_result_from_nlsq(
        {
            "x": np.array([1.0, 2.0]),
            "pcov": np.eye(2),
            "fun": np.array([0.1, 0.1]),
            "success": True,
            "message": "ok",
            "nit": 4,
            "nfev": 9,
        },
        ["a", "b"],
        n_data=20,
    )
    assert res.uncertainties is not None
    assert res.n_iterations == 4
    assert res.final_cost == pytest.approx(0.02)


def test_build_from_nlsq_dict_missing_keys_raises() -> None:
    with pytest.raises(TypeError, match="neither 'x' nor 'popt'"):
        rb.build_result_from_nlsq({"foo": 1}, ["a"], n_data=10)


def test_build_from_nlsq_two_tuple() -> None:
    res = rb.build_result_from_nlsq((np.array([1.0, 2.0]), np.eye(2)), ["a", "b"], n_data=20)
    assert res.covariance is not None
    assert res.final_cost is None  # no residuals in a 2-tuple


def test_build_from_nlsq_three_tuple_with_info() -> None:
    res = rb.build_result_from_nlsq(
        (np.array([1.0]), np.eye(1), {"message": "done", "nfev": 3}),
        ["a"],
        n_data=10,
    )
    assert res.metadata["nfev"] == 3
    assert res.message == "done"


def test_build_from_nlsq_bad_tuple_length_raises() -> None:
    with pytest.raises(TypeError, match="Unexpected tuple length"):
        rb.build_result_from_nlsq((1, 2, 3, 4), ["a"], n_data=10)


def test_build_from_nlsq_object() -> None:
    obj = OptimizeResult(
        x=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        fun=np.array([0.2, 0.2]),
        message="obj ok",
        success=True,
        nfev=5,
        nit=2,
    )
    res = rb.build_result_from_nlsq(obj, ["a", "b"], n_data=30)
    assert res.uncertainties is not None
    assert res.n_function_evals == 5
    assert res.final_cost == pytest.approx(0.08)


def test_build_from_nlsq_plain_object_attributes() -> None:
    # A non-dict object with .x/.pcov/.fun exercises the attribute-extraction
    # path (OptimizeResult is a dict subclass and takes the dict branch instead).
    obj = cast(
        Any,
        SimpleNamespace(
            x=np.array([1.0, 2.0]),
            pcov=np.eye(2),
            fun=np.array([0.3, 0.4]),
            message="ns ok",
            success=True,
            nfev=8,
            nit=3,
            info={"extra": 1},
        ),
    )
    res = rb.build_result_from_nlsq(obj, ["a", "b"], n_data=40)
    assert res.uncertainties is not None
    assert res.n_function_evals == 8
    assert res.metadata["extra"] == 1
    assert res.final_cost == pytest.approx(0.25)


def test_build_from_nlsq_object_missing_popt_raises() -> None:
    obj = cast(Any, SimpleNamespace(x=None))
    with pytest.raises(TypeError, match="neither 'x' nor 'popt'"):
        rb.build_result_from_nlsq(obj, ["a"], n_data=10)


def test_build_from_nlsq_three_tuple_non_dict_info() -> None:
    # Non-dict info object is wrapped under 'raw_info'.
    res = rb.build_result_from_nlsq(
        (np.array([1.0]), np.eye(1), "some-status-string"),
        ["a"],
        n_data=10,
    )
    assert res.metadata["raw_info"] == "some-status-string"


def test_build_from_nlsq_unrecognized_raises() -> None:
    with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
        rb.build_result_from_nlsq(42, ["a"], n_data=10)


# ---------------------------------------------------------------------------
# build_failed_result
# ---------------------------------------------------------------------------


def test_build_failed_result_with_initial() -> None:
    res = rb.build_failed_result(["a", "b"], "boom", initial_params=np.array([1.0, 2.0]))
    assert res.success is False
    assert res.message == "boom"
    np.testing.assert_array_equal(res.parameters, [1.0, 2.0])


def test_build_failed_result_without_initial() -> None:
    res = rb.build_failed_result(["a", "b", "c"], "boom")
    np.testing.assert_array_equal(res.parameters, np.zeros(3))


# ---------------------------------------------------------------------------
# TimedContext
# ---------------------------------------------------------------------------


def test_timed_context_measures_elapsed() -> None:
    timer = rb.TimedContext()
    with timer:
        _ = sum(range(1000))
    assert timer.elapsed >= 0.0


# ---------------------------------------------------------------------------
# _compute_covariance / _status_to_reason
# ---------------------------------------------------------------------------


def test_compute_covariance_well_conditioned() -> None:
    cov = rb._compute_covariance(np.eye(3), np.array([0.1, 0.1, 0.1]), n_data=100, n_params=3)
    assert cov is not None
    assert cov.shape == (3, 3)


def test_compute_covariance_regularizes_near_singular() -> None:
    # Rank-deficient Jacobian -> huge condition number -> Tikhonov path.
    jac = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-16]])
    cov = rb._compute_covariance(jac, np.array([0.1, 0.1]), n_data=10, n_params=2)
    assert cov is not None  # regularization keeps it invertible


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (-1, "Improper input"),
        (0, "Maximum function"),
        (1, "gtol"),
        (2, "xtol"),
        (3, "ftol"),
        (4, "Both xtol and ftol"),
        (99, "Unknown status: 99"),
    ],
)
def test_status_to_reason(status: int, fragment: str) -> None:
    assert fragment in rb._status_to_reason(status)
