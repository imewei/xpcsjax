"""When a solve with a non-finite covariance may be reported as converged."""

import logging

import numpy as np
import pytest
from nlsq.result import CurveFitResult

from xpcsjax.optimization.nlsq.fallback_chain import OptimizationStrategy
from xpcsjax.optimization.nlsq.recovery import execute_with_recovery

_X = np.arange(4.0)
_Y = 2.0 + 3.0 * _X


def _run(popts, strategy=OptimizationStrategy.STANDARD, success=None):
    """Solve a line fit with a fake solver returning ``popts`` and all-inf pcov."""
    queue = list(popts)

    def solver(*_a, **_k):
        return np.asarray(queue.pop(0), dtype=float), np.full((2, 2), np.inf)

    info = {} if success is None else {"success": success}
    return execute_with_recovery(
        lambda x, a, b: a + b * x,
        _X,
        _Y,
        np.array([1.0, 1.0]),
        None,
        strategy,
        logging.getLogger("test_recovery_unknown_covariance"),
        "linear",
        "jac",
        lambda result, _strategy: (*result, dict(info)),
        solver,
        solver,
    )


def test_usable_solve_is_converged_with_unknown_covariance():
    popt, pcov, info, actions, status = _run([[2.0, 3.0]] * 3)

    assert status == "converged"
    assert np.all(np.isnan(pcov))
    assert info["covariance_is_placeholder"] is True
    np.testing.assert_array_equal(popt, [2.0, 3.0])
    assert actions == ["retry_non_finite_covariance"] * 2


def test_lowest_cost_attempt_is_the_one_accepted():
    popt, _, _, _, status = _run([[2.0, 3.1], [2.0, 3.0], [9.0, 9.0]])

    assert status == "converged"
    np.testing.assert_array_equal(popt, [2.0, 3.0])


@pytest.mark.parametrize(
    "popts,strategy,success",
    [
        ([[np.nan, np.nan]] * 3, OptimizationStrategy.STANDARD, None),
        ([[2.0, 3.0]] * 3, OptimizationStrategy.LARGE, False),
    ],
    ids=["non-finite-parameters", "solver-reports-failure"],
)
def test_unusable_solve_stays_failed(popts, strategy, success):
    _, _, info, _, status = _run(popts, strategy, success)

    assert status == "failed"
    assert not info.get("covariance_is_placeholder")


def _solve(solver, model=lambda x, a, b: a + b * x, y=_Y, loss="linear", strategy=None):
    return execute_with_recovery(
        model,
        _X,
        y,
        np.array([1.0, 1.0]),
        None,
        strategy or OptimizationStrategy.STANDARD,
        logging.getLogger("test_recovery_unknown_covariance"),
        loss,
        "jac",
        lambda result, _strategy: (*result, {}),
        solver,
        solver,
    )


def _result(popt, **extra):
    """nlsq's real result type: unpacks to ``(popt, pcov)`` and carries a status."""
    return CurveFitResult(x=np.asarray(popt, dtype=float), pcov=np.full((2, 2), np.inf), **extra)


def test_standard_solve_that_reports_failure_stays_failed():
    _, _, info, _, status = _solve(lambda *_a, **_k: _result([2.0, 3.0], success=False))

    assert status == "failed"
    assert "success" not in info and not info.get("covariance_is_placeholder")


def test_solver_cost_ranks_the_attempts():
    # the plain sum of squares would prefer the second attempt
    queue = [_result([2.0, 3.0], cost=1.0), _result([2.0, 3.1], cost=2.0)]
    queue.append(_result([9.0, 9.0], cost=3.0))

    popt, _, _, _, _ = _solve(lambda *_a, **_k: queue.pop(0), y=2.0 + 3.1 * _X)

    np.testing.assert_array_equal(popt, [2.0, 3.0])


@pytest.mark.parametrize("loss,best", [("huber", [10.0, 0.0]), ("linear", [4.0, 4.0])])
def test_tuple_results_are_ranked_by_the_configured_loss(loss, best):
    # residuals are [a, b, b, b]: sum of squares and Huber disagree on the winner
    queue = [[10.0, 0.0], [4.0, 4.0], [9.0, 9.0]]

    def solver(*_a, **_k):
        return np.asarray(queue.pop(0)), np.full((2, 2), np.inf)

    popt, _, _, _, status = _solve(solver, lambda x, a, b: _Y + np.where(x == 0, a, b), loss=loss)

    assert status == "converged"
    np.testing.assert_array_equal(popt, best)


@pytest.mark.parametrize(
    "error", [ValueError("boom"), RuntimeError("out of memory")], ids=["recoverable", "fatal"]
)
def test_a_later_exception_keeps_the_earlier_usable_solve(error):
    calls = []

    def solver(*_a, **_k):
        calls.append(1)
        if len(calls) > 1:
            raise error
        return np.array([2.0, 3.0]), np.full((2, 2), np.inf)

    popt, pcov, info, actions, status = _solve(solver)

    assert status == "converged"
    assert np.all(np.isnan(pcov)) and info["covariance_is_placeholder"] is True
    np.testing.assert_array_equal(popt, [2.0, 3.0])
    assert actions[0] == "retry_non_finite_covariance"


def test_an_exception_without_a_usable_solve_still_raises():
    def solver(*_a, **_k):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _solve(solver)
