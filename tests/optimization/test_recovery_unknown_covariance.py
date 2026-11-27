"""When a solve with a non-finite covariance may be reported as converged."""

import logging

import numpy as np
import pytest

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
