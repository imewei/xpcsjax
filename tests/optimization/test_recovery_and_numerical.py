"""Tests for error-recovery and numerical-validation helpers.

``recovery_strategies`` maps each NLSQ error type to a prioritized list of
recovery actions; tests verify the dispatch order, the parameter-perturbation
and bounds-tightening transforms, and retry accounting. ``numerical_validation``
raises ``NLSQNumericalError`` (tagged with a detection point) on NaN/Inf in
gradients, parameters, or loss, and on bounds violations — tests cover each
detection point plus the disabled-validation fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.exceptions import NLSQConvergenceError, NLSQNumericalError
from xpcsjax.optimization.numerical_validation import NumericalValidator
from xpcsjax.optimization.recovery_strategies import RecoveryStrategyApplicator

# ---------------------------------------------------------------------------
# RecoveryStrategyApplicator
# ---------------------------------------------------------------------------


def _strategy_name(app: RecoveryStrategyApplicator, err: Exception, params, attempt) -> str:
    out = app.get_recovery_strategy(err, params, attempt)
    assert out is not None
    return out[0]


def test_convergence_error_strategy_order() -> None:
    app = RecoveryStrategyApplicator()
    params = np.array([1.0, 2.0])
    err = NLSQConvergenceError("no converge")
    assert _strategy_name(app, err, params, 0) == "perturb_parameters"
    assert _strategy_name(app, err, params, 1) == "increase_iterations"
    assert _strategy_name(app, err, params, 2) == "relax_tolerance"
    # Beyond the strategy list -> None.
    assert app.get_recovery_strategy(err, params, 3) is None


def test_numerical_error_strategy_order() -> None:
    app = RecoveryStrategyApplicator()
    params = np.array([1.0, 2.0])
    err = NLSQNumericalError("nan")
    assert _strategy_name(app, err, params, 0) == "reduce_step_size"
    assert _strategy_name(app, err, params, 1) == "tighten_bounds"
    assert _strategy_name(app, err, params, 2) == "rescale_data"


def test_unknown_error_type_returns_none() -> None:
    app = RecoveryStrategyApplicator()
    out = app.get_recovery_strategy(ValueError("other"), np.array([1.0]), 0)
    assert out is None


def test_perturb_parameters_reproducible_and_shaped() -> None:
    err = NLSQConvergenceError("x")
    params = np.array([1.0, 2.0, 3.0])
    a = RecoveryStrategyApplicator(seed=7).get_recovery_strategy(err, params, 0)
    b = RecoveryStrategyApplicator(seed=7).get_recovery_strategy(err, params, 0)
    assert a is not None and b is not None
    assert a[1].shape == params.shape
    np.testing.assert_array_equal(a[1], b[1])  # same seed -> identical perturbation
    assert not np.array_equal(a[1], params)  # actually perturbed


def test_perturb_zero_param_uses_additive_fallback() -> None:
    # Zero-valued params must move (multiplicative scaling would leave them at 0).
    err = NLSQConvergenceError("x")
    out = RecoveryStrategyApplicator(seed=1).get_recovery_strategy(err, np.array([0.0, 5.0]), 0)
    assert out is not None
    assert out[1][0] != 0.0


def test_tighten_bounds_clips_params() -> None:
    err = NLSQNumericalError("x")
    bounds = (np.array([0.0]), np.array([10.0]))
    # attempt 1 -> tighten_bounds (0.9): tightened range is [0.5, 9.5].
    out = RecoveryStrategyApplicator().get_recovery_strategy(err, np.array([0.2]), 1, bounds=bounds)
    assert out is not None
    assert out[0] == "tighten_bounds"
    assert out[1][0] == pytest.approx(0.5)  # clipped up to tightened lower


def test_tighten_bounds_without_bounds_is_noop() -> None:
    err = NLSQNumericalError("x")
    params = np.array([0.2])
    out = RecoveryStrategyApplicator().get_recovery_strategy(err, params, 1, bounds=None)
    assert out is not None
    np.testing.assert_array_equal(out[1], params)


def test_passthrough_strategies_return_copy() -> None:
    err = NLSQConvergenceError("x")
    params = np.array([1.0, 2.0])
    # increase_iterations (attempt 1) doesn't modify params but returns a copy.
    out = RecoveryStrategyApplicator().get_recovery_strategy(err, params, 1)
    assert out is not None
    np.testing.assert_array_equal(out[1], params)
    assert out[1] is not params


def test_should_retry() -> None:
    app = RecoveryStrategyApplicator(max_retries=2)
    assert app.should_retry(0) is True
    assert app.should_retry(1) is True
    assert app.should_retry(2) is False


# ---------------------------------------------------------------------------
# NumericalValidator
# ---------------------------------------------------------------------------


def test_validate_gradients_finite_ok() -> None:
    NumericalValidator().validate_gradients(np.array([1.0, 2.0, 3.0]))  # no raise


def test_validate_gradients_detects_nonfinite() -> None:
    with pytest.raises(NLSQNumericalError) as exc:
        NumericalValidator().validate_gradients(np.array([1.0, np.nan, np.inf]))
    assert exc.value.detection_point == "gradient"


def test_validate_gradients_disabled_skips() -> None:
    NumericalValidator(enable_validation=False).validate_gradients(np.array([np.nan]))


def test_validate_parameters_finite_ok() -> None:
    NumericalValidator().validate_parameters(np.array([1.0, 2.0]))


def test_validate_parameters_detects_nonfinite() -> None:
    with pytest.raises(NLSQNumericalError) as exc:
        NumericalValidator().validate_parameters(np.array([1.0, np.nan]))
    assert exc.value.detection_point == "parameter"


def test_validate_parameters_bounds_violation() -> None:
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(NLSQNumericalError) as exc:
        NumericalValidator().validate_parameters(np.array([0.5, 5.0]), bounds=bounds)
    assert exc.value.detection_point == "parameter_bounds"


def test_validate_parameters_within_instance_bounds_ok() -> None:
    bounds = (np.array([0.0]), np.array([10.0]))
    validator = NumericalValidator(bounds=bounds)
    validator.validate_parameters(np.array([5.0]))  # uses instance bounds, no raise


def test_validate_parameters_below_lower_bound() -> None:
    bounds = (np.array([0.0]), np.array([1.0]))
    with pytest.raises(NLSQNumericalError):
        NumericalValidator().validate_parameters(np.array([-1.0]), bounds=bounds)


def test_validate_loss_finite_ok() -> None:
    NumericalValidator().validate_loss(1.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_loss_detects_nonfinite(bad: float) -> None:
    with pytest.raises(NLSQNumericalError) as exc:
        NumericalValidator().validate_loss(bad)
    assert exc.value.detection_point == "loss"


def test_validate_loss_disabled_skips() -> None:
    NumericalValidator(enable_validation=False).validate_loss(float("nan"))


def test_set_bounds_disable_enable() -> None:
    v = NumericalValidator()
    v.set_bounds((np.array([0.0]), np.array([1.0])))
    assert v.bounds is not None
    v.disable()
    assert v.enable_validation is False
    v.validate_loss(float("nan"))  # disabled -> no raise
    v.enable()
    assert v.enable_validation is True
    with pytest.raises(NLSQNumericalError):
        v.validate_loss(float("nan"))
