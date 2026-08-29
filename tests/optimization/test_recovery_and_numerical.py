"""Tests for numerical-validation helpers.

``numerical_validation`` raises ``NLSQNumericalError`` (tagged with a
detection point) on NaN/Inf in gradients, parameters, or loss, and on
bounds violations — tests cover each detection point plus the
disabled-validation fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.exceptions import NLSQNumericalError
from xpcsjax.optimization.numerical_validation import NumericalValidator

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
