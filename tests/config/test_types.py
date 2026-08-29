"""Coverage for xpcsjax/config/types.py's coerce_finite_float bool guard.

bool is an int subclass in Python, so float(True)/float(False) silently
coerce to 1.0/0.0 -- both finite -- unless explicitly rejected first. This
function is the shared load-bearing boundary for parameter bounds/values
across parameter_manager.py, parameter_space.py, heterodyne_parameter_space.py,
and manager.py.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.types import coerce_finite_float


def test_coerce_finite_float_rejects_bool_true() -> None:
    with pytest.raises(ValueError, match="boolean"):
        coerce_finite_float(True, context="test.max")


def test_coerce_finite_float_rejects_bool_false() -> None:
    with pytest.raises(ValueError, match="boolean"):
        coerce_finite_float(False, context="test.min")


def test_coerce_finite_float_accepts_real_numbers() -> None:
    assert coerce_finite_float(1.5, context="test") == 1.5
    assert coerce_finite_float(0, context="test") == 0.0


def test_coerce_finite_float_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        coerce_finite_float(float("nan"), context="test")
    with pytest.raises(ValueError, match="non-finite"):
        coerce_finite_float(float("inf"), context="test")
