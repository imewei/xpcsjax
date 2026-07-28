"""Regression tests for null / non-finite handling in the config subsystem.

Covers a coherent cluster of defects found by the 2026-06-20 ultracode
config-module debug pass — values that are ``None`` (from a blank YAML key) or
non-finite (``NaN`` / ``inf``) silently corrupting or crashing config handling:

* homodyne ``physics_validators`` silently ACCEPTED ``NaN`` parameters because
  IEEE-754 makes every relational comparison with ``NaN`` return ``False``;
* heterodyne ``heterodyne_physics_validators`` caught ``NaN`` only for the
  ``f0`` / ``f3`` range rules (by accident) and missed it everywhere else;
* :meth:`ConfigManager.update_config` raised a cryptic ``TypeError`` when an
  intermediate dot-notation section held a null YAML value;
* :meth:`ParameterManager._load_config_bounds` raised ``TypeError`` from
  ``float(None)`` when a ``parameter_space.bounds`` entry left ``min``/``max``
  blank.
"""

from __future__ import annotations

import math

import pytest

from xpcsjax.config import heterodyne_physics_validators as het_pv
from xpcsjax.config import physics_validators as pv
from xpcsjax.config.manager import ConfigManager
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


def _has_error(violations) -> bool:
    return any(str(v.severity.value) == "error" for v in violations)


# --------------------------------------------------------------------------- #
# #5 — homodyne validators must flag non-finite values as ERROR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("param", ["D0", "contrast", "offset", "alpha"])
def test_homodyne_nan_is_error(param: str) -> None:
    violations = pv.validate_single_parameter(param, math.nan)
    assert _has_error(violations), f"NaN for {param!r} must raise an ERROR violation"


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_homodyne_inf_is_error(value: float) -> None:
    violations = pv.validate_single_parameter("D0", value)
    assert _has_error(violations), "±inf must raise an ERROR violation"


def test_homodyne_finite_value_still_passes() -> None:
    # A healthy value triggers no error (guards against over-flagging).
    assert not _has_error(pv.validate_single_parameter("D0", 1.0e3))


# --------------------------------------------------------------------------- #
# #7 — heterodyne validators must flag non-finite uniformly (not just f0/f3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("param", ["v0", "alpha_ref", "f0"])
def test_heterodyne_nan_is_error(param: str) -> None:
    violations = het_pv.validate_single_parameter(param, math.nan)
    assert _has_error(violations), f"NaN for {param!r} must raise an ERROR violation"


def test_heterodyne_finite_value_still_passes() -> None:
    assert not _has_error(het_pv.validate_single_parameter("v0", 1.0))


# --------------------------------------------------------------------------- #
# #2 / #4 — update_config must handle a null intermediate section
# --------------------------------------------------------------------------- #
def test_update_config_null_intermediate_creates_mapping() -> None:
    cfg = ConfigManager(config_override={"analysis_mode": "static_isotropic", "optimization": None})
    cfg.update_config("optimization.method", "nlsq")
    assert cfg.config["optimization"] == {"method": "nlsq"}


def test_update_config_scalar_intermediate_raises_clearly() -> None:
    cfg = ConfigManager(config_override={"analysis_mode": "static_isotropic", "optimization": 42})
    with pytest.raises(TypeError, match="not a mapping"):
        cfg.update_config("optimization.method", "nlsq")


# --------------------------------------------------------------------------- #
# #3 — _load_config_bounds must tolerate a null bound (treat as unspecified)
# --------------------------------------------------------------------------- #
def test_null_bound_falls_back_to_default() -> None:
    config = {
        "analysis_mode": "static_anisotropic",
        "parameter_space": {
            "bounds": [{"name": "D0", "min": None, "max": 1.0e5}],
        },
    }
    pm = ParameterManager(config, AnalysisMode.STATIC_ANISOTROPIC)
    bound = pm._default_bounds["D0"]
    # The blank min is dropped → default preserved (finite float), not None.
    assert bound["min"] is not None
    assert isinstance(bound["min"], float)
    # The explicitly-set max is still applied.
    assert bound["max"] == pytest.approx(1.0e5)


# --------------------------------------------------------------------------- #
# _load_config_bounds must reject an inverted (min > max) interval
# --------------------------------------------------------------------------- #
def test_inverted_bounds_rejected() -> None:
    config = {
        "analysis_mode": "static_anisotropic",
        "parameter_space": {"bounds": [{"name": "D0", "min": 1.0e5, "max": 100.0}]},
    }
    with pytest.raises(ValueError, match="exceeds max"):
        ParameterManager(config, AnalysisMode.STATIC_ANISOTROPIC)


def test_one_sided_override_inverting_against_default_rejected() -> None:
    """Only 'min' supplied, inverting against the registry default max."""
    config = {
        "analysis_mode": "static_anisotropic",
        "parameter_space": {"bounds": [{"name": "D0", "min": 1.0e9}]},
    }
    with pytest.raises(ValueError, match="exceeds max"):
        ParameterManager(config, AnalysisMode.STATIC_ANISOTROPIC)
