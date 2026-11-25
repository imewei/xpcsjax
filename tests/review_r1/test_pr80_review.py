"""Tests added from the six-pass review of PR #80 (gaps in coverage of new behaviour)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager
from xpcsjax.optimization.nlsq.memory import MEMORY_FRACTION_ENV_VAR, config_memory_fraction


def _cfg(value):
    return {"optimization": {"nlsq": {"memory_fraction": value}}}


@pytest.mark.parametrize(
    "config, expected",
    [
        (_cfg(0.3), 0.3),
        (_cfg(1), 1.0),
        (SimpleNamespace(config=_cfg(0.25)), 0.25),  # ConfigManager-like
        (_cfg(True), None),  # a bool must not become 1.0
        (_cfg("0.5"), None),
        ({"optimization": None}, None),
        ({"optimization": {"nlsq": None}}, None),
        ({}, None),
        (None, None),
    ],
)
def test_config_memory_fraction_reads_only_a_numeric_config_value(monkeypatch, config, expected):
    monkeypatch.delenv(MEMORY_FRACTION_ENV_VAR, raising=False)
    assert config_memory_fraction(config) == expected


def test_config_memory_fraction_yields_to_the_env_var(monkeypatch):
    monkeypatch.setenv(MEMORY_FRACTION_ENV_VAR, "0.5")
    # None -> the selector reads the env var itself (the env "forces" the fraction).
    assert config_memory_fraction(_cfg(0.3)) is None


@pytest.mark.parametrize("flag, expected", [(0, False), (1, True), (False, False), (True, True)])
def test_heterodyne_vary_accepts_bool_and_zero_one(flag, expected):
    pm = HeterodyneParameterManager.from_config(
        {"parameters": {"velocity": {"v0": {"value": 500, "vary": flag}}}}
    )
    assert ("v0" in pm.varying_names) is expected


@pytest.mark.parametrize("flag", ["false", "true", 2, 0.5])
def test_heterodyne_vary_rejects_ambiguous_values(flag):
    with pytest.raises(ValueError, match="vary"):
        HeterodyneParameterManager.from_config(
            {"parameters": {"velocity": {"v0": {"value": 500, "vary": flag}}}}
        )


def test_heterodyne_residual_shape_guard_accepts_the_right_shape_only():
    from tests.review_r1.test_codex_core import _het_model

    model = _het_model()
    n = len(model.t)
    good = np.ones((n, n))
    residuals = np.asarray(model.compute_residuals(good, weights=good))
    assert residuals.size > 0 and np.all(np.isfinite(residuals))
    with pytest.raises(ValueError, match="weights"):
        model.compute_residuals(good, weights=np.ones((n, 1)))


def test_heterodyne_constraints_still_accept_documented_aliases():
    # The typo guard must run AFTER alias mapping: `v_beta` is the user-facing
    # spelling of the canonical `beta`.
    pm = HeterodyneParameterManager.from_config(
        {"initial_parameters": {"fixed_parameters": {"v_beta": 1.0}}}
    )
    assert "beta" not in pm.varying_names
    assert np.isclose(pm.get_parameter_dict()["beta"], 1.0)
