"""Equivalence tests for the typed NLSQ override application (F8)."""

from types import SimpleNamespace

import pytest

from xpcsjax.service.fit import FitOverrides, apply_overrides


def _cm(config: dict) -> SimpleNamespace:
    # apply_overrides only touches `.config`; a namespace stub is sufficient.
    return SimpleNamespace(config=config)


def test_multistart_sets_nested_enable_and_n_starts():
    cm = _cm({})
    apply_overrides(cm, FitOverrides(multistart=True, multistart_n=8))
    nlsq = cm.config["optimization"]["nlsq"]
    assert nlsq["multi_start"]["enable"] is True
    assert nlsq["multi_start"]["n_starts"] == 8


def test_tolerance_relaxes_ftol_xtol_and_gtol_together():
    cm = _cm({})
    apply_overrides(cm, FitOverrides(tolerance=1e-6))
    nlsq = cm.config["optimization"]["nlsq"]
    assert nlsq["ftol"] == 1e-6
    assert nlsq["xtol"] == 1e-6
    assert nlsq["gtol"] == 1e-6  # gtol relaxed too — see apply_cli_overrides rationale


def test_max_iterations_is_coerced_to_int():
    cm = _cm({})
    apply_overrides(cm, FitOverrides(max_iterations=1600))
    assert cm.config["optimization"]["nlsq"]["max_iterations"] == 1600


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [(False, False, None), (True, False, 2), (False, True, 0), (True, True, 0)],
)
def test_verbose_quiet_mapping(verbose, quiet, expected):
    # quiet wins over verbose (0); verbose->2; neither->key absent.
    cm = _cm({})
    apply_overrides(cm, FitOverrides(verbose=verbose, quiet=quiet))
    nlsq = cm.config.get("optimization", {}).get("nlsq", {})
    assert nlsq.get("verbose") == expected


def test_none_fields_write_nothing():
    cm = _cm({})
    apply_overrides(cm, FitOverrides())
    assert cm.config == {}


def test_non_dict_config_is_noop():
    cm = SimpleNamespace(config=None)
    apply_overrides(cm, FitOverrides(multistart=True))  # must not raise
    assert cm.config is None
