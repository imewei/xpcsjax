"""Regression: cmaes.seed in YAML config must actually reach
CMAESWrapperConfig.seed, not silently no-op.

Finding #5 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig
from xpcsjax.optimization.nlsq.config import NLSQConfig


def test_cmaes_seed_field_exists_with_none_default():
    config = NLSQConfig()
    assert config.cmaes_seed is None


def test_cmaes_seed_parsed_from_dict():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    assert config.cmaes_seed == 123


def test_cmaes_seed_round_trips_through_to_dict():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    d = config.to_dict()
    assert d["cmaes"]["seed"] == 123


def test_cmaes_seed_reaches_wrapper_config():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    wrapper_config = CMAESWrapperConfig.from_nlsq_config(config)
    assert wrapper_config.seed == 123


def test_cmaes_seed_defaults_to_none_when_unset():
    config = NLSQConfig.from_dict({})
    wrapper_config = CMAESWrapperConfig.from_nlsq_config(config)
    assert wrapper_config.seed is None
