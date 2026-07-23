"""Regression: present-but-null nested YAML sections must not crash from_dict.

`config_dict.get("section", {})` only substitutes the default for a MISSING
key; a bare YAML header or explicit `null` still yields None, and the
subsequent `.get()`/`float()`/`int()` chains crashed on it. Covers
NLSQConfig.from_dict (nlsq/config.py) and AntiDegeneracyConfig.from_dict
(nlsq/anti_degeneracy_controller.py).
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import AntiDegeneracyConfig
from xpcsjax.optimization.nlsq.config import NLSQConfig


def test_nlsq_config_null_anti_degeneracy_section_falls_back_to_defaults():
    cfg = NLSQConfig.from_dict({"anti_degeneracy": None})
    assert cfg.enable_hierarchical is True  # default, not a crash


def test_nlsq_config_null_hierarchical_subsection_falls_back_to_defaults():
    cfg = NLSQConfig.from_dict({"anti_degeneracy": {"hierarchical": None}})
    assert cfg.hierarchical_outer_tolerance == 1e-6


def test_anti_degeneracy_config_null_scaling_threshold_falls_back_to_default():
    cfg = AntiDegeneracyConfig.from_dict({"constant_scaling_threshold": None})
    assert cfg.constant_scaling_threshold == 3  # DEFAULT_CONSTANT_SCALING_THRESHOLD


def test_anti_degeneracy_config_null_hierarchical_tolerance_falls_back_to_default():
    cfg = AntiDegeneracyConfig.from_dict({"hierarchical": {"outer_tolerance": None}})
    assert cfg.hierarchical_outer_tolerance == 1e-6


def test_anti_degeneracy_config_null_hierarchical_section_falls_back_to_defaults():
    cfg = AntiDegeneracyConfig.from_dict({"hierarchical": None})
    assert cfg.hierarchical_outer_tolerance == 1e-6
