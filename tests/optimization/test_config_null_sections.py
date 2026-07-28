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


# --- Scalar (not section-level) YAML nulls -----------------------------------
# `null`/blank-line YAML means "unset"; `.get(key, default)` only substitutes
# the default for an ABSENT key, so these used to raise TypeError from a bare
# float()/int() instead of falling back.


def test_nlsq_config_null_scalar_fields_fall_back_to_defaults():
    cfg = NLSQConfig.from_dict(
        {"trust_region_scale": None, "ftol": None, "xtol": None, "gtol": None}
    )
    assert (cfg.trust_region_scale, cfg.ftol, cfg.xtol, cfg.gtol) == (1.0, 1e-8, 1e-8, 1e-8)


def test_nlsq_config_null_nested_scalar_fields_fall_back_to_defaults():
    cfg = NLSQConfig.from_dict(
        {
            "cmaes": {"sigma": None, "tol_fun": None},
            "hybrid_streaming": {"warmup_learning_rate": None},
            "multi_start": {"refinement_ftol": None},
            "quality_validation": {"bounds_tolerance": None},
        }
    )
    assert cfg.cmaes_sigma == 0.5
    assert cfg.cmaes_tol_fun == 1e-8
    assert cfg.hybrid_warmup_learning_rate == 0.001
    assert cfg.multi_start_refinement_ftol == 1e-12
    assert cfg.quality_bounds_tolerance == 1e-9


def test_nlsq_config_ftol_still_falls_back_to_tolerance_alias():
    """The `ftol -> tolerance -> 1e-8` chain must survive the safe_float rewrite."""
    assert NLSQConfig.from_dict({"tolerance": 1e-6}).ftol == 1e-6
    assert NLSQConfig.from_dict({}).ftol == 1e-8


def test_nlsq_config_cmaes_none_sentinels_stay_none():
    """`max_generations`/`popsize`/`seed` use None as an ADAPTIVE sentinel --
    they must not be coerced to 0 by a blanket safe_int sweep."""
    cfg = NLSQConfig.from_dict({"cmaes": {"max_generations": None, "popsize": None, "seed": None}})
    assert (cfg.cmaes_max_generations, cfg.cmaes_popsize, cfg.cmaes_seed) == (None, None, None)


def test_extract_nlsq_settings_survives_null_sections():
    """`optimization:` / `nlsq:` with no body parse to None, not {}."""
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

    assert NLSQWrapper._extract_nlsq_settings({"optimization": None}) == {}
    assert NLSQWrapper._extract_nlsq_settings({"optimization": {"nlsq": None}}) == {}
