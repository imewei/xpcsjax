"""Tests for xpcsjax.optimization.nlsq.heterodyne_config.

Covers the safe type-conversion helpers, ``HybridRecoveryConfig`` retry
scaling, every ``__post_init__`` invariant and ``validate()`` error branch, the
nested ``from_dict`` parsing (anti_degeneracy / cmaes / recovery / validation /
x_scale_map sub-dicts, alias keys, non-dict guards), and ``to_dict`` round-trip.
"""

from __future__ import annotations

import warnings

import pytest

from xpcsjax.optimization.nlsq import heterodyne_config as hc
from xpcsjax.optimization.nlsq.heterodyne_config import (
    HybridRecoveryConfig,
    NLSQConfig,
    NLSQValidationConfig,
)

# ---------------------------------------------------------------------------
# safe_float / safe_int
# ---------------------------------------------------------------------------


def test_safe_float() -> None:
    assert hc.safe_float(None, 1.5) == 1.5
    assert hc.safe_float("2.5", 0.0) == 2.5
    assert hc.safe_float(3, 0.0) == 3.0
    assert hc.safe_float("notanumber", 9.0) == 9.0  # conversion fails -> default


def test_safe_int() -> None:
    assert hc.safe_int(None, 7) == 7
    assert hc.safe_int("4", 0) == 4
    assert hc.safe_int(5.0, 0) == 5
    assert hc.safe_int("xx", 3) == 3  # conversion fails -> default


# ---------------------------------------------------------------------------
# HybridRecoveryConfig
# ---------------------------------------------------------------------------


def test_retry_settings_baseline_and_scaled() -> None:
    rc = HybridRecoveryConfig(lr_decay=0.5, lambda_growth=10.0, trust_decay=0.5)
    base = rc.get_retry_settings(0)
    assert base == {"lr_scale": 1.0, "lambda_scale": 1.0, "trust_radius_scale": 1.0}
    s2 = rc.get_retry_settings(2)
    assert s2["lr_scale"] == pytest.approx(0.25)
    assert s2["lambda_scale"] == pytest.approx(100.0)
    assert s2["trust_radius_scale"] == pytest.approx(0.25)


def test_retry_settings_negative_raises() -> None:
    with pytest.raises(ValueError, match="attempt must be >= 0"):
        HybridRecoveryConfig().get_retry_settings(-1)


# ---------------------------------------------------------------------------
# NLSQConfig construction + __post_init__
# ---------------------------------------------------------------------------


def test_defaults_construct_and_validate_clean() -> None:
    cfg = NLSQConfig()
    assert cfg.n_params == 14
    assert cfg.analysis_mode == "two_component"
    assert cfg.validate() == []  # default config is internally consistent


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iterations": 0}, "max_iterations must be >= 1"),
        ({"tolerance": 0.0}, "tolerance must be positive"),
        ({"multistart_n": 0}, "multistart_n must be >= 1"),
        ({"streaming_chunk_size": 0}, "streaming_chunk_size must be >= 1"),
        ({"target_chunk_size": 0}, "target_chunk_size must be >= 1"),
        ({"max_recovery_attempts": -1}, "max_recovery_attempts must be >= 0"),
        ({"loss_scale": 0.0}, "loss_scale must be positive"),
        ({"hierarchical_max_outer_iterations": 0}, "hierarchical_max_outer_iterations"),
        ({"gradient_consecutive_triggers": 0}, "gradient_consecutive_triggers must be >= 1"),
        ({"cmaes_sigma0": 0.0}, "cmaes_sigma0 must be > 0"),
        ({"cmaes_diagonal_filtering": "bad"}, "cmaes_diagonal_filtering must be"),
        ({"cmaes_warmstart_skip_threshold": 0.0}, "cmaes_warmstart_skip_threshold must be > 0"),
        ({"cmaes_restart_strategy": "bad"}, "cmaes_restart_strategy must be"),
        ({"cmaes_max_restarts": -1}, "cmaes_max_restarts must be >= 0"),
        ({"hybrid_warmup_fraction": 1.5}, "hybrid_warmup_fraction must be in"),
        ({"screen_keep_fraction": 2.0}, "screen_keep_fraction must be in"),
        ({"refine_top_k": 0}, "refine_top_k must be >= 1"),
        ({"constant_scaling_threshold": 10}, "constant_scaling_threshold"),
    ],
)
def test_post_init_invariants(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NLSQConfig(**kwargs)


def test_post_init_deprecation_alias_independent() -> None:
    with pytest.warns(DeprecationWarning, match="independent"):
        cfg = NLSQConfig(per_angle_mode="independent")
    assert cfg.per_angle_mode == "individual"  # normalized


# ---------------------------------------------------------------------------
# validate() — fields not enforced in __post_init__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"workflow": "bogus"}, "workflow"),
        ({"goal": "bogus"}, "goal"),
        ({"analysis_mode": "bogus"}, "analysis_mode"),
        ({"per_angle_mode": "bogus"}, "per_angle_mode"),
        ({"fourier_order": 0}, "fourier_order"),
        ({"regularization_mode": "bogus"}, "regularization_mode"),
        ({"hybrid_method": "bogus"}, "hybrid_method"),
        ({"sampling_strategy": "bogus"}, "sampling_strategy"),
        ({"nlsq_stability": "bogus"}, "nlsq_stability"),
        ({"nlsq_memory_fraction": 2.0}, "nlsq_memory_fraction"),
        ({"nlsq_memory_fallback_gb": 0.0}, "nlsq_memory_fallback_gb"),
    ],
)
def test_validate_reports_errors(kwargs: dict, fragment: str) -> None:
    cfg = NLSQConfig(**kwargs)
    errors = cfg.validate()
    assert any(fragment in e for e in errors), errors


def test_validate_fourier_auto_threshold() -> None:
    # fourier_auto_threshold < 1, kept consistent with constant_scaling_threshold.
    cfg = NLSQConfig(fourier_auto_threshold=0, constant_scaling_threshold=-5)
    assert any("fourier_auto_threshold" in e for e in cfg.validate())


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_flat_scalars() -> None:
    cfg = NLSQConfig.from_dict(
        {
            "max_iterations": 500,
            "tolerance": 1e-6,
            "method": "lm",
            "multistart": True,
            "diff_step": None,
            "max_nfev": 200,
            "tr_solver": None,
            "loss_weights": [1.0, 2.0],
            "x_scale": [1.0, 2.0, 3.0],
        }
    )
    assert cfg.max_iterations == 500
    assert cfg.tolerance == pytest.approx(1e-6)
    assert cfg.method == "lm"
    assert cfg.multistart is True
    assert cfg.diff_step is None
    assert cfg.max_nfev == 200
    assert cfg.loss_weights == [1.0, 2.0]
    assert cfg.x_scale == [1.0, 2.0, 3.0]


def test_from_dict_nested_anti_degeneracy() -> None:
    cfg = NLSQConfig.from_dict(
        {
            "anti_degeneracy": {
                "per_angle_mode": "fourier",
                "fourier_order": 3,
                "constant_scaling_threshold": 2,
                "hierarchical": {"enable": True, "max_outer_iterations": 15},
                "regularization": {"mode": "tikhonov", "lambda": 0.05},
                "gradient_monitoring": {"enable": True, "consecutive_triggers": 5},
            }
        }
    )
    assert cfg.per_angle_mode == "fourier"
    assert cfg.fourier_order == 3
    assert cfg.enable_hierarchical is True
    assert cfg.hierarchical_max_outer_iterations == 15
    assert cfg.regularization_mode == "tikhonov"
    assert cfg.group_variance_lambda == pytest.approx(0.05)
    assert cfg.enable_gradient_monitoring is True
    assert cfg.gradient_consecutive_triggers == 5


def test_from_dict_cmaes_alias_keys() -> None:
    cfg = NLSQConfig.from_dict(
        {"cmaes": {"enable": True, "sigma": 0.5, "max_generations": 200, "popsize": 16}}
    )
    assert cfg.enable_cmaes is True
    assert cfg.cmaes_sigma0 == pytest.approx(0.5)  # 'sigma' alias
    assert cfg.cmaes_max_iterations == 200  # 'max_generations' alias
    assert cfg.cmaes_population_size == 16  # 'popsize' alias


def test_from_dict_known_ignored_sections_do_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Canonical optimization.nlsq template sections the heterodyne adapter
    # doesn't translate must not emit "unrecognised key" warnings — that noise
    # fired on every two_component run and obscured genuine typos.
    cfg_dict = {
        "memory_fraction": 0.75,
        "trust_region_scale": 1.0,
        "progress": {"enable": True},
        "diagnostics": {"enabled": False},
        "multi_start": {"enable": False},
        "hybrid_streaming": {"enable": True},
        "quality_validation": {"enable": True},
    }
    with caplog.at_level("WARNING", logger="xpcsjax.optimization.nlsq.heterodyne_config"):
        NLSQConfig.from_dict(cfg_dict)
    assert not [r for r in caplog.records if "unrecognised key" in r.getMessage()]


def test_from_dict_genuine_typo_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    # The warning must still catch real typos so users learn their key was dropped.
    with caplog.at_level("WARNING", logger="xpcsjax.optimization.nlsq.heterodyne_config"):
        NLSQConfig.from_dict({"max_iterationz": 500})
    assert any(
        "unrecognised key" in r.getMessage() and "max_iterationz" in r.getMessage()
        for r in caplog.records
    )


def test_from_dict_nested_recovery_and_validation() -> None:
    cfg = NLSQConfig.from_dict(
        {
            "recovery": {"max_retries": 5, "lambda_growth": 20.0},
            "validation": {"chi2_warn_high": 3.0, "correlation_warn": 0.9},
            "x_scale_map": {"D0": 2.0, "alpha": "3.0"},
        }
    )
    assert cfg.recovery_config.max_retries == 5
    assert cfg.recovery_config.lambda_growth == pytest.approx(20.0)
    assert cfg.validation.chi2_warn_high == pytest.approx(3.0)
    assert cfg.x_scale_map == {"D0": 2.0, "alpha": 3.0}


def test_from_dict_unrecognized_key_ignored() -> None:
    cfg = NLSQConfig.from_dict({"totally_unknown_key": 1, "max_iterations": 100})
    assert cfg.max_iterations == 100  # known key still applied, unknown ignored


def test_from_dict_non_dict_nested_guards() -> None:
    # Non-dict sub-sections are warned about and ignored, not crashed on.
    cfg = NLSQConfig.from_dict(
        {"anti_degeneracy": "not-a-dict", "cmaes": 5, "recovery": [1], "validation": "x"}
    )
    assert isinstance(cfg, NLSQConfig)  # falls back to defaults


# ---------------------------------------------------------------------------
# to_dict + round trip
# ---------------------------------------------------------------------------


def test_to_dict_has_nested_sections() -> None:
    d = NLSQConfig().to_dict()
    assert "recovery" in d and "validation" in d
    assert d["recovery"]["max_retries"] == HybridRecoveryConfig().max_retries
    assert d["validation"]["chi2_warn_high"] == NLSQValidationConfig().chi2_warn_high


def test_from_dict_to_dict_roundtrip_preserves_fields() -> None:
    original = NLSQConfig(
        max_iterations=321,
        cmaes_sigma0=0.42,
        per_angle_mode="fourier",
        recovery_config=HybridRecoveryConfig(max_retries=7),
        validation=NLSQValidationConfig(chi2_fail_high=15.0),
    )
    restored = NLSQConfig.from_dict(original.to_dict())
    assert restored.max_iterations == 321
    assert restored.cmaes_sigma0 == pytest.approx(0.42)
    assert restored.per_angle_mode == "fourier"
    assert restored.recovery_config.max_retries == 7
    assert restored.validation.chi2_fail_high == pytest.approx(15.0)


def test_advisory_warning_does_not_error() -> None:
    # Default gtol=1e-8 with soft_l1 loss triggers the advisory log path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = NLSQConfig(gtol=1e-9, loss="soft_l1").validate()
    assert errors == []  # advisory is a log, not an error
