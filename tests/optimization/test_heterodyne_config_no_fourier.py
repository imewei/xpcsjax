"""Phase 7: heterodyne NLSQConfig drops fourier/independent entirely.

- PerAngleMode Literal no longer admits 'fourier' or 'independent'.
- fourier_order / fourier_auto_threshold dataclass fields are gone.
- The nested-dict parser ignores stray fourier_* keys (does not crash, does not
  resurrect the fields).
- to_dict() round-trip emits no fourier_* keys.
- An explicit per_angle_mode='fourier' fails validate().
"""

from __future__ import annotations

from dataclasses import fields

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


def test_no_fourier_fields_on_dataclass() -> None:
    names = {f.name for f in fields(NLSQConfig)}
    assert "fourier_order" not in names
    assert "fourier_auto_threshold" not in names


def test_to_dict_has_no_fourier_keys() -> None:
    cfg = NLSQConfig(analysis_mode="two_component")
    d = cfg.to_dict()
    assert "fourier_order" not in d
    assert "fourier_auto_threshold" not in d
    assert d["per_angle_mode"] in ("auto", "constant", "averaged", "individual")


def test_validate_rejects_fourier_mode() -> None:
    cfg = NLSQConfig(analysis_mode="two_component")
    cfg.per_angle_mode = "fourier"  # bypass __post_init__
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)


def test_from_dict_ignores_stray_fourier_keys() -> None:
    # A legacy YAML block still carrying fourier_* must not crash or resurrect them.
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "anti_degeneracy": {
                "per_angle_mode": "auto",
                "fourier_order": 2,
                "fourier_auto_threshold": 6,
                "constant_scaling_threshold": 3,
            },
        }
    )
    assert not hasattr(cfg, "fourier_order")
    assert cfg.constant_scaling_threshold == 3


def test_independent_alias_removed() -> None:
    # 'independent' is no longer accepted (alias + DeprecationWarning deleted).
    cfg = NLSQConfig(analysis_mode="two_component")
    cfg.per_angle_mode = "independent"
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)
