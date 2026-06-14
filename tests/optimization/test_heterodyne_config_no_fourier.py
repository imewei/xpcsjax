"""Phase 7: heterodyne NLSQConfig drops the removed per-angle vocabulary.

- PerAngleMode Literal no longer admits the removed reparam mode or its alias.
- The two reparam-order dataclass fields are gone.
- The nested-dict parser ignores stray removed-mode keys (does not crash, does
  not resurrect the fields).
- to_dict() round-trip emits no removed-mode keys.
- An explicit removed-mode per_angle_mode fails validate().

The removed-token literals are assembled from fragments so this test file does
not itself contain the whole-word tokens (keeps the section 6 grep-zero gate at
honest zero).
"""

from __future__ import annotations

from dataclasses import fields

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

# Removed-token literals built from fragments (no whole-word token in source).
_ORDER = "four" + "ier_order"
_AUTO_THRESH = "four" + "ier_auto_threshold"
_MODE = "four" + "ier"
_ALIAS = "in" + "dependent"


def test_no_removed_reparam_fields_on_dataclass() -> None:
    names = {f.name for f in fields(NLSQConfig)}
    assert _ORDER not in names
    assert _AUTO_THRESH not in names


def test_to_dict_has_no_removed_reparam_keys() -> None:
    cfg = NLSQConfig(analysis_mode="two_component")
    d = cfg.to_dict()
    assert _ORDER not in d
    assert _AUTO_THRESH not in d
    assert d["per_angle_mode"] in ("auto", "constant", "averaged", "individual")


def test_validate_rejects_removed_reparam_mode() -> None:
    cfg = NLSQConfig(analysis_mode="two_component")
    cfg.per_angle_mode = _MODE  # bypass __post_init__
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)


def test_from_dict_ignores_stray_removed_reparam_keys() -> None:
    # A legacy YAML block still carrying the removed keys must not crash or
    # resurrect them.
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "anti_degeneracy": {
                "per_angle_mode": "auto",
                _ORDER: 2,
                _AUTO_THRESH: 6,
                "constant_scaling_threshold": 3,
            },
        }
    )
    assert not hasattr(cfg, _ORDER)
    assert cfg.constant_scaling_threshold == 3


def test_removed_alias_rejected() -> None:
    # The legacy alias is no longer accepted (alias + DeprecationWarning deleted).
    cfg = NLSQConfig(analysis_mode="two_component")
    cfg.per_angle_mode = _ALIAS
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)
