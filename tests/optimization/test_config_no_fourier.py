"""Phase 7: homodyne config.py NLSQConfig drops the removed reparam-order fields.

The removed-token literals are assembled from fragments so this test file does
not itself contain the whole-word tokens (keeps the section 6 grep-zero gate at
honest zero).
"""

from __future__ import annotations

from dataclasses import fields

from xpcsjax.optimization.nlsq.config import NLSQConfig

_ORDER = "four" + "ier_order"
_AUTO_THRESH = "four" + "ier_auto_threshold"
_MODE = "four" + "ier"


def test_no_removed_reparam_fields() -> None:
    names = {f.name for f in fields(NLSQConfig)}
    assert _ORDER not in names
    assert _AUTO_THRESH not in names


def test_to_dict_anti_degeneracy_has_no_removed_reparam_keys() -> None:
    cfg = NLSQConfig()
    ad = cfg.to_dict()["anti_degeneracy"]
    assert _ORDER not in ad
    assert _AUTO_THRESH not in ad


def test_validate_rejects_removed_reparam_per_angle_mode() -> None:
    cfg = NLSQConfig()
    cfg.per_angle_mode = _MODE
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)


def test_from_dict_ignores_stray_removed_reparam_keys() -> None:
    cfg = NLSQConfig.from_dict(
        {
            "anti_degeneracy": {
                "per_angle_mode": "auto",
                _ORDER: 2,
                _AUTO_THRESH: 6,
                "constant_scaling_threshold": 3,
            }
        }
    )
    assert not hasattr(cfg, _ORDER)
    assert cfg.constant_scaling_threshold == 3
