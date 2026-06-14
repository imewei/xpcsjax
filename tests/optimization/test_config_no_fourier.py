"""Phase 7: homodyne config.py NLSQConfig drops fourier_order / fourier_auto_threshold."""

from __future__ import annotations

from dataclasses import fields

from xpcsjax.optimization.nlsq.config import NLSQConfig


def test_no_fourier_fields() -> None:
    names = {f.name for f in fields(NLSQConfig)}
    assert "fourier_order" not in names
    assert "fourier_auto_threshold" not in names


def test_to_dict_anti_degeneracy_has_no_fourier() -> None:
    cfg = NLSQConfig()
    ad = cfg.to_dict()["anti_degeneracy"]
    assert "fourier_order" not in ad
    assert "fourier_auto_threshold" not in ad


def test_validate_rejects_fourier_per_angle_mode() -> None:
    cfg = NLSQConfig()
    cfg.per_angle_mode = "fourier"
    errs = cfg.validate()
    assert any("per_angle_mode" in e for e in errs)


def test_from_dict_ignores_stray_fourier_keys() -> None:
    cfg = NLSQConfig.from_dict(
        {"anti_degeneracy": {"per_angle_mode": "auto", "fourier_order": 2,
                             "fourier_auto_threshold": 6, "constant_scaling_threshold": 3}}
    )
    assert not hasattr(cfg, "fourier_order")
    assert cfg.constant_scaling_threshold == 3
