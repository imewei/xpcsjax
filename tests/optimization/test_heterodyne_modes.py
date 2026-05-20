"""Tests for heterodyne per-angle mode vocabulary parity with homodyne."""
from __future__ import annotations

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


def test_individual_mode_accepted() -> None:
    """`individual` is the canonical name (matches homodyne docs)."""
    cfg = NLSQConfig(per_angle_mode="individual")
    assert cfg.per_angle_mode == "individual"
    errors = cfg.validate()
    assert errors == [], f"expected no validation errors, got {errors}"
