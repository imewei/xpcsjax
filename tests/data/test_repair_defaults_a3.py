"""Regression test for review finding A3 (2026-09-15).

``_repair_scaling_issues`` rescaled the whole c2 stack by /10//100/*10 based
on a mean-value heuristic that its own comment admitted could corrupt valid
raw-count data - it has been deleted outright (no safe setting exists).
``repair_nan_values`` used to default to True, so every fit silently
median-filled non-finite correlation values unless a user explicitly opted
out; it now defaults to False, and an explicit repair logs at WARNING with
a count instead of being invisible.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from xpcsjax.data.quality_controller import DataQualityController, QualityControlConfig


def test_repair_nan_values_defaults_to_false():
    cfg = QualityControlConfig()
    assert cfg.repair_nan_values is False


def test_repair_scaling_issues_field_removed():
    cfg = QualityControlConfig()
    assert not hasattr(cfg, "repair_scaling_issues")


def test_repair_scaling_issues_method_removed():
    assert not hasattr(DataQualityController, "_repair_scaling_issues")


def test_repair_nan_values_disabled_by_default_leaves_data_untouched():
    controller = DataQualityController.__new__(DataQualityController)
    controller.quality_config = QualityControlConfig.from_config_dict({})
    assert controller.quality_config.repair_nan_values is False

    data = {"c2_exp": np.array([[1.0, np.nan], [np.nan, 1.0]])}
    repairs_applied: list[str] = []
    # Method itself still exists and works when explicitly invoked/enabled;
    # what changed is that the caller no longer reaches it by default.
    if controller.quality_config.repair_nan_values:
        controller._repair_nan_values(data, repairs_applied)
    assert np.isnan(data["c2_exp"]).any()


def test_repair_nan_values_logs_warning_with_count(caplog: pytest.LogCaptureFixture):
    controller = DataQualityController.__new__(DataQualityController)
    data = {
        "c2_exp": np.array([[1.0, np.nan, np.nan], [1.0, 1.0, 1.0]]).reshape(1, 2, 3),
    }
    repairs_applied: list[str] = []
    with caplog.at_level(logging.WARNING):
        modified = controller._repair_nan_values(data, repairs_applied)
    assert modified is True
    assert any(
        "replaced 2 non-finite value" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )
