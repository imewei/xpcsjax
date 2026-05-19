"""Smoke tests for :mod:`xpcsjax.data.quality_controller`.

The ``DataQualityController`` is wired into ``XPCSDataLoader.load_experimental_data``
at three pipeline stages (raw, filtered, final) but had zero direct test coverage.
These tests pin the public API surface — config construction, controller
instantiation, validate_data_stage at each stage, and the disabled-mode short
circuit — so a regression in any of those is caught without needing the
full real-data characterization run.
"""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data.quality_controller import (
    DataQualityController,
    QualityControlConfig,
    QualityControlResult,
    QualityControlStage,
    QualityLevel,
    create_quality_controller,
    validate_data_with_quality_control,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_data() -> dict:
    """A 4×4 single-angle dataset that satisfies the loader's shape contract."""
    return {
        "c2_exp": np.ones((1, 4, 4), dtype=np.float64) * 1.5,
        "t1": np.arange(4, dtype=np.float64),
        "t2": np.arange(4, dtype=np.float64),
        "wavevector_q_list": np.array([0.01], dtype=np.float64),
        "phi_angles_list": np.array([0.0], dtype=np.float64),
    }


def _baseline_config() -> dict:
    """Minimum config dict the controller's ``__init__`` reads from."""
    return {"quality_control": {"enabled": True, "validation_level": "basic"}}


# ---------------------------------------------------------------------------
# QualityControlConfig
# ---------------------------------------------------------------------------


def test_quality_config_defaults():
    cfg = QualityControlConfig()
    assert cfg.enabled is True
    assert cfg.validation_level == "standard"
    assert 0.0 <= cfg.pass_threshold <= cfg.warn_threshold <= cfg.excellent_threshold <= 100.0


def test_quality_config_from_dict_reads_quality_control_section():
    cfg = QualityControlConfig.from_config_dict(
        {"quality_control": {"enabled": False, "validation_level": "comprehensive"}}
    )
    assert cfg.enabled is False
    assert cfg.validation_level == "comprehensive"


def test_quality_config_from_empty_dict_uses_defaults():
    cfg = QualityControlConfig.from_config_dict({})
    assert cfg.enabled is True
    assert cfg.validation_level == "standard"


# ---------------------------------------------------------------------------
# DataQualityController construction
# ---------------------------------------------------------------------------


def test_controller_instantiates_from_config_dict():
    controller = DataQualityController(_baseline_config())
    assert controller.quality_config.enabled is True


def test_create_quality_controller_factory_returns_controller():
    controller = create_quality_controller(_baseline_config())
    assert isinstance(controller, DataQualityController)


def test_controller_exposes_stage_enum_via_attribute():
    """xpcs_loader uses ``quality_controller.QualityControlStage.RAW_DATA`` —
    the enum must be importable from the module's public surface."""
    assert QualityControlStage.RAW_DATA != QualityControlStage.FINAL_DATA
    assert hasattr(QualityControlStage, "FILTERED_DATA")
    assert hasattr(QualityControlStage, "FINAL_DATA")


# ---------------------------------------------------------------------------
# validate_data_stage — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    [
        QualityControlStage.RAW_DATA,
        QualityControlStage.FILTERED_DATA,
        QualityControlStage.FINAL_DATA,
    ],
)
def test_validate_data_stage_returns_result(stage):
    """Each pipeline stage must return a QualityControlResult without raising."""
    controller = DataQualityController(_baseline_config())
    result = controller.validate_data_stage(_minimal_data(), stage)
    assert isinstance(result, QualityControlResult)
    assert result.stage == stage


def test_validate_data_stage_disabled_short_circuits():
    """When the controller is disabled, every stage must return a minimal
    result without running the full validation pipeline."""
    controller = DataQualityController(
        {"quality_control": {"enabled": False}}
    )
    result = controller.validate_data_stage(
        _minimal_data(), QualityControlStage.RAW_DATA
    )
    assert isinstance(result, QualityControlResult)
    # Minimal results carry the stage but don't run quality checks.
    assert result.stage == QualityControlStage.RAW_DATA


def test_validate_data_with_quality_control_helper():
    """The module-level convenience function constructs the controller and
    validates in one call — used by callers that don't need the
    controller instance beyond a single stage."""
    result = validate_data_with_quality_control(
        _minimal_data(),
        _baseline_config(),
        stage=QualityControlStage.FINAL_DATA,
    )
    assert isinstance(result, QualityControlResult)
    assert result.stage == QualityControlStage.FINAL_DATA


# ---------------------------------------------------------------------------
# Quality level mapping
# ---------------------------------------------------------------------------


def test_quality_levels_are_valid_validation_levels():
    """QualityLevel maps to the ``validation_level`` config string set in
    QualityControlConfig (none / basic / standard / comprehensive)."""
    for name in ("NONE", "BASIC", "STANDARD", "COMPREHENSIVE"):
        assert hasattr(QualityLevel, name), f"missing QualityLevel.{name}"
    # And the value strings line up with the config-side validation_level keys.
    cfg = QualityControlConfig(validation_level=QualityLevel.STANDARD.value)
    assert cfg.validation_level == "standard"
