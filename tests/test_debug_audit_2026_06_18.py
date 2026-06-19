"""Regression tests for the 2026-06-18 whole-codebase debug-audit fixes.

Each test pins a single confirmed finding from the adversarially-verified
audit sweep. Grouped by the module/domain the fix touched.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Config null-section robustness — a bare ``key:`` in YAML parses to ``None``;
# ``.get(key, {})`` returns that ``None`` (the {} default only applies to a
# MISSING key), so downstream membership/sub-.get crashed. (manager.py /
# parameter_manager.py null-deref cluster.)
# ---------------------------------------------------------------------------
def test_config_manager_null_experimental_data() -> None:
    from xpcsjax.config import ConfigManager

    # Crashed at construction (_normalize_schema -> experimental_data block).
    cm = ConfigManager(config_override={"analysis_mode": "laminar_flow", "experimental_data": None})
    assert cm is not None


def test_config_manager_null_optimization_angle_ranges() -> None:
    from xpcsjax.config import ConfigManager

    cm = ConfigManager(config_override={"analysis_mode": "laminar_flow", "optimization": None})
    assert cm.get_target_angle_ranges() == {"enabled": False}


def test_config_manager_null_initial_parameters_per_angle() -> None:
    from xpcsjax.config import ConfigManager

    cm = ConfigManager(
        config_override={"analysis_mode": "laminar_flow", "initial_parameters": None}
    )
    # Must not raise AttributeError on the null section.
    assert cm.validate_per_angle_scaling(5) is not None


def test_parameter_manager_null_parameter_space() -> None:
    from xpcsjax.config.parameter_manager import ParameterManager
    from xpcsjax.config.parameter_registry import AnalysisMode

    # Crashed at construction via _load_config_bounds.
    pm = ParameterManager({"parameter_space": None}, AnalysisMode.LAMINAR_FLOW)
    assert pm is not None


def test_parameter_manager_null_initial_parameters() -> None:
    from xpcsjax.config.parameter_manager import ParameterManager
    from xpcsjax.config.parameter_registry import AnalysisMode

    pm = ParameterManager({"initial_parameters": None}, AnalysisMode.LAMINAR_FLOW)
    # Both accessors dereferenced the null section before the fix.
    assert pm.get_active_parameters() is not None
    assert pm.get_fixed_parameters() == {}


# ---------------------------------------------------------------------------
# Case-variant analysis_mode must not warn "Unknown analysis_mode" — the
# validator allowlist is lowercase, but AnalysisMode.parse() lowercases, so a
# raw "HETERODYNE" is valid. (manager.py:1006-1012)
# ---------------------------------------------------------------------------
def test_validate_config_accepts_case_variant_mode(caplog: pytest.LogCaptureFixture) -> None:
    from xpcsjax.config import ConfigManager

    cm = ConfigManager(config_override={"analysis_mode": "laminar_flow"})
    # Simulate the file-load path where the validator sees the RAW (un-normalized)
    # mode string in a non-canonical case.
    cm.config["analysis_mode"] = "HETERODYNE"
    with caplog.at_level(logging.WARNING):
        cm._validate_config()
    assert not any("Unknown analysis_mode" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Modulo-by-zero guards on user-controllable interval config keys.
# ---------------------------------------------------------------------------
def test_shear_weighting_update_frequency_zero_no_crash() -> None:
    from xpcsjax.optimization.nlsq.shear_weighting import (
        ShearSensitivityWeighting,
        ShearWeightingConfig,
    )

    cfg = ShearWeightingConfig(enable=True, update_frequency=0)
    weighter = ShearSensitivityWeighting(
        phi_angles=np.array([0.0, 45.0, 90.0]),
        n_physical=7,
        phi0_index=4,
        config=cfg,
    )
    # params layout: [2*n_phi per-angle | n_physical physical] = 6 + 7 = 13
    weighter.update_phi0(np.ones(13), iteration=1)  # would ZeroDivisionError pre-fix


def test_gradient_monitor_check_interval_zero_no_crash() -> None:
    from xpcsjax.optimization.nlsq.gradient_monitor import (
        GradientCollapseMonitor,
        GradientMonitorConfig,
    )

    cfg = GradientMonitorConfig(enable=True, check_interval=0)
    monitor = GradientCollapseMonitor(cfg, physical_indices=[2], per_angle_indices=[0, 1])
    status = monitor.check(np.ones(3), iteration=1)  # would ZeroDivisionError pre-fix
    assert status in {"OK", "WARNING", "COLLAPSE_DETECTED"}


# ---------------------------------------------------------------------------
# get_optimal_batch_size must never exceed the dataset size. (device/cpu.py:499)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("data_size", [1, 200, 999, 1000, 5000])
def test_optimal_batch_size_never_exceeds_dataset(data_size: int) -> None:
    from xpcsjax.device.cpu import get_optimal_batch_size

    # Large memory budget so the memory-derived size is huge and the dataset
    # cap is what must bind.
    batch = get_optimal_batch_size(data_size, available_memory_gb=64.0)
    assert 1 <= batch <= data_size


# ---------------------------------------------------------------------------
# detect_degeneracy.basin_labels must align 1:1 with the input results
# (length == len(results); index i corresponds to results[i]). (multistart.py)
# ---------------------------------------------------------------------------
def test_detect_degeneracy_basin_labels_align_with_results() -> None:
    from xpcsjax.optimization.nlsq import multistart as ms

    def _single(chi2: float, success: bool) -> ms.SingleStartResult:
        return ms.SingleStartResult(
            start_idx=0,
            initial_params=np.array([1.0, 1.0]),
            final_params=np.array([1.0, 1.0]),
            chi_squared=chi2,
            reduced_chi_squared=chi2,
            success=success,
        )

    # A failed result in the middle: pre-fix, basin_labels had length == #success
    # and chi2-sorted order, mismatching all_results.
    results = [
        _single(1.0, success=True),
        _single(np.inf, success=False),
        _single(1.01, success=True),
    ]
    _, _, labels = ms.detect_degeneracy(results, chi_sq_threshold=0.1, param_threshold=0.2)
    assert labels is not None
    assert len(labels) == len(results)
    # The failed result carries the -1 sentinel, never a basin id.
    assert labels[1] == -1


# ---------------------------------------------------------------------------
# Nested cmaes.restart_strategy / cmaes.max_restarts must be parsed by
# NLSQConfig.from_dict (heterodyne config). (heterodyne_config.py:1006-1045)
# ---------------------------------------------------------------------------
def test_nested_cmaes_restart_keys_parsed() -> None:
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

    cfg = NLSQConfig.from_dict(
        {"cmaes": {"enable": True, "restart_strategy": "none", "max_restarts": 0}}
    )
    assert cfg.cmaes_restart_strategy == "none"
    assert cfg.cmaes_max_restarts == 0


# ---------------------------------------------------------------------------
# start_frame at/after the last frame must raise loudly, not return an empty
# (n_phi, 0, 0) stack. (xpcs_loader.py:2099-2138)
# ---------------------------------------------------------------------------
def test_frame_slicing_rejects_empty_window() -> None:
    from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader

    loader = XPCSDataLoader.__new__(XPCSDataLoader)  # bypass __init__
    loader.analyzer_config = {"start_frame": 5, "end_frame": -1}
    c2 = np.ones((2, 4, 4))  # 4 frames; start_frame=5 -> empty window
    with pytest.raises(XPCSDataFormatError, match="empty"):
        loader._apply_frame_slicing_to_selected_q(c2)


def test_frame_slicing_valid_window_ok() -> None:
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"start_frame": 2, "end_frame": -1}
    c2 = np.ones((2, 4, 4))
    out = loader._apply_frame_slicing_to_selected_q(c2)
    assert out.shape == (2, 3, 3)  # frames [1:4]


# ---------------------------------------------------------------------------
# Codex adversarial-review follow-ups (result_saving.py).
# ---------------------------------------------------------------------------
# A 0-D scalar uncertainty array passes OptimizationResult.__post_init__
# (ndim==0), and `uncertainties[i]` raises IndexError on a 0-D array even though
# `size == 1`. _extract_parameters must normalize via .ravel() and only use the
# uncertainty when its length matches the parameter count exactly.
def test_extract_parameters_scalar_uncertainties_dropped_no_crash() -> None:
    from xpcsjax.cli.result_saving import _extract_parameters
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    result = OptimizationResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        uncertainties=np.array(0.1),  # 0-D scalar  # type: ignore[arg-type]
        covariance=None,  # type: ignore[arg-type]
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )
    out = _extract_parameters(result, ["a", "b", "c"])
    # size 1 != 3 params -> uncertainty dropped to None, no IndexError.
    assert [out[k]["uncertainty"] for k in ("a", "b", "c")] == [None, None, None]
    assert out["b"]["value"] == 2.0


def test_extract_parameters_scalar_uncertainty_matches_single_param() -> None:
    from xpcsjax.cli.result_saving import _extract_parameters
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    # A size-1 (flattened) uncertainty that matches a single parameter is used.
    result = OptimizationResult(
        parameters=np.array([5.0]),
        uncertainties=np.array(0.1),  # 0-D scalar  # type: ignore[arg-type]
        covariance=None,  # type: ignore[arg-type]
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )
    out = _extract_parameters(result, ["a"])
    assert out["a"]["uncertainty"] == pytest.approx(0.1)


def test_save_results_both_writes_durable_npz_before_json(tmp_path, monkeypatch) -> None:
    """For output_format='both', the durable NPZ must be written before the JSON,
    so a human-readable serialization failure cannot discard the numeric artifact.
    """
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service import (
        persist as result_saving,  # was: from xpcsjax.cli import result_saving
    )

    result = OptimizationResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        uncertainties=None,  # type: ignore[arg-type]
        covariance=None,  # type: ignore[arg-type]
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )

    def _boom(*_a, **_k):
        raise RuntimeError("json serialization failed")

    monkeypatch.setattr(result_saving, "save_results_json", _boom)
    with pytest.raises(RuntimeError, match="json serialization failed"):
        result_saving.save_results(result, tmp_path, "both", None, None)
    # The durable NPZ must already be on disk despite the JSON failure.
    assert list(tmp_path.glob("*.npz")), "NPZ must be written before JSON for output_format='both'"
