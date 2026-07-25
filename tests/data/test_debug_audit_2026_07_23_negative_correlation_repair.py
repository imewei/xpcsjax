"""Regression: negative-correlation repair must not clamp matrices that were
genuinely normalized (legitimate negatives), but must still clamp matrices
that hit the zero-variance/zero-IQR skip branch (never actually
transformed).

Finding #6 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.data.preprocessing import PreprocessingPipeline
from xpcsjax.data.quality_controller import DataQualityController


def _data_with_one_skipped_matrix(n_t: int = 6) -> dict:
    rng = np.random.default_rng(1)
    normal_matrix = 1.0 + 0.3 * rng.standard_normal((n_t, n_t))
    constant_matrix = np.full((n_t, n_t), 2.0)  # zero variance -> skip branch
    c2 = np.stack([normal_matrix, constant_matrix])
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01, 0.01]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }


def test_normalize_data_tracks_per_matrix_mask():
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "statistical"}}}}
    )
    data = _data_with_one_skipped_matrix()
    result = pipeline._normalize_data(data, {"method": "statistical"})
    mask = result.get("_normalized_mask")
    assert mask is not None
    assert list(mask) == [True, False], (
        "matrix 0 (real variance) must be marked normalized, "
        "matrix 1 (zero variance, skip branch) must not be"
    )


def test_repair_negative_correlations_respects_per_matrix_mask():
    controller = DataQualityController.__new__(DataQualityController)
    data = _data_with_one_skipped_matrix()
    # Manufacture negatives: one in the "normalized" matrix (legitimate,
    # e.g. z-score), one in the "skipped" matrix (never transformed, must
    # still be repaired).
    data["c2_exp"] = data["c2_exp"].astype(np.float64)
    data["c2_exp"][0, 0, 0] = -1.5
    data["c2_exp"][1, 0, 0] = -2.5
    data["_normalized_mask"] = [True, False]

    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(data, repairs_applied)

    assert modified is True
    assert data["c2_exp"][0, 0, 0] == -1.5, "normalized matrix must NOT be clamped"
    assert data["c2_exp"][1, 0, 0] == 1e-6, (
        "skipped (never-normalized) matrix must still be clamped"
    )


def test_repair_negative_correlations_clamps_everything_when_unmarked():
    controller = DataQualityController.__new__(DataQualityController)
    data = _data_with_one_skipped_matrix()
    data["c2_exp"] = data["c2_exp"].astype(np.float64)
    data["c2_exp"][0, 0, 0] = -1.5
    data["c2_exp"][1, 0, 0] = -2.5
    # No _normalized_mask key at all -- must behave exactly as before this fix.

    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(data, repairs_applied)

    assert modified is True
    assert data["c2_exp"][0, 0, 0] == 1e-6
    assert data["c2_exp"][1, 0, 0] == 1e-6


def _data_with_one_skipped_matrix_robust(n_t: int = 6) -> dict:
    rng = np.random.default_rng(2)
    normal_matrix = 1.0 + 0.3 * rng.standard_normal((n_t, n_t))
    constant_matrix = np.full((n_t, n_t), 3.0)  # zero IQR -> ROBUST skip branch
    c2 = np.stack([normal_matrix, constant_matrix])
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01, 0.01]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }


def test_normalize_data_tracks_per_matrix_mask_robust():
    """ROBUST method's zero-IQR skip branch must also be tracked per-matrix,
    mirroring the STATISTICAL zero-variance skip branch tested above. The
    design spec covers both normalization methods symmetrically (both gate on
    the same np.finfo-eps skip-guard shape); the plan's first draft only
    exercised STATISTICAL, leaving ROBUST's skip path completely untested."""
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "robust"}}}}
    )
    data = _data_with_one_skipped_matrix_robust()
    result = pipeline._normalize_data(data, {"method": "robust"})
    mask = result.get("_normalized_mask")
    assert mask is not None
    assert list(mask) == [True, False], (
        "matrix 0 (real IQR) must be marked normalized, "
        "matrix 1 (zero IQR, ROBUST skip branch) must not be"
    )


def test_repair_negative_correlations_respects_real_pipeline_mask():
    """End-to-end (design spec Testing item 6): thread the REAL
    _normalize_data output into DataQualityController._repair_negative_correlations,
    not a hand-injected _normalized_mask. The hand-injected-mask test above
    (test_repair_negative_correlations_respects_per_matrix_mask) proves the
    controller reads the mask correctly, but never proves the mask-producing
    pipeline and the mask-consuming controller actually agree on what the
    mask means -- a real seam bug between the two would not be caught there."""
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "statistical"}}}}
    )
    data = _data_with_one_skipped_matrix()
    normalized = pipeline._normalize_data(data, {"method": "statistical"})
    normalized["c2_exp"] = np.asarray(normalized["c2_exp"], dtype=np.float64)

    # Z-scoring matrix 0 legitimately produces negatives (it's mean-centered);
    # confirm the fixture assumption, then inject a definite negative into
    # matrix 1 (the skipped, never-transformed matrix), which would
    # otherwise have no reason to go negative on its own.
    assert np.any(normalized["c2_exp"][0] < 0), (
        "z-score normalization of matrix 0 should legitimately produce a "
        "negative somewhere -- fixture assumption broken"
    )
    normalized["c2_exp"][1, 0, 0] = -2.5

    controller = DataQualityController.__new__(DataQualityController)
    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(normalized, repairs_applied)

    assert modified is True
    assert np.any(normalized["c2_exp"][0] < 0), (
        "matrix 0's real normalization-produced negatives must survive repair"
    )
    assert normalized["c2_exp"][1, 0, 0] == 1e-6, (
        "matrix 1 (never normalized) must still be clamped"
    )
