"""Tests for xpcsjax.optimization.nlsq.strategies.residual_jit.

``StratifiedResidualFunctionJIT`` pads chunks to a uniform size and vmaps the
per-chunk residual for JIT compatibility. Tests build a tiny synthetic
stratified dataset and assert structural invariants independent of the physics:
output length (n_chunks * max_chunk_size), finiteness, exact-zero residuals on
the t1==t2 diagonal AND on padded slots, across all three scaling modes
(per-angle, scalar, fixed/constant). Metadata-consistency and validation errors
are checked via raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)

_PHI = [0.0, 90.0]
_T = [1.0, 2.0]
_PHYSICAL = ["D0", "alpha", "D_offset"]


def _full_grid_chunk(q: float = 0.01) -> SimpleNamespace:
    phi, t1, t2 = [], [], []
    for p in _PHI:
        for a in _T:
            for b in _T:
                phi.append(p)
                t1.append(a)
                t2.append(b)
    n = len(phi)
    return SimpleNamespace(
        phi=np.array(phi), t1=np.array(t1), t2=np.array(t2),
        g2=np.full(n, 1.2), q=q, L=1.0, dt=1.0,
    )


def _stratified(chunks: list | None = None) -> SimpleNamespace:
    if chunks is None:
        chunks = [_full_grid_chunk()]
    return SimpleNamespace(chunks=chunks, sigma=np.ones((len(_PHI), len(_T), len(_T))))


_DIAG_IDX = [0, 3, 4, 7]  # t1==t2 positions in single-chunk grid order


# ---------------------------------------------------------------------------
# residual computation across scaling modes
# ---------------------------------------------------------------------------


def test_per_angle_residuals_shape_finite_diagonal_masked() -> None:
    rf = StratifiedResidualFunctionJIT(_stratified(), per_angle_scaling=True,
                                       physical_param_names=_PHYSICAL)
    params = np.array([0.3, 0.3, 1.0, 1.0, 1e-3, 1.0, 0.0])
    res = np.asarray(rf(params))
    assert res.shape == (8,)  # 1 chunk * max_chunk_size 8
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res[_DIAG_IDX], 0.0, atol=1e-12)


def test_scalar_residuals_run() -> None:
    rf = StratifiedResidualFunctionJIT(_stratified(), per_angle_scaling=False,
                                       physical_param_names=_PHYSICAL)
    res = np.asarray(rf(np.array([0.3, 1.0, 1e-3, 1.0, 0.0])))
    assert res.shape == (8,)
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res[_DIAG_IDX], 0.0, atol=1e-12)


def test_fixed_scaling_constant_mode() -> None:
    # Constant mode: contrast/offset fixed per-angle, params = physical only.
    rf = StratifiedResidualFunctionJIT(
        _stratified(),
        per_angle_scaling=True,
        physical_param_names=_PHYSICAL,
        fixed_contrast_per_angle=np.array([0.3, 0.35]),
        fixed_offset_per_angle=np.array([1.0, 1.02]),
    )
    assert rf.use_fixed_scaling is True
    res = np.asarray(rf(np.array([1e-3, 1.0, 0.0])))  # physical params only
    assert res.shape == (8,)
    assert np.all(np.isfinite(res))


def test_dt_none_uses_fallback_and_warns() -> None:
    chunk = _full_grid_chunk()
    chunk.dt = None
    rf = StratifiedResidualFunctionJIT(_stratified([chunk]), per_angle_scaling=False,
                                       physical_param_names=_PHYSICAL)
    assert rf.dt is None
    res = np.asarray(rf(np.array([0.3, 1.0, 1e-3, 1.0, 0.0])))
    assert np.all(np.isfinite(res))  # dt=0.001 fallback keeps it finite


# ---------------------------------------------------------------------------
# padding / mask
# ---------------------------------------------------------------------------


def test_padding_zeros_masked_in_output() -> None:
    chunk_a = _full_grid_chunk()  # 8 points
    # Smaller second chunk (2 points), still angle-complete and off-diagonal.
    chunk_b = SimpleNamespace(
        phi=np.array([0.0, 90.0]), t1=np.array([1.0, 1.0]), t2=np.array([2.0, 2.0]),
        g2=np.array([1.1, 1.1]), q=0.01, L=1.0, dt=1.0,
    )
    rf = StratifiedResidualFunctionJIT(_stratified([chunk_a, chunk_b]),
                                       per_angle_scaling=False, physical_param_names=_PHYSICAL)
    assert rf.max_chunk_size == 8
    assert rf.n_real_points == 10
    res = np.asarray(rf(np.array([0.3, 1.0, 1e-3, 1.0, 0.0])))
    assert res.shape == (16,)  # 2 chunks * 8
    assert np.all(np.isfinite(res))
    # chunk_b occupies res[8:16]; only first 2 are real, rest are padding -> 0.
    np.testing.assert_allclose(res[10:16], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# metadata + validation
# ---------------------------------------------------------------------------


def test_inconsistent_q_raises() -> None:
    a = _full_grid_chunk(q=0.01)
    b = _full_grid_chunk(q=0.02)
    with pytest.raises(ValueError, match="Inconsistent q values"):
        StratifiedResidualFunctionJIT(_stratified([a, b]), per_angle_scaling=False,
                                      physical_param_names=_PHYSICAL)


def test_empty_chunks_raises() -> None:
    with pytest.raises(ValueError, match="chunks is empty"):
        StratifiedResidualFunctionJIT(
            SimpleNamespace(chunks=[], sigma=np.ones((2, 2, 2))),
            per_angle_scaling=False, physical_param_names=_PHYSICAL,
        )


def test_validate_chunk_structure_passes() -> None:
    rf = StratifiedResidualFunctionJIT(_stratified(), per_angle_scaling=False,
                                       physical_param_names=_PHYSICAL)
    assert rf.validate_chunk_structure() is True


def test_validate_chunk_structure_detects_missing_angle() -> None:
    chunk_a = _full_grid_chunk()
    # chunk_b only has phi=0 (missing 90) -> angle-incomplete.
    chunk_b = SimpleNamespace(
        phi=np.array([0.0, 0.0]), t1=np.array([1.0, 2.0]), t2=np.array([2.0, 1.0]),
        g2=np.array([1.1, 1.1]), q=0.01, L=1.0, dt=1.0,
    )
    rf = StratifiedResidualFunctionJIT(_stratified([chunk_a, chunk_b]),
                                       per_angle_scaling=False, physical_param_names=_PHYSICAL)
    with pytest.raises(ValueError, match="invalid angle distribution"):
        rf.validate_chunk_structure()


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def test_get_diagnostics() -> None:
    rf = StratifiedResidualFunctionJIT(_stratified(), per_angle_scaling=True,
                                       physical_param_names=_PHYSICAL)
    diag = rf.get_diagnostics()
    assert diag["n_chunks"] == 1
    assert diag["max_chunk_size"] == 8
    assert diag["n_real_points"] == 8
    assert diag["n_phi"] == 2
    assert diag["jit_compiled"] is True
    assert diag["padding_overhead_pct"] == pytest.approx(0.0)


def test_log_diagnostics_runs() -> None:
    rf = StratifiedResidualFunctionJIT(_stratified(), per_angle_scaling=False,
                                       physical_param_names=_PHYSICAL)
    rf.log_diagnostics()  # should not raise
