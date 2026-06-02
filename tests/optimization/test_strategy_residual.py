"""Tests for xpcsjax.optimization.nlsq.strategies.residual.

``StratifiedResidualFunction`` computes weighted residuals over angle-stratified
chunks via real JAX g2 evaluation. Tests build a tiny synthetic stratified
dataset (full phi x t1 x t2 grid in one chunk) and assert structural invariants
that hold regardless of the physics: correct residual length, finiteness,
exact-zero residuals on the t1==t2 diagonal (autocorrelation mask), and
``jax_residual`` agreeing with ``__call__``. Construction-time validation
(empty chunks, inconsistent angles/shapes) is checked via raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.strategies.residual import (
    StratifiedResidualFunction,
    create_stratified_residual_function,
)

_PHI = [0.0, 90.0]
_T = [1.0, 2.0]
_PHYSICAL = ["D0", "alpha", "D_offset"]


def _full_grid_chunk() -> SimpleNamespace:
    """One chunk holding the full phi x t1 x t2 cartesian grid (all angles)."""
    phi, t1, t2 = [], [], []
    for p in _PHI:
        for a in _T:
            for b in _T:
                phi.append(p)
                t1.append(a)
                t2.append(b)
    n = len(phi)
    return SimpleNamespace(
        phi=np.array(phi),
        t1=np.array(t1),
        t2=np.array(t2),
        g2=np.full(n, 1.2),
        q=0.01,
        L=1.0,
        dt=1.0,
    )


def _stratified(sigma: np.ndarray | None = None) -> SimpleNamespace:
    chunk = _full_grid_chunk()
    if sigma is None:
        sigma = np.ones((len(_PHI), len(_T), len(_T)))  # (n_phi, n_t1, n_t2)
    return SimpleNamespace(chunks=[chunk], sigma=sigma)


# Indices on the t1==t2 diagonal given the grid construction order.
_DIAG_IDX = [0, 3, 4, 7]


# ---------------------------------------------------------------------------
# residual computation
# ---------------------------------------------------------------------------


def test_per_angle_residuals_shape_finite_and_diagonal_masked() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=True,
                                    physical_param_names=_PHYSICAL)
    # [c0, c1, o0, o1, D0, alpha, D_offset]
    params = np.array([0.3, 0.3, 1.0, 1.0, 1e-3, 1.0, 0.0])
    res = rf(params)
    assert res.shape == (8,)
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res[_DIAG_IDX], 0.0, atol=1e-12)  # diagonal masked


def test_scalar_residuals_run() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    params = np.array([0.3, 1.0, 1e-3, 1.0, 0.0])  # [contrast, offset, *physical]
    res = rf(params)
    assert res.shape == (8,)
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res[_DIAG_IDX], 0.0, atol=1e-12)


def test_jax_residual_matches_call() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    params = np.array([0.3, 1.0, 1e-3, 1.0, 0.0])
    import jax.numpy as jnp

    np.testing.assert_allclose(np.asarray(rf.jax_residual(jnp.asarray(params))), rf(params))


def test_zero_sigma_point_is_masked_no_nan() -> None:
    # An off-diagonal point with sigma=0 must produce a finite (masked) residual,
    # not NaN/Inf from division by zero.
    sigma = np.ones((len(_PHI), len(_T), len(_T)))
    sigma[0, 0, 1] = 0.0  # flat index 1 -> point (phi=0, t1=1, t2=2), off-diagonal
    rf = StratifiedResidualFunction(_stratified(sigma), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    res = rf(np.array([0.3, 1.0, 1e-3, 1.0, 0.0]))
    assert np.all(np.isfinite(res))
    assert res[1] == pytest.approx(0.0, abs=1e-12)  # masked by zero sigma


def test_residuals_are_deterministic() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=True,
                                    physical_param_names=_PHYSICAL)
    params = np.array([0.3, 0.4, 1.0, 1.1, 1e-3, 1.0, 0.0])
    np.testing.assert_array_equal(rf(params), rf(params))


# ---------------------------------------------------------------------------
# diagnostics / validation (cached post-init paths)
# ---------------------------------------------------------------------------


def test_get_diagnostics_uses_cached_values() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=True,
                                    physical_param_names=_PHYSICAL)
    diag = rf.get_diagnostics()
    assert diag["n_chunks"] == 1
    assert diag["n_total_points"] == 8
    assert diag["n_angles"] == 2
    assert diag["per_angle_scaling"] is True
    assert diag["chunk_sizes"] == [8]


def test_log_diagnostics_runs() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    rf.log_diagnostics()  # should not raise


def test_validate_chunk_structure_cached_true() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    # chunks were freed after inline validation -> returns cached True.
    assert rf.validate_chunk_structure() is True


def test_deprecated_paths_raise() -> None:
    rf = StratifiedResidualFunction(_stratified(), per_angle_scaling=False,
                                    physical_param_names=_PHYSICAL)
    import jax.numpy as jnp

    with pytest.raises(RuntimeError, match="_call_jax_chunked is unavailable"):
        rf._call_jax_chunked(jnp.zeros(5))


# ---------------------------------------------------------------------------
# construction-time validation
# ---------------------------------------------------------------------------


def test_empty_chunks_raises() -> None:
    with pytest.raises(ValueError, match="chunks is empty"):
        StratifiedResidualFunction(
            SimpleNamespace(chunks=[], sigma=np.ones((2, 2, 2))),
            per_angle_scaling=False,
            physical_param_names=_PHYSICAL,
        )


def test_inconsistent_angles_across_chunks_raises() -> None:
    good = _full_grid_chunk()
    # Second chunk missing the 90-degree angle entirely.
    bad = SimpleNamespace(
        phi=np.zeros(2), t1=np.array(_T), t2=np.array(_T),
        g2=np.ones(2), q=0.01, L=1.0, dt=1.0,
    )
    data = SimpleNamespace(chunks=[good, bad], sigma=np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="inconsistent angles"):
        StratifiedResidualFunction(data, per_angle_scaling=False, physical_param_names=_PHYSICAL)


def test_inconsistent_shapes_raises() -> None:
    chunk = _full_grid_chunk()
    chunk.g2 = np.ones(3)  # mismatched length vs phi/t1/t2 (8)
    data = SimpleNamespace(chunks=[chunk], sigma=np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="inconsistent array shapes"):
        StratifiedResidualFunction(data, per_angle_scaling=False, physical_param_names=_PHYSICAL)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def test_factory_with_validation() -> None:
    rf = create_stratified_residual_function(
        _stratified(), per_angle_scaling=True, physical_param_names=_PHYSICAL, validate=True
    )
    assert isinstance(rf, StratifiedResidualFunction)
    assert rf.n_phi == 2


def test_factory_without_validation() -> None:
    rf = create_stratified_residual_function(
        _stratified(), per_angle_scaling=False, physical_param_names=_PHYSICAL, validate=False
    )
    assert isinstance(rf, StratifiedResidualFunction)
