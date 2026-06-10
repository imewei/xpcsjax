"""Tests for the pointwise scattered-eval seam on the heterodyne evaluator.

Also defines the shared fixtures (_params14, _make_single_chunk_chunked) reused
by tests/parity/test_engine_heterodyne_pointwise_parity.py.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.model_adapter import (
    HeterodynePointEvaluator,
    HeterodynePointwiseEvaluator,
    PointEvaluator,
)


def _params14() -> jnp.ndarray:
    """A physically-plausible 14-param heterodyne vector (registry order).

    Values are only a fixed evaluation point for numeric-equivalence tests, not a
    fit. If the canonical registry layout changes, update this vector; the tests
    assert grid==scattered at this point, so the exact values are not load-bearing.
    """
    return jnp.asarray(
        [1.0e-3, 0.0, 1.0, 1.0e-3, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float64,
    )


def _make_single_chunk_chunked(phi_vals, t, c2_grids):
    """Build a 1-chunk ``StratifiedChunkedData`` for the engine.

    Parameters
    ----------
    phi_vals : list[float]   per-angle phi values
    t : np.ndarray           shared time grid, shape (n_t,)
    c2_grids : list[np.ndarray]  observed (n_t, n_t) grid per angle

    Returns a real ``StratifiedChunkedData`` (chunks=[Chunk(...)], sigma=ones 3-D).
    """
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        Chunk,
        StratifiedChunkedData,
    )

    n_t = len(t)
    n_phi = len(phi_vals)
    phi_list, t1_list, t2_list, g2_list = [], [], [], []
    for a, phi in enumerate(phi_vals):
        for i in range(n_t):
            for j in range(n_t):
                phi_list.append(float(phi))
                t1_list.append(float(t[i]))
                t2_list.append(float(t[j]))
                g2_list.append(float(c2_grids[a][i, j]))
    chunk = Chunk(
        phi=np.asarray(phi_list, dtype=np.float64),
        t1=np.asarray(t1_list, dtype=np.float64),
        t2=np.asarray(t2_list, dtype=np.float64),
        g2=np.asarray(g2_list, dtype=np.float64),
        q=0.0123,
        L=1.0,
        dt=0.1,
    )
    sigma = np.ones((n_phi, n_t, n_t), dtype=np.float64)
    return StratifiedChunkedData(chunks=[chunk], sigma=sigma)


def test_meshgrid_evaluator_does_not_advertise_scattered():
    ev = HeterodynePointEvaluator(analysis_mode="two_component", q=0.01, dt=0.1)
    assert getattr(ev, "supports_scattered", False) is False


def test_pointwise_evaluator_advertises_scattered_and_is_a_point_evaluator():
    ev = HeterodynePointwiseEvaluator(analysis_mode="two_component", q=0.01, dt=0.1)
    assert ev.supports_scattered is True
    assert isinstance(ev, PointEvaluator)  # still satisfies the grid Protocol


def test_scattered_matches_meshgrid_gather_pointwise():
    """eval_scattered at (phi_idx, t1_idx, t2_idx) == meshgrid grid[t1, t2]."""
    q, dt = 0.0123, 0.1
    t = jnp.asarray(np.arange(6, dtype=np.float64))
    phi_unique = jnp.asarray([0.0, 45.0], dtype=np.float64)
    contrast = jnp.asarray([0.30, 0.31], dtype=np.float64)
    offset = jnp.asarray([1.00, 1.02], dtype=np.float64)
    params = _params14()

    mesh = HeterodynePointEvaluator(analysis_mode="two_component", q=q, dt=dt)
    grid0 = np.asarray(mesh.eval_points(params, phi_unique[0], t, t, contrast[0], offset[0])[0])
    grid1 = np.asarray(mesh.eval_points(params, phi_unique[1], t, t, contrast[1], offset[1])[0])

    phi_idx = jnp.asarray([0, 0, 1, 1], dtype=jnp.int64)
    t1_idx = jnp.asarray([1, 4, 2, 5], dtype=jnp.int64)
    t2_idx = jnp.asarray([3, 0, 5, 1], dtype=jnp.int64)

    point = HeterodynePointwiseEvaluator(analysis_mode="two_component", q=q, dt=dt)
    got = np.asarray(
        point.eval_scattered(params, phi_unique, t, phi_idx, t1_idx, t2_idx, contrast, offset)
    )
    expected = np.asarray(
        [grid0[1, 3], grid0[4, 0], grid1[2, 5], grid1[5, 1]], dtype=np.float64
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_engine_takes_scattered_branch_and_matches_grid():
    """The scattered branch (a) is actually exercised and (b) equals the grid path."""
    from xpcsjax.optimization.nlsq.strategies.residual_jit import (
        StratifiedResidualFunctionJIT,
    )

    q, dt = 0.0123, 0.1
    t = np.arange(5, dtype=np.float64)
    phi_vals = [0.0, 30.0]
    n_phi = len(phi_vals)
    params = _params14()
    mesh = HeterodynePointEvaluator(analysis_mode="two_component", q=q, dt=dt)
    grids = [
        np.asarray(mesh.eval_points(params, p, jnp.asarray(t), jnp.asarray(t), 0.3, 1.0)[0])
        for p in phi_vals
    ]
    chunked = _make_single_chunk_chunked(phi_vals, t, grids)

    common = dict(
        stratified_data=chunked,
        per_angle_scaling=True,
        physical_param_names=["dummy"] * 14,
        fixed_contrast_per_angle=None,
        fixed_offset_per_angle=None,
    )
    # [contrast(n_phi) | offset(n_phi) | physical(14)]
    pvec = jnp.asarray(
        [*([0.30] * n_phi), *([1.00] * n_phi), *list(np.asarray(params))],
        dtype=jnp.float64,
    )

    class _SpyEval(HeterodynePointwiseEvaluator):
        calls = {"n": 0}

        def eval_scattered(self, *a, **k):
            type(self).calls["n"] += 1
            return super().eval_scattered(*a, **k)

    grid_engine = StratifiedResidualFunctionJIT(
        **common, evaluator=HeterodynePointEvaluator(analysis_mode="two_component", q=q, dt=dt)
    )
    spy = _SpyEval(analysis_mode="two_component", q=q, dt=dt)
    point_engine = StratifiedResidualFunctionJIT(**common, evaluator=spy)

    r_grid = np.asarray(grid_engine(pvec))
    r_point = np.asarray(point_engine(pvec))

    assert _SpyEval.calls["n"] >= 1  # scattered branch was actually used
    np.testing.assert_allclose(r_point, r_grid, rtol=1e-10, atol=1e-14)
