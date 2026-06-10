"""Parity guard for the column-blocked covariance Jacobian.

The heterodyne stratified-LS covariance previously called ``jax.jacfwd`` on the
joint residual, which pushes all ``n_params`` basis tangents through the
pointwise kernel at once — every ``(N,)`` intermediate materialised at width
``n_params``. At >=1M points that ``n_params``-wide tangent is the dominant
post-solve memory spike. ``_chunked_jacfwd_dense`` computes the SAME Jacobian in
small column blocks (a vmapped JVP per block), capping the tangent width.

This module pins that the chunked Jacobian is numerically identical to
``jax.jacfwd`` (it only touches post-solve covariance, never the fit
trajectory). If these drift, the covariance contract is broken.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import _chunked_jacfwd_dense


def _nonlinear_residual(p: np.ndarray) -> jnp.ndarray:
    """A nonlinear R^n_in -> R^n_out map that exercises mixed partials.

    Shaped like a real residual: more outputs than inputs, every output depends
    on every parameter through transcendental terms so the Jacobian is dense.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    n_out = 137
    k = jnp.arange(n_out, dtype=jnp.float64)[:, None]  # (n_out, 1)
    # (n_out, n_in): sin(k * p_j) * p_j^2 + exp(-p_j / (k+1))
    terms = jnp.sin(k * p[None, :]) * p[None, :] ** 2 + jnp.exp(-p[None, :] / (k + 1.0))
    return jnp.sum(terms, axis=1)  # (n_out,)


@pytest.mark.parametrize("col_block", [1, 2, 4, 8, 16])
def test_chunked_jacfwd_matches_jacfwd(col_block):
    """Chunked Jacobian == jax.jacfwd for every column-block width."""
    x = np.array([0.7, -1.3, 2.1, 0.05, -0.9, 1.7, 3.2], dtype=np.float64)

    reference = np.asarray(jax.jacfwd(_nonlinear_residual)(x), dtype=np.float64)
    chunked = _chunked_jacfwd_dense(_nonlinear_residual, x, col_block=col_block)

    assert chunked.shape == reference.shape == (137, x.size)
    # Forward-mode JVP is exact; the only possible delta is XLA fusion noise
    # across batch widths, which is <= a few ULP. Pin tightly.
    np.testing.assert_allclose(chunked, reference, rtol=1e-12, atol=1e-12)


def test_chunked_jacfwd_works_through_jit():
    """The residual is jax.jit-wrapped in production; the helper must handle it."""
    x = np.array([1.1, 0.3, -2.0, 0.8], dtype=np.float64)
    jitted = jax.jit(_nonlinear_residual)

    reference = np.asarray(jax.jacfwd(jitted)(x), dtype=np.float64)
    chunked = _chunked_jacfwd_dense(jitted, x, col_block=2)

    np.testing.assert_allclose(chunked, reference, rtol=1e-12, atol=1e-12)


def test_chunked_jacfwd_default_block_is_identical():
    """The production default block (no col_block kwarg) is also exact."""
    x = np.linspace(-1.0, 1.0, 11, dtype=np.float64)

    reference = np.asarray(jax.jacfwd(_nonlinear_residual)(x), dtype=np.float64)
    chunked = _chunked_jacfwd_dense(_nonlinear_residual, x)

    np.testing.assert_allclose(chunked, reference, rtol=1e-12, atol=1e-12)
