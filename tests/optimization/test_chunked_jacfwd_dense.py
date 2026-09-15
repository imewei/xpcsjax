"""E1: ``covariance._chunked_jacfwd_dense`` (moved here from
``heterodyne_stratified_ls.py``, now also used by ``strategies/stratified_ls.py``
at the >=1M-point post-solve Jacobian) must be bit/ULP-identical to
``jax.jacfwd`` -- that's the whole point of introducing it (caps live tangent
width without changing the assembled Jacobian).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.covariance import _chunked_jacfwd_dense


def _nonlinear_residual(params: jnp.ndarray) -> jnp.ndarray:
    a, b, c, d = params
    x = jnp.linspace(0.0, 1.0, 37)
    return a * jnp.sin(b * x + c) + d * x**2 - 0.3


def test_matches_jax_jacfwd_exactly():
    x0 = jnp.array([1.3, 2.1, -0.4, 0.7])
    expected = np.asarray(jax.jacfwd(_nonlinear_residual)(x0))
    actual = _chunked_jacfwd_dense(_nonlinear_residual, x0)
    np.testing.assert_array_equal(actual, expected)


def test_matches_for_various_col_block_sizes():
    x0 = jnp.array([0.5, -1.2, 3.3, 0.1, 2.0])

    def fn(p):
        return jnp.cumsum(p) ** 2 + jnp.sin(p)

    expected = np.asarray(jax.jacfwd(fn)(x0))
    for col_block in (1, 2, 3, 5, 8):
        actual = _chunked_jacfwd_dense(fn, x0, col_block=col_block)
        np.testing.assert_array_equal(actual, expected)


def test_empty_params():
    def fn(_p):
        return jnp.zeros(3)

    result = _chunked_jacfwd_dense(fn, jnp.array([]))
    assert result.shape == (0, 0)
