"""Regression: ``_jax_jacobian`` works under ``jax.jacfwd``.

The inner ``jax_func`` wrapper used to call ``np.asarray(p)`` on ``p``, but under
``jax.jacfwd`` ``p`` is a tracer — ``np.asarray`` on it raises
``TracerArrayConversionError`` (a ``TypeError`` subclass), which the caller's
broad ``except`` silently swallowed, always returning ``None`` for the
``initial_jacobian_norms`` diagnostic. The fix passes the tracer straight
through. This pins that the Jacobian is now actually computed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.strategies.sequential import _jax_jacobian


def test_jax_jacobian_returns_real_jacobian_under_jacfwd() -> None:
    def residual(p: np.ndarray) -> jnp.ndarray:
        return jnp.asarray([p[0] ** 2, 2.0 * p[1], p[0] * p[1]])

    params = np.array([1.5, 2.0], dtype=np.float64)
    jac = _jax_jacobian(residual, params)

    assert jac.shape == (3, 2)
    assert np.all(np.isfinite(jac))
    # Analytic: d/dp0 = [2p0, 0, p1]; d/dp1 = [0, 2, p0] at (1.5, 2.0).
    expected = np.array([[3.0, 0.0], [0.0, 2.0], [2.0, 1.5]], dtype=np.float64)
    np.testing.assert_allclose(jac, expected, atol=1e-10)
