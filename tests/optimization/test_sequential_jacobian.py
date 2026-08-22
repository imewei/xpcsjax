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

from xpcsjax.optimization.nlsq.strategies.sequential import (
    _jax_jacobian,
    optimize_per_angle_sequential,
)


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


def test_all_physical_fixed_hits_zero_free_vector_fast_path() -> None:
    """Every parameter fixed (issue #58 follow-up coverage gap).

    Not reachable through ``NLSQWrapper``/``fit_nlsq_jax`` in practice, since
    the public wrapper always keeps at least the scaling parameters free --
    but the lower-level strategy API accepts a fully-degenerate bounds pair
    and must not crash. `strip_fixed_parameters` reduces `initial_params` to
    a length-0 free vector, and `optimize_single_angle`'s own zero-length
    fast path (``strategies/sequential.py``, "every parameter fixed" branch)
    evaluates `residual_func` directly on that empty vector WITHOUT ever
    invoking NLSQ's solver/JIT trace -- exercising the `has_fixed` wrapper's
    `restore_by_mask_jax` call on a length-0 concrete array, not a tracer.
    """

    def residual(
        params: np.ndarray, phi: np.ndarray, t1: np.ndarray, t2: np.ndarray, g2: np.ndarray
    ):
        # Trivial JAX-safe residual: params is always the fully-restored
        # 2-length vector here regardless of what free-length slice this
        # closure's wrapper was called with.
        return jnp.asarray(g2 - (params[0] + params[1] * t1))

    n_points = 12  # sequential.py enforces a min_points_per_angle floor of 10
    phi = np.zeros(n_points, dtype=np.float64)
    t1 = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    t2 = np.zeros(n_points, dtype=np.float64)
    g2 = 2.0 + 3.0 * t1

    fixed_values = np.array([2.0, 3.0], dtype=np.float64)
    bounds = (fixed_values.copy(), fixed_values.copy())  # lower == upper -> all fixed

    result = optimize_per_angle_sequential(
        phi=phi,
        t1=t1,
        t2=t2,
        g2_exp=g2,
        residual_func=residual,
        initial_params=fixed_values.copy(),
        bounds=bounds,
    )

    np.testing.assert_allclose(result.combined_parameters, fixed_values, atol=1e-12)
    # The per-angle covariance itself IS exactly zero at this level (nothing
    # was estimated) -- verified directly on the raw per-angle result, which
    # combine_angle_results' own "dead"-column inverse-variance fallback
    # then perturbs to a tiny placeholder (well-known, already handled at
    # the wrapper.py layer via its own post-solve re-zero step, exercised by
    # test_fixed_parameter_survives_sequential_fit). This low-level
    # strategy-level API does not itself guarantee exact zero, only "did not
    # crash and reports finite, small values" -- so assert that contract,
    # not the wrapper's stronger one.
    assert np.all(result.per_angle_results[0]["covariance"] == 0.0)
    assert np.all(np.isfinite(result.combined_covariance))
    assert np.all(np.abs(result.combined_covariance) < 1e-6)
