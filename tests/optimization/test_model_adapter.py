"""Tests for the model-agnostic ``PointEvaluator`` adapter.

Phase 1.1 introduces a thin ``PointEvaluator`` Protocol that decouples the
stratification engine from a specific physics kernel. ``HomodynePointEvaluator``
is the homodyne (``laminar_flow``) adapter; it must be a byte-for-byte pass-through
to the real 9-arg ``compute_g2_scaled`` so that threading it through
``StratifiedResidualFunctionJIT`` is behavior-preserving.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.model_adapter import (
    HomodynePointEvaluator,
    PointEvaluator,
)


def test_homodyne_evaluator_matches_compute_g2_scaled():
    """``HomodynePointEvaluator.eval_points`` must equal the raw kernel exactly."""
    from xpcsjax.core.physics_nlsq import compute_g2_scaled

    q, L, dt = 0.0237, 2_000_000.0, 0.1
    ev = HomodynePointEvaluator(analysis_mode="laminar_flow", q=q, L=L, dt=dt)
    params = jnp.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    t1 = jnp.array([1.0, 2.0, 3.0])
    t2 = jnp.array([2.0, 3.0, 4.0])
    phi = jnp.array([0.0, 45.0, 90.0])
    contrast = jnp.array([0.3, 0.3, 0.3])
    offset = jnp.array([1.0, 1.0, 1.0])

    got = np.asarray(ev.eval_points(params, phi, t1, t2, contrast, offset))
    want = np.asarray(
        compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)
    )
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_homodyne_evaluator_is_a_point_evaluator():
    """The adapter satisfies the runtime-checkable Protocol."""
    ev = HomodynePointEvaluator(analysis_mode="laminar_flow", q=0.0237, L=2e6, dt=0.1)
    assert isinstance(ev, PointEvaluator)
