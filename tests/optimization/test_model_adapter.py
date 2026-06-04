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
    HeterodynePointEvaluator,
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


def test_heterodyne_evaluator_matches_pointwise_kernel():
    """``HeterodynePointEvaluator.eval_points`` must equal the raw pointwise kernel.

    The evaluator receives *value* arrays ``(phi, t1, t2)`` per scattered point
    and bridges them to the index-based ``compute_c2_heterodyne_pointwise``
    kernel. It carries its OWN per-angle ``contrast_arr/offset_arr`` gathered by
    ``phi_idx`` inside the kernel, so the per-point ``contrast``/``offset`` args
    of the Protocol are deliberately ignored.
    """
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne_pointwise

    model, _c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    params = jnp.asarray(model.get_params(), dtype=jnp.float64)

    t = np.asarray(model.t)
    phi_unique = np.unique(np.asarray(phi))  # ascending unique detector angles

    # Fixed per-angle scaling carried by the evaluator (shape (n_phi,)).
    contrast_arr = jnp.asarray([0.18, 0.22, 0.25, 0.30], dtype=jnp.float64)
    offset_arr = jnp.asarray([1.05, 1.10, 1.02, 1.08], dtype=jnp.float64)

    # Pick a chunk of scattered points: (phi_value, t1_value, t2_value) triples,
    # drawn from the model's grids with a NON-monotonic phi subset.
    point_phi_idx = np.array([2, 0, 3, 1, 0, 3], dtype=np.int32)
    point_t1_idx = np.array([0, 5, 11, 3, 7, 2], dtype=np.int32)
    point_t2_idx = np.array([4, 5, 0, 8, 1, 11], dtype=np.int32)

    phi_vals = jnp.asarray(phi_unique[point_phi_idx], dtype=jnp.float64)
    t1_vals = jnp.asarray(t[point_t1_idx], dtype=jnp.float64)
    t2_vals = jnp.asarray(t[point_t2_idx], dtype=jnp.float64)

    ev = HeterodynePointEvaluator(
        analysis_mode="two_component",
        t=t,
        q=float(model.q),
        dt=float(model.dt),
        phi_unique=phi_unique,
        contrast_arr=contrast_arr,
        offset_arr=offset_arr,
    )

    got = np.asarray(
        ev.eval_points(params, phi_vals, t1_vals, t2_vals, contrast=None, offset=None)
    )

    want = np.asarray(
        compute_c2_heterodyne_pointwise(
            params,
            jnp.asarray(t, dtype=jnp.float64),
            float(model.q),
            float(model.dt),
            phi_unique=jnp.asarray(phi_unique, dtype=jnp.float64),
            phi_idx=jnp.asarray(point_phi_idx),
            t1_idx=jnp.asarray(point_t1_idx),
            t2_idx=jnp.asarray(point_t2_idx),
            contrast=contrast_arr,
            offset=offset_arr,
        )
    )
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_heterodyne_evaluator_is_a_point_evaluator():
    """The heterodyne adapter satisfies the runtime-checkable Protocol."""
    ev = HeterodynePointEvaluator(
        analysis_mode="two_component",
        t=np.array([0.1, 0.2, 0.3]),
        q=0.0054,
        dt=0.1,
        phi_unique=np.array([0.0, 45.0]),
        contrast_arr=np.array([0.3, 0.3]),
        offset_arr=np.array([1.0, 1.0]),
    )
    assert isinstance(ev, PointEvaluator)
