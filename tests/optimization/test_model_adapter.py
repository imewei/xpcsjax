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
    want = np.asarray(compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_homodyne_evaluator_is_a_point_evaluator():
    """The adapter satisfies the runtime-checkable Protocol."""
    ev = HomodynePointEvaluator(analysis_mode="laminar_flow", q=0.0237, L=2e6, dt=0.1)
    assert isinstance(ev, PointEvaluator)


def test_heterodyne_evaluator_returns_per_angle_meshgrid():
    """``HeterodynePointEvaluator`` must plug into the stratification ENGINE.

    The shared engine ``StratifiedResidualFunctionJIT`` calls the evaluator with
    a SCALAR ``phi``, the FULL ``t1``/``t2`` grids, and a SCALAR per-angle
    ``contrast``/``offset`` (sliced from the optimizer's param vector), then
    ``squeeze``\\s axis 0 and gathers from a ``(n_phi, n_t, n_t)`` grid. So
    ``eval_points`` must (a) return the full per-angle ``(n_t, n_t)`` meshgrid
    of the heterodyne MESHGRID kernel and (b) USE the supplied scaling. This test
    mirrors the engine's exact call convention.
    """
    import jax

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

    model, _c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    params = jnp.asarray(model.get_params(), dtype=jnp.float64)

    t = jnp.asarray(model.t, dtype=jnp.float64)
    n_t = t.shape[0]
    phi_unique = jnp.asarray(np.unique(np.asarray(phi)), dtype=jnp.float64)
    n_phi = phi_unique.shape[0]

    contrast_arr = jnp.asarray([0.18, 0.22, 0.25, 0.30], dtype=jnp.float64)
    offset_arr = jnp.asarray([1.05, 1.10, 1.02, 1.08], dtype=jnp.float64)

    ev = HeterodynePointEvaluator(
        analysis_mode="two_component", q=float(model.q), dt=float(model.dt)
    )

    # (3) Single-angle: eval_points returns (1, n_t, n_t) (length-1 phi axis,
    #     matching the homodyne adapter); after squeeze(axis=0) it must equal the
    #     raw meshgrid kernel's (n_t, n_t) exactly.
    phi_scalar = phi_unique[1]
    contrast_scalar = contrast_arr[1]
    offset_scalar = offset_arr[1]
    got_single = ev.eval_points(params, phi_scalar, t, t, contrast_scalar, offset_scalar)
    want_single = compute_c2_heterodyne(
        params,
        t,
        float(model.q),
        float(model.dt),
        phi_scalar,
        contrast_scalar,
        offset_scalar,
    )
    assert got_single.shape == (1, n_t, n_t)
    assert want_single.shape == (n_t, n_t)
    np.testing.assert_allclose(
        np.asarray(jnp.squeeze(got_single, axis=0)),
        np.asarray(want_single),
        rtol=1e-12,
        atol=0.0,
    )

    # (4) Engine-convention test: replicate the engine's exact vmap call pattern.
    def engine_call(ph, c, o):
        return jnp.squeeze(ev.eval_points(params, jnp.asarray(ph), t, t, c, o), axis=0)

    grid = jax.vmap(engine_call, in_axes=(0, 0, 0))(phi_unique, contrast_arr, offset_arr)
    assert grid.shape == (n_phi, n_t, n_t)

    # ...and it matches a per-angle loop over the meshgrid kernel.
    want_grid = np.stack(
        [
            np.asarray(
                compute_c2_heterodyne(
                    params,
                    t,
                    float(model.q),
                    float(model.dt),
                    phi_unique[i],
                    contrast_arr[i],
                    offset_arr[i],
                )
            )
            for i in range(n_phi)
        ]
    )
    np.testing.assert_allclose(np.asarray(grid), want_grid, rtol=1e-12, atol=0.0)

    # (5) The optimizer drives scaling: two different contrasts -> different output.
    out_a = np.asarray(ev.eval_points(params, phi_scalar, t, t, jnp.asarray(0.10), offset_scalar))
    out_b = np.asarray(ev.eval_points(params, phi_scalar, t, t, jnp.asarray(0.40), offset_scalar))
    assert not np.allclose(out_a, out_b), "contrast must affect the output"


def test_heterodyne_evaluator_is_a_point_evaluator():
    """The heterodyne adapter satisfies the runtime-checkable Protocol."""
    ev = HeterodynePointEvaluator(analysis_mode="two_component", q=0.0054, dt=0.1)
    assert isinstance(ev, PointEvaluator)
