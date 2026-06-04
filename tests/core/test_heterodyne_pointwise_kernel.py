"""Pointwise heterodyne kernel must exactly match the meshgrid path."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.core.heterodyne_jax_backend import (
    compute_c2_heterodyne,  # meshgrid (N,N)
    compute_c2_heterodyne_pointwise,  # NEW pointwise
)


def _params():
    # [D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample, D_offset_sample,
    #  v0, beta, v_offset, f0, f1, f2, f3, phi0]
    return jnp.asarray(
        [1000.0, 0.9, 0.0, 1500.0, 1.0, 0.0, 50.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float64,
    )


@pytest.mark.parametrize("phi_angle", [0.0, 5.0, -7.5])
def test_pointwise_matches_meshgrid(phi_angle):
    n_t = 12
    t = jnp.arange(1, n_t + 1, dtype=jnp.float64) * 0.1
    q, dt = 0.0054, 0.1
    contrast, offset = 0.18, 1.05
    p = _params()

    mesh = np.asarray(compute_c2_heterodyne(p, t, q, dt, phi_angle, contrast, offset))

    ii, jj = np.meshgrid(np.arange(n_t), np.arange(n_t), indexing="ij")
    phi_unique = jnp.asarray([phi_angle], dtype=jnp.float64)
    t1_idx = ii.reshape(-1).astype(np.int32)
    t2_idx = jj.reshape(-1).astype(np.int32)

    pw = np.asarray(
        compute_c2_heterodyne_pointwise(
            p,
            t,
            q,
            dt,
            phi_unique=phi_unique,
            phi_idx=jnp.asarray(t1_idx * 0),  # all zeros -> single phi
            t1_idx=jnp.asarray(t1_idx),
            t2_idx=jnp.asarray(t2_idx),
            contrast=jnp.asarray([contrast], dtype=jnp.float64),
            offset=jnp.asarray([offset], dtype=jnp.float64),
        )
    ).reshape(n_t, n_t)

    max_diff = np.max(np.abs(mesh - pw))
    assert max_diff < 1e-10, f"phi={phi_angle}: max diff = {max_diff:.3e}"


def test_pointwise_multi_phi_gather():
    """Two phi angles, scattered points; each must match its meshgrid value."""
    n_t = 6
    t = jnp.arange(1, n_t + 1, dtype=jnp.float64) * 0.1
    q, dt = 0.0054, 0.1
    p = _params()
    phis = [0.0, 8.0]
    contrasts = [0.18, 0.22]
    offsets = [1.05, 1.10]

    meshes = [
        np.asarray(compute_c2_heterodyne(p, t, q, dt, ph, c, o))
        for ph, c, o in zip(phis, contrasts, offsets, strict=True)
    ]

    # Build scattered points across both angles
    rows = []
    for a in (0, 1):
        for i in range(n_t):
            for j in range(n_t):
                rows.append((a, i, j))
    a_arr = np.array([r[0] for r in rows], dtype=np.int32)
    i_arr = np.array([r[1] for r in rows], dtype=np.int32)
    j_arr = np.array([r[2] for r in rows], dtype=np.int32)

    pw = np.asarray(
        compute_c2_heterodyne_pointwise(
            p,
            t,
            q,
            dt,
            phi_unique=jnp.asarray(phis, dtype=jnp.float64),
            phi_idx=jnp.asarray(a_arr),
            t1_idx=jnp.asarray(i_arr),
            t2_idx=jnp.asarray(j_arr),
            contrast=jnp.asarray(contrasts, dtype=jnp.float64),
            offset=jnp.asarray(offsets, dtype=jnp.float64),
        )
    )
    expected = np.array([meshes[a][i, j] for a, i, j in rows])
    assert np.max(np.abs(pw - expected)) < 1e-10
