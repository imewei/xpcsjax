"""Regression tests for the degenerate 1x1-grid branch in physics_nlsq.py.

The per-point stratified residual path (xpcsjax/optimization/nlsq/wrapper.py's
vmap-based model function) calls compute_g2_scaled with single-element 1D
t1/t2 arrays (e.g. ``t1=jnp.array([t1_val])``), never a 2D meshgrid. Both
degenerate-grid branches must handle that exact shape without crashing, and
must agree with the general grid-based path for the same physical (t1, t2).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.core.physics_nlsq import (
    _compute_g1_diffusion_meshgrid,
    _compute_g1_shear_meshgrid,
)

DT = 0.1
Q = 0.005
WQ2H_DT = 0.5 * Q**2 * DT


def _grid_reference_diffusion(params, t1_val, t2_val):
    n = int(round(t2_val / DT)) + 1
    t_grid = jnp.asarray(np.linspace(0.0, DT * (n - 1), n))
    i, j = int(round(t1_val / DT)), int(round(t2_val / DT))
    g1_grid = _compute_g1_diffusion_meshgrid(params, t_grid, t_grid, WQ2H_DT, DT)
    return g1_grid[i, j]


@pytest.mark.parametrize("t1_val,t2_val", [(0.5, 0.8), (0.2, 1.4), (0.3, 0.3)])
def test_diffusion_degenerate_matches_grid(t1_val, t2_val):
    params = jnp.array([1000.0, -0.8, 0.0])  # D0, alpha, D_offset
    t1 = jnp.array([t1_val])
    t2 = jnp.array([t2_val])

    g1_degenerate = _compute_g1_diffusion_meshgrid(params, t1, t2, WQ2H_DT, DT)
    assert g1_degenerate.shape == (1, 1)
    assert jnp.all(jnp.isfinite(g1_degenerate))

    g1_grid = _grid_reference_diffusion(params, t1_val, t2_val)
    assert jnp.allclose(g1_degenerate[0, 0], g1_grid, rtol=1e-3, atol=1e-3)


def _grid_reference_shear(params, phi, sinc_prefactor, t1_val, t2_val):
    n = int(round(t2_val / DT)) + 1
    t_grid = jnp.asarray(np.linspace(0.0, DT * (n - 1), n))
    i, j = int(round(t1_val / DT)), int(round(t2_val / DT))
    g1_grid = _compute_g1_shear_meshgrid(params, t_grid, t_grid, phi, sinc_prefactor, DT)
    return g1_grid[0, i, j]


@pytest.mark.parametrize("t1_val,t2_val", [(0.5, 0.8), (0.2, 1.4), (0.3, 0.3)])
def test_shear_degenerate_matches_grid(t1_val, t2_val):
    # D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0
    params = jnp.array([1000.0, -0.8, 0.0, 0.05, 1.0, 0.0, 30.0])
    L = 2e6
    sinc_prefactor = 0.5 / np.pi * Q * L * DT
    phi = jnp.array([10.0])
    t1 = jnp.array([t1_val])
    t2 = jnp.array([t2_val])

    g1_degenerate = _compute_g1_shear_meshgrid(params, t1, t2, phi, sinc_prefactor, DT)
    assert g1_degenerate.shape == (1, 1, 1)
    assert jnp.all(jnp.isfinite(g1_degenerate))

    g1_grid = _grid_reference_shear(params, phi, sinc_prefactor, t1_val, t2_val)
    assert jnp.allclose(g1_degenerate[0, 0, 0], g1_grid, rtol=1e-3, atol=1e-3)
