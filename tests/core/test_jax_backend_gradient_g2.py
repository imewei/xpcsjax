"""Regression test for gradient_g2 (audit C12).

`gradient_g2` was wired as `grad(compute_g2_scaled)`, but compute_g2_scaled
returns an array (n_phi, n_t, n_t); `jax.grad` requires scalar output, so any
call raised `TypeError: Gradient only defined for scalar-output functions`.
The correct AD of an array-valued model w.r.t. params is the Jacobian.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.jax_backend import gradient_g2


def test_gradient_g2_returns_finite_jacobian():
    params = jnp.array([100.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    t1 = jnp.array([0.0])
    t2 = jnp.array([1.0])
    phi = jnp.array([0.0])

    jac = np.asarray(gradient_g2(params, t1, t2, phi, 0.01, 1e6, 0.8, 1.0, 0.001))

    # Jacobian of g2 output w.r.t. the parameter vector: last axis == n_params.
    assert jac.shape[-1] == params.shape[0]
    assert np.all(np.isfinite(jac))
